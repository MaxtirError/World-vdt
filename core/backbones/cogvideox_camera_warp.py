from typing import Any, Dict, List, Tuple

import torch
from core.models import CogVideoXCameraWarpTransformer, CameraEncoder3D, CogVideoXWarpEncoder
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.loaders import CogVideoXLoraLoaderMixin
from pathlib import Path
logger = get_logger(LOG_NAME, LOG_LEVEL)


class CogVideoXCameraWarpDiffusion(torch.nn.Module, CogVideoXLoraLoaderMixin):
    def __init__(self, model_path: str, 
            cache_dir : str,
            warp_num_layers: int,
            train_height: int,
            train_width: int):
        super().__init__()
        self.model_path = model_path
        self.warp_num_layers = warp_num_layers
        try:
            pretrained_transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", cache_dir=cache_dir)
        except:
            pretrained_transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
        self.transformer : CogVideoXCameraWarpTransformer = CogVideoXCameraWarpTransformer.from_transformer(pretrained_transformer)
        
        self.warp_encoder : CogVideoXWarpEncoder = CogVideoXWarpEncoder.from_transformer(
            pretrained_transformer, warp_num_layers,
            attention_head_dim=pretrained_transformer.config.attention_head_dim,
            num_attention_heads=pretrained_transformer.config.num_attention_heads)
        del pretrained_transformer
        
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.attention_head_dim
        self.camera_encoder = CameraEncoder3D(
            resolution=(train_height, train_width), 
            out_channel = inner_dim)
    
    def forward(self, noisy_video_latents : torch.Tensor,
        noisy_model_input : torch.Tensor,
        timesteps : torch.Tensor,
        warp_latents : torch.Tensor, 
        warp_masks : torch.Tensor,
        prompt_embedding : torch.Tensor,
        image_rotary_emb : torch.Tensor,
        extrinsics : torch.Tensor = None, 
        intrinsics : torch.Tensor = None,
        camera_hidden_states : torch.Tensor = None,
        drop_out_camera=False) -> torch.Tensor:
        weight_dtype = noisy_model_input.dtype
        # camera condition
        if not drop_out_camera:
            if camera_hidden_states is None:
                camera_hidden_states = self.camera_encoder(extrinsics, intrinsics)
            # print(image_latents.shape, camera_latents.shape)
        else:
            camera_hidden_states = None

        # logger.info(f"noisy_model_input.shape: {noisy_model_input.shape}; prompt_embeds.shape: {prompt_embeds.shape}")
        branch_block_samples = self.warp_encoder(
            hidden_states=noisy_video_latents,
            encoder_hidden_states=prompt_embedding,
            branch_cond=warp_latents,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        branch_block_samples = [block_sample.to(dtype=weight_dtype) for block_sample in branch_block_samples]
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            branch_block_samples=branch_block_samples,
            branch_block_masks=warp_masks,
            camera_hidden_states=camera_hidden_states,
        )[0]
        return model_output
    
    def enable_gradient_checkpointing(self):
        """Enables gradient checkpointing for the model."""
        self.transformer.enable_gradient_checkpointing()
        self.warp_encoder.enable_gradient_checkpointing()
    
    def save_pretrained(self, save_path: str):
        save_path = Path(save_path)
        transformer_lora_layers_to_save = get_peft_model_state_dict(self.transformer)
        self.save_lora_weights(save_path / "transformer", transformer_lora_layers=transformer_lora_layers_to_save,)
        self.warp_encoder.save_pretrained(save_path / "warp_encoder", subfolder="warp_encoder")
        self.camera_encoder.save_pretrained(save_path / "camera_encoder", subfolder="camera_encoder")
    
    def from_pretrained(self, load_path: str):
        load_path = Path(load_path)
        # load transformer lora
        lora_state_dict = self.load_lora_weights(load_path / "transformer")
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        incompatible_keys = set_peft_model_state_dict(self.transformer, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        self.warp_encoder.from_pretrained(load_path / "warp_encoder", subfolder="warp_encoder")
        self.camera_encoder.from_pretrained(load_path / "camera_encoder", subfolder="camera_encoder")