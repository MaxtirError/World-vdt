from typing import Any, Dict, List, Tuple

import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger
# from core.pipe import CogVideoXI2VCameraWarpPipeline
from pathlib import Path
from core.utils.debug_utils import CUDATimer
from diffusers.models.modeling_utils import ModelMixin
from diffusers_helper.models import (
    HunyuanVideoTransformer3DModelPacked, 
    HunyuanVideoTransformer3DModelBranch, 
    CameraWarpFramePack,
    CameraEncoder3D)
logger = get_logger(LOG_NAME, LOG_LEVEL)


class FramePackCameraWarpDiffusion(ModelMixin):
    def __init__(self,
            cache_dir : str,
            branch_num_layers: int,
            branch_num_single_layers: int,
            train_height: int,
            train_width: int,
            latent_window_size : int = 4,
            distilled_guidance_scale: float = 10.0,):
        super().__init__()
        pretrained_transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", cache_dir=cache_dir)
        self.transformer : CameraWarpFramePack = CameraWarpFramePack.from_transformer(transformer=pretrained_transformer)
        self.branch : HunyuanVideoTransformer3DModelBranch = HunyuanVideoTransformer3DModelBranch.from_transformer(
            transformer=pretrained_transformer,
            branch_num_layers=branch_num_layers,
            branch_num_single_layers=branch_num_single_layers,)
        del pretrained_transformer
        
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.attention_head_dim
        self.camera_encoder : CameraEncoder3D = CameraEncoder3D(
            resolution=(train_height, train_width), 
            out_channel = inner_dim)
        self.distilled_guidance_scale = distilled_guidance_scale
        self.latent_window_size = latent_window_size
        self.train_height = train_height
        self.train_width = train_width
        
    def forward(self, 
        noisy_latents : torch.Tensor,
        timesteps : torch.Tensor,
        warp_latents : torch.Tensor, 
        start_latent : torch.Tensor,
        history_latents : torch.Tensor,
        warp_masks : torch.Tensor,
        pooled_projections : torch.Tensor,
        encoder_hidden_states : torch.Tensor,
        encoder_attention_mask : torch.Tensor,
        image_embedings : torch.Tensor,
        extrinsics : torch.Tensor = None, 
        intrinsics : torch.Tensor = None,
        camera_hidden_states : torch.Tensor = None,
        drop_out_camera=False) -> torch.Tensor:
        weight_dtype = noisy_latents.dtype
        # camera condition
        if not drop_out_camera:
            if camera_hidden_states is None:
                camera_hidden_states = self.camera_encoder(extrinsics, intrinsics)
            # print(image_latents.shape, camera_latents.shape)
        else:
            camera_hidden_states = None

        indices = torch.arange(0, sum([1, 16, 2, 1, self.latent_window_size])).unsqueeze(0)
        clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, self.latent_window_size], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

        clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
        clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

        batch_size = noisy_latents.shape[0]
        distilled_guidance = torch.tensor([self.distilled_guidance_scale * 1000.0] * batch_size)

        extra_kwargs = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "pooled_projections": pooled_projections,
            "guidance": distilled_guidance,
            "image_embeddings": image_embedings,
        }

        # logger.info(f"noisy_model_input.shape: {noisy_model_input.shape}; prompt_embeds.shape: {prompt_embeds.shape}")
        branch_transformer_block_samples, branch_single_transformer_block_samples = self.branch(
            hidden_states=noisy_latents, 
            timestep=timesteps,
            branch_cond=warp_latents, 
            guidance=distilled_guidance,
            latent_indices=latent_indices,
            return_dict=False,
            **extra_kwargs,
        )
        branch_block_samples = [block_sample.to(dtype=weight_dtype) for block_sample in branch_block_samples]
        model_output = self.transformer(
            hidden_states=noisy_latents, 
            timestep=timesteps, 
            camera_hidden_states=camera_hidden_states,
            branch_transformer_block_samples=branch_transformer_block_samples,
            branch_single_transformer_block_samples=branch_single_transformer_block_samples,
            branch_block_masks=warp_masks,
            latent_indices=latent_indices,
            clean_latents=clean_latents, 
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x, 
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x, 
            clean_latent_4x_indices=clean_latent_4x_indices,
            return_dict=False,
            **extra_kwargs,
        )[0]
        return model_output
    
    def enable_gradient_checkpointing(self):
        """Enables gradient checkpointing for the model."""
        self.transformer.enable_gradient_checkpointing()
        self.branch.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disables gradient checkpointing for the model."""
        self.transformer.disable_gradient_checkpointing()
        self.branch.disable_gradient_checkpointing()
    
    def save_pretrained(self, save_path: str):
        save_path = Path(save_path)
        self.branch.save_pretrained(save_path / "branch", subfolder="branch")
        self.camera_encoder.save_pretrained(save_path / "camera_encoder", subfolder="camera_encoder")
        self.transformer.save_pretrained(save_path / "transformer", subfolder="transformer")
    
    def from_pretrained(self, load_pah: str):
        load_path = Path(load_path)
        load_branch = HunyuanVideoTransformer3DModelBranch.from_pretrained(
            load_path / "branch", subfolder="branch"
        )
        self.branch.load_state_dict(load_branch.state_dict())
        camera_encoder = CameraEncoder3D.from_pretrained(
            load_path / "camera_encoder", subfolder="camera_encoder"
        )
        self.camera_encoder.load_state_dict(camera_encoder.state_dict())
        transformer = CameraWarpFramePack.from_pretrained(
            load_path / "transformer", subfolder="transformer"
        )
        self.transformer.load_state_dict(transformer.state_dict())