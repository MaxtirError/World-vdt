from dataclasses import dataclass
from typing import *

import torch
import torch.nn as nn

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import BaseOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from ..utils import zero_module
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.configuration_utils import register_to_config
import numpy as np
import torch.nn.functional as F
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.modeling_utils import ModelMixin
from easydict import EasyDict as edict

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class CogVideoxWarpOutput(BaseOutput):
    branch_block_samples: Tuple[torch.Tensor]


class CogVideoXWarpEncoder(CogVideoXTransformer3DModel):
    @register_to_config
    def __init__(self,
        num_layers: int = 2,
        in_channels: int = 16,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64, 
        sample_frames: int = 49,
        sample_width: int = 90,
        sample_height: int = 60,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        dropout: float = 0.0,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        max_text_seq_length: int = 226,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        ofs_embed_dim: int = None,
        out_channels: int = 16,
        patch_bias: bool = True,
        patch_size: int = 2,
        patch_size_t: int = None,
        spatial_interpolation_scale: float = 1.875,
        temporal_compression_ratio: int = 4,
        temporal_interpolation_scale: float = 1.0,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        timestep_activation_fn: str = "silu",
        use_learned_positional_embeddings: bool = True,
        use_rotary_positional_embeddings: bool = True,
        **kwargs):
        super().__init__(
            num_layers=num_layers,
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim, 
            sample_frames=sample_frames,
            sample_width=sample_width,
            sample_height=sample_height,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            dropout=dropout,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            max_text_seq_length=max_text_seq_length,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            ofs_embed_dim=ofs_embed_dim,
            out_channels=out_channels,
            patch_bias=patch_bias,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_compression_ratio=temporal_compression_ratio,
            temporal_interpolation_scale=temporal_interpolation_scale,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            timestep_activation_fn=timestep_activation_fn,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            **kwargs)
        
        inner_dim = num_attention_heads * attention_head_dim
        # branch_blocks
        self.branch_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.branch_blocks.append(zero_module(nn.Linear(inner_dim, inner_dim)))
        # replace the patch embedding layer
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels * 2 + 1 if in_channels == 16 else in_channels + 1,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 2,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
        train_frames: int = 49,
        sample_width: int = 90,
        sample_height: int = 60,
    ):
        config = transformer.config
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        config["sample_frames"] = train_frames
        config["sample_width"] = sample_width
        config["sample_height"] = sample_height
        branch = cls(**config)

        if load_weights_from_transformer:
            conv_in_condition_weight=torch.zeros_like(branch.patch_embed.proj.weight)
            
            if config.in_channels == 16:
                conv_in_condition_weight[:,:config.in_channels,...]=transformer.patch_embed.proj.weight
                conv_in_condition_weight[:,config.in_channels:2*config.in_channels,...]=transformer.patch_embed.proj.weight
            elif config.in_channels == 32:
                conv_in_condition_weight[:,:config.in_channels//2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...]
                conv_in_condition_weight[:,config.in_channels//2:config.in_channels,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...]
            else:
                raise ValueError(f"in_channels {config.in_channels} is not supported")
            branch.patch_embed.proj.weight.data=torch.nn.Parameter(conv_in_condition_weight)
            branch.patch_embed.proj.bias.data=transformer.patch_embed.proj.bias

            branch.embedding_dropout.load_state_dict(transformer.embedding_dropout.state_dict())
            branch.time_proj.load_state_dict(transformer.time_proj.state_dict())
            branch.time_embedding.load_state_dict(transformer.time_embedding.state_dict())
            
            branch.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            

        return branch

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        branch_cond: torch.Tensor = None,
        conditioning_scale: torch.bfloat16 = 1.0,
        timestep: Union[int, float, torch.LongTensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, CogVideoxWarpOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            branch_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            branch_mode (`torch.Tensor`):
                The mode tensor of shape `(batch_size, 1)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for branch outputs.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding 
        branch_cond = torch.concat([hidden_states, branch_cond], dim=-3)
        hidden_states = self.patch_embed(encoder_hidden_states, branch_cond)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            block_samples = block_samples + (hidden_states,)
        # branch block
        branch_block_samples = ()
        for block_sample, branch_block in zip(block_samples, self.branch_blocks):
            block_sample = branch_block(block_sample)
            branch_block_samples = branch_block_samples + (block_sample,)

        branch_block_samples = [sample * conditioning_scale for sample in branch_block_samples]
        branch_block_samples = [sample.to(dtype=hidden_states.dtype) for sample in branch_block_samples]

        branch_block_samples = None if len(branch_block_samples) == 0 else branch_block_samples

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (branch_block_samples, )

        return CogVideoxWarpOutput(
            branch_block_samples=branch_block_samples
        )


class CogVideoXCameraWarpTransformer(CogVideoXTransformer3DModel):
    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @classmethod
    def from_transformer(cls, transformer, train_frames : int = 25, sample_width: int = 90, sample_height: int = 60):
        config = transformer.config
        ori_sample_width = config.sample_width
        ori_sample_height = config.sample_height
        # set the config for the transformer
        config["sample_frames"] = train_frames
        config["sample_width"] = sample_width
        config["sample_height"] = sample_height
        warp_model = cls(**config)
        state_dict = transformer.state_dict()
        # remove the patch embedding from the state dict
        if train_frames != 49:
            logger.warning("Warning: The model is trained with 49 frames, but the current model is set to 25 frames. The patch embedding will be removed from the state dict.")
            if ori_sample_width != sample_width or ori_sample_height != sample_height:
                state_dict.pop("patch_embed.pos_embedding", None)
            else:
                old_pos_embedding = state_dict["patch_embed.pos_embedding"]
                spatail_pos_embedding = old_pos_embedding[:, config.max_text_seq_length:, :]
                # print("spatial_pos_embedding.shape", spatail_pos_embedding.shape)
                embed_dim = config.num_attention_heads * config.attention_head_dim
                p = config.patch_size
                spatila_pos_embedding = spatail_pos_embedding.reshape(1, -1, sample_width * sample_height // (p * p), embed_dim)
                # print("spatila_pos_embedding.shape", spatila_pos_embedding.shape)
                cur_temporal_size = (train_frames - 1) // config.temporal_compression_ratio + 1
                spatila_pos_embedding = spatila_pos_embedding[:, :cur_temporal_size, :, :].reshape(1, -1, embed_dim)
                new_pos_embedding = torch.cat([old_pos_embedding[:, :config.max_text_seq_length, :], spatila_pos_embedding], dim=1)
                state_dict["patch_embed.pos_embedding"] = new_pos_embedding
                
                
        warp_model.load_state_dict(state_dict, strict=False)
        return warp_model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        camera_hidden_states: torch.Tensor = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        branch_block_samples: Optional[torch.Tensor] = None,
        branch_block_masks: Optional[torch.Tensor] = None,
        return_hidden_states: Optional[bool] = False,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        if branch_block_masks is not None:
            # Process mask
            masks = branch_block_masks.reshape(-1, 1, height, width)
            masks = F.avg_pool2d(masks, kernel_size=self.patch_size, stride=self.patch_size)
            masks = masks.view(batch_size, num_frames, *masks.shape[1:])
            masks = masks.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            masks = masks.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            masks = (masks > 0.0).bool()
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
            
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        if camera_hidden_states is not None:
            hidden_states = hidden_states + camera_hidden_states

        # 3. Transformer blocks
        hidden_states_list = []
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
            
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if branch_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(branch_block_samples)
                interval_control = int(np.ceil(interval_control))
                if branch_block_masks is None:
                    hidden_states = hidden_states + branch_block_samples[i // interval_control]
                else:
                    hidden_states = torch.where(masks == False, hidden_states + branch_block_samples[i // interval_control], hidden_states)

            if return_hidden_states:
                hidden_states_list.append(torch.cat([encoder_hidden_states, hidden_states], dim=1))
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if return_hidden_states:
                return (output, hidden_states_list)
            else:
                return (output,)
        return Transformer2DModelOutput(sample=output)
