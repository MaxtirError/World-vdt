from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn
import numpy as np

from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import logging
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from .hunyuan_video_packed import *
from dataclasses import dataclass
from torch.nn import functional as F


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class HunyuanVideoTransformerBranchOutput(BaseOutput):
    branch_transformer_block_sample: Tuple[torch.Tensor]
    branch_single_transformer_block_sample: Tuple[torch.Tensor]

class HunyuanVideoTransformer3DModelBranch(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        has_image_proj=False,
        image_proj_dim=1152,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels * 2 + 1, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(inner_dim, pooled_projection_dim)

        self.image_projection = None

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        self.branch_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.branch_blocks.append(zero_module(nn.Linear(inner_dim, inner_dim)))
        
        for _ in range(len(self.single_transformer_blocks)):
            self.branch_blocks.append(zero_module(nn.Linear(inner_dim, inner_dim)))


        self.inner_dim = inner_dim
        self.use_gradient_checkpointing = False

        if has_image_proj:
            self.install_image_projection(image_proj_dim)

        self.high_quality_fp32_output_for_inference = False

    def install_image_projection(self, in_channels):
        self.image_projection = ClipVisionProjection(in_channels=in_channels, out_channels=self.inner_dim)
        self.config['has_image_proj'] = True
        self.config['image_proj_dim'] = in_channels

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True
        print('self.use_gradient_checkpointing = True')

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False
        print('self.use_gradient_checkpointing = False')
    def initialize_patch_embedding(self, another_conv : nn.Conv3d):
        another_weight = another_conv.weight.data
        another_bias = another_conv.bias.data
        new_weight = torch.zeros_like(self.x_embedder.proj.weight.data)
        new_weight[:, :another_conv.in_channels, :, :, :] = another_weight
        self.x_embedder.proj.weight.data = new_weight
        self.x_embedder.proj.bias.data = another_bias

    def gradient_checkpointing_method(self, block, *args):
        if self.use_gradient_checkpointing:
            result = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        else:
            result = block(*args)
        return result

    def process_input_hidden_states(
            self,
            latents,
            branch_cond,
            latent_indices=None,
    ):
        latents = torch.cat([latents, branch_cond], dim=-3)
        hidden_states = self.gradient_checkpointing_method(self.x_embedder.proj, latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)
        return hidden_states, rope_freqs

    def forward(
            self,
            hidden_states, branch_cond, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
            latent_indices=None,
            image_embeddings=None,
            attention_kwargs=None, return_dict=True
    ):

        if attention_kwargs is None:
            attention_kwargs = {}

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config['patch_size'], self.config['patch_size_t']
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, branch_cond, latent_indices)

        temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
        encoder_hidden_states = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)

        if self.image_projection is not None:
            assert image_embeddings is not None, 'You must use image embeddings!'
            extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
            extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

            # must cat before (not after) encoder_hidden_states, due to attn masking
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        with torch.no_grad():
            if batch_size == 1:
                # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
                # If they are not same, then their impls are wrong. Ours are always the correct one.
                text_len = encoder_attention_mask.sum().item()
                encoder_hidden_states = encoder_hidden_states[:, :text_len]
                attention_mask = None, None, None, None
            else:
                img_seq_len = hidden_states.shape[1]
                txt_seq_len = encoder_hidden_states.shape[1]

                cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = img_seq_len + txt_seq_len
                max_seqlen_kv = max_seqlen_q

                attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        
        block_samples = ()

        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs
            )
            block_samples += (hidden_states,)

        for block_id, block in enumerate(self.single_transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs
            )
            block_samples += (hidden_states,)

        transformer_block_samples = ()
        single_transformer_block_samples = ()
        
        # branch blocks
        for block_idx, block_sample, branch_block in enumerate(zip(block_samples, self.branch_blocks)):
            block_sample = self.gradient_checkpointing_method(
                branch_block,
                block_sample
            )
            if block_idx < len(self.transformer_blocks):
                transformer_block_samples += (block_sample,)
            else:
                single_transformer_block_samples += (block_sample,)


        if return_dict:
            return HunyuanVideoTransformerBranchOutput(
                branch_transformer_block_sample=transformer_block_samples,
                branch_single_transformer_block_sample=single_transformer_block_samples,
            )

        return (transformer_block_samples, single_transformer_block_samples)

    @classmethod
    def from_transformer(
        cls,
        transformer : HunyuanVideoTransformer3DModelPacked, 
        num_layers: int = 2,
        num_single_layers: int = 2,
        load_weights_from_transformer: bool = True,
    ):
        config = transformer.config
        
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        # pop x_embedder and context_embedder
        config.pop("has_clean_x_embedder")


        branch = cls(**config)
        if load_weights_from_transformer:
            # initialize patch_embedding from original conv
            branch.initialize_patch_embedding(transformer.x_embedder.proj)
            # load other weights
            branch.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            branch.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            branch.rope.load_state_dict(transformer.rope.state_dict())

        return branch

# borrowed from diffusers framepack, only changing the forward function
class CameraWarpFramePack(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        has_image_proj=False,
        image_proj_dim=1152,
        has_clean_x_embedder=False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(inner_dim, pooled_projection_dim)

        self.clean_x_embedder = None
        self.image_projection = None

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.inner_dim = inner_dim
        self.use_gradient_checkpointing = False
        self.enable_teacache = False

        if has_image_proj:
            self.install_image_projection(image_proj_dim)

        if has_clean_x_embedder:
            self.install_clean_x_embedder()

        self.high_quality_fp32_output_for_inference = False

    def install_image_projection(self, in_channels):
        self.image_projection = ClipVisionProjection(in_channels=in_channels, out_channels=self.inner_dim)
        self.config['has_image_proj'] = True
        self.config['image_proj_dim'] = in_channels

    def install_clean_x_embedder(self):
        self.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(self.inner_dim)
        self.config['has_clean_x_embedder'] = True

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True
        print('self.use_gradient_checkpointing = True')

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False
        print('self.use_gradient_checkpointing = False')

    def initialize_teacache(self, enable_teacache=True, num_steps=25, rel_l1_thresh=0.15):
        self.enable_teacache = enable_teacache
        self.cnt = 0
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.teacache_rescale_func = np.poly1d([7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02])

    def gradient_checkpointing_method(self, block, *args):
        if self.use_gradient_checkpointing:
            result = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        else:
            result = block(*args)
        return result

    def process_input_hidden_states(
            self,
            latents, latent_indices=None,
            clean_latents=None, clean_latent_indices=None,
            clean_latents_2x=None, clean_latent_2x_indices=None,
            clean_latents_4x=None, clean_latent_4x_indices=None
    ):
        hidden_states = self.gradient_checkpointing_method(self.x_embedder.proj, latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_latents = self.gradient_checkpointing_method(self.clean_x_embedder.proj, clean_latents)
            clean_latents = clean_latents.flatten(2).transpose(1, 2)

            clean_latent_rope_freqs = self.rope(frame_indices=clean_latent_indices, height=H, width=W, device=clean_latents.device)
            clean_latent_rope_freqs = clean_latent_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            clean_latents_2x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_2x, clean_latents_2x)
            clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device)
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            clean_latents_4x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_4x, clean_latents_4x)
            clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device)
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        return hidden_states, rope_freqs

    def forward(
            self,
            hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
            camera_hidden_states=None,
            branch_transformer_block_samples=None,
            branch_single_transformer_block_samples=None,
            branch_block_masks=None,
            latent_indices=None,
            clean_latents=None, clean_latent_indices=None,
            clean_latents_2x=None, clean_latent_2x_indices=None,
            clean_latents_4x=None, clean_latent_4x_indices=None,
            image_embeddings=None,
            attention_kwargs=None, return_dict=True
    ):

        if attention_kwargs is None:
            attention_kwargs = {}

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config['patch_size'], self.config['patch_size_t']
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

        if branch_block_masks is not None:
            # Process mask
            assert p_t == 1, "branch_block_masks only support p_t == 1"
            masks = branch_block_masks.reshape(-1, 1, height, width)
            masks = F.avg_pool2d(masks, kernel_size=p, stride=p)
            masks = masks.view(batch_size, num_frames, *masks.shape[1:])
            masks = masks.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            masks = masks.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            masks = (masks > 0.0).bool()
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
        else:
            masks = None
            
        if camera_hidden_states is not None:
            # zero padding camera_hidden_states into hidden_states
            assert camera_hidden_states.shape[1] == original_context_length, f"camera_hidden_states shape {camera_hidden_states.shape[1]} != original_context_length {original_context_length}"
            hidden_states[:, -original_context_length:, :] += camera_hidden_states

        temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
        encoder_hidden_states = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)

        if self.image_projection is not None:
            assert image_embeddings is not None, 'You must use image embeddings!'
            extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
            extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

            # must cat before (not after) encoder_hidden_states, due to attn masking
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        with torch.no_grad():
            if batch_size == 1:
                # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
                # If they are not same, then their impls are wrong. Ours are always the correct one.
                text_len = encoder_attention_mask.sum().item()
                encoder_hidden_states = encoder_hidden_states[:, :text_len]
                attention_mask = None, None, None, None
            else:
                img_seq_len = hidden_states.shape[1]
                txt_seq_len = encoder_hidden_states.shape[1]

                cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = img_seq_len + txt_seq_len
                max_seqlen_kv = max_seqlen_q

                attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

        def mask_add(mask, x, y):
            if mask is None:
                return x + y
            else:
                return torch.where(mask == True, x + y, x)
        if self.enable_teacache:
            modulated_inp = self.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                curr_rel_l1 = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
                self.accumulated_rel_l1_distance += self.teacache_rescale_func(curr_rel_l1)
                should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh

                if should_calc:
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            self.cnt += 1

            if self.cnt == self.num_steps:
                self.cnt = 0

            if not should_calc:
                hidden_states = hidden_states + self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()

                for block_id, block in enumerate(self.transformer_blocks):
                    hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        rope_freqs
                    )

                for block_id, block in enumerate(self.single_transformer_blocks):
                    hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        rope_freqs
                    )

                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for block_id, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )
                if branch_transformer_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(branch_transformer_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, -original_context_length:, :] = mask_add(masks, hidden_states[:, -original_context_length:, :], branch_transformer_block_samples[block_id // interval_control])

            for block_id, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )
                if branch_single_transformer_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(branch_single_transformer_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, -original_context_length:, :] = mask_add(masks, hidden_states[:, -original_context_length:, :], branch_single_transformer_block_samples[block_id // interval_control])

        hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)

        hidden_states = hidden_states[:, -original_context_length:, :]

        if self.high_quality_fp32_output_for_inference:
            hidden_states = hidden_states.to(dtype=torch.float32)
            if self.proj_out.weight.dtype != torch.float32:
                self.proj_out.to(dtype=torch.float32)

        hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)

        hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                         t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                         pt=p_t, ph=p, pw=p)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)

        return (hidden_states,)

    @classmethod
    def from_transformer(
        cls,
        transformer
    ):
        config = transformer.config
        model = cls(**config)
        model.load_state_dict(transformer.state_dict())
        return model