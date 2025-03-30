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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class CogVideoxWarpOutput(BaseOutput):
    branch_block_samples: Tuple[torch.Tensor]

class CogVideoXWarpEncoder(CogVideoXTransformer3DModel):
    @register_to_config()
    def __init__(self,
        in_channels: int = 16,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64, 
        **kwargs):
        super().__init__(in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim, 
            **kwargs)
        
        inner_dim = num_attention_heads * attention_head_dim
        # branch_blocks
        self.branch_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.branch_blocks.append(zero_module(nn.Linear(inner_dim, inner_dim)))

        self.branch_x_embedder = zero_module(torch.nn.Linear(in_channels, inner_dim))
        
    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = transformer.config
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
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


class CogVideoXCameraWarpModel(CogVideoXTransformer3DModel):
    @register_to_config()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        parent_transformer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        transformer = cls(**parent_transformer.config)
        transformer.load_state_dict(parent_transformer.state_dict())
        return transformer
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        branch_block_samples: Optional[torch.Tensor] = None,
        branch_block_masks: Optional[torch.Tensor] = None,
        self_guidance_hidden_states: Optional[torch.Tensor] = None,
        self_guidance_masks: Optional[torch.Tensor] = None,
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
        if self_guidance_masks is not None:
            hidden_states, masks = self.patch_embed(encoder_hidden_states, hidden_states, masks=self_guidance_masks)
            # hidden_states: (B, Len_t + Len_v, C); masks: (B, Len_v, C)
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
        elif branch_block_masks is not None:
            hidden_states, masks = self.patch_embed(encoder_hidden_states, hidden_states, masks=branch_block_masks)
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

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
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                current_block_kwargs = attention_kwargs.copy() if attention_kwargs else {}
                if attention_kwargs and "prev_hidden_states" in attention_kwargs:
                    layer_states = attention_kwargs["prev_hidden_states"].get(i)
                    if layer_states is not None:
                        current_block_kwargs["prev_hidden_states"] = layer_states
                        current_block_kwargs["prev_clip_weight"] = attention_kwargs["prev_clip_weight"]
                    prev_resample_mask = attention_kwargs.get("prev_resample_mask")
                    if prev_resample_mask is not None:
                        current_block_kwargs["prev_resample_mask"] = prev_resample_mask
            
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=current_block_kwargs,
                )
            if self_guidance_hidden_states is not None:
                hidden_states = torch.where(masks == False, self_guidance_hidden_states[i], hidden_states)

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
