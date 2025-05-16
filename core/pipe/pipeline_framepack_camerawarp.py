# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from diffusers.models import AutoencoderKLCogVideoX
import torch
from tqdm import tqdm

@torch.no_grad()
def vae_decode(latents, vae, image_mode=False):
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)

    return image

class FramePackValidationPipeline:

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        backbone
    ):
        self.vae = vae
        self.backbone = backbone
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        self.vae = self.vae.to(device, dtype=dtype)
        self.backbone = self.backbone.to(device, dtype=dtype)
        return self

    @staticmethod
    def postprocess(frames: torch.Tensor):
        frames = frames.permute(0, 2, 1, 3, 4)[0].float()
        frames = ((frames.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
        return frames
    
    @torch.no_grad()
    def __call__(self, 
        latent_shape : Tuple[int, int, int, int],
        t_scale = 1000.0,
        steps=50,
        noise=None,
        verbose=True,
        return_latents=False,
        **extra_kwargs):
        weight_dtype = self.vae.dtype
        if noise is not None:
            frame_latents = noise
        else:
            frame_latents = torch.randn(latent_shape, device=self.vae.device)
        t_seq = np.linspace(1, 0, steps + 1)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Validation", disable=not verbose):
            t = torch.tensor([t], device=self.vae.device)
            t_prev = torch.tensor([t_prev], device=self.vae.device)
            pred_v = self.backbone(
                frame_latents.to(dtype=weight_dtype),
                t * t_scale,
                **extra_kwargs
            )
            if not isinstance(pred_v, torch.Tensor):
                pred_v = pred_v[0]
            frame_latents = frame_latents - pred_v * (t - t_prev).view(-1, *([1] * (len(latent_shape) - 1)))
        if return_latents:
            return frame_latents
        cat_prev_frames = extra_kwargs["history_latents"][:, :, -2:].to(frame_latents)
        frame_latents_to_decode = torch.cat([cat_prev_frames, frame_latents], dim=2)
        frames = vae_decode(frame_latents_to_decode, self.vae)
        frames = self.postprocess(frames)
        warp_frames_to_decode = torch.cat([cat_prev_frames, extra_kwargs["warp_latents"].to(frame_latents)], dim=2)
        warp_frames = vae_decode(warp_frames_to_decode, self.vae)
        warp_frames = self.postprocess(warp_frames)
        return {"frames": frames, "warp_frames": warp_frames}

        
            
