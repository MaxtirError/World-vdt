from core.trainers.base import Trainer
from typing_extensions import override
from core.schemas import Components
from typing import *
import torch
from diffusers import AutoencoderKLHunyuanVideo

class FramePackSFTTrainer(Trainer):
    """Trainer class for FramePack SFT (Supervised Fine-Tuning) training."""
    
    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)
        cache_dir = str(self.args.cache_dir)
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', cache_dir=cache_dir)
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', cache_dir=cache_dir)

        vae_scale_factor_spatial = 2 ** (len(components.vae.config.block_out_channels) - 1)
        sample_height = self.state.train_height // vae_scale_factor_spatial
        sample_width = self.state.train_width // vae_scale_factor_spatial
        if self.args.debug:
            print(f"sample height: {sample_height}, sample width: {sample_width}")
        components.backbone = CogVideoXCameraWarpDiffusion(
            model_path=model_path,
            cache_dir=cache_dir,
            warp_num_layers=self.args.warp_num_layers,
            train_height=self.state.train_height,
            train_width=self.state.train_width,
            train_frames=self.state.train_frames,
            sample_height=sample_height,
            sample_width=sample_width,
        )
        
        try:
            components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=cache_dir)
        except:
            components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

        