from typing import Any, Dict, List, Tuple, Union
from core.datasets import TartanAirCameraWarpDataset
from core.pipe import CogVideoXI2VCameraWarpPipeline
import click
from core.schemas import Components
from core.models import CogVideoXCameraWarpDiffusion
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

@click.command()
@click.option("--model_path", type=str, required=True, help="Path to the model directory.")
def load_components(model_path, cache_dir) -> Dict[str, Any]:
    components = Components()
    model_path = str(model_path)
    cache_dir = str(cache_dir)

    components.pipeline_cls = CogVideoXI2VCameraWarpPipeline
    try:
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", cache_dir=cache_dir)
    except:
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    
    try:
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", cache_dir=cache_dir)
    except:
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

    try:
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", cache_dir=cache_dir)
    except:
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        
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

def initialze_pipeline(components: Dict[str, Any], args: Dict[str, Any]) -> CogVideoXI2VCameraWarpPipeline:
    pipeline = components.pipeline_cls(
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        backbone=components.backbone,
        scheduler=components.scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(args.device)
    return pipeline

def main():
    # load components