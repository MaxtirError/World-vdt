import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from core.datasets import TartanAirCameraWarpDataset
from utils import *
import pipeline
from diffusers import AutoencoderKLCogVideoX
debug=True

def encode_video(vae, video: torch.Tensor) -> torch.Tensor:
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent

@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--log_path", type=click.Path(), default=None, help="Path to log file")
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--num_threads", type=int, default=1, help="Number of threads to use for data loading and processing")
@click.option("--world_size", type=int, default=32, help="Number of processes to use for distributed training")
@click.option("--local_rank", type=int, default=0, help="Local rank for distributed training")
def main(root: str, log_path: Optional[str], output_path: Optional[str], num_threads: int, world_size: int, local_rank: int):
    np.random.seed(0)
    root = os.path.expanduser(root)
    output_path = os.path.expanduser(output_path)
    model_path = "THUDM/CogVideoX-5b-I2V"
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", cache_dir="./cache").to("cuda")
    train_height = 480 
    train_width = 720
    dataset = TartanAirCameraWarpDataset(
        root=root,
        num_frames=49,
        height=train_height,
        width=train_width,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale_factor_temporal = vae.config.temporal_compression_ratio
    all_index = list(range(len(dataset.instances)))
    start_index = local_rank * len(all_index) // world_size
    end_index = (local_rank + 1) * len(all_index) // world_size
    all_index = all_index[start_index:end_index]
    
    def _provider():
        for index in all_index:
            yield index
    
    # @catch_exception
    def _load_data(index):
        data = dataset.__getitem__(index)
        instance = dataset.instances[index]
        save_path = Path(output_path) / instance
        data['save_dir'] = save_path
        return data
    
    @torch.no_grad()
    def _process_scene(data):
        # F C H W -> B C F H W
        latent = encode_video(vae, data['frames'].transpose(0, 1)[None, ...])[0].transpose(0, 1).cpu()
        warp_latent = encode_video(vae, data['warp_frames'].transpose(0, 1)[None, ...])[0].transpose(0, 1).cpu()
        
        masks = data["masks"].transpose(0, 1)[None, ...]
        masks = torch.nn.functional.interpolate(
            masks, 
            size=(
                (masks.shape[-3] - 1) // vae_scale_factor_temporal + 1, 
                train_height // vae_scale_factor_spatial, 
                train_width // vae_scale_factor_spatial
            )
        )[0].transpose(0, 1).to(dtype=warp_latent.dtype)
        data['latent'] = latent
        data['warp_latent'] = warp_latent
        data['masks'] = masks
        return data
    
    # @catch_exception
    def _write_data(data):
        save_dir = Path(data['save_dir'])
        if debug:
            save_dir = "./debugs/visualize_latent"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data['latent'], save_dir / "latent.pt")
        torch.save(data['warp_latent'], save_dir / "warp_latent.pt")
        torch.save(data['masks'], save_dir / "masks.pt")
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_process_scene] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(len(all_index)):
            pipe.get()
    log_file = Path(log_path) / f"log_{local_rank}.txt"
    with open(log_file, "w") as f:
        f.write(f"Processed {len(all_index)} scenes from {start_index} to {end_index}\n")

if __name__ == '__main__':
    main()
