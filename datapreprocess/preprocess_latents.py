import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
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



@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(vae.device, dtype=vae.dtype)
    latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents

    frames = vae.decode(latents).sample
    return frames

def visualze_frames(frames : torch.Tensor, save_path):
    #[C, F, H, W] -> [F, H, W C]
    frames = frames.permute(1, 2, 3, 0)
    # [-1, 1] to [0, 255]
    frames = torch.clamp((frames + 1) * 0.5 * 255, 0, 255).to(torch.uint8)
    frames = frames.cpu().numpy()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=14)
    

def visualize_latent(vae, latent: torch.Tensor, save_path) -> None:
    # Decode the latent representation back to frames
    frames = decode_latents(vae, latent[None, ...])[0]
    # Convert frames to numpy  array and save as video
    visualze_frames(frames, save_path)
    

@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--log_path", type=click.Path(), default=None, help="Path to log file")
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--num_threads", type=int, default=1, help="Number of threads to use for data loading and processing")
@click.option("--world_size", type=int, default=32, help="Number of processes to use for distributed training")
@click.option("--local_rank", type=int, default=0, help="Local rank for distributed training")
@click.option("--target_frames", type=int, default=49, help="Number of frames to use for each video")
@click.option("--target_height", type=int, default=480, help="Height of the video frames")
@click.option("--target_width", type=int, default=720, help="Width of the video frames")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
def main(root: str, log_path: Optional[str], output_path: Optional[str], num_threads: int, world_size: int, local_rank: int, target_frames: int, target_height: int, target_width: int, debug: bool):
    np.random.seed(0)
    root = os.path.expanduser(root)
    if output_path is None:
        output_path = root
    else:
        output_path = os.path.expanduser(output_path)
    if debug:
        print("Original root:", root)
        print("Output path:", output_path)
        output_path = "./debugs/visualize_latent"
        print("Changed output path to:", output_path)
    
    assert target_frames == 49 or target_frames == 25, "Only support 49 and 25 frames for now"
    sample_temporal = (target_frames == 25)
    
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
    
    num_latents_to_process = len(dataset.instances)
    if sample_temporal:
        # 25 frames will sample from 49 frames twice
        num_latents_to_process = num_latents_to_process * 2
    
    all_index = list(range(num_latents_to_process))
    
    start_index = local_rank * len(all_index) // world_size
    end_index = (local_rank + 1) * len(all_index) // world_size
    all_index = all_index[start_index:end_index]
    
    def _provider():
        for index in all_index:
            yield index
    
    # @catch_exception
    def _load_data(index):
        if sample_temporal:
            prefix = index % 2
            index = index // 2
        data = dataset.__getitem__(index)
        data['prefix'] = prefix if sample_temporal else None
        instance = dataset.instances[index]
        save_path = Path(output_path) / instance / "_fitting_latent"
        data['save_dir'] = save_path
        
        if target_width != target_height:
            data['frames'] = torch.nn.functional.interpolate(
                data['frames'],
                size=(
                    target_height,
                    target_width
                ))
            data['warp_frames'] = torch.nn.functional.interpolate(
                data['warp_frames'],
                size=(
                    target_height,
                    target_width
                ))
            
        if sample_temporal:
            if prefix == 0:
                data['frames'] = data['frames'][:target_frames]
                data['warp_frames'] = data['warp_frames'][:target_frames]
            else:
                data['frames'] = data['frames'][::2]
                data['warp_frames'] = data['warp_frames'][::2]
        
        assert data['frames'].shape == (target_frames, 3, target_height, target_width), f"frames shape mismatch: {data['frames'].shape} != {(target_frames, 3, target_height, target_width)}"
        assert data['warp_frames'].shape == (target_frames, 3, target_height, target_width), f"warp_frames shape mismatch: {data['warp_frames'].shape} != {(target_frames, 3, target_height, target_width)}"
        return data
    
    @torch.no_grad()
    def _process_scene(data):
        # F C H W -> B C F H W
        latent = encode_video(vae, data['frames'].transpose(0, 1)[None, ...])[0].transpose(0, 1).cpu()
        warp_latent = encode_video(vae, data['warp_frames'].transpose(0, 1)[None, ...])[0].transpose(0, 1).cpu()
        
        data['latent'] = latent
        data['warp_latent'] = warp_latent
        return data
    
    # @catch_exception
    def _write_data(data):
        save_dir = Path(data['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = "" if not sample_temporal else f"_{data['prefix']}"
        resolution = f"{target_frames}x{target_height}x{target_width}"
        torch.save(data['latent'], save_dir / f"latent_{resolution}{prefix}.pt")
        torch.save(data['warp_latent'], save_dir / f"warp_latent_{resolution}{prefix}.pt")
        if debug:
            # print shape of all components
            print(f"latent shape: {data['latent'].shape}")
            print(f"warp_latent shape: {data['warp_latent'].shape}")
            print(f"masks shape: {data['masks'].shape}")
            torch.cuda.empty_cache()
            #get memory stats
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024} MB")
            # decode for results
            visualize_latent(vae, data['latent'], save_dir / "latent_small.mp4")
            visualize_latent(vae, data['warp_latent'], save_dir / "warp_latent_small.mp4")
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_process_scene] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    if debug:
        for index in range(10):
            data = _load_data(index)
            _process_scene(data)
            _write_data(data)
        exit(0)
    with pipe:
        for i in trange(len(all_index)):
            pipe.get()
    log_file = Path(log_path) / f"log_{local_rank}.txt"
    with open(log_file, "w") as f:
        f.write(f"Processed {len(all_index)} scenes from {start_index} to {end_index}\n")

if __name__ == '__main__':
    main()
