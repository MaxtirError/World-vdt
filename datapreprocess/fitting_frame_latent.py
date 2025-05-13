

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from utils import *
import pipeline

from diffusers import (
    AutoencoderKLHunyuanVideo
)

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


@torch.no_grad()
def vae_encode(image, vae):
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents

def _load_video(video_path):
    '''
    Args:
        video_path: path to the video
    Returns:
        video: (num_frames, 3, H, W) tensor of images
    '''
    video = imageio.get_reader(str(video_path))
    # get the first num_frames frames
    video = [frame for frame in video.iter_data()]
    video = torch.tensor(np.array(video)).permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
    return video.float()
@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--local_rank", type=int, default=0, help="Rank of the current process")
@click.option("--world_size", type=int, default=1, help="Total number of processes")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
def main(root: str, num_threads: int, local_rank : int, world_size : int, debug: bool):
    np.random.seed(0)
    root = Path(os.path.expanduser(root))
    instances = (root / ".index.txt").read_text().splitlines()
    start_index = local_rank * len(instances) // world_size
    end_index = (local_rank + 1) * len(instances) // world_size
    instances = instances[start_index:end_index]
    print("Total instances:", len(instances))
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    vae.eval()
    vae.enable_slicing()
    # vae.enable_tiling()
    vae.to(device='cuda')
    def _provider():
        for instance in instances:
            yield instance
    
    # @catch_exception
    def _load_data(instance):
        video_path = root / "raw" / instance / "frames.mp4"
        video = _load_video(video_path)
        save_dir = root / "frame_latents" / instance 
        return {
            'video': video,
            "save_dir": save_dir,
        }

    def _encode_latents(data):
        video = data['video'].transpose(0, 1).unsqueeze(0)
        video = video.to(device=vae.device, dtype=vae.dtype)
        latent = vae_encode(video, vae).squeeze(0)
        data['latent'] = latent
        return data
    
    # @catch_exception
    def _write_data(data):
        save_dir = Path(data['save_dir'])
        if debug:
            save_dir = Path("debugs/vis_tartanairclip")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data['latent'].cpu(), save_dir / "latents.pt")
        if debug:
            recon_video = vae_decode(data['latent'].unsqueeze(0), vae).squeeze(0).transpose(0, 1)
            recon_video = ((recon_video.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "recon_frames.mp4", recon_video)
            original_video = data['video'].cpu().permute(0, 2, 3, 1)
            original_video = ((original_video * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "original_frames.mp4", original_video)

    
    if debug:
        data = _load_data(instances[0])
        data = _encode_latents(data)
        _write_data(data)
        exit()
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_encode_latents]),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(len(instances)):
            pipe.get()

if __name__ == '__main__':
    main()
