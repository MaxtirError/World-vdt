import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from renderers.gaussian_renderer import GaussianRenderer
from datasets.tartanair_framepack import TartanairFramePackLoader
from diffusers import (
    AutoencoderKLHunyuanVideo
)
from utils import *
import pipeline
@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
@click.option("--local_rank", type=int, default=0, help="Rank of the current process")
@click.option("--world_size", type=int, default=1, help="Total number of processes")
def main(root: str, output_path : str, num_threads: int, debug: bool, local_rank: int, world_size: int):
    np.random.seed(0)
    root = Path(os.path.expanduser(root))
    output_path = Path(os.path.expanduser(output_path))
    renderer = GaussianRenderer(resolution=(544, 704))
    loader = TartanairFramePackLoader(root, image_size=(544, 704), latent_window_size=9)
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    vae.eval()
    vae.enable_slicing()
    vae.to('cuda')
    gen_instances = []
    for idx in range(len(loader.instances)):
        num_frames = loader.instances[idx][1]
        gen_instances.extend([(idx, batch_idx) for batch_idx in range(num_frames)])
    # sort by the batch index
    gen_instances.sort(key=lambda x: x[1])

    start_index = local_rank * len(gen_instances) // world_size
    end_index = (local_rank + 1) * len(gen_instances) // world_size
    gen_instances = gen_instances[start_index:end_index]
    print("Total instances:", len(gen_instances))
    def _provider():
        for instance in gen_instances:
            yield instance
    
    # @catch_exception
    def _load_data(instance):
        idx, batch_idx = instance
        data = loader(idx)
        save_path = output_path / data['video_info']['scene_path'] / f"{batch_idx:06d}" if not debug else Path("debugs/vis_warp_dataset")
        data['save_dir'] = save_path
        return data
    
    @torch.no_grad()
    def _process_scene(data):
        pc_scene = data['history_scene']
        pc_scene = pc_scene.to('cuda')
        video = pc_scene.warp_to_rgba(renderer)
        data['warp_frames'] = video.cpu()
        return data

    @torch.no_grad()
    def _encode_latents(data):
        frames = data['frames'][None, ...] # [B, F, C, H, W], [0, 1]
        frames = frames.permute(0, 2, 1, 3, 4) * 2 - 1.0 # [B, C, F, H, W], [-1, 1]
        section_latents = vae_encode(frames, vae) # [B, C, F, H, W], [-1, 1]
        data['section_latents'] = section_latents[0].cpu()
        warp_frames = data['warp_frames'][:, :3][None, ...] # [B, F, C, H, W], [0, 1]
        warp_frames = warp_frames.permute(0, 2, 1, 3, 4) * 2 - 1.0
        warp_latents = vae_encode(warp_frames, vae)
        data['warp_latents'] = warp_latents[0].cpu()
        return data
    
    # @catch_exception
    def _write_data(data):
        save_dir = Path(data['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        warp_frames = (data['warp_frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        meta = {
            "extrinsics": data['extrinsics'].tolist(),
            "intrinsics": data['intrinsics'].tolist(),
            "video_info": data['video_info']
        }
        write_meta(save_dir / "meta.json", meta)
        mask = warp_frames[..., -1] > 0
        np.save(save_dir / "mask.npy", mask)
        write_video(save_dir / "warp_frames.mp4", warp_frames[..., :-1])
        # save latents
        torch.save(data['section_latents'], save_dir / "section_latents.pt")
        torch.save(data['warp_latents'], save_dir / "warp_latents.pt")
        start_image = (data['frames'][0].permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
        start_image = start_image
        imageio.imwrite(save_dir / "start_image.png", start_image)
        if debug:
            print(data['section_latents'].shape)
            print(data['warp_latents'].shape)
            print(data['frames'].shape)
            print(data['warp_frames'].shape)
            # save original frames
            frames = (data['frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "frames.mp4", frames)
            recon_video = vae_decode(data['section_latents'][None, ...], vae).permute(0, 2, 1, 3, 4).squeeze(0)
            recon_video = ((recon_video.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "recon_frames.mp4", recon_video)
            recon_warp_video = vae_decode(data['warp_latents'][None, ...], vae).permute(0, 2, 1, 3, 4).squeeze(0)
            recon_warp_video = ((recon_warp_video.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "recon_warp_frames.mp4", recon_warp_video)
    
    if debug:
        data = _load_data((0, 0))
        data = _process_scene(data)
        data = _encode_latents(data)
        _write_data(data)
        return
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_process_scene] * 4),
        pipeline.Parallel([_encode_latents]),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(len(gen_instances)):
            pipe.get()

if __name__ == '__main__':
    main()
