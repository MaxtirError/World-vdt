import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from renderers.gaussian_renderer import GaussianRenderer
from datasets.tartanair import TartanairWarpBatchGenerator
from diffusers import (
    AutoencoderKLHunyuanVideo
)
from utils import *
import pipeline
@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
@click.option("--num_generate_batch", type=int, default=10, help="Number of batches to generate")
@click.option("--local_rank", type=int, default=0, help="Rank of the current process")
@click.option("--world_size", type=int, default=1, help="Total number of processes")
def main(root: str, num_threads: int, debug: bool, num_generate_batch, local_rank: int, world_size: int):
    np.random.seed(233)
    root = Path(os.path.expanduser(root))
    renderer = GaussianRenderer(resolution=(544, 704))
    Generator = TartanairWarpBatchGenerator(root)
    if debug:
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        vae.eval()
        vae.to('cuda')
    clip_index = list(range(Generator.__len__()))
    start_index = local_rank * len(clip_index) // world_size
    end_index = (local_rank + 1) * len(clip_index) // world_size
    clip_index = clip_index[start_index:end_index]
    num_batchs = len(clip_index) * num_generate_batch
    print("Expected to generate total batch:", num_batchs)
    def _provider():
        for idx in clip_index:
            yield idx
    
    # @catch_exception
    def _load_clip(index):
        clip_data = Generator.fetch_clip_data(index)
        save_path = root / "warp_batchs" / clip_data['video_info']['scene_path'] if not debug else Path("debugs/warp_batchs")
        clip_data['save_dir'] = save_path
        return [(clip_data, i) for i in range(num_generate_batch)]
    
    def _generate_batch(clip_data):
        clip_data, batch_index = clip_data
        batch_data = Generator.generate_from_clip_data(clip_data)
        batch_data['save_dir'] = clip_data['save_dir'] / f"batch_{batch_index:03d}"
        return batch_data
    
    @torch.no_grad()
    def _process_scene(data):
        pc_scene = data['history_scene']
        pc_scene = pc_scene.to('cuda')
        video = pc_scene.warp_to_rgba(renderer)
        data['warp_frames'] = video.cpu()
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
        torch.save(data['history_latents'], save_dir / "history_latents.pt")
        torch.save(data['frame_latents'], save_dir / "frame_latents.pt")
        # write start image
        start_image = (data['start_image'].permute(0, 2, 3, 1) * 127.5 + 127.5).clamp(0, 255).numpy().astype(np.uint8)
        imageio.imwrite(save_dir / "start_image.png", start_image[0])
        if debug:
            print(data['frame_latents'].shape)
            start_latent = vae_encode(data['start_image'].unsqueeze(2), vae)
            his_latent = torch.cat([start_latent, data['history_latents'][None, ...].to("cuda")], dim=2)
            print(his_latent.shape)
            recon_video = vae_decode(his_latent, vae).squeeze(0).transpose(0, 1)
            recon_video = ((recon_video.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "recon_his_frames.mp4", recon_video)
            frame_latent = torch.cat([start_latent, data['frame_latents'][None, ...].to("cuda")], dim=2)
            recon_video = vae_decode(frame_latent, vae).squeeze(0).transpose(0, 1)
            recon_video = ((recon_video.cpu().permute(0, 2, 3, 1) * 127.5 + 127.5)).clamp(0, 255).numpy().astype(np.uint8)
            write_video(save_dir / "recon_frames.mp4", recon_video)
    
    if debug:
        data = _load_clip(0)
        data = _generate_batch(data[0])
        data = _process_scene(data)
        _write_data(data)
        return
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_clip] * 2),
        pipeline.Unbatch(),
        pipeline.Parallel([_generate_batch] * num_threads),
        pipeline.Parallel([_process_scene] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(num_batchs):
            pipe.get()

if __name__ == '__main__':
    main()
