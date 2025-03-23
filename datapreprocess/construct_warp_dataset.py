import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from renderers.gaussian_renderer import GaussianRenderer
from datasets.tartanair import TartanairLoader
from utils import *
import pipeline

@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--log_path", type=click.Path(exists=False), default=None, help="Path to log file")
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
def main(root: str, log_path: Optional[str], output_path: Optional[str], num_threads: int):
    np.random.seed(0)
    root = os.path.expanduser(root)
    output_path = os.path.expanduser(output_path)
    renderer = GaussianRenderer(resolution=(480, 720))
    loader = TartanairLoader(root,
        window_size=100,
        max_num_history=4,
        frame_size=49,
        image_size=(480, 720),
        max_depth_range=128,
        debug=False)
    
    total_clips = 0
    for idx in range(len(loader.instances)):
        num_frames = loader.instances[idx][1]
        total_clips += (num_frames - loader.window_size) // loader.frame_size * (loader.max_num_history + 1)
    print("Expected to generate total clips:", total_clips)
    def _provider():
        for idx in range(len(loader.instances)):
            num_frames = loader.instances[idx][1]
            clip_idx = 0
            for st_window in range(0, num_frames - loader.window_size, loader.frame_size):
                for num_history in range(0, loader.max_num_history + 1):
                    yield (idx, st_window, num_history, clip_idx)
                    clip_idx += 1
    
    # @catch_exception
    def _load_data(index_info):
        idx, st_window, num_history, clip_idx = index_info
        data = loader(idx, st_window=st_window, num_history=num_history)
        save_path = Path(output_path) / data['video_info']['scene_path'] / f"{clip_idx:06d}"
        data['save_dir'] = save_path
        return data
    
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
        frames = (data['frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        warp_frames = (data['warp_frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        meta = {
            "extrinsics": data['extrinsics'].tolist(),
            "intrinsics": data['intrinsics'].tolist(),
            "video_info": data['video_info']
        }
        write_meta(save_dir / "meta.json", meta)
        mask = warp_frames[..., -1] > 0
        np.save(save_dir / "mask.npy", mask)
        write_video(save_dir / "frames.mp4", frames)
        write_video(save_dir / "warp_frames.mp4", warp_frames[..., :-1])
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_process_scene] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(total_clips):
            pipe.get()

if __name__ == '__main__':
    main()
