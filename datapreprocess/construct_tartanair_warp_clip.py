import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *

import numpy as np
import click
from tqdm import trange, tqdm
from renderers.gaussian_renderer import GaussianRenderer
from datasets.tartanair import TartanairSimpleLoader
from utils import *
import pipeline
@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--window_size", type=int, default=153, help="Window size for the data loader")
@click.option("--sliding_step", type=int, default=80, help="Number of overlapped frames between the clips")
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
def main(root: str, output_path: Optional[str], window_size: int, sliding_step: int, num_threads: int, debug: bool):
    np.random.seed(0)
    root = os.path.expanduser(root)
    output_path = os.path.expanduser(output_path)
    loader = TartanairSimpleLoader(root, image_size=(544, 704), window_size=window_size)
    total_frames = 0
    total_clips = 0
    for idx in range(len(loader.instances)):
        num_frames = loader.instances[idx][1]
        total_frames += num_frames
        total_clips += len(range(0, num_frames - window_size + 1, sliding_step))
        
    print("Total frames:", total_frames)
    def _provider():
        for idx in range(len(loader.instances)):
            num_frames = loader.instances[idx][1]
            clip_idx = 0
            for st_window in range(0, num_frames - window_size + 1, sliding_step):
                yield (idx, st_window, clip_idx)
                clip_idx += 1
    
    # @catch_exception
    def _load_data(index_info):
        idx, st_window, clip_idx = index_info
        data = loader(idx, st_window=st_window)
        save_path = Path(output_path) / "raw" / data['video_info']['scene_path'] / f"{clip_idx:06d}"
        data['save_dir'] = save_path
        return data
    
    # @catch_exception
    def _write_data(data):
        save_dir = Path(data['save_dir'])
        if debug:
            save_dir = Path("debugs/vis_tartanairclip")
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "extrinsics": data['extrinsics'].tolist(),
            "intrinsics": data['intrinsics'].tolist(),
            "video_info": data['video_info']
        }
        write_meta(save_dir / "meta.json", meta)
        frames = (data['images'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        write_video(save_dir / "frames.mp4", frames)
        torch.save(data['depths'], save_dir / "depths.pt")

    
    if debug:
        data = _load_data((0, 200, 0))
        _write_data(data)
        exit()
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(total_clips):
            pipe.get()

if __name__ == '__main__':
    main()
