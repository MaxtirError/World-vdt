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
from utils import *
import pipeline
@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--log_path", type=click.Path(exists=False), default=None, help="Path to log file")
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode")
@click.option("--total_batch", type=int, default=50000, help="Total generated batch size")
def main(root: str, log_path: Optional[str], output_path: Optional[str], num_threads: int, debug: bool, total_batch: int):
    np.random.seed(0)
    root = os.path.expanduser(root)
    output_path = os.path.expanduser(output_path)
    renderer = GaussianRenderer(resolution=(480, 832))
    loader = TartanairFramePackLoader(root)
    total_frames = 0
    for idx in range(len(loader.instances)):
        num_frames = loader.instances[idx][1]
        total_frames += num_frames
        
    print("Total frames:", total_frames)
    cur_frame_num = 0
    instance_sample_batch_num = []
    for idx in range(len(loader.instances)):
        num_frames = loader.instances[idx][1]
        sample_num = total_batch * (cur_frame_num + num_frames) // total_frames - total_batch * cur_frame_num // total_frames
        instance_sample_batch_num.append(sample_num)
        cur_frame_num += num_frames
    assert sum(instance_sample_batch_num) == total_batch, f"Total batch size {total_batch} is not equal to the sum of instance sample batch num {sum(instance_sample_batch_num)}"
    print("Expected to generate total clips:", total_batch)
    def _provider():
        for idx in range(len(loader.instances)):
            num_frames = loader.instances[idx][1]
            sample_num = instance_sample_batch_num[idx]
            clip_idx = 0
            start_windows = np.linspace(0, num_frames - 1, sample_num, dtype=int)
            for st_window in start_windows:
                yield (idx, st_window, clip_idx)
                clip_idx += 1
    
    # @catch_exception
    def _load_data(index_info):
        idx, st_window, clip_idx = index_info
        data = loader(idx, st_window=st_window)
        if debug:
            data = loader(idx, st_window=st_window, section_size=16)
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
        if debug:
            save_dir = Path("debugs/vis_framepack")
        save_dir.mkdir(parents=True, exist_ok=True)
        frames = (data['frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        warp_frames = (data['warp_frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
        history_frames = (data['history_frames'].permute(0, 2, 3, 1) * 255).clamp(0, 255).numpy().astype(np.uint8)
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
        write_video(save_dir / "history_frames.mp4", history_frames)
    
    if debug:
        data = _load_data((0, 200, 0))
        data = _process_scene(data)
        _write_data(data)
        exit()
    
    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_load_data] * num_threads),
        pipeline.Parallel([_process_scene] * num_threads),
        pipeline.Parallel([_write_data] * num_threads)
    ])
    with pipe:
        for i in trange(total_batch):
            pipe.get()

if __name__ == '__main__':
    main()
