
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import pandas as pd
import torch
from utils import *
import utils3d
from pathlib import Path
from representations.pc_scene import PcScene

class TartanairLoader:
    def __init__(
        self,
        root,
        window_size=48,
        max_num_history=4,
        frame_size=24,
        image_size=128,
        max_depth_range=128,
        debug=False,
        use_mask=False,
    ):
        self.root = root
        self.frame_size = frame_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        metadata = pd.read_csv(Path(root, 'metadata.csv'))
        self.instances = [(row['path'], row['num_frames']) for _, row in metadata.iterrows()]
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.instances))
        
        self.window_size = window_size
        self.max_num_history = max_num_history
        self.max_depth_range = max_depth_range
        
        self.debug = debug
        self.use_mask = use_mask
    
    @staticmethod
    def load_raw_datas(scene_path : Path, frame_list):
        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        for frame in frame_list:
            frame_path = Path(scene_path, f"{frame:06d}_left")
            image_path = frame_path / 'image.jpg'
            depth_path = frame_path / 'depth.png'
            meta_path = frame_path / 'meta.json'
            image = read_image(image_path)
            depth, _ = read_depth(depth_path)
            meta = read_meta(meta_path)
            images.append(image)
            depths.append(depth)
            extrinsics.append(meta['extrinsics'])
            intrinsics.append(meta['intrinsics'])
        return {
            'images': np.stack(images),
            'depths': np.stack(depths),
            'extrinsics': np.stack(extrinsics),
            'intrinsics': np.stack(intrinsics)
        }
    
    def __len__(self):
        return len(self.instances)
    
    def __call__(self, idx, st_window=None, num_history=None, frame_size=None):
        scene_path, num_frames = self.instances[idx]
        video_info = {"scene_path": scene_path}
        scene_path = Path(self.root, scene_path)
        frame_size = frame_size or self.frame_size
        assert num_frames >= frame_size, f'Instance {scene_path} has less than {frame_size} frames.'
        if st_window is None:
            st_window = np.random.randint(num_frames - self.window_size + 1)
        else:
            assert st_window >= 0 and st_window <= num_frames - self.window_size, f'Invalid st_window {st_window} for instance {scene_path}'
        st_frame = np.random.randint(st_window, st_window + self.window_size - frame_size + 1)
        if num_history is None:
            num_history = np.random.randint(0, self.max_num_history + 1)
        frame_list = list(range(st_frame, st_frame + frame_size))
        video_info['frame_list'] = frame_list
        if num_history > 0:
            history_frames = [st_window + v * self.window_size // (num_history + 1)for v in range(1, num_history + 1)]
            video_info['history_frames'] = history_frames
            frame_list = history_frames + frame_list
        else:
            video_info['history_frames'] = []
        raw_data = self.load_raw_datas(scene_path, frame_list)
        # first resize to target size
        images = torch.tensor(raw_data['images'], dtype=torch.float32).permute(0, 3, 1, 2).float() / 255.0
        depths = torch.tensor(raw_data['depths'], dtype=torch.float32).unsqueeze(1)
        extrinsics_raw = torch.tensor(raw_data['extrinsics'], dtype=torch.float32)
        intrinsics_raw = torch.tensor(raw_data['intrinsics'], dtype=torch.float32)
        # print(images.dtype, depths.dtype, extrinsics.dtype, intrinsics.dtype)
        images, intrinsics_raw, depths = crop_images(images, self.image_size, mode='random', intrinsics=intrinsics_raw, depth=depths)
        depths = depths.squeeze(1)
        depth_mask = torch.isfinite(depths)
        mask = depth_mask & (depths < depths[depth_mask].min() * self.max_depth_range)
        # get the raw point cloud in camera space
        pcs_world_raw = utils3d.torch.depth_to_points(depths, intrinsics_raw, extrinsics_raw)
        
        gt_world_raw = pcs_world_raw[num_history:][mask[num_history:]].reshape(-1, 3)
        mean = gt_world_raw.mean(0)
        scale = (gt_world_raw - mean).abs().max()
        # normalize extrinsics
        
        extrinsics = extrinsics_raw[num_history:].clone()
        extrinsics = normalize_extrinsics(extrinsics, scale, mean)
        intrinsics = intrinsics_raw[num_history:]
        frames = images[num_history:]
        # transform to relative pose
        inv_extrinsics = torch.inverse(extrinsics[0])
        extrinsics = extrinsics @ inv_extrinsics
        
        history_pcs = pcs_world_raw[:(num_history + 1)][mask[:(num_history + 1)]].reshape(-1, 3)
        history_rgb = images[:(num_history + 1)].permute(0, 2, 3, 1)[mask[:(num_history + 1)]]
        
        history_scene = PcScene(
            xyz=history_pcs, 
            rgb=history_rgb, 
            scale=scale.item(), 
            extrinsics=extrinsics_raw[num_history:], 
            intrinsics=intrinsics_raw[num_history:],
            history_frames=images[:(num_history+1)] if self.debug else None)
        
        return {'frames': frames, 
                'extrinsics' : extrinsics,
                'intrinsics' : intrinsics, 
                'history_scene' : history_scene, 
                'video_info': video_info}