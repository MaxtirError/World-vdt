
import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from core.utils.io_utils import read_image, read_depth, read_meta
from core.utils.image_utils import crop_images
from core.utils.camera_utils import normalize_extrinsics
import copy
import utils3d
from pathlib import Path
from representations.pc_scene import PcScene
class TartanairVideoSTD(Dataset):
    def __init__(
        self,
        root,
        frame_size=24,
        image_size=128,
        max_depth_range=128,
        mode='rgb_vae',
        debug=False,
        use_mask=False,
    ):
        super().__init__()
        assert mode in ['rgb_vae', 'xyz_vae', "rgb_diffusion", 'xyz_diffusion', 'cross_diffusion', 'history_diffusion'], f'Invalid mode {mode}.'
        self.root = root
        self.frame_size = frame_size
        self.image_size = image_size
        metadata = pd.read_csv(Path(root, 'metadata.csv'))
        self.instances = [(row['path'], row['num_frames']) for _, row in metadata.iterrows()]
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.instances))
        
        self.mode = mode
        self.max_depth_range = max_depth_range
        print(f'Using training {mode} mode.')
        
        self.value_range = (-1, 1) if 'xyz' in mode else (0, 1)
        self.debug = debug
        self.use_mask = use_mask
        
    
    def __len__(self):
        return len(self.instances)
    
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
            
        
    def _get_video(self, idx, st_frame=None):
        scene_path, num_frames = self.instances[idx]
        scene_path = Path(self.root, scene_path)
        assert num_frames >= self.frame_size, f'Instance {scene_path} has less than {self.frame_size} frames.'
        if st_frame is None:
            st_frame = np.random.randint(num_frames - self.frame_size + 1)
        # print(st_frame, scene_path)
        # print(scene_path, st_frame)
        frame_list = list(range(st_frame, st_frame + self.frame_size))
        raw_data = self.load_raw_datas(scene_path, frame_list)
        # first resize to target size
        images = torch.tensor(raw_data['images'], dtype=torch.float32).permute(0, 3, 1, 2).float() / 255.0
        depths = torch.tensor(raw_data['depths'], dtype=torch.float32).unsqueeze(1)
        extrinsics = torch.tensor(raw_data['extrinsics'], dtype=torch.float32)
        intrinsics = torch.tensor(raw_data['intrinsics'], dtype=torch.float32)
        # print(images.dtype, depths.dtype, extrinsics.dtype, intrinsics.dtype)
        images, intrinsics, depths = crop_images(images, self.image_size, mode='random', intrinsics=intrinsics, depth=depths)
        depths = depths.squeeze(1)
        depth_mask = torch.isfinite(depths)
        mask = depth_mask & (depths < depths[depth_mask].min() * self.max_depth_range)
        # get the raw point cloud in camera space
        pcs_camera_raw = utils3d.torch.depth_to_points(depths, intrinsics)
        pcs_world_raw = utils3d.torch.depth_to_points(depths, intrinsics, extrinsics)
        pcs_world_raw = pcs_world_raw[mask].reshape(-1, 3)
        mean = pcs_world_raw.mean(0)
        scale = (pcs_world_raw - mean).abs().max()
        # normalize extrinsics
        extrinsics = normalize_extrinsics(extrinsics, scale, mean)
        # set all invalid points to 0        
        pcs_camera_raw[~mask] = 0
        xyzs = pcs_camera_raw.permute(0, 3, 1, 2) / scale
        if self.use_mask:
            xyzs = torch.cat([xyzs, mask.unsqueeze(1).float()], dim=1)

        frames = torch.cat([images, xyzs], dim=1)
        
        # transform to relative pose
        inv_extrinsics = torch.inverse(extrinsics[0])
        extrinsics = extrinsics @ inv_extrinsics
        
        cond = torch.cat([extrinsics.flatten(-2), intrinsics.flatten(-2)], dim=-1)
            
        return {'frames': frames, 'cond' : cond, 'scale' : scale, 'mean' : mean}
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        if 'vae' in self.mode:
            if self.mode == 'xyz_vae' and self.use_mask:
                return sample['image'][:, :3] * sample['image'][:, 3:]
            return sample['image']
        if self.mode == 'cross_diffusion':
            frames = sample['frames'].flatten(0, 1)
            return {
                'image': frames[:, :3],
                'xyz': frames[:, 3:]
            }
        return sample['frames'].flatten(0, 1)

    @staticmethod
    def collate_fn_rgb_vae(batch):
        pack = {
            'image': torch.cat([b['frames'][:, :3] for b in batch])
        }
        return pack
    
    @staticmethod
    def collate_fn_xyz_vae(batch):
        frame_size = batch[0]['frames'].shape[0]
        pack = {
            'image': torch.cat([b['frames'][:, 3:] for b in batch]),
            'scale' : torch.cat([b['scale'].repeat(frame_size) for b in batch]),
            'mean' : torch.cat([b['mean'].repeat(frame_size, 1) for b in batch])
        }
        return pack
    
    @staticmethod
    def collate_fn_rgb_diffusion(batch):
        pack = {
            'frames': torch.stack([b['frames'][:, :3] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch]),
        }
        return pack
    
    @staticmethod
    def collate_fn_xyz_diffusion(batch):
        pack = {
            'frames': torch.stack([b['frames'][:, 3:] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch]),
        }
        return pack
    
    @staticmethod
    def collate_fn_cross_diffusion(batch):
        pack = {
            'frames': torch.stack([b['frames'] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch]),
        }
        return pack
    
    @property
    def collate_fn(self):
        return getattr(self, f'collate_fn_{self.mode}')

    def __getitem__(self, index):
        if self.debug:
            return self._get_video(index)
        try:
            data = self._get_video(index)
        except Exception as e:
            print(f'Error loading {self.instances[index]}: {e}')
            return self.__getitem__(np.random.randint(len(self.instances)))
        return data
    



class TartanairHistory(TartanairVideoSTD):
    def __init__(
        self,
        root,
        window_size=48,
        max_num_history=4,
        **kwargs):
        super().__init__(root, mode='history_diffusion', **kwargs)
        self.window_size = window_size
        self.max_num_history = max_num_history
    
    def _get_video(self, idx, st_frame=None, num_history=None, frame_size=None):
        scene_path, num_frames = self.instances[idx]
        scene_path = Path(self.root, scene_path)
        frame_size = frame_size or self.frame_size
        assert num_frames >= frame_size, f'Instance {scene_path} has less than {frame_size} frames.'
        st_window = np.random.randint(num_frames - self.window_size + 1)
        if st_frame is None:
            st_frame = np.random.randint(st_window, st_window + self.window_size - frame_size + 1)
        if num_history is None:
            num_history = np.random.randint(0, self.max_num_history + 1)
        # print(st_frame, scene_path)
        # print(scene_path, st_frame)
        frame_list = list(range(st_frame, st_frame + frame_size))
        if num_history > 0:
            history_frames = [st_window + v * self.window_size // (num_history + 1)for v in range(1, num_history + 1)]
            frame_list = history_frames + frame_list
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
        pcs_camera_raw = utils3d.torch.depth_to_points(depths[num_history:], intrinsics)        
        # set all invalid points to 0    
        pcs_camera_raw[~mask[num_history:]] = 0
        xyzs = pcs_camera_raw.permute(0, 3, 1, 2) / scale
        if self.use_mask:
            xyzs = torch.cat([xyzs, mask[num_history:].unsqueeze(1).float()], dim=1)
        frames = torch.cat([images[num_history:], xyzs], dim=1)
        # transform to relative pose
        inv_extrinsics = torch.inverse(extrinsics[0])
        extrinsics = extrinsics @ inv_extrinsics
        cond = torch.cat([extrinsics.flatten(-2), intrinsics.flatten(-2)], dim=-1)
        
        history_pcs = pcs_world_raw[:(num_history + 1)][mask[:(num_history + 1)]].reshape(-1, 3)
        history_rgb = images[:(num_history + 1)].permute(0, 2, 3, 1)[mask[:(num_history + 1)]]
        
        history_scene = PcScene(
            xyz=history_pcs, 
            rgb=history_rgb, 
            scale=scale.item(), 
            extrinsics=extrinsics_raw[num_history:], 
            intrinsics=intrinsics_raw[num_history:],
            history_frames=images[:(num_history+1)] if self.debug else None)
        
        return {'frames': frames[:frame_size], 'cond' : cond[:frame_size], 'history_scenes' : history_scene}
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        frames = sample['frames'].flatten(0, 1)
        return {
            'image': frames[:, :3],
            'xyz': frames[:, 3:]
        }

    @staticmethod
    def collate_fn_history_diffusion(batch):
        pack = {
            'frames': torch.stack([b['frames'] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch]),
            'history_scenes' : [b['history_scenes'] for b in batch]
        }
        return pack