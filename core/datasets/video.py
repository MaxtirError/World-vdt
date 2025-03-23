import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from core.utils.image_utils import *
from core.utils.camera_utils import *

class VideoSTD(Dataset):
    def __init__(
        self,
        root,
        frame_size=16,
        image_size=128,
        mode='train',
        image_type='rgb',
        folder = "dust3r",
        debug=False
    ):
        assert image_type in ['rgb', 'xyz', 'rgbxyz', 'history']
        super().__init__()
        self.root = root
        self.frame_size = frame_size
        self.image_size = image_size
        self.mode = mode
        self.metadata = pd.read_csv(os.path.join(root, f'{mode}_metadata.csv'))
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.metadata))
        self.instances = self.metadata[self.metadata['num_frames'] >= frame_size]['scene_id'].tolist()
        # self.instances = open(os.path.join(root, f'{mode}_instances.txt')).read().splitlines()
        print(f'Loaded {len(self.instances)} instances that more then {frame_size} for {mode} mode.')
        self.image_type = image_type
        self.folder = folder
        print(f'Using {image_type} images.')
        self.root = os.path.join(root, folder)
        self.value_range = (0, 1)
        if image_type == 'xyz':
            self.value_range = (-1, 1)
        self.debug = debug
    
    def __len__(self):
        return len(self.instances)
    
    @staticmethod
    def load_frames_metadata(meta, st_frame, frame_size):
        meta_view = {}
        frame_keys = ['focals', 'intrinsics', 'cams2world']
        for k, v in meta.items():
            if k in frame_keys:
                meta_view[k] = torch.tensor(v[st_frame:st_frame+frame_size]).float()
            else:
                meta_view[k] = torch.tensor(v).float()
        return meta_view
        
    def _get_video(self, idx):
        
        instance = self.instances[idx]
        with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
            meta = json.load(f)
        num_frames = len(meta['focals'])
        image_size = meta["image_size"]
        H, W = image_size[0], image_size[1]
        assert num_frames >= self.frame_size, f'Instance {instance} has less than {self.frame_size} frames.'

        st_frame = np.random.randint(num_frames - self.frame_size + 1)
        meta = self.load_frames_metadata(meta, st_frame, self.frame_size)
        if torch.isnan(intrinsics).any() or torch.isnan(extrinsics).any():
            raise ValueError('NaN in condition')
        
        image_paths = [os.path.join(self.root, self.mode, instance, f"{view:03d}_{self.image_type}.png") for view in range(st_frame, st_frame + self.frame_size)]
        frames = read_images(image_paths)
        frames, intrinsics = crop_images(frames, self.image_size, mode='random', intrinsics=intrinsics)

        if self.image_type == 'xyz':
            frames = transform_xyz(frames, extrinsics)
            frames = frames[:, :3]
            
        # transform to relative pose
        inv_extrinsics = torch.inverse(extrinsics[0])
        extrinsics = extrinsics @ inv_extrinsics
        
        cond = torch.cat([extrinsics.flatten(-2), intrinsics.flatten(-2)], dim=-1)
        if torch.isnan(cond).any():
            #save the instance for further analysis
            
            with open("nan_instances.txt", "w") as f:
                f.write(instance)
            # save metadata
            with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
                remeta = json.load(f)
            with open("nan_metadata.json", "w") as f:
                json.dump(remeta, f, indent=4)
            raise ValueError('NaN in condition')
            
        return {'frames': frames, 'cond' : cond}
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['frames'].flatten(0, 1)

    @staticmethod
    def collate_fn(batch):
        pack = {
            'frames': torch.stack([b['frames'] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch])
        }
        return pack

    def __getitem__(self, index):
        if self.debug:
            return self._get_video(index)
        try:
            data = self._get_video(index)
        except Exception as e:
            print(f'Error loading {self.instances[index]}: {e}')
            return self.__getitem__(np.random.randint(len(self.instances)))
        return data

class VideoRGBXYZ(VideoSTD):
    def _get_video(self, idx):
        instance = self.instances[idx]
        with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
            meta = json.load(f)
        num_frames = len(meta['focals'])
        image_size = meta["image_size"]
        H, W = image_size[0], image_size[1]
        assert num_frames >= self.frame_size, f'Instance {instance} has less than {self.frame_size} frames.'

        st_frame = np.random.randint(num_frames - self.frame_size + 1)
        meta = self.load_frames_metadata(meta, st_frame, self.frame_size)
        extrinsics = normalize_extrinsics(torch.inverse(meta['cams2world']), meta["scale"], meta["mu"])
        intrinsics = normalize_intrinsics(meta['intrinsics'], H, W)
        if torch.isnan(intrinsics).any() or torch.isnan(extrinsics).any():
            raise ValueError('NaN in condition')
        
        rgb_images = [os.path.join(self.root, self.mode, instance, f"{view:03d}_rgb.png") for view in range(st_frame, st_frame + self.frame_size)]
        xyz_images = [os.path.join(self.root, self.mode, instance, f"{view:03d}_xyz.png") for view in range(st_frame, st_frame + self.frame_size)]
        rgb_frames = read_images(rgb_images)
        xyz_frames = read_images(xyz_images)
        frames = torch.cat([rgb_frames, xyz_frames], dim=1)
        frames, intrinsics = crop_images(frames, self.image_size, mode='random', intrinsics=intrinsics)
        # split to xyzframe and rgbframe
        rgb_frames = frames[:, :3]
        xyz_frames = frames[:, 3:]
        
        xyz_frames = transform_xyz(xyz_frames, extrinsics)
        xyz_frames = xyz_frames[:, :3]
            
        # transform to relative pose
        inv_extrinsics = torch.inverse(extrinsics[0])
        extrinsics = extrinsics @ inv_extrinsics
        
        cond = torch.cat([extrinsics.flatten(-2), intrinsics.flatten(-2)], dim=-1)
        if torch.isnan(cond).any():
            raise ValueError('NaN in condition')
        
        frames = torch.cat([rgb_frames, xyz_frames], dim=1)
            
        return {'frames' : frames, 'cond' : cond}
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        rgb_frames, xyz_frames = sample['frames'].chunk(2, dim=2)
        return {
            "rgb" : rgb_frames.flatten(0, 1),
            "xyz" : (xyz_frames.flatten(0, 1) + 1.0) / 2.0,
        }

class VideoHistory(VideoSTD):
    def __init__(
        self,
        root,
        max_num_history=4,
        **kwargs,
        ):
        super().__init__(root, **kwargs)
        self.max_num_history = max_num_history
    
    @staticmethod
    def load_frames_metadata(meta, frame_list):
        meta_view = {}
        frame_keys = ['focals', 'intrinsics', 'cams2world']
        for k, v in meta.items():
            if k in frame_keys:
                meta_view[k] = torch.tensor([v[f] for f in frame_list]).float()
            else:
                meta_view[k] = torch.tensor(v).float()
        return meta_view    
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        rgb_frames, xyz_frames = sample['frames'].chunk(2, dim=2)
        return {
            "rgb" : rgb_frames.flatten(0, 1),
            "xyz" : (xyz_frames.flatten(0, 1) + 1.0) / 2.0,
        }
    
    def _get_video(self, idx):
        instance = self.instances[idx]
        with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
            meta = json.load(f)
        num_frames = len(meta['focals'])
        image_size = meta["image_size"]
        H, W = image_size[0], image_size[1]
        assert num_frames >= self.frame_size, f'Instance {instance} has less than {self.frame_size} frames.'

        st_frame = np.random.randint(num_frames - self.frame_size + 1)
        num_history = np.random.randint(self.max_num_history + 1)
        num_history = min(num_history, st_frame)
        frame_list = list(range(st_frame, st_frame + self.frame_size))
        if self.mode == 'test':
            num_history = 3
        if num_history > 0:
            if self.mode == 'train':
                # random select num_history frames from [0, st_frame - 1]
                history_list = list(np.random.choice(range(st_frame), num_history, replace=False))
                frame_list = history_list + frame_list
            if self.mode == 'test':
                history_list = list(range(st_frame + self.frame_size, min(st_frame + self.frame_size + num_history, num_frames)))
                num_history = len(history_list)
                frame_list = history_list + frame_list
        meta = self.load_frames_metadata(meta, frame_list)
        extrinsics = normalize_extrinsics(torch.inverse(meta['cams2world']), meta["scale"], meta["mu"])
        intrinsics = normalize_intrinsics(meta['intrinsics'], H, W)
        # print(extrinsics.shape, intrinsics.shape)
        if torch.isnan(intrinsics).any() or torch.isnan(extrinsics).any():
            raise ValueError('NaN in condition')
        
        rgb_images = [os.path.join(self.root, self.mode, instance, f"{view:03d}_rgb.png") for view in frame_list]
        xyz_images = [os.path.join(self.root, self.mode, instance, f"{view:03d}_xyz.png") for view in frame_list]
        rgb_frames = read_images(rgb_images)
        xyz_frames = read_images(xyz_images)
        frames = torch.cat([rgb_frames, xyz_frames], dim=1)
        frames, intrinsics = crop_images(frames, self.image_size, mode='random', intrinsics=intrinsics)
        # split to xyzframe and rgbframe
        rgb_frames = frames[num_history:, :3]
        xyz_frames = frames[num_history:, 3:]
        # st frame is also included in history
        hisotry_frames = frames[:(num_history + 1)]
        history_rgb = hisotry_frames[:, :3]
        history_xyz = hisotry_frames[:, 3:]
        
        extrinsics = extrinsics[num_history:]
        intrinsics = intrinsics[num_history:]
        xyz_frames = transform_xyz(xyz_frames, extrinsics)
        xyz_frames = xyz_frames[:, :3]
            
        # transform to relative pose
        ref_extrinsics = extrinsics[0]
        inv_extrinsics = torch.inverse(ref_extrinsics)
        extrinsics = extrinsics @ inv_extrinsics
        
        cond = torch.cat([extrinsics.flatten(-2), intrinsics.flatten(-2)], dim=-1)
        if torch.isnan(cond).any():
            raise ValueError('NaN in condition')
        
        frames = torch.cat([rgb_frames, xyz_frames], dim=1)
        
        history_xyz = history_xyz.permute(0, 2, 3, 1)
        history_rgb = history_rgb.permute(0, 2, 3, 1)
        history_mask = history_xyz[..., 3] > 0
        history_pcs = history_xyz[history_mask][..., :3] * 2 - 1
        history_rgb = history_rgb[history_mask]
        history_pcs = apply_transforms(history_pcs, ref_extrinsics)
        if history_pcs.shape[0] < 8192:
            # skip too small history pcs
            raise ValueError('Too small history pcs')
            
        # num_history from 0-4
        return {'frames' : frames, 'cond' : cond, 'history_pcs': history_pcs, 'history_rgb': history_rgb, 'history_frames': hisotry_frames}
    
    @staticmethod
    def collate_fn(batch):
        pack = {
            'frames': torch.stack([b['frames'] for b in batch]),
            'cond': torch.stack([b['cond'] for b in batch]),
            'history_pcs': [b['history_pcs'] for b in batch],
            'history_rgb': [b['history_rgb'] for b in batch],
        }
        return pack
    