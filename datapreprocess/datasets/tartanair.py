
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
from easydict import EasyDict as edict

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


class TartanairSimpleLoader:
    '''
    for constructing a clip video for the conviniance of training
    '''
    def __init__(
        self,
        root,
        image_size=128,
        window_size=160,
    ):
        self.root = root
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        metadata = pd.read_csv(Path(root, 'metadata.csv'))
        self.instances = [(row['path'], row['num_frames']) for _, row in metadata.iterrows()]
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.instances))
        
        self.window_size = window_size
    
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
    
    def __call__(self, idx, st_window=None):
        scene_path, num_frames = self.instances[idx]
        video_info = {"scene_path": scene_path}
        scene_path = Path(self.root, scene_path)
        frame_size = self.window_size
        assert num_frames >= frame_size, f'Instance {scene_path} has less than {frame_size} frames.'
        if st_window is None:
            st_window = np.random.randint(num_frames - self.window_size + 1)
        else:
            st_window = min(st_window, num_frames - self.window_size)
        frame_list = list(range(st_window, st_window + frame_size))
        video_info['frame_list'] = frame_list
        raw_data = self.load_raw_datas(scene_path, frame_list)
        
        images = torch.tensor(raw_data['images'], dtype=torch.float32).permute(0, 3, 1, 2).float() / 255.0
        depths = torch.tensor(raw_data['depths'], dtype=torch.float32).unsqueeze(1)
        extrinsics_raw = torch.tensor(raw_data['extrinsics'], dtype=torch.float32)
        intrinsics_raw = torch.tensor(raw_data['intrinsics'], dtype=torch.float32)
        # print(images.dtype, depths.dtype, extrinsics.dtype, intrinsics.dtype)
        images, intrinsics_raw, depths = crop_images(images, self.image_size, mode='random', intrinsics=intrinsics_raw, depth=depths)
        
        return {
            'images': images,
            'depths': depths.squeeze(1),
            'extrinsics' : extrinsics_raw,
            'intrinsics' : intrinsics_raw,
            'video_info': video_info
        }


class TartanairWarpBatchGenerator:
    def __init__(
        self,
        root,
        raw_subfolder = "raw",
        latent_subfolder = "frame_latents",
        warp_subfolder = "warp_batchs",
        clip_window_size=39,
        latent_window_size=9,
        max_num_history_latent=16+2+1,
        max_depth_range=128,
        debug=False,
        use_mask=False,
        start_section_ratio=0.4,
    ):
        self.root = Path(root)
        self.raw_subfolder = raw_subfolder
        self.latent_subfolder = latent_subfolder
        self.warp_subfolder = warp_subfolder
        self.instances = (root / ".index.txt").read_text().splitlines()
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.instances))
        
        self.max_depth_range = max_depth_range
        
        self.debug = debug
        self.use_mask = use_mask

        self.start_section_ratio = start_section_ratio
        self.clip_window_size = clip_window_size
        self.max_num_history_latent = max_num_history_latent
        self.latent_window_size = latent_window_size
    
    def greedy_sample_history(self, history_extrinsics, generate_extrinsics):
        
        history_origin = self.get_camera_orgin_from_extrinsics(history_extrinsics)
        generate_origin = self.get_camera_orgin_from_extrinsics(generate_extrinsics)

        # calculate the distance between the camera origin of the history frames and the generate frame
        history_origin_numpy = history_origin.cpu().numpy()
        generate_origin_numpy = generate_origin.cpu().numpy()
        # greedy sample the best history subset
        subset_size = min(self.max_num_warp_cache, len(history_origin_numpy))
        _, history_sample = greedy_subset_selection_numpy(history_origin_numpy, generate_origin_numpy, k=subset_size)
        return history_sample
    
    def sparse_sample_history(self, history_list):
        # log sample frames
        # get log2 of num_history
        log_num_history = int(np.log2(len(history_list)))
        return [history_list[-2 ** i] for i in range(log_num_history + 1)]
        
    def load_raw_datas(self, scene_path : str):
        raw_scene_path = Path(self.root, self.raw_subfolder, scene_path)
        latent_scene_path = Path(self.root, self.latent_subfolder, scene_path)
        clip_frames = load_video(raw_scene_path / "frames.mp4")
        depths = torch.load(raw_scene_path / "depths.pt")
        clip_latents = torch.load(latent_scene_path / "latents.pt")
        meta = read_meta(raw_scene_path / "meta.json")
        extrinsics = torch.tensor(meta['extrinsics'], dtype=torch.float32)
        intrinsics = torch.tensor(meta['intrinsics'], dtype=torch.float32)
        return edict({
            "frames": clip_frames,
            "depths": depths,
            "latents": clip_latents,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics
        })

    def __len__(self):
        return len(self.instances)
    
    def fetch_clip_data(self, idx):
        scene_path = self.instances[idx]
        video_info = {"scene_path": scene_path}
        clip_data =  self.load_raw_datas(scene_path)
        clip_data['video_info'] = video_info
        return clip_data

    def lift_depth_to_points(self, depths, extrinsics, intrinsics, frames=None):
        depth_mask = torch.isfinite(depths)
        mask = depth_mask & (depths < depths[depth_mask].min() * self.max_depth_range)
        # get the raw point cloud in camera space
        pcs_world_raw = utils3d.torch.depth_to_points(depths, intrinsics, extrinsics)
        pcs = pcs_world_raw[mask].reshape(-1, 3)
        if frames is not None:
            rgb = frames.permute(0, 2, 3, 1)[mask]
        else:
            rgb = None
        return pcs, rgb
    
    def generate_from_clip_data(self, clip_data, st_latent=None, section_size=None):
        # else random history_latent
        if section_size is None:
            if np.random.rand() < self.start_section_ratio:
                section_size = 1 + self.latent_window_size
            else:
                section_size = np.random.randint(1 + self.latent_window_size, self.clip_window_size + 1)

        if st_latent is None:
            st_latent = np.random.randint(self.clip_window_size - section_size + 1)
        else:
            st_latent = min(st_latent, self.clip_window_size - section_size)
        section_latents = clip_data.latents[:, st_latent:st_latent + section_size]
        generate_latents = section_latents[:, -self.latent_window_size:]
        history_latents = section_latents[:, 1:-self.latent_window_size]
        if len(history_latents) > self.max_num_history_latent:
            history_latents = history_latents[:, -self.max_num_history_latent:]
        print("section_size", section_size)
        print("history_latents", history_latents.shape)
        print("generate_latents", generate_latents.shape)
        
        st_window = st_latent * 4
        end_window = st_window + section_size * 4 - 3
        start_generate = end_window - self.latent_window_size * 4
        start_history = max(st_window + 1, start_generate - self.max_num_history_latent * 4)
        
        # generate_frames = clip_data.frames[start_generate:end_window]
        generate_depth = clip_data.depths[start_generate:end_window]
        generate_extrinsics = clip_data.extrinsics[start_generate:end_window]
        generate_intrinsics = clip_data.intrinsics[start_generate:end_window]
        gen_world_raw, _ = self.lift_depth_to_points(generate_depth, generate_extrinsics, generate_intrinsics)

        mean = gen_world_raw.mean(0)
        scale = (gen_world_raw - mean).abs().max()
        extrinsics_normalized = generate_extrinsics.clone()
        extrinsics_normalized = normalize_extrinsics(extrinsics_normalized, scale, mean)
        inv_extrinsics = torch.inverse(extrinsics_normalized[0])
        extrinsics_normalized = extrinsics_normalized @ inv_extrinsics
        
        history_list = [st_window] + list(range(start_history, start_generate))
        # greedy sample the best history subset
        history_sample = self.sparse_sample_history(history_list)
        # construct history scene
        history_images = (clip_data.frames[history_sample] + 1) / 2.0
        history_depth = clip_data.depths[history_sample]
        history_extrinsics = clip_data.extrinsics[history_sample]
        history_intrinsics = clip_data.intrinsics[history_sample]
        history_pcs, history_rgb = self.lift_depth_to_points(history_depth, history_extrinsics, history_intrinsics, frames=history_images)
        
        history_scene = PcScene(
            xyz=history_pcs.clone(), 
            rgb=history_rgb.clone(), 
            scale=scale.item(), 
            extrinsics=clip_data.extrinsics[start_generate-1:end_window].clone(), 
            intrinsics=clip_data.intrinsics[start_generate-1:end_window].clone(),
            history_frames=history_images if self.debug else None)
         
        
        return {'start_image': clip_data.frames[st_window:st_window + 1],
                "history_latents": history_latents,
                "frame_latents": generate_latents,
                'extrinsics' : extrinsics_normalized,
                'intrinsics' : generate_intrinsics, 
                'history_scene' : history_scene,
                'video_info': clip_data.video_info,}