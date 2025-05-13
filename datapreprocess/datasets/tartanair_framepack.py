
import os
import sys
import numpy as np
import pandas as pd
import torch
from utils import *
import utils3d
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from representations.pc_scene import PcScene

class TartanairFramePackLoader:
    def __init__(
        self,
        root,
        latent_window_size=4,
        max_section_latent_window_size=40,
        max_num_warp_cache=4,
        max_num_history_latent=16+2+1,
        image_size=(480, 832),
        max_depth_range=128,
        debug=False,
        use_mask=False,
        start_section_ratio=0.4,
    ):
        self.root = root
        self.max_section_latent_window_size = max_section_latent_window_size
        self.latent_window_size = latent_window_size
        self.frame_size = latent_window_size * 4
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        metadata = pd.read_csv(Path(root, 'metadata.csv'))
        self.instances = [(row['path'], row['num_frames']) for _, row in metadata.iterrows()]
        # filter out instances with less than frame_size frames
        print("total valid instances:", len(self.instances))
        
        self.max_depth_range = max_depth_range
        
        self.debug = debug
        self.use_mask = use_mask

        self.start_section_ratio = start_section_ratio
        self.max_num_history_frames = max_num_history_latent * 4
        self.max_num_warp_cache = max_num_warp_cache
    
    @staticmethod
    def get_camera_orgin_from_extrinsics(extrinsics):
        # extrinsics: (N, 4, 4)
        # camera origin in world space
        inv_extrinsics = torch.inverse(extrinsics)
        camera_origin = inv_extrinsics[:, :3, 3]  # (N, 3)
        return camera_origin
    
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
    
    def visualize_camera_trajectory(self, history_extrinsics, generate_extrinsics):
        import matplotlib.pyplot as plt
        # extrinsics: (N, 4, 4)
        camera_origin = self.get_camera_orgin_from_extrinsics(history_extrinsics)
        camera_origin_numpy = camera_origin.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(camera_origin_numpy[:, 0], camera_origin_numpy[:, 1], camera_origin_numpy[:, 2], c='r', label='history')
        camera_origin_gen = self.get_camera_orgin_from_extrinsics(generate_extrinsics)
        camera_origin_gen_numpy = camera_origin_gen.cpu().numpy()
        ax.scatter(camera_origin_gen_numpy[:, 0], camera_origin_gen_numpy[:, 1], camera_origin_gen_numpy[:, 2], c='b', label='generate')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # save the figure
        plt.savefig('debugs/vis_camera_trajectory.png')
    
    def sparse_sample_history(self, num_history):
        # log sample frames
        # get log2 of num_history
        log_num_history = int(np.log2(num_history))
        return [num_history - 2 ** i for i in range(log_num_history + 1)]
        
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
    
    def __call__(self, idx, st_window=None, section_size=None):
        scene_path, num_frames = self.instances[idx]
        video_info = {"scene_path": scene_path}
        scene_path = Path(self.root, scene_path)

        num_latents = (num_frames + 3) // 4
        max_window_size = min(num_latents, self.max_section_latent_window_size)
        
        # section_size = 1(start_latent) + history_latent + latent_window_size
        # for start section, history_latent = 0
        # else random history_latent
        if section_size is None:
            if np.random.rand() < self.start_section_ratio:
                section_size = 1 + self.latent_window_size
            else:
                section_size = np.random.randint(1 + self.latent_window_size, max_window_size + 1)
        
        total_frame_size = section_size * 4 - 3

        if st_window is None:
            st_window = np.random.randint(num_frames - total_frame_size + 1)
        else:
            st_window = min(st_window, num_frames - total_frame_size)

        # if the real_num_history_frames <= max_num_history_frames, then the frame list is st_window and its following total_frame_size frames
        # else we only hold the st_frames, ..., max_num_history_frames, latent_window_size
        end_window = st_window + total_frame_size
        start_generate = end_window - self.latent_window_size * 4
        start_history = max(st_window + 1, start_generate - self.max_num_history_frames)
        
        frame_list_to_load = [st_window] + list(range(start_history, start_generate)) + list(range(start_generate, end_window))
        video_info['frame_list'] = frame_list_to_load
        num_history = len(frame_list_to_load) - self.latent_window_size * 4
        raw_data = self.load_raw_datas(scene_path, frame_list_to_load)
        # first resize to target size
        images = torch.tensor(raw_data['images'], dtype=torch.float32).permute(0, 3, 1, 2).float() / 255.0
        depths = torch.tensor(raw_data['depths'], dtype=torch.float32).unsqueeze(1)
        extrinsics_raw = torch.tensor(raw_data['extrinsics'], dtype=torch.float32)
        intrinsics_raw = torch.tensor(raw_data['intrinsics'], dtype=torch.float32)
        images, intrinsics_raw, depths = crop_images(images, self.image_size, mode='random', intrinsics=intrinsics_raw, depth=depths)
        depths = depths.squeeze(1)
        depth_mask = torch.isfinite(depths)
        mask = depth_mask & (depths < depths[depth_mask].min() * self.max_depth_range)
        # get the raw point cloud in camera space
        pcs_world_raw = utils3d.torch.depth_to_points(depths[num_history:], intrinsics_raw[num_history:], extrinsics_raw[num_history:])
        
        gt_world_raw = pcs_world_raw[mask[num_history:]].reshape(-1, 3)
        mean = gt_world_raw.mean(0)
        scale = (gt_world_raw - mean).abs().max()
        extrinsics_normalized = extrinsics_raw.clone()
        extrinsics_normalized = normalize_extrinsics(extrinsics_normalized, scale, mean)
        inv_extrinsics = torch.inverse(extrinsics_normalized[0])
        extrinsics_normalized = extrinsics_normalized @ inv_extrinsics
        
        history_extrinsics = extrinsics_raw[:num_history]
        generate_extrinsics = extrinsics_raw[num_history:]
        # greedy sample the best history subset
        history_sample = self.sparse_sample_history(num_history)
        
        # construct history scene
        history_images = images[:num_history][history_sample]
        history_depth = depths[:num_history][history_sample]
        history_extrinsics = extrinsics_raw[:num_history][history_sample]
        history_intrinsics = intrinsics_raw[:num_history][history_sample]
        history_mask = mask[:num_history][history_sample]
        history_pcs_world = utils3d.torch.depth_to_points(history_depth, history_intrinsics, history_extrinsics)
        history_pcs = history_pcs_world[history_mask].reshape(-1, 3)
        history_rgb = history_images.permute(0, 2, 3, 1)[history_mask]
        
        history_scene = PcScene(
            xyz=history_pcs, 
            rgb=history_rgb, 
            scale=scale.item(), 
            extrinsics=extrinsics_raw[num_history:], 
            intrinsics=intrinsics_raw[num_history:],
            history_frames=history_images[:num_history] if self.debug else None)
         
        
        return {'frames': images[num_history:], 
                'extrinsics' : extrinsics_normalized,
                'intrinsics' : intrinsics_raw[num_history:],
                'history_frames': images[:num_history], 
                'history_scene' : history_scene, 
                'video_info': video_info}