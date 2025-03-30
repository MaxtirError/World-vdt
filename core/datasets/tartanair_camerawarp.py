
from torch.utils.data import Dataset
import imageio
from pathlib import Path
import torch
import numpy as np
import json

class TartanAirCameraWarpDataset(Dataset):
    def __init__(self,
        root : str,
        num_frames : int,
        height : int,
        width : int,):
        '''
        This class loads TartanAir dataset for camera warp task.
        dataset structure:
        root
        ├── .index.txt
        ├── instance_path
        │   ├── frames.mp4
        │   ├── meta.json
        |   ├── warp_frames.mp4
        |   ├── mask.npy
        Args:
            root: path to the dataset
            num_frames: number of frames to load
            height: height of the images
            width: width of the images
            
        '''
        self.root = Path(root)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        # read .index.txt file and splitlines for instances path
        self.instances = (self.root / ".index.txt").read_text().splitlines()

    def __len__(self):
        return len(self.instances)
    
    def _load_video(self, video_path):
        '''
        Args:
            video_path: path to the video
        Returns:
            video: (num_frames, 3, H, W) tensor of images
        '''
        video = imageio.get_reader(str(video_path))
        video = [frame for frame in video.iter_data()]
        video = torch.tensor(np.array(video)).permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
        return video.float()
    
    def _load_mask(self, mask_path):
        '''
        Args:
            mask_path: path to the mask
        Returns:
            mask: (num_frames, H, W) tensor of mask
        '''
        mask = torch.tensor(np.load(mask_path) / 255.0).float()
        return mask
    
    def _load_meta(self, meta_path):
        '''
        Args:
            meta_path: path to the meta file
        Returns:
            meta: dictionary of meta information
        '''
        return json.load(open(meta_path, "r"))
        

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the instance
        Returns:
            frames: (num_frames, H, W, 3) tensor of images
            intrinsics: (num_frames, 3, 3) tensor of intrinsics
            extrinsics: (num_frames, 4, 4) tensor of extrinsics
        '''
        instance = self.instances[idx]
        instance_path = self.root / instance
        frames = self._load_video(instance_path / "frames.mp4")
        warp_frames = self._load_video(instance_path / "warp_frames.mp4")
        masks = self._load_mask(instance_path / "mask.npy")
        meta = self._load_meta(instance_path / "meta.json")
        intrinsics = torch.tensor(meta["intrinsics"])
        extrinsics = torch.tensor(meta["extrinsics"])
        return {
            "frames": frames,
            "warp_frames": warp_frames,
            "masks": masks.unsqueeze(1),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }
    
    @staticmethod
    def collate_fn(batch):
        frames = torch.stack([sample["frames"] for sample in batch])
        warp_frames = torch.stack([sample["warp_frames"] for sample in batch])
        masks = torch.stack([sample["masks"] for sample in batch])
        intrinsics = torch.stack([sample["intrinsics"] for sample in batch])
        extrinsics = torch.stack([sample["extrinsics"] for sample in batch])
        return {
            "frames": frames,
            "warp_frames": warp_frames,
            "masks": masks,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }