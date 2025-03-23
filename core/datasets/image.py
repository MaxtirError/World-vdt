import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from core.utils.image_utils import read_image, crop_images, load_images_dust3r, apply_mask
from core.utils.camera_utils import normalize_extrinsics, apply_transforms
import utils3d

class ImageSTDVAE(Dataset):
    def __init__(
        self,
        root,
        image_size=128,
        mode='train'
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.instances = open(os.path.join(root, f'{mode}_instances.txt')).read().splitlines()
        self.value_range = (0, 1)
        print(f'Loaded {len(self.instances)} instances for {mode} mode.')

 
    def __len__(self):
        return len(self.instances)

    def _get_image(self, idx):
        instance = self.instances[idx]
        with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata)
        view = np.random.randint(n_views)
        metadata = metadata[view]

        image_path = os.path.join(self.root, self.mode, instance, metadata['image'])
        image = load_images_dust3r(image_path)
        image = crop_images(image, self.image_size, mode='random')
        
        return {'image': image}

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {
            'image': torch.stack([b['image'] for b in batch]),
        }
        return pack

    def __getitem__(self, index):
        try:
            data = self._get_image(index)
        except Exception as e:
            print(f'Error loading {self.instances[index]}: {e}')
            return self.__getitem__(np.random.randint(len(self.instances)))
        return data

class RGBXYZVAE(ImageSTDVAE):
    def __init__(
        self,
        root,
        image_size=128,
        mode='train',
        image_type='rgb',
        folder = "dust3r",
        no_mask = False
    ):
        assert image_type in ['rgb', 'xyz']
        self.image_type = image_type
        self.folder = folder
        print(f'Using {image_type} images.')
        super().__init__(root, image_size, mode)
        self.root = os.path.join(root, folder)
        self.no_mask = no_mask
        if image_type == 'xyz':
            self.value_range = (-1, 1)
    
    @staticmethod
    def load_view_metadata(meta, view):
        meta_view = {}
        frame_keys = ['focals', 'intrinsics', 'cams2world']
        for k, v in meta.items():
            if k in frame_keys:
                meta_view[k] = torch.tensor(v[view]).float()
            else:
                meta_view[k] = torch.tensor(v).float()
        return meta_view
        
    def _get_image(self, idx):
        
        instance = self.instances[idx]
        with open(os.path.join(self.root, self.mode, instance, 'meta.json')) as f:
            meta = json.load(f)        
        image_size = meta["image_size"]
        H, W = image_size[0], image_size[1]
        view = np.random.randint(len(meta["focals"]))
        # print(view, meta['focals'][view])
        image_path = os.path.join(self.root, self.mode, instance, f"{view:03d}_{self.image_type}.png")

        if self.image_type == 'rgb':
            image = read_image(image_path)
            image = crop_images(image, self.image_size, mode='random')
            return {'image': image}
        # transform xyz
        meta = self.load_view_metadata(meta, view)
        # intrinsics = meta["intrinsics"]
        extrinsics = normalize_extrinsics(meta["cams2world"].inverse(), meta["scale"], meta["mu"])
        # normalize extrinsics
        xyz = torch.tensor(np.array(Image.open(image_path))) / 255
        xyz = crop_images(xyz.permute(2, 0, 1), self.image_size, mode='random').permute(1, 2, 0)
        mask = xyz[..., 3]
        xyz = xyz[..., :3] * 2 - 1
        xyz = apply_transforms(xyz.flatten(0, 1), extrinsics).reshape(self.image_size, self.image_size, 3)
        xyz = apply_mask(xyz, mask).permute(2, 0, 1)
        if self.no_mask:
            xyz = xyz[:3]
        return {'image': xyz}
        
        
        