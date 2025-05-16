
from torch.utils.data import Dataset
import imageio
from pathlib import Path
import torch
import numpy as np
import json
from transformers import SiglipImageProcessor

class TartanAirCameraWarpDataset(Dataset):
    def __init__(self,
        root : str,
        num_frames : int,
        height : int,
        width : int,
        use_precompute_vae_latent : bool = False):
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
        |   ├── latent.pt (optional)
        |   ├── ...
        Args:
            root: path to the dataset
            num_frames: number of frames to load
            height: height of the images
            width: width of the images
            use_precompute_vae_latent: whether to use precomputed VAE latent
            
        '''
        assert num_frames == 49 or num_frames == 25, "Only 49 and 25 frames are supported"
        assert (height, width) == (480, 720) or (height, width) == (320, 480), "Only 480x720 and 240x360 are supported"
        self.root = Path(root)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.use_precompute_vae_latent = use_precompute_vae_latent
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
        # get the first num_frames frames
        video = [frame for frame in video.iter_data()]
        video = torch.tensor(np.array(video)).permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
        F, C, H, W = video.shape
        if self.height != H or self.width != W:
            video = torch.nn.functional.interpolate(video, size=(self.height, self.width), mode='bilinear')
        # assert video.shape == (self.num_frames, 3, self.height, self.width), f"video shape mismatch: {video.shape} != {(self.num_frames, 3, self.height, self.width)}"
        return video.float()
    
    def _load_mask(self, mask_path):
        '''
        Args:
            mask_path: path to the mask
        Returns:
            mask: (num_frames, 1, H, W) tensor of mask
        '''
        mask = torch.tensor(np.load(mask_path)).float().unsqueeze(1)
        F, C, H, W = mask.shape
        if self.height != H or self.width != W:
            mask = torch.nn.functional.interpolate(mask, size=(self.height, self.width), mode="nearest")
        # assert mask.shape == (self.num_frames, 1, self.height, self.width), f"mask shape mismatch: {mask.shape} != {(self.num_frames, 1, self.height, self.width)}"
        return mask
    
    def _load_meta(self, meta_path):
        '''
        Args:
            meta_path: path to the meta file
        Returns:
            meta: dictionary of meta information
        '''
        return json.load(open(meta_path, "r"))
        

    def _get_video(self, idx):
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
        sample_type=""
        
        if self.num_frames < frames.shape[0]:
            assert frames.shape[0] == 49
            if np.random.rand() > 0.5:
                frames = frames[::2]
                warp_frames = warp_frames[::2]
                masks = masks[::2]
                intrinsics = intrinsics[::2]
                extrinsics = extrinsics[::2]
                sample_type = "_1"
            else:
                frames = frames[:self.num_frames]
                warp_frames = warp_frames[:self.num_frames]
                masks = masks[:self.num_frames]
                intrinsics = intrinsics[:self.num_frames]
                extrinsics = extrinsics[:self.num_frames]
                sample_type = "_0"
                
        assert frames.shape == (self.num_frames, 3, self.height, self.width), f"frames shape mismatch: {frames.shape} != {(self.num_frames, 3, self.height, self.width)}"
        assert warp_frames.shape == (self.num_frames, 3, self.height, self.width), f"warp_frames shape mismatch: {warp_frames.shape} != {(self.num_frames, 3, self.height, self.width)}"
        assert masks.shape == (self.num_frames, 1, self.height, self.width), f"masks shape mismatch: {masks.shape} != {(self.num_frames, 1, self.height, self.width)}"
        assert intrinsics.shape == (self.num_frames, 3, 3), f"intrinsics shape mismatch: {intrinsics.shape} != {(self.num_frames, 3, 3)}"
        assert extrinsics.shape == (self.num_frames, 4, 4), f"extrinsics shape mismatch: {extrinsics.shape} != {(self.num_frames, 4, 4)}"
        
        if self.use_precompute_vae_latent:
            resolution =f"{self.num_frames}x{self.height}x{self.width}"
            latent_path = instance_path / "_fitting_latent" / f"latent_{resolution}{sample_type}.pt"
            warp_latent_path = instance_path / "_fitting_latent" / f"warp_latent_{resolution}{sample_type}.pt"
            if latent_path.exists() and warp_latent_path.exists():
                latent = torch.load(latent_path)
                warp_latent = torch.load(warp_latent_path)
            else:
                raise FileNotFoundError(f"Precomputed VAE latent not found: {latent_path}")
            return {
                "frames": frames,
                "warp_frames": warp_frames,
                "masks": masks,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "latents": latent,
                "warp_latents": warp_latent
            }

        return {
            "frames": frames,
            "warp_frames": warp_frames,
            "masks": masks,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }
    
    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the instance
        Returns:
            data: dictionary of data
        '''
        try:
            data = self._get_video(idx)
        except Exception as e:
            print(f'Error loading {self.instances[idx]}: {e}')
            return self.__getitem__(np.random.randint(len(self.instances)))
        return data
    
    
    @staticmethod
    def collate_fn(batch):
        return {k : torch.stack([d[k] for d in batch]) for k in batch[0].keys()}
    


class NaiveTestDataset(Dataset):
    def __init__(self,
        num_frames : int,
        height : int,
        width : int,
        length : int = 1000,):
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
        |   ├── latent.pt (optional)
        |   ├── ...
        Args:
            root: path to the dataset
            num_frames: number of frames to load
            height: height of the images
            width: width of the images
            use_precompute_vae_latent: whether to use precomputed VAE latent
            
        '''
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.length = length
        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')

    def __len__(self):
        return self.length

    def _get_video(self, idx):
        '''
        Args:
            idx: index of the instance
        Returns:
            frames: (num_frames, H, W, 3) tensor of images
        '''
        frames = torch.rand(self.num_frames, 3, self.height, self.width) * 2 - 1.0
        cond_image_np = (frames[0, ...].permute(1, 2, 0).numpy() + 1.0) * 127.5
        cond_image_np = np.clip(cond_image_np, 0, 255).astype(np.uint8)
        preprocessed_image = self.feature_extractor(cond_image_np, return_tensors="pt").pixel_values
        return {"frames" : frames, "preprocessed_image" : preprocessed_image[0]}
    
    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the instance
        Returns:
            data: dictionary of data
        '''
        try:
            data = self._get_video(idx)
        except Exception as e:
            print(f'Error loading {self.instances[idx]}: {e}')
            return self.__getitem__(np.random.randint(len(self.instances)))
        return data
    
    
    @staticmethod
    def collate_fn(batch):
        return {k : torch.stack([d[k] for d in batch]) for k in batch[0].keys()}

class TartanAirFramePackDataset(Dataset):
    def __init__(self,
        root : str,
        height : int,
        width : int,
        latent_size : int = 1, 
        latent_window_size : int = 9,
        max_history_latent_size : int = 16+2+1):
        '''
        This class loads TartanAir dataset for camera warp task.
        dataset structure:
        root
        ├── .index.txt
        ├── instance_path
        │   ├── section_latents.pt
        │   ├── meta.json
        │   ├── start_image.png
        |   ├── warp_frames.mp4
        |   ├── warp_latent.pt
        |   ├── ...
        the section_latents shape is (C, F, H // 8, W // 8)
        first latent align with the start_image
        F can be split to [1 + history_latent_size + latent_window_size]
        Args:
            root: path to the dataset
            height: height of the images
            width: width of the images
            latent_size: size of the latent to pass to the model
            latent_window_size: size of the latent window
            max_history_latent_size: max size of the history latent
        '''
        self.root = Path(root)
        self.height = height
        self.width = width
        self.instances = (self.root / ".index.txt").read_text().splitlines()
        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        self.latent_size = latent_size
        self.latent_window_size = latent_window_size
        self.max_history_latent_size = max_history_latent_size

    def __len__(self):
        return len(self.instances)

    def check_output(self, outputs):
        latent_height = self.height // 8
        latent_width = self.width // 8
        assert outputs["start_latents"].shape[1:] == (1, latent_height, latent_width), f"start_latents shape mismatch: {outputs['start_latents'].shape[1:]} != {(1, latent_height, latent_width)}"
        assert outputs["warp_latents"].shape[1:] == (self.latent_size, latent_height, latent_width), f"warp_latents shape mismatch: {outputs['warp_latents'].shape[1:]} != {(self.latent_size, latent_height, latent_width)}"
        assert outputs["frame_latents"].shape[1:] == (self.latent_size, latent_height, latent_width), f"frame_latents shape mismatch: {outputs['frame_latents'].shape[1:]} != {(self.latent_size, latent_height, latent_width)}"
    
    def _load_batch_data(self, index, return_raw=False):
        '''
        return a dict of:
        {
            "start_latent": (C, 1 , h, w) tensor of start latent
            "frame_latent": (C, latent_size, h, w) tensor of frame latents
            "warp_latent": (C, latent_size, h, w) tensor of warp latents
            "history_latent": (C, history_size, H, W) tensor of history latents
            "preprocessed_image": (3, H, W) tensor of preprocessed image
            "extrinsics": (F, 4, 4) tensor of extrinsics
            "intrinsics": (F, 3, 3) tensor of intrinsics
            "section_latents": (C, F, h, w) tensor of section latents (Optional)
            "video_info": (dict) video info (Optional)
        }
        '''
        instance = self.instances[index]
        instance_path = self.root / instance

        # process start image
        start_image = imageio.imread(str(instance_path / "start_image.png"))
        assert start_image.shape[0] == self.height and start_image.shape[1] == self.width, f"start_image shape mismatch: {start_image.shape} != {(self.height, self.width)}"
        preprocessed_image = self.feature_extractor(start_image, return_tensors="pt").pixel_values
        preprocessed_image = preprocessed_image.squeeze(0)

        # process warp frames
        warp_latent = torch.load(instance_path / "warp_latents.pt")
        mask = torch.tensor(np.load(instance_path / "mask.npy")).float()
        # interpolate the mask to the latent size
        mask = mask[None, None, ...]
        warp_mask = torch.nn.functional.interpolate(
            mask,
            size=(
                (mask.shape[-3] - 1) // 4 + 1,
                self.height // 8,
                self.width // 8,
            )
        ).squeeze(0)
        assert warp_mask.shape[1:] == warp_latent.shape[1:], f"mask shape mismatch: {mask.shape[1:]} != {warp_latent.shape[1:]}"
        warp_mask = warp_mask[:, 1:1 + self.latent_size, ...]
        warp_latent = warp_latent[:, 1:1 + self.latent_size, ...]

        # process section latents
        section_latents = torch.load(instance_path / "section_latents.pt")
        section_frames = section_latents.shape[1]
        start_frame = section_frames - self.latent_window_size
        start_history = max(0, start_frame - self.max_history_latent_size)
        start_latent = section_latents[:, :1]
        frame_latent = section_latents[:, start_frame:start_frame+self.latent_size]
        history_latent = section_latents[:, start_history:start_frame]

        # process camera intrinsics and extrinsics
        meta = json.load(open(instance_path / "meta.json", "r"))
        intrinsics = torch.tensor(meta["intrinsics"], dtype=torch.float32)
        extrinsics = torch.tensor(meta["extrinsics"], dtype=torch.float32)
        assert extrinsics.shape[0] == self.latent_window_size * 4 and intrinsics.shape[0] == self.latent_window_size * 4
        intrinsics = intrinsics[:self.latent_size * 4]
        extrinsics = extrinsics[:self.latent_size * 4]
        
        return_dict = {
            "start_latents": start_latent,
            "warp_latents": warp_latent,
            "history_latents": history_latent,
            "frame_latents": frame_latent,
            "preprocessed_images": preprocessed_image,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "warp_masks": warp_mask,
        }
        self.check_output(return_dict)
        if return_raw:
            return_dict["section_latents"] = section_latents
            return_dict["video_info"] = meta
        return return_dict

    def __getitem__(self, index):
        return self._load_batch_data(index)        

    def collate_fn(self, batch):
        # align the history latent size
        for b in batch:
            history_latent = b["history_latents"]
            zero_padding = torch.zeros((history_latent.shape[0], self.max_history_latent_size - history_latent.shape[1], history_latent.shape[2], history_latent.shape[3]), dtype=history_latent.dtype)
            b["history_latents"] = torch.cat([zero_padding, history_latent], dim=1)
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


