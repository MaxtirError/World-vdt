from typing import *
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import utils3d
from PIL import Image
import PIL
import os
from core.utils.camera_utils import apply_transforms

def read_image(path: str) -> torch.Tensor:
    image = Image.open(path)
    # image = image.convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image

def read_images(paths: List[str]) -> torch.Tensor:
    images = [read_image(path) for path in paths]
    return torch.stack(images, dim=0)

def crop_images(image : torch.Tensor, 
    crop_size : Union[int, Tuple[int, int]],
    mode='random',
    intrinsics : Union[None, torch.Tensor] = None,
    depth = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Crop images to a fixed size.

    Args:
        images: The [N x 3 x H x W] tensor of images.
        crop_size: The size to crop the images to.
        mode: The crop mode, either 'center' or 'random'.
        (Optional) intrinsics: The [N x 3 x 3] tensor of intrinsics matrices.
        (Optional) depth: The [N x 1 x H x W] tensor of depth maps.

    Returns:
        The [N x 3 x crop_size x crop_size] tensor of cropped images and the [N x 3 x 3] tensor of cropped intrinsics matrices.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    original_h, original_w = image.shape[-2:]
    h, w = crop_size
    crop_h, crop_w = min(int(h / w * original_w), original_h), min(int(w / h * original_h), original_w)
    if mode == 'center':
        crop_top, crop_left = original_h // 2 - crop_h // 2, original_w // 2 - crop_w // 2
    if mode == 'random':
        crop_top, crop_left = np.random.randint(0, original_h - crop_h + 1), np.random.randint(0, original_w - crop_w + 1)
    image = F.crop(image, crop_top, crop_left, crop_h, crop_w)
    image = F.resize(image, crop_size, antialias=True)
    result = [image]
    if intrinsics is not None:
        transform = torch.Tensor([
            original_w / crop_w, 0, -crop_left / crop_w,
            0, original_h / crop_h, -crop_top / crop_h,
            0, 0, 1
        ]).reshape(3, 3)
        intrinsics = transform @ intrinsics
        result.append(intrinsics)
    if depth is not None:
        depth = F.crop(depth, crop_top, crop_left, crop_h, crop_w)
        depth = F.resize(depth, crop_size, interpolation=InterpolationMode.NEAREST)
        result.append(depth)
    return tuple(result) if len(result) > 1 else result[0]


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images_dust3r(path, square_ok=False):
    """ open and convert image into proper input format for DUSt3R
    """
    img = Image.open(os.path.join(path)).convert('RGB')
    W1, H1 = img.size
    img = _resize_pil_image(img, 512)
    W, H = img.size
    cx, cy = W//2, H//2
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not (square_ok) and W == H:
        halfh = 3*halfw/4
    img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img

def apply_mask(img : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    mask = (mask.unsqueeze(-1) > 0.5).float()
    img = img * mask
    return torch.cat([img, mask], dim=-1)
    
def transform_xyz(xyz : torch.Tensor, extrinsics : torch.Tensor):
    # xyz [4, H, W]
    # extrinsics [4, 4]
    if len(xyz.shape) == 3:
        xyz = xyz.permute(1, 2, 0)
        H, W = xyz.shape[:2]
        mask = xyz[..., 3]
        xyz = xyz[..., :3] * 2 - 1
        xyz = apply_transforms(xyz.flatten(0, 1), extrinsics).reshape(H, W, 3)
        return apply_mask(xyz, mask).permute(2, 0, 1)
    else:
        assert len(xyz.shape) == 4
        xyz = xyz.permute(0, 2, 3, 1)
        H, W = xyz.shape[1:3]
        mask = xyz[..., 3]
        xyz = xyz[..., :3] * 2 - 1
        xyz = apply_transforms(xyz.flatten(1, 2), extrinsics).reshape(-1, H, W, 3)
        return apply_mask(xyz, mask).permute(0, 3, 1, 2)

def save_to_gif(images : torch.Tensor, nrow : int, save_path : str, value_range : Tuple[float, float] = (0, 1)):   
    from torchvision.utils import make_grid
    import imageio 
    video_frames = []
    for frame in range(images.shape[1]):
        image = images[:, frame]
        if image.shape[0] != 1:
            image = make_grid(image, nrow=nrow , normalize=True, value_range=value_range)
        else:
            # map value range to (0, 1)
            image = (image[0] - value_range[0]) / (value_range[1] - value_range[0])
            image = image.clamp(0, 1)
        # print(image.shape)
        video_frames.append((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimsave(save_path, video_frames, fps=10)
        