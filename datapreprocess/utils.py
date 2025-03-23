import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import IO
import zipfile
import json
import io
from typing import *
from pathlib import Path
import re
from PIL import Image, PngImagePlugin
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import imageio
import numpy as np
import cv2 
import time
from functools import wraps

def read_image(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a image, return uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()  
    else:
        data = path.read()
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def write_image(path: Union[str, os.PathLike, IO], image: np.ndarray, quality: int = 95):
    """
    Write a image, input uint8 RGB array of shape (H, W, 3).
    """
    data = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)


def read_depth(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, float]:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth, unit


def write_depth(
    path: Union[str, os.PathLike, IO], 
    depth: np.ndarray, 
    unit: float = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Encode and write a depth image as 16-bit PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W).
    - `unit: float = None`
        The unit of the depth values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    mask_values, mask_nan, mask_inf = np.isfinite(depth), np.isnan(depth),np.isinf(depth)

    depth = depth.astype(np.float32)
    mask_finite = depth
    near = max(depth[mask_values].min(), 1e-5)
    far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = Image.fromarray(depth)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    if unit is not None:
        pnginfo.add_text('unit', str(unit))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)


def read_meta(path: Union[str, os.PathLike, IO]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def write_meta(path: Union[str, os.PathLike, IO], meta: Dict[str, Any]):
    Path(path).write_text(json.dumps(meta))
    

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


def normalize_extrinsics(extrinsics : torch.Tensor, scale : Union[float, torch.Tensor], mu : torch.Tensor) -> torch.Tensor:
    if len(extrinsics.shape) == 2:
        assert 4 == extrinsics.shape[0] == extrinsics.shape[1]
        R = extrinsics[:3, :3]
        T = extrinsics[:3, 3]
        normed_T = (T + mu @ R.T) / scale
        extrinsics[:3, 3] = normed_T
        return extrinsics
    else:
        assert 4 == extrinsics.shape[1] == extrinsics.shape[2]
        R = extrinsics[:, :3, :3]
        T = extrinsics[:, :3, 3]
        normed_T = (T + mu @ R.transpose(1, 2)) / scale
        extrinsics[:, :3, 3] = normed_T
        return extrinsics

def normalize_intrinsics(intrinsics : torch.Tensor, H : int, W : int) -> torch.Tensor:
    intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / W
    intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / H
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / W
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / H
    return intrinsics



def catch_exception(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"Exception in {fn.__name__}",  end='r')
            # print({', '.join(repr(arg) for arg in args)}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})
            traceback.print_exc(chain=False)
            time.sleep(0.1)
            return None
    return wrapper

def write_image(path: Union[str, os.PathLike, IO], image: np.ndarray, quality: int = 95):
    """
    Write a image, input uint8 RGB array of shape (H, W, 3).
    """
    if image.shape[2] == 3:
        data = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        if isinstance(path, (str, os.PathLike)):
            Path(path).write_bytes(data)
        else:
            path.write(data)
    elif image.shape[2] == 4:
        data = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, quality])[1].tobytes()
        if isinstance(path, (str, os.PathLike)):
            Path(path).write_bytes(data)
        else:
            path.write(data)
    else:
        raise ValueError("Invalid image shape")
    
def write_video(path: Union[str, os.PathLike, IO], video: np.ndarray, fps: int = 10):
    """
    Write a video, input uint8 RGB array of shape (T, H, W, 3).
    """
    writer = imageio.get_writer(path, fps=fps)
    for frame in video:
        writer.append_data(frame)
    writer.close()