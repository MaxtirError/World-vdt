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

import numpy as np
import cv2 

    
def write_rgbd_data(
    file: Union[IO, os.PathLike], 
    image: Union[np.ndarray, bytes], 
    depth: Union[np.ndarray, bytes], depth_mask: Union[np.ndarray, bytes] = None, depth_mask_inf: Union[np.ndarray, bytes] = None, 
    intrinsics: np.ndarray = None, 
    segmentation_mask: Union[np.ndarray, bytes] = None, segmentation_labels: Union[Dict[str, int], bytes] = None, 
    normal: np.ndarray = None, normal_mask: np.ndarray = None,
    meta: Union[Dict[str, Any], bytes] = None, 
    *, max_depth_range: float = 1e4, png_compression: int = 7, jpg_quality: int = 95,
):
    """
    Write RGBD data as zip archive containing the image, depth, mask, segmentation_mask, and meta data.
    In the zip file there will be:
    - `meta.json`: The meta data as a JSON file.
    - `image.jpg`: The RGB image as a JPEG file.
    - `depth.png/exr`: The depth map as a PNG or EXR file, depending on the `depth_type`.
    - `mask.png` (optional): The mask as a uint8 PNG file.
    - `segmentation_mask.png` (optional): The segformer mask as a uint8/uint16 PNG file.

    You can provided those data as np.ndarray or bytes. If you provide them as np.ndarray, they will be properly processed and encoded.
    If you provide them as bytes, they will be written as is, assuming they are already encoded.
    """
    if meta is None:
        meta = {}
    elif isinstance(meta, bytes):
        meta = json.loads(meta.decode())

    if isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])[1].tobytes()

    if isinstance(depth, bytes):
        depth_bytes = depth
    elif isinstance(depth, np.ndarray):
        if depth_mask is None:
            depth_mask = np.ones_like(depth, dtype=bool)
        if depth_mask_inf is None:
            depth_mask_inf = np.zeros_like(depth, dtype=bool)
        assert not (depth_mask & depth_mask_inf).any(), "depth_mask and depth_mask_inf conflict."
        assert depth_mask.any(), "depth_mask is empty."
        depth = depth.astype(np.float32)
        near = max(depth[depth_mask].min(), 1e-3)
        far = max(near * 1.1, min(depth[depth_mask].max(), near * max_depth_range))
        depth = 1 + ((np.log(depth.clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
        depth = np.where(depth_mask, depth, np.where(depth_mask_inf, 65535, 0))
        depth_bytes = cv2.imencode('.png', depth.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()
        meta['depth_near'] = float(near)
        meta['depth_far'] = float(far)
    else:
        raise ValueError("Depth must be provided as bytes or numpy.ndarray.")

    if segmentation_mask is not None:
        if isinstance(segmentation_mask, bytes):
            segmentation_mask_bytes = segmentation_mask
        else:
            segmentation_mask_bytes = cv2.imencode('.png', segmentation_mask)[1].tobytes()

    if intrinsics is not None:
        meta['intrinsics'] = intrinsics.tolist()

    if normal is not None:
        if isinstance(normal, bytes):
            normal_bytes = normal
        elif isinstance(normal, np.ndarray):
            if normal_mask is None:
                normal_mask = np.ones_like(normal, dtype=bool)
            normal = ((normal * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 65535).astype(np.uint16)
            normal = np.where(normal_mask[..., None], normal, 0)
            normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
            normal_bytes = cv2.imencode('.png', normal, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1].tobytes()

    meta_bytes = meta if isinstance(meta, bytes) else json.dumps(meta).encode()

    with zipfile.ZipFile(file, 'w') as z:
        z.writestr('meta.json', meta_bytes)
        z.writestr('image.jpg', image_bytes)
        z.writestr('depth.png', depth_bytes)
        if normal is not None:
            z.writestr('normal.png', normal_bytes)
        if segmentation_mask is not None:
            z.writestr('segmentation_mask.png', segmentation_mask_bytes)


def read_rgbd_data(file: Union[str, Path, IO], return_bytes: bool = False) -> Dict[str, Union[np.ndarray, Dict[str, Any], bytes]]:   
    """
    Read an RGBD zip file and return the image, depth, mask, segmentation_mask, intrinsics, and meta data.
    
    ### Parameters:
    - `file: Union[str, Path, IO]`
        The file path or file object to read from.
    - `return_bytes: bool = False`
        If True, return the image, depth, mask, and segmentation_mask as raw bytes.

    ### Returns:
    - `Tuple[Dict[str, Union[np.ndarray, Dict[str, Any]]], Dict[str, bytes]]`
        A dictionary containing: (If missing, the value will be None; if return_bytes is True, the value will be bytes)
        - `image`: RGB numpy.ndarray of shape (H, W, 3).
        - `depth`: float32 numpy.ndarray of shape (H, W).
        - `mask`: bool numpy.ndarray of shape (H, W). 
        - `segformer_mask`: uint8 numpy.ndarray of shape (H, W).
        - `intrinsics`: float32 numpy.ndarray of shape (3, 3).
        - `meta`: Dict[str, Any].
    """
    # Load & extract archive
    with zipfile.ZipFile(file, 'r') as z:
        meta = z.read('meta.json')
        if not return_bytes:
            meta = json.loads(z.read('meta.json'))

        image = z.read('image.jpg')
        if not return_bytes:
            image = cv2.imdecode(np.frombuffer(z.read('image.jpg'), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = z.read('depth.png')
        if not return_bytes:
            depth = cv2.imdecode(np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED)
            depth_mask = (0 < depth) & (depth < 65535)
            depth_mask_inf = depth == 65535
            near, far = meta['depth_near'], meta['depth_far']
            depth = (depth.astype(np.float32) - 1) / 65533
            depth = near ** (1 - depth) * far ** depth
        else:
            depth_mask = None
            depth_mask_inf = None
        
        if 'segmentation_mask.png' in z.namelist():
            segmentation_mask = z.read('segmentation_mask.png')
            if not return_bytes:
                segmentation_mask = cv2.imdecode(np.frombuffer(segmentation_mask, np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            segmentation_mask = None
        
        if 'normal.png' in z.namelist():
            normal = z.read('normal.png')
            if not return_bytes:
                normal = cv2.imdecode(np.frombuffer(z.read('normal.png'), np.uint8), cv2.IMREAD_UNCHANGED)
                normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
                normal = (normal.astype(np.float32) / 65535 - 0.5) * [2.0, -2.0, -2.0]
                normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        
                normal_mask = (normal != 0).all(axis=-1)
            else:
                normal_mask = None
        else:
            normal, normal_mask = None, None
    
    # intrinsics
    if not return_bytes and 'intrinsics' in meta:
        intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
    else:
        intrinsics = None

    return_dict = {
        'image': image,
        'depth': depth,
        'depth_mask': depth_mask,
        'depth_mask_inf': depth_mask_inf,
        'normal': normal,
        'normal_mask': normal_mask,
        'segmentation_mask': segmentation_mask,
        'intrinsics': intrinsics,
        'meta': meta,
    }
    return_dict = {k: v for k, v in return_dict.items() if v is not None}
    
    return return_dict


def save_glb(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(save_path)


def save_ply(
    save_path: Union[str, os.PathLike], 
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_colors: np.ndarray,
):
    import trimesh
    import trimesh.visual
    from PIL import Image

    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(save_path)



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


def read_segmentation(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Read a segmentation mask
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to read from.
    ### Returns:
    - `Tuple[np.ndarray, Dict[str, int]]`
        A tuple containing:
        - `mask`: uint8 or uint16 numpy.ndarray of shape (H, W).
        - `labels`: Dict[str, int]. The label mapping, a dictionary of {label_name: label_id}.
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    labels = json.loads(pil_image.info['labels']) if 'labels' in pil_image.info else None
    mask = np.array(pil_image)
    return mask, labels


def write_segmentation(path: Union[str, os.PathLike, IO], mask: np.ndarray, labels: Dict[str, int] = None, compression_level: int = 7):
    """
    Write a segmentation mask and label mapping, as PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `mask: np.ndarray`
        The segmentation mask, uint8 or uint16 array of shape (H, W).
    - `labels: Dict[str, int] = None`
        The label mapping, a dictionary of {label_name: label_id}.
    - `compression_level: int = 7`
        The compression level for PNG compression.
    """
    assert mask.dtype == np.uint8 or mask.dtype == np.uint16, f"Unsupported dtype {mask.dtype}"
    pil_image = Image.fromarray(mask)
    pnginfo = PngImagePlugin.PngInfo()
    if labels is not None:
        labels_json = json.dumps(labels, ensure_ascii=True, separators=(',', ':'))
        pnginfo.add_text('labels', labels_json)
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)



def read_normal(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a normal image, return float32 normal array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    normal = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    mask_nan = np.all(normal == 0, axis=-1)
    normal = (normal.astype(np.float32) / 65535 - 0.5) * [2.0, -2.0, -2.0]
    normal = normal / (np.sqrt(np.square(normal[..., 0]) + np.square(normal[..., 1]) + np.square(normal[..., 2])) + 1e-12)
    normal[mask_nan] = np.nan
    return normal


def write_normal(path: Union[str, os.PathLike, IO], normal: np.ndarray, compression_level: int = 7) -> np.ndarray:
    """
    Write a normal image, input float32 normal array of shape (H, W, 3).
    """
    mask_nan = np.isnan(normal).any(axis=-1)
    normal = ((normal * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 65535).astype(np.uint16)
    normal[mask_nan] = 0
    data = cv2.imencode('.png', cv2.cvtColor(normal, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression_level])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)


def read_meta(path: Union[str, os.PathLike, IO]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def write_meta(path: Union[str, os.PathLike, IO], meta: Dict[str, Any]):
    Path(path).write_text(json.dumps(meta))