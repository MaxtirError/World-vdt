import torch
from typing import *
import utils3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def apply_transforms(xyz : torch.Tensor, extrinsics : torch.Tensor) -> torch.Tensor:
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    return xyz @ R.transpose(-1, -2) + T.unsqueeze(-2)


def get_plucker_ray(extrinsics: torch.Tensor, intrinsics: torch.Tensor, 
        H: int = None, W: int = None,
        uv: torch.Tensor = None,
        raw: bool = False) -> torch.Tensor:
    """
    Args:
        extrinsics: (B, V, 4, 4) extrinsic matrices.
        intrinsics: (B, V, 3, 3) intrinsic matrices.
        H (optional): height of the image.
        W (optional): width of the image.
        uv (optional): (..., n_rays, 2) uv coordinates. Batch shape can be omitted and will be broadcasted to extrinsics and intrinsics.
        
    Returns:
        plucker: (B, V, H, W, 6) plucker coordinates.
    """
    if uv is None:
        assert H is not None and W is not None
        uv = utils3d.torch.image_uv(H, W).flatten(0, 1).to(extrinsics)
        
    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)
    with torch.cuda.amp.autocast(enabled=False):
        inv_transformation = (intrinsics @ extrinsics[:, :, :3, :3]).inverse()
        inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)       
    # normalize rays_d
    rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-6)                                         
    rays_o = inv_extrinsics[..., :3, 3] # B, V, 3                       
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW                
    if raw:
        return rays_o.reshape(*extrinsics.shape[:2], H, W, 3), rays_d.reshape(*extrinsics.shape[:2], H, W, 3)                                                  
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(*extrinsics.shape[:2], H, W, 6)

def visualize_camera_orbit(extrinsics, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    inv_extrinsics = extrinsics.inverse()
    rays_o = inv_extrinsics[..., :3, 3]      
    x = rays_o[:, 0].cpu().numpy()
    y = rays_o[:, 1].cpu().numpy()
    z = rays_o[:, 2].cpu().numpy()
    ax.scatter(x, y, z, c='r', marker='o', s=5)
    ax.plot(x, y, z, c='b', linewidth=2)  # 连接点的线，蓝色，宽度为 2
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')      
    plt.savefig(save_path)