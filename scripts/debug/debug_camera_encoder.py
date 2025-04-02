from core.datasets import TartanAirCameraWarpDataset
import torch
dataset = TartanAirCameraWarpDataset(
    root="/home/t-zelonglv/data/TartanAir_Warp",
    num_frames=49,
    height=480,
    width=720
)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=dataset.collate_fn
)
data = next(iter(data_loader))

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
        uv = utils3d.torch.image_uv(W, H).flatten(0, 1).to(extrinsics)
        
        
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
        return rays_o.reshape(*extrinsics.shape[:2], W, H, 3).transpose(-2, -3), rays_d.reshape(*extrinsics.shape[:2], W, H, 3).transpose(-2, -3)                                                  
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(*extrinsics.shape[:2], W, H, 6).transpose(-2, -3)

import torch
import numpy as np
import utils3d
if __name__ == "__main__":
    extrinsics = data["extrinsics"].to("cuda")
    intrinsics = data["intrinsics"].to("cuda")
    H = 480
    W = 720
    rays_o, rays_d = get_plucker_ray(extrinsics, intrinsics, H, W, raw=True)
    rnd_frames = np.random.randint(0, extrinsics.shape[1])
    rnd_u = np.random.randint(0, H)
    rnd_v = np.random.randint(0, W)
    print("random choice", rnd_frames, rnd_u, rnd_v)
    rays_o = rays_o[:, rnd_frames, rnd_u, rnd_v]
    rays_d = rays_d[:, rnd_frames, rnd_u, rnd_v]
    point = rays_o + rays_d
    recover_uv, _ = utils3d.torch.project_cv(point, extrinsics[0, rnd_frames], intrinsics[0, rnd_frames])
    recover_uv[0][0] *= H
    recover_uv[0][1] *= W
    print("recover uv", recover_uv)