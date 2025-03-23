from pathlib import Path
from .image_utils import save_to_gif
from .camera_utils import apply_transforms
import utils3d
import torch
import os
from core.utils.pc_utils import GaussianRenderer, PointCloudModel
def save_pcs(xyzs, rgbs, extrinsics, output_path):
    xyzs = xyzs.cpu()
    rgbs = rgbs.cpu()
    extrinsics = extrinsics.cpu()
    os.makedirs(output_path, exist_ok=True)
    save_to_gif(rgbs.unsqueeze(0), 0, Path(output_path, 'rgb.gif'))
    save_to_gif(xyzs.unsqueeze(0), 0, Path(output_path, 'xyz.gif'), value_range=(-1, 1))
    rgbs = rgbs.permute(0, 2, 3, 1)
    xyzs = xyzs.permute(0, 2, 3, 1)
    # print(xyzs.norm(dim=-1).min(), xyzs.norm(dim=-1).max())
    mask = xyzs.norm(dim=-1) > 0
    xyzs[~mask] = 1
    save_to_gif(xyzs.permute(0, 3, 1, 2).unsqueeze(0), 0, Path(output_path, 'xyz_mask.gif'), value_range=(-1, 1))
    xyzs = apply_transforms(xyzs.flatten(1, 2), torch.inverse(extrinsics)).reshape(*xyzs.shape)
    xyzs = xyzs[mask]
    rgbs = rgbs[mask]
    utils3d.io.write_ply(Path(output_path, 'pointcloud.ply'), xyzs.numpy(), vertex_colors=rgbs.numpy())

@torch.no_grad()
def rerender_pointclouds(xyzs, rgbs, intrinsics, output_path, renderer):
    render_images = []
    for i in range(24):
        xyz = xyzs[i].permute(1, 2, 0)
        rgb = rgbs[i].permute(1, 2, 0)
        mask = xyz.norm(dim=-1) > 0
        xyz = xyz[..., :3][mask]
        rgb = rgb[mask]
        min_z = max(xyz[:, 2].min(), 1e-3)
        xyz = xyz / min_z
        pc = PointCloudModel(xyz, rgb, device=intrinsics.device)
        extrinsic = torch.eye(4).to(intrinsics.device)
        render_images.append(renderer.render(pc, extrinsic, intrinsics[i], colors_overwrite=pc.get_rgbs)["color"])
    render_images = torch.stack(render_images, dim=0)
    print(render_images.shape)
    save_to_gif(render_images[None, ...], 0, Path(output_path, 'rerender.gif'))
    