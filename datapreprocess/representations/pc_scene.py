# a point cloud renderer using gaussian splatting
import torch
import utils3d
from utils import normalize_extrinsics

class PointCloudModel:

    def __init__(self, 
                xyz,
                rgb,
                scale = 0.005,
                device='cuda'):
        self.active_sh_degree = 0
        self._xyz = xyz.to(device).contiguous().float()
        self._rgb = rgb.to(device).contiguous().float()
        self._scales = torch.ones((xyz.shape[0], 3), device=device) * scale
        self._rotation = torch.zeros((xyz.shape[0], 4), device=device)
        self._rotation[:, 0] = 1
        self._opacity = torch.ones((xyz.shape[0], 1), device=device)

    @property
    def get_scaling(self):
        return self._scales
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_rgbs(self):
        return self._rgb
    
    @property
    def get_opacity(self):
        return self._opacity

class PcScene:
    def __init__(
        self,
        xyz : torch.Tensor,
        rgb : torch.Tensor,
        scale : float,
        extrinsics : torch.Tensor,
        intrinsics : torch.Tensor,
        history_frames=None):
        self.xyz = xyz
        self.rgb = rgb
        self.scale = scale
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.history_frames = history_frames
    
    def to(self, device):
        return PcScene(
            self.xyz.to(device),
            self.rgb.to(device),
            self.scale,
            self.extrinsics.to(device),
            self.intrinsics.to(device)
        )
    
    @staticmethod
    def render_dict_to_frame(color, depth, intrinsic, scale):
        """
        Convert render dict to frames.

        Args:
            colors (torch.Tensor): (3, H, W) colors
            depth (torch.Tensor): (1, H, W) depth

        Returns:
            (torch.Tensor): (7, H, W) frames
        """
        # gaussian renderers provide inv depth
        depth = 1.0 / depth[0]
        mask = torch.isfinite(depth)
        xyzs = utils3d.torch.depth_to_points(depth, intrinsic, None) / scale
        xyzs[~mask] = 0
        frames = torch.cat([color, xyzs.permute(2, 0, 1), mask[None, ...]], dim=0)
        return frames

    @torch.no_grad()
    def warp_to_rgba(self, renderer, frame_list=None):
        pc = PointCloudModel(self.xyz, self.rgb)
        wrap_rgba = []
        if frame_list is None:
            frame_list = range(len(self.extrinsics))
        for frame in frame_list:
            if frame >= len(self.extrinsics):
                frame = -1
            extrinsic = self.extrinsics[frame]
            intrinsic = self.intrinsics[frame]
            render_dict = renderer.render(pc, extrinsic, intrinsic, colors_overwrite=pc.get_rgbs)
            rgb = render_dict["color"].clamp(0, 1)
            mask = (render_dict["depth"] > 0).float()
            rgba = torch.cat([rgb * mask, mask], dim=0)
            wrap_rgba.append(rgba)
        wrap_rgba = torch.stack(wrap_rgba, dim=0)
        return wrap_rgba
    
    def rescale(self, new_scale : float):
        self.xyz = self.xyz / self.scale * new_scale
        self.extrinsics = normalize_extrinsics(self.extrinsics, self.scale / new_scale, torch.zeros(3).to(self.extrinsics))
        self.scale = new_scale