# a point cloud renderer using gaussian splatting
import torch
import math
from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict as edict
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
import utils3d
class PointCloudModel:

    def __init__(self, 
                xyz,
                rgb,
                scale = 0.001,
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


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render(viewpoint_camera, pc : PointCloudModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return edict({"color": rendered_image,
            "depth": depth})


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "antialiasing" : False,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: PointCloudModel,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        # Render
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
        return render_ret