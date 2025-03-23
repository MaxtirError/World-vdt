import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from typing import *
import math
from functools import partial
import copy
import io

import numpy as np
import torch
import click
import cv2
import utils3d
from tqdm import trange, tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndimage

from moge.utils.io import read_image, read_depth, read_meta, write_image, write_depth, write_meta
from moge.utils import pipeline
from moge.utils.geometry_torch import mask_aware_nearest_resize
from moge.utils.geometry_numpy import mask_aware_nearest_resize_numpy, mask_aware_area_resize_numpy, depth_occlusion_edge_numpy
from moge.utils.alignment import align_affine_lstsq, align_depth_affine, align_points_scale_z_shift, align_points_scale_xyz_shift
from moge.utils.vis import colorize_depth
from moge.model.moge_model.v1 import MoGeModel
from moge.utils.tools import timeit, catch_exception


def minmax_normalize(x: torch.FloatTensor, m: torch.BoolTensor):
    *batch_size, height, width = x.shape
    x = x.flatten(-2)
    m = m.flatten(-2)
    min_ = torch.where(m, x, torch.inf).min(dim=-1).values
    max_ = torch.where(m, x, -torch.inf).max(dim=-1).values
    return ((x - min_[..., None]) / (max_[..., None] - min_[..., None])).clamp(min=0, max=1).unflatten(-1, (height, width)), min_, max_


def filter_global(points: torch.Tensor, mask: torch.Tensor, pseudo_points: torch.Tensor):
    (points_lr, pseudo_points_lr), lr_mask = mask_aware_nearest_resize((points, pseudo_points), mask, (64, 64))
    scale, shift = align_points_scale_z_shift(points_lr.flatten(-3, -2), pseudo_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / pseudo_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=1)
    valid = scale > 0
    scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

    points = scale[..., None, None, None] * points + shift[..., None, None, :]

    err = (points - pseudo_points).norm(dim=-1) / pseudo_points[:, :, 2]
    mask = mask & (err < 0.2)
    return mask, err


def compute_anchor_sampling_weight(points: torch.Tensor, mask: torch.Tensor, radius_2d: torch.Tensor, radius_3d: torch.Tensor, num_test: int = 64) -> torch.Tensor:
    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def filter_local(points_a: torch.Tensor, points_b: torch.Tensor, mask: torch.Tensor, focal: torch.Tensor, level: Literal[4, 16, 64], align_resolution: int = 32, num_patches: int = 16, tolerance: float = 0.4):
    device, dtype = points_a.device, points_a.dtype
    *batch_shape, height, width, _ = points_a.shape
    batch_size = math.prod(batch_shape)
    points_a, points_b, mask, focal = points_a.reshape(-1, height, width, 3), points_b.reshape(-1, height, width, 3), mask.reshape(-1, height, width), focal.reshape(-1)
    
    # Importance anchor sampling
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d_a = 0.5 / level / focal * points_a[..., 2]
    radius_3d_b = 0.5 / level / focal * points_b[..., 2]
    anchor_sampling_weights_a = compute_anchor_sampling_weight(points_a, mask, radius_2d, radius_3d_a, num_test=16)
    anchor_sampling_weights_b = compute_anchor_sampling_weight(points_b, mask, radius_2d, radius_3d_b, num_test=16)
    anchor_sampling_weights = (anchor_sampling_weights_a + anchor_sampling_weights_b) / 2

    # Sample patch anchor points indices 
    where_mask = torch.where(mask)
    if where_mask[0].shape[0] == 0:
        return mask.reshape(*batch_shape, height, width)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_radius = 0.5 / level
    patch_radius_in_pixel = math.ceil(patch_radius * (height ** 2 + width ** 2) ** 0.5)
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-patch_radius_in_pixel, patch_radius_in_pixel + 1, device=device), 
        torch.arange(-patch_radius_in_pixel, patch_radius_in_pixel + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None] 

    # Get patch points [num_total_patches, patch_h, patch_w, 3] & patch anchor points [num_total_patches, 3]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    patch_anchor_points_a = points_a[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    patch_anchor_points_b = points_b[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    patch_points_a = points_a[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_points_b = points_b[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask = patch_mask & mask[patch_batch_idx[:, None, None], patch_i, patch_j]

    # Include only patches within the radius [batch_size, num_patches, patch_h, patch_w]
    patch_dist_a = (patch_points_a - patch_anchor_points_a[:, None, None, :]).norm(dim=-1)
    patch_radius_3d_a = 0.5 / level / focal[patch_batch_idx] * patch_anchor_points_a[:, 2]
    patch_dist_b = (patch_points_b - patch_anchor_points_b[:, None, None, :]).norm(dim=-1)    
    patch_radius_3d_b = 0.5 / level / focal[patch_batch_idx] * patch_anchor_points_b[:, 2]
    patch_mask &= (patch_dist_b <= patch_radius_3d_b[:, None, None]) | (patch_dist_a <= patch_radius_3d_a[:, None, None])

    # Pick only non-empty patches [all_patches, ...]
    MINIMUM_POINTS_PER_PATCH = 16
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return mask.reshape(*batch_shape, height, width)
    
    patch_i, patch_j = patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                       # [num_nonempty_patches, patch_h, patch_w]
    patch_points_a = patch_points_a[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w, 3]
    patch_points_b = patch_points_b[nonempty]                               # [num_nonempty_patches, 3]
    patch_radius_3d_b = patch_radius_3d_b[nonempty]                         # [num_nonempty_patches]
    patch_anchor_points_b = patch_anchor_points_b[nonempty]                 # [num_nonempty_patches, 3]
    patch_batch_idx = patch_batch_idx[nonempty]                             # [num_nonempty_patches]

    # Low resolution for scale and shift alignment
    (patch_points_a_lr, patch_points_b_lr), patch_lr_mask = mask_aware_nearest_resize((patch_points_a, patch_points_b), patch_mask, (align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(patch_points_a_lr.flatten(-3, -2), patch_points_b_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / patch_radius_3d_b[:, None].add(1e-7), trunc=tolerance)

    patch_points_a = local_scale[:, None, None, None] * patch_points_a + local_shift[:, None, None, :]                                          # [num_nonempty_patches, patch_h, patch_w, 3]

    patch_err = (patch_points_a - patch_points_b).norm(dim=-1) / patch_radius_3d_b[:, None, None]                                           # [num_nonempty_patches, patch_h, patch_w]
    patch_filter_out_mask = (patch_err > tolerance) & patch_mask                                                                                
    # for i_patch in trange(num_nonempty_patches):
    #     mask[patch_batch_idx[i_patch], patch_i[i_patch], patch_j[i_patch]] &= ~patch_filter_out_mask[i_patch]

    index = patch_batch_idx[:, None, None].expand_as(patch_i) * height * width + patch_i * width + patch_j
    mask = mask.flatten().float().scatter_reduce_(dim=0, index=index.flatten(), src=(~patch_filter_out_mask.flatten()).float(), reduce='prod', include_self=True)
    mask = mask.reshape(*batch_shape, height, width)
    
    return mask > 0


def constant_coefficients(width: int, height: int) -> sp.csr_array:
    return sp.eye(height * width, format='csr')


def grad_coefficients(width: int, height: int) -> sp.csr_array:
    grid_index = np.arange(width * height).reshape(height, width)
    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]           
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]           
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    A = sp.csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))

    return A


def sp_irls_lsmr(A: sp.csr_array, b: np.ndarray, x0: np.ndarray, w: np.ndarray = None, damp: float = 0, max_irls_iter: int = 20, max_lsmr_iter: int = 1000, atol: float = 1e-5, btol: float = 1e-5, delta: float = 1e-3, show: bool = False):
    x = x0
    W_sqrt = sp.eye(A.shape[0], format='csr')
    for i in trange(max_irls_iter, leave=False):
        x, *_ = spla.lsmr(
            W_sqrt @ A, W_sqrt @ b, damp=damp, atol=atol, btol=btol, x0=x, maxiter=max_lsmr_iter, show=show,
        )
        r = b - A @ x
        W_sqrt = sp.diags_array(1 / np.maximum(np.abs(r), delta) ** 0.5, format='csr')
    return x, i


def make_equation(gt_depth_log: np.ndarray, gt_mask: np.ndarray, pred_depth_log: np.ndarray, pred_mask: np.ndarray, x0: np.ndarray = None, beta: float = 1) -> Tuple[sp.csr_array, np.ndarray]:
    height, width = gt_depth_log.shape
    
    # if x0 is not None:
    #     is_frozen = (np.abs(x0 - gt_depth_log.reshape(-1)) < 1e-2) & gt_mask.reshape(-1)
    # else:
    #     is_frozen = np.zeros_like(gt_depth_log.reshape(-1), dtype=np.bool)
    is_frozen = gt_mask.reshape(-1)
    const_ids = np.where(is_frozen)[0]
    var_ids = np.where(~is_frozen)[0]

    A_constant = constant_coefficients(width, height)[gt_mask.reshape(-1)]
    A_grad = grad_coefficients(width, height)

    b_constant = gt_depth_log.reshape(-1)[gt_mask.reshape(-1)]
    b_grad = A_grad @ pred_depth_log.reshape(-1)

    grad_mask = np.concatenate([(pred_mask[:, :-1] * pred_mask[:, 1:]).reshape(-1), (pred_mask[:-1, :] * pred_mask[1:, :]).reshape(-1)])
    grad_weight = beta * grad_mask * np.exp(-10 * np.abs(b_grad))
    
    A = sp.vstack([
        A_constant, 
        sp.diags_array(grad_weight, format='csr') @ A_grad, 
    ])
    b = np.concatenate([
        b_constant, 
        grad_weight * b_grad, 
    ])

    # Filter out constants
    A, b = A[:, var_ids].tocsr(), b - A[:, const_ids] @ gt_depth_log.reshape(-1)[const_ids]

    # Filter out zero rows
    nonzero = np.where(spla.norm(A, axis=1, ord=1) > 0)[0]
    A, b = A[nonzero], b[nonzero]

    return A, b, var_ids, const_ids


def complete_depth(gt_depth: np.ndarray, gt_mask: np.ndarray, pred_depth: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    height, width = gt_depth.shape

    gt_depth_log = np.nan_to_num(np.log(gt_depth), nan=0, posinf=0, neginf=0)
    pred_depth_log = np.nan_to_num(np.log(pred_depth), nan=0, posinf=0, neginf=0)
    pred_depth_log = pred_depth_log - pred_depth_log[gt_mask].mean() + gt_depth_log[gt_mask].mean()
    n_downsample = int(math.ceil(math.log(max(height, width) / 64, 2)))

    # Pyramid
    gt_pyramid = [(gt_depth_log, gt_mask)]
    pred_pyramid = [(pred_depth_log, pred_mask)]
    for i in range(1, n_downsample + 1):
        height_lr, width_lr = height // 2 ** i, width // 2 ** i
        gt_pyramid.append(mask_aware_area_resize_numpy(gt_pyramid[-1][0], gt_pyramid[-1][1], width_lr, height_lr))
        pred_pyramid.append(mask_aware_area_resize_numpy(pred_pyramid[-1][0], pred_pyramid[-1][1], width_lr, height_lr))

        gt_depth_log_lr, gt_mask_lr = gt_pyramid[-1]
        pred_depth_log_lr, pred_mask_lr = pred_pyramid[-1]

    # Low-res to high-res iterations
    for i in range(n_downsample, -1, -1):
        height_lr, width_lr = height // 2 ** i, width // 2 ** i

        gt_depth_log_lr, gt_mask_lr = gt_pyramid[i]
        pred_depth_log_lr, pred_mask_lr = pred_pyramid[i]
        if i == n_downsample:
            x = pred_depth_log_lr.reshape(-1)
        
        A, b, var_ids, const_ids = make_equation(gt_depth_log_lr, gt_mask_lr, pred_depth_log_lr, pred_mask_lr.astype(np.float32), x0=x, beta=1)
        x[var_ids], *_ = spla.lsmr(A, b, x0=x[var_ids], damp=0 if i == n_downsample else 1e-3, atol=2.5e-5, btol=2.5e-5, show=False, maxiter=1000)
        x[const_ids] = gt_depth_log_lr.reshape(-1)[const_ids]

        if i > 0:
            next_height_lr, next_width_lr = height // 2 ** (i - 1), width // 2 ** (i - 1)
            _, next_pred_mask_lr, index = mask_aware_nearest_resize_numpy(None, pred_mask_lr, (next_width_lr, next_height_lr), return_index=True)
            x = x.reshape(height_lr, width_lr)
            x = (x[index] + (pred_pyramid[i - 1][0] - pred_depth_log_lr[index]))
            x = x.reshape(-1)

    # A, b, var_ids, const_ids = make_equation(gt_depth_log, gt_mask, pred_depth_log, pred_mask.astype(np.float32), x0=x, beta=1)
    # x[var_ids], *_ = sp_irls_lsmr(A, b, x0=x[var_ids], damp=1e-2, atol=2e-5, btol=2e-5, delta=1e-1, show=False, max_irls_iter=1, max_lsmr_iter=1000)
    # x[const_ids] = gt_depth_log.reshape(-1)[const_ids]

    depth_completed = np.exp(x).reshape(height, width)
    
    return depth_completed


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--log_path", type=click.Path(exists=False), default=None, help="Path to log file")
@click.option("--output_path", type=click.Path(), default=None)
@click.option("--resize_to", 'resize_to', type=int, default=None, help="Resize the input images to this size")
@click.option("--num_threads", type=int, default=8, help="Number of threads to use for data loading and processing")
@click.option("--level", type=int, default=3, help="Level of local filterring")
@click.option("--world_size", type=int, default=1, help="Number of processes to use for data loading and processing")
@click.option("--rank", type=int, default=0, help="Rank of current process")
def main(input_path: str, log_path : str, output_path: str, resize_to: int, num_threads: int, level: int, world_size: int, rank: int):
    filenames = Path(input_path, '.index.txt').read_text().splitlines()
    num_files = len(filenames)
    start_idx = rank * num_files // world_size
    end_idx = (rank + 1) * num_files // world_size
    filenames = filenames[start_idx:end_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    if output_path is None:
        output_path = input_path
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        log_path = Path(log_path, f"rank{rank}.log")
    # reverse the filenames
    filenames = filenames[::-1]
    def _provider():
        for fname in filenames:
            yield fname

    @catch_exception
    def _read_data(filename: str):
        if filename is None:
            return None
        
        depth, depth_unit = read_depth(Path(input_path, filename, 'depth.png'))
        depth_syn, _ = read_depth(Path(input_path, filename, 'depth_syn.png'))
        meta = read_meta(Path(input_path, filename, 'meta.json'))
        
        return {
            'filename': filename, 
            'depth': depth, 
            'depth_syn': depth_syn, 
            'depth_unit': depth_unit, 
            'meta': meta
        }

    @catch_exception
    def _filter_depth(instance: Dict[str, np.ndarray]):
        if instance is None:
            return None
        intrinsics = torch.tensor(instance['meta']['intrinsics'], dtype=torch.float32)
        depth, depth_syn = torch.from_numpy(instance['depth']), torch.from_numpy(instance['depth_syn'])
        mask = torch.isfinite(depth) & torch.isfinite(depth_syn)

        points = utils3d.torch.depth_to_points(torch.where(mask, depth, 1), intrinsics=intrinsics).to(device)
        points_syn = utils3d.torch.depth_to_points(torch.where(mask, depth_syn, 1), intrinsics=intrinsics).to(device)

        focal = 1 / (1 / intrinsics[0, 0] ** 2 + 1 / intrinsics[1, 1] ** 2) ** 0.5
        focal = focal.to(device)
        mask_filtered = mask.to(device)
        if level >= 1:
            mask_filtered = filter_local(points, points_syn, mask_filtered, focal, 4, align_resolution=16, num_patches=64, tolerance=1)
        if level >= 2:
            mask_filtered = filter_local(points, points_syn, mask_filtered, focal, 16, align_resolution=8, num_patches=1024, tolerance=1)
        if level >= 3:
            mask_filtered = filter_local(points, points_syn, mask_filtered, focal, 64, align_resolution=4, num_patches=8192, tolerance=1)
        mask_filtered = mask_filtered.cpu().numpy()

        instance['mask_filtered'] = mask_filtered
            
        return instance

    @catch_exception
    def _complete_depth(instance: Dict[str, np.ndarray]):
        if instance is None:
            return None
        depth, depth_syn, mask_filtered = instance['depth'], instance['depth_syn'], instance['mask_filtered']
        mask_syn = np.isfinite(depth_syn)
        depth_completed = complete_depth(depth, mask_filtered, depth_syn, mask_syn)
        instance['depth_completed'] = depth_completed
        # fg_edge_mask, bg_edge_mask = depth_occlusion_edge_numpy(depth_completed, mask_syn, 3, 0.05)
        instance['mask_completed'] = mask_syn #& ~(fg_edge_mask | bg_edge_mask)
        
        return instance

    @catch_exception
    def _write_data(instance: Dict[str, np.ndarray]):
        if instance is None:
            return
        save_path = Path(output_path, instance['filename'])
        save_path.mkdir(parents=True, exist_ok=True)

        mask_inf = np.isinf(instance['depth']) | np.isinf(instance['depth_syn'])
        mask_syn = np.isfinite(instance['depth_syn'])
        
        depth_filtered = np.where(mask_inf, np.inf, np.where(instance['mask_filtered'], instance['depth'], np.nan))
        depth_completed = np.where(mask_inf, np.inf, np.where(instance['mask_completed'], instance['depth_completed'], np.nan))

        write_depth(save_path.joinpath('depth_filtered.png'), depth_filtered, instance['depth_unit'])
        write_depth(save_path.joinpath('depth_completed.png'), depth_completed, instance['depth_unit'])


    pipe = pipeline.Sequential([
        _provider,
        pipeline.Parallel([_read_data] * num_threads),
        pipeline.Parallel([_filter_depth] * num_threads),
        pipeline.Parallel([_complete_depth] * num_threads),
        pipeline.Parallel([_write_data] * num_threads),
    ])

    with pipe:
        for i in trange(len(filenames)):
            if i % 100 == 0:
                # check if depth_complete is exist
                save_path = Path(output_path, filenames[i])
                if os.path.exists(save_path.joinpath('depth_completed.png')):
                    if log_path is not None:
                        with open(log_path, 'a+') as f:
                            f.write(f"Meet existing depth_completed.png, Task Completed!!!!\n")
                    else:
                        print("Meet existing depth_completed.png, Task Completed!!!!")
                    break
            pipe.get()
            if log_path is not None:
                if i % 1000 == 0:
                    with open(log_path, 'a+') as f:
                        f.write(f"Currently Processing {i}/ {len(filenames)} files\n")
    if log_path is not None:
        with open(log_path, 'a+') as f:
            f.write("Task Complete!!!!\n")
    else:
        print("Task Complete!!!!")


if __name__ == '__main__':
    main()
