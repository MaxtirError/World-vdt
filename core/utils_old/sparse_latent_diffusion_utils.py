import os
import json
import torch
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import utils3d
from rembg import remove
from PIL import Image

from renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from representations import Octree, Gaussian
from representations.geometry import MeshExtractResult
from core import backbones
from core import frameworks
from core import samplers
from core.modules import sparse as sp
from core.utils.random_utils import sphere_hammersley_sequence

from extensions.tsdf_fusion import fusion


def parse_seeds(seeds):
    seeds_str = seeds.split(',')
    res = []
    for seed in seeds_str:
        if '-' in seed:
            start, end = map(int, seed.split('-'))
            res.extend(range(start, end+1))
        else:
            res.append(int(seed))
    return res


def prompt_to_dir_name(prompt):
    # keep only alphanumeric characters and spaces
    prompt = ''.join([c for c in prompt if c.isalnum() or c.isspace()])
    prompt = '_'.join(prompt.split())
    return prompt[:50]


def preprocess_image(path):
    input = Image.open(path)
    # if has alpha channel, use it directly; otherwise, remove background
    if input.mode == 'RGBA':
        output = input
    else:
        input = input.convert('RGB')
        output = remove(input)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return output


def load_model(path, ckpt, use_fp32=False):
    print(f'Loading model from {path} with checkpoint {ckpt} ...', end=' ')
    cfg = edict(json.load(open(os.path.join(path, 'config.json'), 'r')))
    is_multi_backbone = hasattr(cfg, 'backbones')
    if not is_multi_backbone:
        backbone = getattr(backbones, cfg.backbone.name)(**cfg.backbone.args).cuda() # type: ignore
        framework = getattr(frameworks, cfg.framework.name)(backbone, **cfg.framework.args) # type: ignore
    else:
        backbone = {}
        for name, cfg_backbone in cfg.backbones.items():
            backbone[name] = getattr(backbones, cfg_backbone.name)(**cfg_backbone.args).cuda()
        framework = getattr(frameworks, cfg.framework.name)(backbone, **cfg.framework.args)
        backbone = list(backbone.values())[0]
    if use_fp32:
        backbone.use_fp16 = False
        backbone.dtype = torch.float32
        backbone.convert_to_fp32()
    try:
        backbone.load_state_dict(torch.load(os.path.join(path, 'ckpts', ckpt), map_location='cpu'))
    except Exception as e:
        print(f'\nError: {e}')
        print('Failed to load model, trying strict=False ...', end=' ')
        backbone.load_state_dict(torch.load(os.path.join(path, 'ckpts', ckpt), map_location='cpu'), strict=False)
    print('Done')
    return framework


def get_sampler(framework):
    if isinstance(framework, frameworks.GaussianDiffusion):
        sampler = samplers.DdimSampler(framework)
    elif isinstance(framework, frameworks.FlowMatching):
        sampler = samplers.EulerSampler(framework)
    else:
        raise NotImplementedError(f'Unknown framework: {framework.__class__.__name}')
    return sampler


def decode_occ_latent(occ_vae, z):
    samples = occ_vae.decode(z)
    coords = torch.argwhere(samples>0).cuda()[:, [0, 2, 3, 4]].int()
    return coords


def encode_sparse_latent(feats, model):
    feats = sp.SparseTensor(
        feats = torch.from_numpy(feats['patchtokens']).float(),
        coords = torch.cat([
            torch.zeros(feats['patchtokens'].shape[0], 1).int(),
            torch.from_numpy(feats['indices']).int(),
        ], dim=1),
    ).cuda()
    z = model.encode(feats, sample_posterior=False)
    return z


def decode_sparse_latent(ldm_vae, z):
    samples = ldm_vae.decode(z)
    samples = samples[0] if isinstance(samples, list) else list(samples.values())[0][0]
    return samples


def sample_sparse_structure(occ_sampler, occ_vae, cond, return_process=False, sampler_kwargs={}):
    # Sample occupancy latent
    reso = occ_sampler.framework.backbone.resolution
    noise = torch.randn(1, occ_sampler.framework.backbone.in_channels, reso, reso, reso).cuda()
    occ_samples = occ_sampler.sample(
        1,
        **occ_sampler.framework.get_inference_cond([cond] if not isinstance(cond, torch.Tensor) else cond.unsqueeze(0)),
        noise=noise,
        **sampler_kwargs,
        verbose=True,
    )
    
    # Decode occupancy latent
    coords = decode_occ_latent(occ_vae, occ_samples.samples)

    pack = {
        'occ': occ_samples,
        'coords': coords,
    }
    
    if return_process:
        process = []
        for x in occ_samples.pred_x_0:
            process.append(decode_occ_latent(occ_vae, x))
        pack['occ_process'] = process
    
    return pack


def sample_structured_latent(ldm_sampler, vae, cond, coords, mean, std, return_process=False, sampler_kwargs={}):
    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], ldm_sampler.framework.backbone.in_channels).cuda(),
        coords=coords,
    )
    latent_samples = ldm_sampler.sample(
        1,
        **ldm_sampler.framework.get_inference_cond([cond] if not isinstance(cond, torch.Tensor) else cond.unsqueeze(0)),
        noise=noise,
        **sampler_kwargs,
        verbose=True,
    )
    
    # Decode sparse latent
    latent_samples.samples = latent_samples.samples * std + mean
    samples = decode_sparse_latent(vae, latent_samples.samples)
    
    pack = {
        'ldm': latent_samples,
        'samples': samples,
    }
    
    if return_process:
        process = []
        for x in latent_samples.pred_x_0:
            x = x * std + mean
            process.append(decode_sparse_latent(vae, x))
        pack['process'] = process
        
    return pack


def sample_one(occ_sampler, occ_vae, ldm_sampler, vae, cond, mean, std, args):
    # Sample occupancy latent
    reso = occ_sampler.framework.backbone.resolution
    noise = torch.randn(1, occ_sampler.framework.backbone.in_channels, reso, reso, reso).cuda()
    occ_samples = occ_sampler.sample(
        1,
        **occ_sampler.framework.get_inference_cond([cond] if not isinstance(cond, torch.Tensor) else cond.unsqueeze(0)),
        noise=noise,
        steps=args.step,
        strength=args.strength,
        cfg_interval=(0.5, 0.95) if isinstance(occ_sampler, samplers.EulerSampler) else None,
        rescale_t=3.0,
        verbose=True,
    )
    
    # Decode occupancy latent
    coords = decode_occ_latent(occ_vae, occ_samples.samples)

    # Sample sparse latent
    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], ldm_sampler.framework.backbone.in_channels).cuda(),
        coords=coords,
    )
    latent_samples = ldm_sampler.sample(
        1,
        **ldm_sampler.framework.get_inference_cond([cond] if not isinstance(cond, torch.Tensor) else cond.unsqueeze(0)),
        noise=noise,
        steps=args.step,
        eta=1.0,
        strength=args.strength,
        cfg_interval=(0.5, 0.95) if isinstance(occ_sampler, samplers.EulerSampler) else None,
        rescale_t=3.0,
        verbose=True,
    )
    
    # Decode sparse latent
    if args.normalization != 'none':
        latent_samples.samples = latent_samples.samples * std + mean
    samples = decode_sparse_latent(vae, latent_samples.samples)
    samples_process = []
    if hasattr(args, 'save_process') and args.save_process:
        for x in latent_samples.pred_x_0:
            if args.normalization != 'none':
                x = x * std + mean
            samples_process.append(decode_sparse_latent(vae, x))
            
    pack = {
        'occ': occ_samples,
        'ldm': latent_samples,
        'samples': samples,
        'process': samples_process,
    }

    return pack


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering'):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'rgb' not in rets: rets['rgb'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['rgb'].append(np.clip(res['rgb'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            if 'mask' not in rets: rets['mask'] = []
            rets['normal'].append(np.clip(res['normals'][0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
            rets['mask'].append(np.clip(res['mask'][0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_process_video(process, resolution=512, bg_color=(0, 0, 0), num_frames=300):
    r = 2
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, 40)
    frames = None
    for i, rep in enumerate(process):
        part_s = i * len(extrinsics) // len(process)
        part_e = (i + 1) * len(extrinsics) // len(process)
        res = render_frames(rep, extrinsics[part_s:part_e], intrinsics[part_s:part_e], {'resolution': resolution, 'bg_color': bg_color})
        if frames is None:
            frames = {k: [] for k in res.keys()}
        for k in res.keys():
            frames[k].extend(res[k])
    return frames


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def crop_to_foreground(image, bg_color=(0, 0, 0), resolution=None):
    if isinstance(image, list):
        resolution = resolution or image[0].shape[0]
        foreground_masks = [(img != np.array(bg_color)).any(axis=-1) for img in image]
        ymin, xmin = np.stack([np.min(np.argwhere(mask), axis=0) for mask in foreground_masks]).min(axis=0)
        ymax, xmax = np.stack([np.max(np.argwhere(mask), axis=0) for mask in foreground_masks]).max(axis=0)
        center = (xmin + xmax) / 2, (ymin + ymax) / 2
        size = max(xmax - xmin, ymax - ymin) * 1.05
        size = int(size)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        image = [img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] for img in image]
        image = [Image.fromarray(img) for img in image]
        image = [img.resize((resolution, resolution), Image.Resampling.LANCZOS) for img in image]
        image = [np.array(img) for img in image]
        return image
    else:
        resolution = resolution or image.shape[0]
        foreground_mask = (image != np.array(bg_color)).any(axis=-1)
        ymin, xmin = np.min(np.argwhere(foreground_mask), axis=0)
        ymax, xmax = np.max(np.argwhere(foreground_mask), axis=0)
        center = (xmin + xmax) / 2, (ymin + ymax) / 2
        size = max(xmax - xmin, ymax - ymin) * 1.05
        size = int(size)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image = Image.fromarray(image)
        image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        image = np.array(image)
        return image


def extract_mesh(sample, resolution=512):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, 100) for i in range(100)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    
    tsdf_vol = fusion.TSDFVolume(
        np.array([
            [sample.aabb[0].item(), sample.aabb[0].item() + sample.aabb[3].item()],
            [sample.aabb[1].item(), sample.aabb[1].item() + sample.aabb[4].item()],
            [sample.aabb[2].item(), sample.aabb[2].item() + sample.aabb[5].item()],
        ]),
        sample.aabb[3].item() / resolution,
    )
    
    for i, (rgb, depth) in tqdm(enumerate(zip(res['rgb'], res['depth'])), desc='Extracting mesh'):
        intr = intrinsics[i].cpu().numpy()
        intr[0:2] *= resolution
        _extr = extrinsics[i].cpu().numpy()
        R = _extr[:3, :3]
        T = _extr[:3, 3]
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = R.T
        extr[:3, 3] = -R.T @ T

        tsdf_vol.integrate(rgb, depth, intr, extr)
        
    verts, faces, _, colors = tsdf_vol.get_mesh(level=0)
    return verts, faces, colors


def construct_voxel(occ):
    rep = Octree(
        depth=10,
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
        device='cuda',
        primitive='voxel',
        sh_degree=0,
        voxel_config={'solid': True},
    )
    coords = torch.nonzero(occ > 0, as_tuple=False)
    resolution = occ.shape[-1]
    rep.position = coords.float() / resolution
    rep.depth = torch.full((rep.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')
    rep.features_dc = torch.where((coords.sum(dim=-1, keepdim=True) % 2) == 0, torch.ones_like(rep.position), torch.zeros_like(rep.position)).reshape(-1, 1, 3).float()
    return rep
