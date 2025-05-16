import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from typing import *
import torch
from diffusers import (
    AutoencoderKLHunyuanVideo,
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from core.datasets import TartanAirFramePackDataset
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipVisionModel
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask
from core.backbones import FramePackCameraWarpDiffusion
from core.pipe import FramePackValidationPipeline
from core.utils import write_video
import copy
logger = get_logger(LOG_NAME, LOG_LEVEL)
model_path = "hunyuanvideo-community/HunyuanVideo"
cache_dir = "./cache/framepack/"
backbone = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cuda()
text_encoder = LlamaModel.from_pretrained(model_path, subfolder='text_encoder', cache_dir=cache_dir)
text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder_2', cache_dir=cache_dir)
tokenizer = LlamaTokenizerFast.from_pretrained(model_path, subfolder='tokenizer', cache_dir=cache_dir)
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2', cache_dir=cache_dir)
vae = AutoencoderKLHunyuanVideo.from_pretrained(model_path, subfolder='vae', cache_dir=cache_dir)
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', cache_dir=cache_dir)
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.enable_slicing()
vae.enable_tiling()
device = torch.device("cuda")
weight_dtype = torch.bfloat16
vae.to(device, dtype=weight_dtype)
image_encoder.to(device, dtype=weight_dtype)
@torch.no_grad()
def encode_video(vae, video: torch.Tensor) -> torch.Tensor:
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent
    
@torch.no_grad()
def vae_decode(latents, vae, image_mode=False):
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)

    return image

@torch.no_grad()
def encode_text(prompt):#, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]
    
    max_length = 256

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    return llama_vec, clip_l_pooler

dataset = TartanAirFramePackDataset(
    root="/home/t-zelonglv/blob/zelong/data/TartanAir_544x704",
    height=544,
    width=704,
    latent_size=1,
)
text_encoder = text_encoder.to(device, dtype=weight_dtype)
text_encoder_2 = text_encoder_2.to(device, dtype=weight_dtype)
llama_vec, clip_l_pooler = encode_text("")

llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
text_encoder = text_encoder

data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=dataset.collate_fn,
    batch_size=1,
    num_workers=1,
    pin_memory=8,
    shuffle=True,
)
data = next(iter(data_loader))

pipe = FramePackValidationPipeline(
    vae=vae,
    backbone=backbone
)
with torch.no_grad():
    image_encoder_last_hidden_state = image_encoder(data['preprocessed_images'].to(device=device, dtype=weight_dtype)).last_hidden_state
    history_latents = data["history_latents"].to(device=device, dtype=weight_dtype)
    start_latent = data["start_latents"].to(device=device, dtype=weight_dtype)
    indices = torch.arange(0, sum([1, 16, 2, 1, 1])).unsqueeze(0)
    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, 1], dim=1)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
    
    distilled_guidance = torch.tensor([10.0 * 1000.0] * 1)
    extra_kwargs = {
        "pooled_projections": clip_l_pooler.to(device=device, dtype=weight_dtype),
        "encoder_hidden_states": llama_vec.to(device=device, dtype=weight_dtype),
        "encoder_attention_mask": llama_attention_mask.to(device=device),
        "image_embeddings": image_encoder_last_hidden_state.to(device=device, dtype=weight_dtype),
        "guidance": distilled_guidance.to(device=device, dtype=weight_dtype),}
    extra_kwargs.update(
        latent_indices=latent_indices,
        clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices,
        clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices,
        clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices,
    )
    # move extra_kwargs to device
    for key in extra_kwargs:
        if isinstance(extra_kwargs[key], torch.Tensor):
            extra_kwargs[key] = extra_kwargs[key].to(device=device)
    latent_shape = data["frame_latents"].shape
    frame_latents = pipe(
        latent_shape=latent_shape,
        **extra_kwargs,
        return_latents=True,
    )
    prev_frames = vae_decode(history_latents, vae)
    prev_frames = pipe.postprocess(prev_frames)
    print(prev_frames.shape)
    os.makedirs("debugs/vis_warp_dataset", exist_ok=True)
    write_video("debugs/vis_warp_dataset/prev_frames.mp4", prev_frames)
    frame_latents_to_decode = torch.cat([history_latents[:, :, -2:].to(frame_latents), frame_latents], dim=2)
    frames = vae_decode(frame_latents_to_decode, vae)
    print(frames.shape)
    frames = pipe.postprocess(frames)
    os.makedirs("debugs/vis_warp_dataset", exist_ok=True)
    write_video("debugs/vis_warp_dataset/frames.mp4", frames)