from core.trainers.base import Trainer
from typing_extensions import override
from core.schemas import Components
from typing import *
import torch
from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from core.datasets import NaiveTestDataset
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask
from core.trainers.utils import register
from diffusers.training_utils import compute_density_for_timestep_sampling
from core.backbones import FramePackCameraWarpDiffusion


from core.utils import (
    free_memory,
    unload_model,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)
class FramePackSFTTrainer(Trainer):
    """Trainer class for FramePack SFT (Supervised Fine-Tuning) training."""
    UNLOAD_LIST = ["text_encoder", "text_encoder_2"]
    
    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)
        cache_dir = str(self.args.cache_dir)
        components.backbone = FramePackCameraWarpDiffusion(cache_dir=cache_dir, 
            branch_num_layers=self.args.branch_num_layers,
            branch_num_single_layers=self.args.branch_num_single_layers,
            train_height=self.state.train_height,
            train_width=self.state.train_width,
            latent_window_size=self.args.latent_window_size,)
        
        components.text_encoder = LlamaModel.from_pretrained(model_path, subfolder='text_encoder', cache_dir=cache_dir)
        components.text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder_2', cache_dir=cache_dir)
        components.tokenizer = LlamaTokenizerFast.from_pretrained(model_path, subfolder='tokenizer', cache_dir=cache_dir)
        components.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2', cache_dir=cache_dir)
        components.vae = AutoencoderKLHunyuanVideo.from_pretrained(model_path, subfolder='vae', cache_dir=cache_dir)
        components.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', cache_dir=cache_dir)
        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=cache_dir)

        return components
    
    @override
    def prepare_models(self) -> None:
        logger.info("Initializing models")

        self.components.vae.eval()
        self.components.text_encoder.eval()
        self.components.text_encoder_2.eval()
        self.components.image_encoder.eval()
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.text_encoder_2.requires_grad_(False)
        self.components.image_encoder.requires_grad_(False)

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent
    
    @override
    @torch.no_grad()
    def encode_text(self, prompt):#, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
        assert isinstance(prompt, str)

        prompt = [prompt]
        
        max_length = 256
        text_encoder = self.components.text_encoder
        text_encoder_2 = self.components.text_encoder_2
        tokenizer = self.components.tokenizer
        tokenizer_2 = self.components.tokenizer_2

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

    @override
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "framepack":
            self.dataset = NaiveTestDataset(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder_2 = self.components.text_encoder_2.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        llama_vec, clip_l_pooler = self.encode_text("")
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        self.llama_vec = llama_vec
        self.clip_l_pooler = clip_l_pooler
        self.llama_attention_mask = llama_attention_mask

        unload_model(self.components.text_encoder)
        unload_model(self.components.text_encoder_2)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # model input
        # latent not in the batch
        weight_dtype = self.state.weight_dtype
        if self.args.use_precompute_vae_latent:
            latents = batch["latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
            history_latents = batch["history_latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
            warp_latents = batch["warp_latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
        else:
            latents = self.encode_video(batch["frames"].permute(0, 2, 1, 3, 4))
            history_latents = self.encode_video(batch["history_frames"].permute(0, 2, 1, 3, 4))
            warp_latents = self.encode_video(batch["warp_frames"].permute(0, 2, 1, 3, 4))

        # image condition
        images = batch["condition_images"]  # [B, C, F, H, W]
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(images.size(0),), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        start_latent = self.encode_video(noisy_images)
<<<<<<< HEAD
        image_encoder_last_hidden_state = self.components.image_encoder(batch['preprocessed_condition_image'].to(device=self.accelerator.device, dtype=weight_dtype)).last_hidden_state

        t = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=self.args.batch_size,
            logit_mean=self.args.logit_mean,
            logit_std=self.args.logit_std
        )
        timesteps = t * 1000.0
        t = t.view(-1, *[1 for _ in range(len(latents.shape) - 1)])
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t) * latents + (self.args.sigma_min + (1 - self.args.sigma_min) * t) * noise

        
        masks = batch["masks"].permute(0, 2, 1, 3, 4)
        masks = torch.nn.functional.interpolate(
            masks, 
            size=(
                (masks.shape[-3] - 1) // 4 + 1, 
                self.state.train_height // 8, 
                self.state.train_width // 8
            )
        )

=======
        image_encoder_last_hidden_state = self.components.image_encoder(batch['preprocessed_image'].to(device=self.accelerator.device, dtype=weight_dtype)).last_hidden_state
        height = self.state.train_height
        width = self.state.train_width
        history_latents = torch.zeros(size=(self.args.batch_size, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).to(device=self.accelerator.device, dtype=weight_dtype)

        latent_window_size = 1
        latent_padding = 2
        latent_padding_size = latent_padding * latent_window_size

        clean_latents_pre = start_latent.to(history_latents)
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0).repeat(self.args.batch_size, 1)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        x = torch.randn((self.args.batch_size, 16, latent_window_size, height // 8, width // 8), device=self.accelerator.device, dtype=weight_dtype)
        t = torch.tensor([1.0] * self.args.batch_size, device=self.accelerator.device, dtype=weight_dtype)
        batch_size = x.shape[0]
        distilled_guidance_scale = 10.0
        distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size)

        model_kwargs = {
            "pooled_projections": self.clip_l_pooler.to(device=self.accelerator.device, dtype=weight_dtype).repeat(batch_size, 1),
            "encoder_hidden_states": self.llama_vec.to(device=self.accelerator.device, dtype=weight_dtype).repeat(batch_size, 1, 1),
            "encoder_attention_mask": self.llama_attention_mask.repeat(batch_size, 1),
            "image_embeddings": image_encoder_last_hidden_state,
            "latent_indices": latent_indices,
            "clean_latents": clean_latents,
            "clean_latent_indices": clean_latent_indices,
            "clean_latents_2x": clean_latents_2x,
            "clean_latent_2x_indices": clean_latent_2x_indices,
            "clean_latents_4x": clean_latents_4x,
            "clean_latent_4x_indices": clean_latent_4x_indices,
            "guidance": distilled_guidance.to(device=self.accelerator.device, dtype=weight_dtype),
        }
>>>>>>> 940eef7365d8c14b7c54da2c0f42759696baf7ed
        model_predict = self.components.backbone(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            warp_latents=warp_latents,
            start_latent=start_latent,
            history_latents=history_latents,
            warp_masks=masks,
            pooled_projections=self.clip_l_pooler.to(device=self.accelerator.device, dtype=weight_dtype),
            encoder_hidden_states=self.llama_vec.to(device=self.accelerator.device, dtype=weight_dtype),
            encoder_attention_mask=self.llama_attention_mask.to(device=self.accelerator.device, dtype=weight_dtype),
            image_embedings=image_encoder_last_hidden_state.to(device=self.accelerator.device, dtype=weight_dtype),
        )[0]
        
        loss = torch.mean((model_predict - latents) ** 2)
        return loss


register(
    model_name="framepack",
    training_type="sft",
    trainer_cls=FramePackSFTTrainer
)