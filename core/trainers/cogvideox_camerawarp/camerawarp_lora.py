from typing import Any, Dict, List, Tuple, Union

import torch
import copy
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from core.backbones import CogVideoXCameraWarpDiffusion
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from core.schemas import Components
from core.trainers.lora_base import Trainer
from core.utils import unwrap_model
import random
from core.pipe import CogVideoXI2VCameraWarpPipeline
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger
from core.trainers.utils import register
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from core.utils.debug_utils import CUDATimer
logger = get_logger(LOG_NAME, LOG_LEVEL)


class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]
    
    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)
        cache_dir = str(self.args.cache_dir)

        components.pipeline_cls = CogVideoXI2VCameraWarpPipeline
        try:
            components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", cache_dir=cache_dir)
        except:
            components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        
        try:
            components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", cache_dir=cache_dir)
        except:
            components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        try:
            components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", cache_dir=cache_dir)
        except:
            components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
            
        vae_scale_factor_spatial = 2 ** (len(components.vae.config.block_out_channels) - 1)
        sample_height = self.state.train_height // vae_scale_factor_spatial
        sample_width = self.state.train_width // vae_scale_factor_spatial
        print(f"sample height: {sample_height}, sample width: {sample_width}")
        components.backbone = CogVideoXCameraWarpDiffusion(
            model_path=model_path,
            cache_dir=cache_dir,
            warp_num_layers=self.args.warp_num_layers,
            train_height=self.state.train_height,
            train_width=self.state.train_width,
            sample_height=sample_height,
            sample_width=sample_width,
        )
        
        try:
            components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=cache_dir)
        except:
            components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXI2VCameraWarpPipeline:
        pipe = CogVideoXI2VCameraWarpPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            backbone=unwrap_model(self.accelerator, self.components.backbone),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # model input
        # latent not in the batch
        if self.args.use_precompute_vae_latent:
            latent = batch["latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
        else:
            latent = self.encode_video(batch["frames"].permute(0, 2, 1, 3, 4))
            latent = latent.permute(0, 2, 1, 3, 4)
        weight_dtype = self.state.weight_dtype
        
        # image condition
        images = batch["frames"][:, :1, :, :, :].permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(images.size(0),), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latents = self.encode_video(noisy_images)
        # from [B, C, F, H, W] to [B, F, C, H, W]
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        if random.random() < self.args.noised_image_dropout:
            image_latents = torch.zeros_like(image_latents)
        
        # warp condition
        if self.args.use_precompute_vae_latent:
            conditioning_latents = batch["warp_latents"].to(device=self.accelerator.device, dtype=weight_dtype)
        else:
            conditioning_latents = self.encode_video(batch["warp_frames"].permute(0, 2, 1, 3, 4))
            conditioning_latents = conditioning_latents.permute(0, 2, 1, 3, 4)
            
        torch.cuda.empty_cache()
        
        # process mask
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        vae_scale_factor_temporal = self.components.vae.config.temporal_compression_ratio
        
        masks = batch["masks"].permute(0, 2, 1, 3, 4)
        masks = torch.nn.functional.interpolate(
            masks, 
            size=(
                (masks.shape[-3] - 1) // vae_scale_factor_temporal + 1, 
                self.state.train_height // vae_scale_factor_spatial, 
                self.state.train_width // vae_scale_factor_spatial
            )
        ).permute(0, 2, 1, 3, 4).to(dtype=conditioning_latents.dtype)
        conditioning_latents=torch.concat([conditioning_latents, masks], -3).to(dtype=weight_dtype)

        # Sample noise that will be added to the latents
        noise = torch.randn_like(latent).to(dtype=weight_dtype)
        batch_size, num_frames, num_channels, height, width = latent.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=latent.device
        )
        timesteps = timesteps.long()
        transformer_config = self.state.transformer_config
        
        # Prepare rotary embeds
        image_rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_video_latents = self.components.scheduler.add_noise(latent, noise, timesteps)
        noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

        model_output = self.components.backbone(
            noisy_video_latents=noisy_video_latents,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            warp_latents=conditioning_latents,
            warp_masks=masks,
            extrinsics=batch["extrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            intrinsics=batch["intrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            prompt_embedding=self.prompt_embedding.repeat(batch_size, *([1] * (len(self.prompt_embedding.shape) - 1))),
            image_rotary_emb=image_rotary_emb,
            drop_out_camera=random.random() < self.args.camera_condition_dropout,
        )
        
        model_pred = self.components.scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
        
        weights = 1 / (1 - self.components.scheduler.alphas_cumprod[timesteps])
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = latent

        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        if self.args.loss_warp > 0:
            warp_loss = torch.mean((weights * (model_pred*masks - target*masks) ** 2).reshape(batch_size, -1), dim=1)
            warp_loss = warp_loss.mean()
            loss = loss + self.args.loss_warp * warp_loss
        return loss
    
    @override
    def get_validation_data(self):
        num_validation_samples = 1
        train_dataset = copy.deepcopy(self.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=num_validation_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=train_dataset.collate_fn
        )
        # load data
        data = next(iter(train_loader))
        return data, num_validation_samples
        
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXI2VCameraWarpPipeline
    ) -> List[Tuple[str, Union[Image.Image, List[Image.Image]]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # reference image is the first frame of the video
        image = eval_data["frames"][:, 0, :, :, :] # [B, C, H, W]
        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt_embeds=self.prompt_embedding,
            image=image,
            warp_frames=eval_data["warp_frames"],
            extrinsics=eval_data["extrinsics"],
            intrinsics=eval_data["intrinsics"],
            masks=eval_data["masks"],
            generator=self.state.generator,
            device=self.accelerator.device,
            dtype=self.state.weight_dtype,
        ).frames[0]
        video_gt = pipe.video_processor.postprocess_video(eval_data['frames'].permute(0, 2, 1, 3, 4), output_type="pil")[0]
        video_warp = pipe.video_processor.postprocess_video(eval_data['warp_frames'].permute(0, 2, 1, 3, 4), output_type="pil")[0]
        return  {
            "video_gt": {"type" : "video", "value": video_gt},
            "video_warp": {"type" : "video", "value": video_warp},
            "video_generate": {"type" : "video", "value": video_generate},
        }
    
    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin
    
    @override
    def compute_loss_debug(self, batch) -> torch.Tensor:
        with CUDATimer("vae encoding operations"):
            # model input
            if self.args.use_precompute_vae_latent:
                latent = batch["latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
            else:
                latent = self.encode_video(batch["frames"].permute(0, 2, 1, 3, 4))
                latent = latent.permute(0, 2, 1, 3, 4)
            weight_dtype = self.state.weight_dtype
        
            # image condition
            images = batch["frames"][:, :1, :, :, :].permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(images.size(0),), device=self.accelerator.device)
            image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
            noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
            image_latents = self.encode_video(noisy_images)
            # from [B, C, F, H, W] to [B, F, C, H, W]
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
            # Padding image_latents to the same frame number as latent
            padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
            latent_padding = image_latents.new_zeros(padding_shape)
            image_latents = torch.cat([image_latents, latent_padding], dim=1)
            if random.random() < self.args.noised_image_dropout:
                image_latents = torch.zeros_like(image_latents)
            
            # warp condition
            if self.args.use_precompute_vae_latent:
                conditioning_latents = batch["warp_latents"].to(device=self.accelerator.device, dtype=weight_dtype)
            else:
                conditioning_latents = self.encode_video(batch["warp_frames"].permute(0, 2, 1, 3, 4))
                conditioning_latents = conditioning_latents.permute(0, 2, 1, 3, 4)
            torch.cuda.empty_cache()
        
        
            # process mask
            vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
            vae_scale_factor_temporal = self.components.vae.config.temporal_compression_ratio
            
            masks = batch["masks"].permute(0, 2, 1, 3, 4)
            masks = torch.nn.functional.interpolate(
                masks, 
                size=(
                    (masks.shape[-3] - 1) // vae_scale_factor_temporal + 1, 
                    self.state.train_height // vae_scale_factor_spatial, 
                    self.state.train_width // vae_scale_factor_spatial
                )
            ).permute(0, 2, 1, 3, 4).to(dtype=conditioning_latents.dtype)
            conditioning_latents=torch.concat([conditioning_latents, masks], -3).to(dtype=weight_dtype)

            # Sample noise that will be added to the latents
            noise = torch.randn_like(latent).to(dtype=weight_dtype)
            batch_size, num_frames, num_channels, height, width = latent.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=latent.device
            )
            timesteps = timesteps.long()
            transformer_config = self.state.transformer_config
        
        with CUDATimer("rotary positional embedding operations"):
            # Prepare rotary embeds
            image_rotary_emb = (
                self.prepare_rotary_positional_embeddings(
                    height=height * vae_scale_factor_spatial,
                    width=width * vae_scale_factor_spatial,
                    num_frames=num_frames,
                    transformer_config=transformer_config,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    device=self.accelerator.device,
                )
                if transformer_config.use_rotary_positional_embeddings
                else None
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_video_latents = self.components.scheduler.add_noise(latent, noise, timesteps)
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

        model_output = self.components.backbone.forward_debug(
            noisy_video_latents=noisy_video_latents,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            warp_latents=conditioning_latents,
            warp_masks=masks,
            extrinsics=batch["extrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            intrinsics=batch["intrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            prompt_embedding=self.prompt_embedding.repeat(batch_size, *([1] * (len(self.prompt_embedding.shape) - 1))),
            image_rotary_emb=image_rotary_emb,
            drop_out_camera=random.random() < self.args.camera_condition_dropout,
        )
        
        with CUDATimer("loss final computing process"):
            print(model_output.shape, noisy_video_latents.shape, timesteps.shape)
            model_pred = self.components.scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
            
            weights = 1 / (1 - self.components.scheduler.alphas_cumprod[timesteps])
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            target = latent

            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
            loss = loss.mean()
            if self.args.loss_warp > 0:
                warp_loss = torch.mean((weights * (model_pred*masks - target*masks) ** 2).reshape(batch_size, -1), dim=1)
                warp_loss = warp_loss.mean()
                loss = loss + self.args.loss_warp * warp_loss
        return loss
    
    
register(
    model_name="cogvideox-camerawarp",
    training_type="lora",
    trainer_cls=CogVideoXI2VLoraTrainer
)