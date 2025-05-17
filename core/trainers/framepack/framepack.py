from core.trainers.base import Trainer
from typing_extensions import override
from core.schemas import Components
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
from core.trainers.utils import register
from diffusers.training_utils import compute_density_for_timestep_sampling
from core.backbones import FramePackCameraWarpDiffusion
from core.pipe import FramePackValidationPipeline
import copy
import torch.nn.functional as F
from core.utils import (
    free_memory,
    unload_model,
    unwrap_model
)

logger = get_logger(LOG_NAME, LOG_LEVEL)
class FramePackSFTTrainer(Trainer):
    """Trainer class for FramePack SFT (Supervised Fine-Tuning) training."""
    UNLOAD_LIST = ["text_encoder", "text_encoder_2", "vae"]
    
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
            latent_window_size=self.state.latent_size,
            training_type=self.args.training_type,)
        
        components.text_encoder = LlamaModel.from_pretrained(model_path, subfolder='text_encoder', cache_dir=cache_dir)
        components.text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder_2', cache_dir=cache_dir)
        components.tokenizer = LlamaTokenizerFast.from_pretrained(model_path, subfolder='tokenizer', cache_dir=cache_dir)
        components.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2', cache_dir=cache_dir)
        components.vae = AutoencoderKLHunyuanVideo.from_pretrained(model_path, subfolder='vae', cache_dir=cache_dir)
        components.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', cache_dir=cache_dir)

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
        self.state.transformer_config = self.components.backbone.transformer.config

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
            self.dataset = TartanAirFramePackDataset(
                root=self.args.data_root,
                height=self.state.train_height,
                width=self.state.train_width,
                latent_size=self.state.latent_size,
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

    def repeat_to_batch_size(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Repeat the tensor to match the batch size.
        """
        batch_size = self.args.batch_size
        if tensor.shape[0] == batch_size:
            return tensor
        else:
            return tensor.repeat(batch_size, *[1 for _ in range(len(tensor.shape) - 1)])

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # model input
        # latent not in the batch
        weight_dtype = self.state.weight_dtype
        latents = batch["frame_latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
        history_latents = batch["history_latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
        warp_latents = batch["warp_latents"].to(device=self.accelerator.device, dtype=self.state.weight_dtype)
        warp_mask = batch["warp_masks"].to(device=self.accelerator.device, dtype=weight_dtype)


        # image condition
        start_latents = batch["start_latents"].to(device=self.accelerator.device, dtype=weight_dtype)
        image_encoder_last_hidden_state = self.components.image_encoder(batch['preprocessed_images'].to(device=self.accelerator.device, dtype=weight_dtype)).last_hidden_state

        t = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=self.args.batch_size,
            logit_mean=self.args.logit_mean,
            logit_std=self.args.logit_std
        )
        timesteps = t * 1000.0
        t = t.view(-1, *[1 for _ in range(len(latents.shape) - 1)]).to(latents)
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t) * latents + (self.args.sigma_min + (1 - self.args.sigma_min) * t) * noise
        model_predict = self.components.backbone(
            noisy_latents=noisy_latents,
            timesteps=timesteps.to(device=self.accelerator.device, dtype=weight_dtype),
            warp_latents=warp_latents,
            start_latents=start_latents,
            history_latents=history_latents,
            warp_masks=warp_mask,
            pooled_projections=self.repeat_to_batch_size(self.clip_l_pooler.to(device=self.accelerator.device, dtype=weight_dtype)),
            encoder_hidden_states=self.repeat_to_batch_size(self.llama_vec.to(device=self.accelerator.device, dtype=weight_dtype)),
            encoder_attention_mask=self.repeat_to_batch_size(self.llama_attention_mask.to(device=self.accelerator.device)),
            image_embeddings=image_encoder_last_hidden_state.to(device=self.accelerator.device, dtype=weight_dtype),
            extrinsics=batch["extrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            intrinsics=batch["intrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
        )
        target = (1 - self.args.sigma_min) * noise - latents
        loss = F.mse_loss(model_predict, target)
        if self.args.debug:
            print(f"loss: {loss.item()}")
        return loss
    
    @override
    def initialize_pipeline(self):
        pipe = FramePackValidationPipeline(
            vae=self.components.vae,
            backbone=unwrap_model(self.accelerator, self.components.backbone)
        )
        return pipe

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
    @torch.no_grad()
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: FramePackCameraWarpDiffusion
    ):
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # reference image is the first frame of the video
        weight_dtype = self.state.weight_dtype
        image_encoder_last_hidden_state = self.components.image_encoder(eval_data['preprocessed_images'].to(device=self.accelerator.device, dtype=weight_dtype)).last_hidden_state
        
        extra_kwargs = {
            "warp_latents": eval_data["warp_latents"].to(device=self.accelerator.device, dtype=weight_dtype),
            "start_latents": eval_data["start_latents"].to(device=self.accelerator.device, dtype=weight_dtype),
            "history_latents": eval_data["history_latents"].to(device=self.accelerator.device, dtype=weight_dtype),
            "warp_masks": eval_data["warp_masks"].to(device=self.accelerator.device, dtype=weight_dtype),
            "pooled_projections": self.clip_l_pooler.to(device=self.accelerator.device, dtype=weight_dtype),
            "encoder_hidden_states": self.llama_vec.to(device=self.accelerator.device),
            "encoder_attention_mask": self.llama_attention_mask.to(device=self.accelerator.device),
            "image_embeddings": image_encoder_last_hidden_state.to(device=self.accelerator.device, dtype=weight_dtype),
            "extrinsics": eval_data["extrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
            "intrinsics": eval_data["intrinsics"].to(device=self.accelerator.device, dtype=weight_dtype),
        }
        latent_shape = eval_data["frame_latents"].shape
        result_dict = pipe(
            latent_shape=latent_shape,
            **extra_kwargs,
        )
        return  {
            key : {"type" : "video", "value": value} for key, value in result_dict.items()
        }
    

class FramePackLoraTrainer(FramePackSFTTrainer):
    pass


register(
    model_name="framepack",
    training_type="sft",
    trainer_cls=FramePackSFTTrainer
)

register(
    model_name="framepack",
    training_type="lora",
    trainer_cls=FramePackLoraTrainer
)