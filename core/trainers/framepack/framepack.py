from core.trainers.base import Trainer
from typing_extensions import override
from core.schemas import Components
from typing import *
import torch
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from core.datasets import NaiveTestDataset
from core.constants import LOG_LEVEL, LOG_NAME
from accelerate.logging import get_logger

from core.utils import (
    free_memory,
    unload_model,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)
class FramePackSFTTrainer(Trainer):
    """Trainer class for FramePack SFT (Supervised Fine-Tuning) training."""
    
    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)
        cache_dir = str(self.args.cache_dir)
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_path, cache_dir=cache_dir)
        return components

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
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        if self.args.model_type == "camerawarp":
            # precompute prompt embedding
            self.prompt_embedding = self.encode_text("")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )
