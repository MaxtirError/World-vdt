import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import os
import diffusers
import torch
import transformers
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np

from core.constants import LOG_LEVEL, LOG_NAME
from core.datasets import TartanAirCameraWarpDataset, NaiveTestDataset
from core.schemas import Args, Components, State
import time
from core.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
    write_video
)
from core.utils.general_utils import *
from core.utils.debug_utils import CUDATimer

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}

class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
            latent_size=self.args.train_resolution[0] // 4, # only for framepack
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        print(accelerator.logging_dir)

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.backbone.transformer.config
        

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "camerawarp":
            self.dataset = TartanAirCameraWarpDataset(
                root=self.args.data_root,
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                use_precompute_vae_latent=self.args.use_precompute_vae_latent,
            )
        elif self.args.model_type == "framepack":
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

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "backbone":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
        
        # enable backbone's gradient except for transformer

        if self.args.training_type == "lora":
            # add LoRA to backbone's transformer
            self.components.backbone.requires_grad_(True)
            self.components.backbone.transformer.requires_grad_(False)
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.backbone.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)
        
        # self.components.backbone.to(self.accelerator.device, dtype=weight_dtype)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            if self.args.debug:
                print("Gradient checkpointing enabled.")
            self.components.backbone.enable_gradient_checkpointing()
            
        model_summary = get_model_summary(self.components.backbone)
        # dump summary to output dir
        if self.accelerator.is_main_process:
            summary_path = self.args.output_dir / "model_summary.txt"
            with open(summary_path, "w") as f:
                f.write(model_summary)
        logger.info(f"Model summary:\n{model_summary}")

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")
        
        # Make sure the trainable params are in float32
        # cast_training_params([self.components.backbone], dtype=torch.float32)

        backbone_parameters = list(filter(lambda p: p.requires_grad, self.components.backbone.parameters()))
        parameters_with_lr = {
            "params": backbone_parameters,
            "lr": self.args.learning_rate,
        }
        
        params_to_optimize = [parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in backbone_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.backbone, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.backbone, self.optimizer, self.data_loader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        # print(self.args.model_dump())
        # valid_config = {
        #     k: v for k, v in self.args.model_dump().items()
        #     if isinstance(v, (int, float, str, bool, torch.Tensor))
        # }
        self.accelerator.init_trackers(tracker_name)
        # dump config to the output dir
        if self.accelerator.is_main_process:
            config_path = self.args.output_dir / "config.json"
            with open(config_path, "w") as f:
                f.write(self.args.model_dump_json(indent=4))
                # json.dump(self.args.model_dump_json(), f, indent=4)
    


    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            if self.args.load_checkpoint_only:
                logger.info(f"Loading checkpoint from {resume_from_checkpoint_path}")
                self.components.backbone.from_pretrained(resume_from_checkpoint_path)
            else:
                self.accelerator.load_state(resume_from_checkpoint_path)
        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator
        
        if self.args.validation_only:
            logger.info("Validation only mode. Skipping training.")
            for i in range(0, 100):
                self.validate(i)
            return 0
            
        if global_step > 0 or self.args.debug:
            self.validate(global_step)

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.backbone.train()
            models_to_accumulate = [self.components.backbone]

            log = []
            time_last_print = 0.0
            time_elapsed = 0.0
            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                step_log = {}
                time_start = time.time()
                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    # if self.args.debug:
                    #     loss = self.compute_loss_debug(batch)
                    # else:
                    loss = self.compute_loss(batch)

                    if self.args.debug:
                        # get memory usage before backward
                        print("memory before backward: ", torch.cuda.memory_allocated() / 1024 / 1024)
                        with CUDATimer("backward"):
                            accelerator.backward(loss)
                    else:
                        accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.backbone.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.backbone.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        step_log["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                time_end = time.time()
                time_elapsed += time_end - time_start

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                step_log["loss"] = loss.detach().item()
                step_log["lr"] = self.lr_scheduler.get_last_lr()[0]

                # Maybe run validation
                should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)
                    
                
                # Print progress
                if accelerator.is_main_process and global_step % self.args.i_print == 0:
                    speed = self.args.i_print / (time_elapsed - time_last_print) * 3600
                    max_memory_allocated = torch.cuda.max_memory_allocated(accelerator.device)
                    columns = [
                        f'Step: {global_step}/{self.args.train_steps} ({global_step / self.args.train_steps * 100:.2f}%)',
                        f'Elapsed: {time_elapsed / 3600:.2f} h',
                        f'Speed: {speed:.2f} steps/h',
                        f'ETA: {(self.args.train_steps - global_step) / speed:.2f} h',
                        f"Max mem: {max_memory_allocated / 1024 / 1024:.2f} MB",
                    ]
                    time_last_print = time_elapsed
                    description = '| '.join([c.ljust(25) for c in columns])
                    logger.info(description)

                if accelerator.is_main_process:
                    log.append((global_step, {}))

                    # Log time
                    log[-1][1]['time'] = {
                        'step': time_end - time_start,
                        'elapsed': time_elapsed,
                    }

                    # Log losses
                    if step_log is not None:
                        log[-1][1].update(step_log)

                if accelerator.is_main_process and global_step % self.args.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log)}' for step, log in log
                    ])
                    with open(self.args.output_dir / "logs.txt", "a+") as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    accelerator.log(log_show, step=global_step)
                    log = []

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")
        data, num_validation_samples = self.get_validation_data()
        accelerator = self.accelerator

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.backbone.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()
        pipe = pipe.to(accelerator.device, dtype=self.state.weight_dtype)

        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}.",
                main_process_only=False,
            )
            batch_data = {k : v[i:i+1] for k, v in data.items()}
            validation_artifacts = self.validation_step(batch_data, pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(validation_artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(validation_artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-sample{i:03d}-{key}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    # export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    write_video(filename, artifact_value, fps=self.args.gen_fps)
                    
        del pipe

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.backbone.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Union[Image.Image, List[Image.Image]]]]:
        raise NotImplementedError
    
    def get_validation_data(self) -> Tuple[Dict[str, Any], int]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.backbone)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        model.save_pretrained(output_dir)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                if isinstance(
                    unwrap_model(self.accelerator, model),
                    type(unwrap_model(self.accelerator, self.components.backbone)),
                ):
                    model = unwrap_model(self.accelerator, model)
                    model.from_pretrained(input_dir)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state("./cache/temp_state/")
                # for deepspeed
                # first wait main process to save the state
                if self.accelerator.is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                    os.system(f"cp -r ./cache/temp_state/* {save_path}")  
                    # clear cache
                    os.system("rm -rf ./cache/temp_state/*")
                self.accelerator.wait_for_everyone()
                # for deepspeed
                if not self.accelerator.is_main_process and self.accelerator.local_process_index == 0:
                    # save the remaining processes, each node will save its own state
                    os.system(f"cp -r ./cache/temp_state/* {save_path}")
                    os.system("rm -rf ./cache/temp_state/*")
                if self.args.debug:
                    # for debug version, quit after saving
                    logger.info("Debug version, quitting after saving.")
                    exit(0)
                    
