from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel


class State(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    train_frames: int
    train_height: int
    train_width: int

    transformer_config: Dict[str, Any] = None

    weight_dtype: torch.dtype = torch.float32  # dtype for mixed precision training
    num_trainable_parameters: int = 0
    overwrote_max_train_steps: bool = False
    num_update_steps_per_epoch: int = 0
    total_batch_size_count: int = 0

    generator: Optional[torch.Generator] = None

    using_deepspeed: bool = False
    using_fsdp: bool = False
