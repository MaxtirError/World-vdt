# Performance log
## CogVideoX
| Resolution   |Training Type|Distributed Type| CUDA Memory | GPU Type | Speed |BS/GPU|Precompute latent|Graident Checkpointing
| -------      |--- |---| ----        | ---      |------ |---|---|---|
| 25x480x720   |lora|Deepspeed| 60GB        | 8x4A100  |456.01 steps/h|1|yes|no|
| 25x320x480   |sft |---| 72GB        | 1x1A100  |1132.51 steps/h|1|no|no|
| 25x320x480   |sft |---| 57GB        | 1x1A100  |814.48 steps/h|2|yes|yes|
| 25x320x480   |sft |Deepspeed| ----        | 2x4A100  |101 steps/h|2|yes|no|
| 25x320x480   |sft |DDP |   68GB    | 2x4A100  |202steps/h| 1 | yes | yes|
| 25x320x480   |sft |Deepspeed| 60GB        | 1x4A100  |382 step/h |4|yes| yes|
| 25x320x480   |sft |Deepspeed| OOM         | 1x4A100  |- |2|yes| no|
| 25x320x480   |sft | 58GB        | 1x4A100  | 1018.71 steps/h| 1 | yes | no|

## framepack
|Model Size| CUDA Memory| Speed| Zero Stage|offlaod|Batch Size|Graident Checkpoint|
|---|---|---|---|---|---|---|
|5.6B|40GB|267.65 steps/h|2|none|1|enable|
|5.6B|OOM|-|2|none|4|enable|

5.6B Memory before training start: {
    "memory_allocated": 17.142,
    "memory_reserved": 25.117,
    "max_memory_allocated": 17.142,
    "max_memory_reserved": 25.117
}

ooled_projections: torch.Size([1, 768])
05/04/2025 03:23:45 - INFO - trainer - encoder_hidden_states: torch.Size([1, 512, 4096])
05/04/2025 03:23:45 - INFO - trainer - encoder_attention_mask: torch.Size([1, 512])
05/04/2025 03:23:45 - INFO - trainer - image_embeddings: torch.Size([1, 729, 1152])
05/04/2025 03:23:45 - INFO - trainer - latent_indices: torch.Size([1, 9])
05/04/2025 03:23:45 - INFO - trainer - clean_latents: torch.Size([1, 16, 2, 52, 120])
05/04/2025 03:23:45 - INFO - trainer - clean_latent_indices: torch.Size([1, 2])
05/04/2025 03:23:45 - INFO - trainer - clean_latents_2x: torch.Size([1, 16, 2, 52, 120])
05/04/2025 03:23:45 - INFO - trainer - clean_latent_2x_indices: torch.Size([1, 2])
05/04/2025 03:23:45 - INFO - trainer - clean_latents_4x: torch.Size([1, 16, 16, 52, 120])
05/04/2025 03:23:45 - INFO - trainer - clean_latent_4x_indices: torch.Size([1, 16])
05/04/2025 03:23:45 - INFO - trainer - guidance: torch.Size([1])
