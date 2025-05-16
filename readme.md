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

frame pack lora
single gpu
setting : g1-bs1-gc
05/16/2025 12:13:07 - INFO - trainer - Step: 40/2577640 (0.00%) | Elapsed: 0.03 h          | Speed: 1206.60 steps/h   | ETA: 2136.25 h           | Max mem: 38124.05 MB 
setting : g1-bs8-gc
05/16/2025 12:47:05 - INFO - trainer - Step: 10/644410 (0.00%)  | Elapsed: 0.03 h          | Speed: 305.60 steps/h    | ETA: 2108.67 h           | Max mem: 51998.49 MB 
setting : g1-bs4-gc
05/16/2025 12:42:32 - INFO - trainer - Step: 10/322210 (0.00%)  | Elapsed: 0.06 h          | Speed: 154.81 steps/h    | ETA: 2081.23 h           | Max mem: 69150.77 MB   

setting g4-bs1-gc
05/16/2025 12:50:45 - INFO - trainer - Step: 10/2577640 (0.00%) | Elapsed: 0.02 h          | Speed: 588.33 steps/h    | ETA: 4381.25 h           | Max mem: 42209.24 MB