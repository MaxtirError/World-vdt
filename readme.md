# Performance log
| Resolution   |Training Type|Distributed Type| CUDA Memory | GPU Type | Speed |BS/GPU|Precompute latent|Graident Checkpointing
| -------      |--- |---| ----        | ---      |------ |---|---|---|
| 25x480x720   |lora|Deepspeed| 60GB        | 8x4A100  |456.01 steps/h|1|yes|no|
| 25x320x480   |sft |---| 72GB        | 1x1A100  |1132.51 steps/h|1|no|no|
| 25x320x480   |sft |---| 57GB        | 1x1A100  |814.48 steps/h|2|yes|yes|
| 25x320x480   |sft |Deepspeed| ----        | 2x4A100  |101 steps/h|2|yes|no|
| 25x320x480   |sft |DDP |OOM       | 2x4A100  | -| 1 | yes | no|
| 25x320x480   |sft |Deepspeed| 60GB        | 1x4A100  |382 step/h |4|yes| yes|
| 25x320x480   |sft |Deepspeed| OOM         | 1x4A100  |- |2|yes| no|
| 25x320x480   |sft | 58GB        | 1x4A100  | 1018.71 steps/h| 1 | yes | no|
