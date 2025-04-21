# Performance log
| Resolution   | CUDA Memory | GPU Type | Speed |BS/GPU|Precompute latent|Graident Checkpointing
| -------      | ----        | ---      |------ |---|---|---|
| 25x480x720   | 60GB        | 8x4A100  |456.01 steps/h|1|yes|no|
| 25x320x480   | 72GB        | 1x1A100  |1132.51 steps/h|1|no|no|
| 25x320x480   | 57GB        | 1x1A100  |814.48 steps/h|yes|yes|