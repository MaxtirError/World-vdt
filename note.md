# Finetune CogVideoX
一个学习Finetune CogvideoX的技巧
从main开始
## 搭建accelerator
Accelerator的setting code:
```python
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[kwargs],
)
```
这里report_to 可选tensorboard等框架，其中有个我没见过的新框架wandb，是个类似Tensorboard的东西
## 构建基础模型和CogVideoX Branch分支
```python
logger.info("Initializing CogVideoX branch weights from transformer")
branch = CogvideoXBranchModel.from_transformer(
    transformer=transformer,
    num_layers=args.branch_layer_num,
    attention_head_dim=transformer.config.attention_head_dim,
    num_attention_heads=transformer.config.num_attention_heads,
    load_weights_from_transformer=True,
    wo_text=args.wo_text,
)
```

vae参数配置
```python
if args.enable_slicing:
    vae.enable_slicing()
if args.enable_tiling:
    vae.enable_tiling()
```
1. ​**enable_slicing：分片解码**
​功能：将输入的高分辨率视频帧或图像切分为多个小块（切片）进行分步处理，避免一次性加载完整数据到显存中。
​适用场景：当处理超高分辨率（如 4K 或长视频序列）时，显存可能无法容纳完整的张量，此时分片处理可显著降低显存峰值占用。
​实现逻辑：
前向传播（编码）时，保持完整输入；
反向传播（解码生成）时，按切片逐步重建输出，通过梯度累积保持一致性。
​优势：支持在有限显存下生成更高质量的视频，避免因显存不足（OOM）导致中断
2. ​**enable_tiling：分块计算**
​功能：将输入数据划分为重叠的“瓦片”（tiles），以滑动窗口方式分块处理，降低单次计算复杂度。
​适用场景：针对具有局部相关性的大尺寸视频帧（如动态纹理、局部运动），分块可复用相邻区域的中间特征。
​实现逻辑：
编码阶段：将输入划分为多个瓦片，独立提取特征；
解码阶段：通过加权融合各瓦片的输出，消除块间边界伪影。
​优势：在保持生成效果的同时，减少显存占用约 30-50%（具体数值因输入尺寸而异）。
## 优化器的构建
```python
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
```
简单来说这玩意儿是个占位符，平台会自动帮你吧这个优化器部署上去

