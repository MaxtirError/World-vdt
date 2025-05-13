
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
import torch

model_kwargs = torch.load("model_input.pt")
for k, v in model_kwargs.items():
    print("key:", k)
    print("value:", v.shape)
    print("dtype:", v.dtype)

cache_dir = "./cache/framepack"
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", cache_dir=cache_dir)
backbone = HunyuanVideoTransformer3DModelPacked.from_transformer_debug(transformer).to("cuda")
backbone = backbone.to(torch.bfloat16)
# simulate all input
model_kwargs = {k: v for k, v in model_kwargs.items()}
batch_size = 2
height = 416
width = 960
x = torch.randn((batch_size, 16, 9, height // 8, width // 8), device="cuda").to(torch.bfloat16)
t = torch.tensor([1.0] * batch_size, device="cuda").to(torch.bfloat16)

for k, v in model_kwargs.items():
    model_kwargs[k] = v.repeat(batch_size, *([1] * (len(v.shape) - 1)))


with torch.no_grad():
    model_output = backbone(
        x,
        t,
        **model_kwargs,
    )