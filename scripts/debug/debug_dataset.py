from core.datasets import TartanAirCameraWarpDataset
import torch

from diffusers.video_processor import VideoProcessor
dataset = TartanAirCameraWarpDataset(
    root="/home/t-zelonglv/data/TartanAir_Warp",
    num_frames=49,
    height=480,
    width=720
)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=dataset.collate_fn
)
from diffusers.utils.export_utils import export_to_video
data = next(iter(data_loader))
video_processor = VideoProcessor()

print(data['warp_frames'].shape)  # (1, 49, 3, 480, 720)
warp_frames = video_processor.postprocess_video(data["warp_frames"].permute(0, 2, 1, 3, 4), output_type="pil")
# get PIL Image's shape
print(warp_frames[0][0].size)  # (1, 49, 3, 480, 720)
export_to_video(warp_frames[0], "debugs/test.mp4", fps=10)