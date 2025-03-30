from core.models.camera_encoder import CameraEncoder3D
import torch
if __name__ == "__main__":
    model = CameraEncoder3D((480, 720))
    extrinsics = torch.randn(1, 49, 4, 4)
    intrinsics = torch.randn(1, 49, 3, 3)
    out = model(extrinsics, intrinsics)
    print(out.shape)  # (2, 12, 3072)