import torch
import torch.nn as nn
from .utils import zero_module
from core.utils.camera_utils import get_plucker_ray
from typing import *
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, hidden_size, in_channels=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x):
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x):
        """
        Args:
            x (torch.Tensor) (:N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.hidden_size:
            embed = torch.cat([embed, torch.zeros(N, self.hidden_size - embed.shape[1], device=embed.device)], dim=-1)
        return embed

class CameraEncoder3D(ModelMixin, ConfigMixin):
    @register_to_config()
    def __init__(self, 
            resolution : Tuple[int, int],
            pos_emb_dim : int = 144, 
            hidden_channel : int = 288, 
            out_channel : int =3072):
        super().__init__()
        self.resolution = resolution
        self.pos_encoding = AbsolutePositionEmbedder(pos_emb_dim, 6)
        self.conv3d = nn.Conv3d(
            in_channels=pos_emb_dim,
            out_channels=hidden_channel,
            kernel_size=(4,8,8),
            stride=(4,8,8),
            padding=(2,0,0))
        
        self.bn3d = nn.BatchNorm3d(hidden_channel)
        self.act3d = nn.GELU()
        
        self.conv2d = nn.Conv2d(
            in_channels=hidden_channel,
            out_channels=out_channel,
            kernel_size=2,
            stride=2)
        
        self.bn2d = nn.BatchNorm2d(out_channel)
        self.act2d = nn.GELU()
        
        # 投影层
        self.proj = zero_module(nn.Linear(out_channel, out_channel))
        
    def forward(self, extrinsics : torch.Tensor, intrinsics : torch.Tensor):
        dtype = extrinsics.dtype
        extrinsics = extrinsics.type(torch.float32)
        intrinsics = intrinsics.type(torch.float32)
        plucker = get_plucker_ray(extrinsics, intrinsics, H = self.resolution[0], W = self.resolution[1])
        x = self.pos_encoding(plucker).to(dtype)
        
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.act3d(x)
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.conv2d(x)  # => (B*12, 3072, 29,44)
        x = self.bn2d(x)
        x = self.act2d(x)
        x = x.mean(dim=[2,3])  # 全局平均池化 => (B*12, 3072)
        x = self.proj(x)
        x = x.reshape(B, T, -1)  # => (B, 12, 3072)
        return x