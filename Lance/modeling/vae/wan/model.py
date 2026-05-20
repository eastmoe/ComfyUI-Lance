# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

__all__ = ['WanVideoVAE']

from typing import List
import torch
from torch import Tensor
from einops import rearrange

from common.utils.logging import get_logger
from common.utils.distributed import get_device
from common.utils.misc import AutoEncoderParams
from .vae2_2 import Wan2_2_VAE


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


class WanVideoVAE(object):
    __version__ = "v2.2"
    __name__ = "WanVideoVAE"
    __logger__ = None

    def __init__(self, config_path: str = "", **kwargs) -> None:
        if self.__class__.__logger__ is None:
            self.__class__.__logger__ = get_logger(self.__class__.__name__)
        self.logger = self.__class__.__logger__

        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.configure_vae_model()
        self.use_sample = kwargs.get("use_sample", True)

        # wan vae2.2 config is equal to seedance vae
        self.vae_config = AutoEncoderParams(
            downsample_spatial=16,
            downsample_temporal=4,
            z_channels=48,
            # scale_factor=1.0,
            # shift_factor=0.012,
        )

    def configure_vae_model(self):
        device = get_device()

        # 从 path_default.yaml 读取 VAE 路径
        try:
            from config.config_factory import get_model_path
            vae_path = get_model_path("vae.wan")
        except Exception as e:
            # 降级到默认路径
            vae_path = "downloads/Wan2.2_VAE.pth"
        
        self.vae: Wan2_2_VAE = Wan2_2_VAE(vae_pth=vae_path, device=device, dtype=self.dtype)
        # self.vae.requires_grad_(False).eval()
        # self.vae.to(device=get_device())

    @staticmethod
    def _tile_starts(size: int, tile: int, overlap: int) -> List[int]:
        tile = max(1, min(int(tile), int(size)))
        if tile >= size:
            return [0]
        overlap = max(0, min(int(overlap), tile - 1))
        stride = max(1, tile - overlap)
        starts = []
        pos = 0
        while True:
            start = min(pos, size - tile)
            if not starts or starts[-1] != start:
                starts.append(start)
            if start + tile >= size:
                break
            pos += stride
        return starts

    @staticmethod
    def _tile_weight(
        tile_shape: torch.Size,
        *,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        latent_h: int,
        latent_w: int,
        scale_y: int,
        scale_x: int,
        overlap_latent: int,
    ) -> Tensor:
        tile_h = int(tile_shape[-2])
        tile_w = int(tile_shape[-1])
        weight = torch.ones((1, 1, 1, tile_h, tile_w), dtype=torch.float32)
        overlap_y = min(overlap_latent * scale_y, tile_h // 2)
        overlap_x = min(overlap_latent * scale_x, tile_w // 2)
        if y0 > 0 and overlap_y > 0:
            ramp = torch.linspace(0.0, 1.0, overlap_y, dtype=torch.float32).view(1, 1, 1, overlap_y, 1)
            weight[..., :overlap_y, :] *= ramp
        if y1 < latent_h and overlap_y > 0:
            ramp = torch.linspace(1.0, 0.0, overlap_y, dtype=torch.float32).view(1, 1, 1, overlap_y, 1)
            weight[..., -overlap_y:, :] *= ramp
        if x0 > 0 and overlap_x > 0:
            ramp = torch.linspace(0.0, 1.0, overlap_x, dtype=torch.float32).view(1, 1, 1, 1, overlap_x)
            weight[..., :, :overlap_x] *= ramp
        if x1 < latent_w and overlap_x > 0:
            ramp = torch.linspace(1.0, 0.0, overlap_x, dtype=torch.float32).view(1, 1, 1, 1, overlap_x)
            weight[..., :, -overlap_x:] *= ramp
        return weight

    def _vae_decode_tiled(self, u: Tensor, tile_size: int, tile_overlap: int) -> Tensor:
        _, _, _, latent_h, latent_w = u.shape
        spatial_scale = int(getattr(self.vae_config, "downsample_spatial", 16) or 16)
        tile_latent = max(1, (max(1, int(tile_size)) + spatial_scale - 1) // spatial_scale)
        overlap_latent = max(0, (max(0, int(tile_overlap)) + spatial_scale - 1) // spatial_scale)

        if tile_latent >= latent_h and tile_latent >= latent_w:
            return self.vae.decode(u)

        tile_latent = min(tile_latent, max(latent_h, latent_w))
        overlap_latent = min(overlap_latent, max(tile_latent - 1, 0))
        y_starts = self._tile_starts(latent_h, min(tile_latent, latent_h), overlap_latent)
        x_starts = self._tile_starts(latent_w, min(tile_latent, latent_w), overlap_latent)

        output = None
        weights = None
        for y0 in y_starts:
            y1 = min(y0 + tile_latent, latent_h)
            for x0 in x_starts:
                x1 = min(x0 + tile_latent, latent_w)
                tile = u[..., y0:y1, x0:x1]
                decoded = self.vae.decode(tile).float().clamp_(-1, 1).cpu()

                scale_y = max(1, decoded.shape[-2] // max(1, y1 - y0))
                scale_x = max(1, decoded.shape[-1] // max(1, x1 - x0))
                full_h = latent_h * scale_y
                full_w = latent_w * scale_x
                py0, py1 = y0 * scale_y, y0 * scale_y + decoded.shape[-2]
                px0, px1 = x0 * scale_x, x0 * scale_x + decoded.shape[-1]

                if output is None:
                    output = torch.zeros(
                        (decoded.shape[0], decoded.shape[1], decoded.shape[2], full_h, full_w),
                        dtype=torch.float32,
                    )
                    weights = torch.zeros((1, 1, 1, full_h, full_w), dtype=torch.float32)

                weight = self._tile_weight(
                    decoded.shape,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    scale_y=scale_y,
                    scale_x=scale_x,
                    overlap_latent=overlap_latent,
                )
                output[..., py0:py1, px0:px1] += decoded * weight
                weights[..., py0:py1, px0:px1] += weight
                del tile, decoded, weight
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return output.div_(weights.clamp_min_(1e-6)).clamp_(-1, 1)

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], **kwargs) -> List[Tensor]:
        device = get_device()

        latents = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for x in samples:
                x = x.to(device=device).unsqueeze(0)  # 1CTHW

                u, log_var = self.vae.encode(x)  # [1,48,t,h,w], [1,48,t,h,w]

                if self.use_sample:
                    u = reparameterize(u, log_var)  # [1,48,t,h,w]

                u = rearrange(u, "b c ... -> b ... c")  # -> [1,t,h,w,48] for 兼容

                latents.append(u.squeeze(0))  # -> [t,h,w,48]

            return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor], **kwargs) -> List[Tensor]:
        device = get_device()
        tiled = bool(kwargs.get("tiled", False))
        tile_size = int(kwargs.get("tile_size", 384))
        tile_overlap = int(kwargs.get("tile_overlap", 64))

        samples = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for u in latents:
                u = u.unsqueeze(0).to(device=device)  # -> [1,t,h,w,48]
                u = rearrange(u, "b ... c -> b c ...")  # -> [1,48,t,h,w]

                if tiled and tile_size > 0:
                    x_hat = self._vae_decode_tiled(u, tile_size, tile_overlap)  # -> [1,3,T,H,W]
                else:
                    x_hat = self.vae.decode(u)  # -> [1,3,T,H,W]

                samples.append(x_hat.squeeze(0))  # -> List[[3,T,H,W]]

            return samples
