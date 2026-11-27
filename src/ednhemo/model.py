"""The 2D U-Net used by both published checkpoints.

The architecture is copied verbatim from the training notebooks so the released
``state_dict`` files load without any key remapping:

* four encoder levels (32, 64, 128, 256 channels), bottleneck at 512,
* two 3x3 convolutions + batch norm + ReLU per block,
* transposed convolutions in the decoder, bilinear fallback when the spatial
  size does not match the skip connection,
* 7,765,409 trainable parameters.

The model takes a single-channel 512x512 slice normalized to ``[-1, 1]`` and
returns one logit map per pixel (apply ``sigmoid`` and a 0.5 threshold).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DoubleConv", "UNet2D", "build_unet", "count_parameters"]


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """Four-level 2D U-Net, identical to the one used for training."""

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
            ch = f
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip = skip[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skip[i // 2]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, s], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


def build_unet(**kwargs: object) -> UNet2D:
    """Return a freshly initialized :class:`UNet2D`."""
    return UNet2D(**kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (7,765,409 for the published checkpoints)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
