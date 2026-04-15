from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for _ in range(3):
            next_channels = min(channels * 2, 512)
            layers.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels

        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Three-scale patch discriminator similar to StyleGAN/Pix2Pix training setups."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_scales: int = 3) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator(in_channels, base_channels) for _ in range(n_scales)]
        )
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        current = x
        for disc in self.discriminators:
            outputs.append(disc(current))
            current = self.downsample(current)
        return outputs
