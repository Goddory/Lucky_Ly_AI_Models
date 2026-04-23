from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, max_tokens: int = 4096) -> None:
        super().__init__()
        reduced_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.max_tokens = max_tokens

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).reshape(b, -1, h * w).transpose(1, 2)
        k = self.key(x).reshape(b, -1, h * w)
        v = self.value(x).reshape(b, -1, h * w)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        return torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        num_tokens = h * w

        if num_tokens > self.max_tokens:
            scale = (num_tokens / float(self.max_tokens)) ** 0.5
            target_h = max(1, int(h / scale))
            target_w = max(1, int(w / scale))
            x_small = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            attended_small = self._attend(x_small)
            attended = F.interpolate(
                attended_small,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            attended = self._attend(x)

        return x + self.gamma * attended


class TextureGeneratorG(nn.Module):
    """
    Geometry-aware texture prior network G.

    Inputs:
    - uv_texture: [B, 3, H, W]
    - z_g: [B, latent_dim]

    Outputs:
    - M: skin biophysical control map [B, 2, H, W] (melanin/hemoglobin)
    - H: high-frequency detail map [B, 3, H, W] (wrinkles/pores)
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.res = nn.Sequential(
            ResidualBlock(base_channels * 4),
            SelfAttention2D(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.cond = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 8),
            nn.GELU(),
            nn.Linear(base_channels * 8, base_channels * 8),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 4, base_channels * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 2, base_channels),
        )

        self.head_m = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.head_h = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, uv_texture: torch.Tensor, z_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc1(uv_texture)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res(x)

        cond = self.cond(z_g)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        x = x * (1 + gamma) + beta

        x = self.up1(x)
        x = self.up2(x)

        m = torch.sigmoid(self.head_m(x))
        h = torch.tanh(self.head_h(x))
        return m, h


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = nn.Sequential(
            ConvBlock(in_channels + skip_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SkinToneControlGA(nn.Module):
    """U-Net G_A: transforms M + alpha into base albedo map A."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        # M has 2 channels, alpha adds one channel.
        self.down1 = UNetDown(3, base_channels)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = UNetDown(base_channels, base_channels * 2)
        self.pool2 = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            UNetDown(base_channels * 2, base_channels * 4),
            SelfAttention2D(base_channels * 4),
        )

        self.up1 = UNetUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = UNetUp(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, m: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=m.dtype, device=m.device)

        if alpha.dim() == 0:
            alpha = alpha[None, None, None, None].expand(m.size(0), 1, m.size(2), m.size(3))
        elif alpha.dim() == 1:
            alpha = alpha[:, None, None, None].expand(m.size(0), 1, m.size(2), m.size(3))

        x = torch.cat([m, alpha], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        # Gradient checkpointing: recompute activations during backward
        # instead of storing them → saves ~40% VRAM at skip connections.
        # We check torch.is_grad_enabled() because in Stage 1, GA is frozen (eval mode)
        # but the input from G still has gradients.
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            b  = checkpoint(self.bottleneck, self.pool2(d2), use_reentrant=False)
            u1 = checkpoint(self.up1, b, d2,                 use_reentrant=False)
            u2 = checkpoint(self.up2, u1, d1,                use_reentrant=False)
        else:
            b  = self.bottleneck(self.pool2(d2))
            u1 = self.up1(b, d2)
            u2 = self.up2(u1, d1)

        return torch.sigmoid(self.head(u2))


class ReflectanceNetworkGC(nn.Module):
    """Final reflectance network G_C combining A and H into C."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(6, base_channels),
            ResidualBlock(base_channels),
            SelfAttention2D(base_channels),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=1),
        )

    def forward(self, a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, h], dim=1)
        return torch.sigmoid(self.net(x))


class PBRExtractorGE(nn.Module):
    """PBR decomposition network G_E that predicts Specular S and Normal N maps."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            ConvBlock(3, base_channels),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2),
            SelfAttention2D(base_channels * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 2, base_channels),
        )

        self.spec_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.normal_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(c)
        s = torch.sigmoid(self.spec_head(x))

        n = torch.tanh(self.normal_head(x))
        n = F.normalize(n, dim=1)
        return s, n


class LoRAConv2d(nn.Module):
    """LoRA adapter wrapper for Conv2d layers used in PEFT fine-tuning."""

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.conv = conv
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the base convolution for parameter-efficient adaptation.
        for param in self.conv.parameters():
            param.requires_grad = False

        if conv.groups == 1:
            down_kernel = conv.kernel_size
            down_stride = conv.stride
            down_padding = conv.padding
            down_dilation = conv.dilation
        else:
            down_kernel = (1, 1)
            down_stride = (1, 1)
            down_padding = (0, 0)
            down_dilation = (1, 1)

        self.lora_down = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=rank,
            kernel_size=down_kernel,
            stride=down_stride,
            padding=down_padding,
            dilation=down_dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.lora_up(self.lora_down(x)) * self.scaling


class LoRAMultiheadAttention(nn.Module):
    """LoRA adapter for MultiheadAttention projection weights."""

    def __init__(self, attn: nn.MultiheadAttention, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.attn = attn
        self.rank = rank
        self.scaling = alpha / rank

        for param in self.attn.parameters():
            param.requires_grad = False

        embed_dim = attn.embed_dim
        self.q_down = nn.Linear(embed_dim, rank, bias=False)
        self.q_up = nn.Linear(rank, embed_dim, bias=False)
        self.k_down = nn.Linear(embed_dim, rank, bias=False)
        self.k_up = nn.Linear(rank, embed_dim, bias=False)
        self.v_down = nn.Linear(embed_dim, rank, bias=False)
        self.v_up = nn.Linear(rank, embed_dim, bias=False)

        nn.init.kaiming_uniform_(self.q_down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.k_down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_down.weight, a=5**0.5)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.k_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ):
        q = query + self.q_up(self.q_down(query)) * self.scaling
        k = key + self.k_up(self.k_down(key)) * self.scaling
        v = value + self.v_up(self.v_down(value)) * self.scaling
        return self.attn(q, k, v, **kwargs)


def inject_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_keywords: Optional[Sequence[str]] = None,
) -> nn.Module:
    """
    Recursively inject LoRA adapters into convolution and attention modules.

    target_keywords filters modules by fully-qualified name. If omitted, all
    eligible Conv2d and MultiheadAttention modules are adapted.
    """

    keywords = tuple((target_keywords or ()))

    def _matches(name: str) -> bool:
        if not keywords:
            return True
        lowered = name.lower()
        return any(keyword.lower() in lowered for keyword in keywords)

    def _inject(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            fq_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Conv2d) and _matches(fq_name):
                setattr(parent, child_name, LoRAConv2d(child, rank=rank, alpha=alpha))
                continue

            if isinstance(child, nn.MultiheadAttention) and _matches(fq_name):
                setattr(
                    parent,
                    child_name,
                    LoRAMultiheadAttention(child, rank=rank, alpha=alpha),
                )
                continue

            _inject(child, fq_name)

    _inject(module)
    return module


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if "lora_" in name and param.requires_grad:
            yield param
