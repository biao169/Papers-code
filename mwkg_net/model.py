# -*- coding: utf-8 -*-
"""MWKG-Net reconstruction from Jia et al., MSSP 242 (2026), 113662.

Paper: A multi-scale wavelet and spectral kurtosis-guided framework for
noise-resistant rolling bearing fault diagnosis.

The paper fixes the topology and equations but omits hidden channel widths,
MWConv1D kernel size/decomposition depth, and the gamma used by Eq. (14).
Those values are therefore explicit, documented reconstruction choices.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _haar_filters(dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    scale = 1.0 / math.sqrt(2.0)
    low = torch.tensor([scale, scale], dtype=dtype).view(1, 1, 2)
    high = torch.tensor([scale, -scale], dtype=dtype).view(1, 1, 2)
    return low, high


class HaarDWT1D(nn.Module):
    """Fixed Haar analysis/synthesis transform used by paper Eq. (1)-(6)."""

    def __init__(self) -> None:
        super().__init__()
        low, high = _haar_filters()
        self.register_buffer("low_filter", low)
        self.register_buffer("high_filter", high)

    def decompose(self, x: Tensor) -> tuple[Tensor, Tensor, int]:
        original_length = x.size(-1)
        if original_length % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        channels = x.size(1)
        low_filter = self.low_filter.to(dtype=x.dtype).repeat(channels, 1, 1)
        high_filter = self.high_filter.to(dtype=x.dtype).repeat(channels, 1, 1)
        low = F.conv1d(x, low_filter, stride=2, groups=channels)
        high = F.conv1d(x, high_filter, stride=2, groups=channels)
        return low, high, original_length

    def reconstruct(self, low: Tensor, high: Tensor, output_length: int) -> Tensor:
        if low.shape != high.shape:
            raise ValueError("Haar reconstruction requires matching low/high shapes")
        channels = low.size(1)
        low_filter = self.low_filter.to(dtype=low.dtype).repeat(channels, 1, 1)
        high_filter = self.high_filter.to(dtype=low.dtype).repeat(channels, 1, 1)
        output = F.conv_transpose1d(low, low_filter, stride=2, groups=channels)
        output = output + F.conv_transpose1d(high, high_filter, stride=2, groups=channels)
        return output[..., :output_length]


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MWConv1D(nn.Module):
    """Paper Eq. (1)-(8): multi-level Haar pyramid and reverse reconstruction."""

    def __init__(self, channels: int, levels: int = 3, kernel_size: int = 5) -> None:
        super().__init__()
        if channels < 1 or levels < 1:
            raise ValueError("channels and levels must be positive")
        self.channels = channels
        self.levels = levels
        self.wavelet = HaarDWT1D()
        # Fig. 1 contains a direct Conv1D branch parallel to the wavelet path.
        self.base_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.scale_convs = nn.ModuleList(
            [DepthwiseSeparableConv1d(2 * channels, kernel_size) for _ in range(levels)]
        )
        self.scale_weights = nn.ParameterList(
            [nn.Parameter(torch.full((1, 2 * channels, 1), 0.1)) for _ in range(levels)]
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"Expected [batch,{self.channels},length], got {tuple(x.shape)}")
        current = x
        scale_features: list[Tensor] = []
        lengths: list[int] = []
        for conv, alpha in zip(self.scale_convs, self.scale_weights):
            low, high, original_length = self.wavelet.decompose(current)
            combined = torch.cat([low, high], dim=1)
            scale_features.append(conv(combined) * alpha)
            lengths.append(original_length)
            current = low

        reconstructed: Tensor | None = None
        for feature, output_length in zip(reversed(scale_features), reversed(lengths)):
            low_feature, high_feature = feature.chunk(2, dim=1)
            if reconstructed is not None:
                if reconstructed.size(-1) != low_feature.size(-1):
                    reconstructed = F.interpolate(
                        reconstructed, size=low_feature.size(-1), mode="linear", align_corners=False
                    )
                low_feature = low_feature + reconstructed
            reconstructed = self.wavelet.reconstruct(low_feature, high_feature, output_length)
        assert reconstructed is not None
        return self.base_conv(x) + reconstructed


class MWCBlock(nn.Module):
    """Residual Conv-BN-ReLU-MWConv-BN block shown in paper Fig. 4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        wavelet_levels: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.wavelet = MWConv1D(out_channels, levels=wavelet_levels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        output = self.norm(self.wavelet(self.pre(x)))
        return self.activation(output + residual)


def channel_kurtosis(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Paper Eq. (9), evaluated over the temporal dimension."""
    if x.ndim != 3:
        raise ValueError("channel_kurtosis expects [batch,channels,length]")
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1).clamp_min(eps)
    fourth_moment = centered.pow(4).mean(dim=-1)
    return fourth_moment / variance.square()


class KGFR(nn.Module):
    """Paper Eq. (9)-(11): kurtosis-guided channel re-weighting."""

    def __init__(self, temperature: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        per_sample_kurtosis = channel_kurtosis(x, eps=self.eps)
        weights = torch.softmax(per_sample_kurtosis / self.temperature, dim=1)
        return x * weights.unsqueeze(-1), per_sample_kurtosis, weights


class MWKGNet(nn.Module):
    """MWKG-Net topology reconstructed from paper Fig. 4.

    Defaults use a 64/128-channel two-stage backbone, four wavelet levels,
    two shallow blocks, one transition block and three deep blocks. The paper
    does not publish those numerical widths/depths; all are configurable.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 64,
        deep_channels: int = 128,
        wavelet_levels: int = 4,
        wavelet_kernel_size: int = 5,
        shallow_blocks: int = 2,
        deep_blocks: int = 3,
        kgfr_temperature: float = 1.0,
        name: str = "mwkg_net",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.name = name
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            MWConv1D(base_channels, levels=wavelet_levels, kernel_size=wavelet_kernel_size),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.shallow = nn.Sequential(
            *[
                MWCBlock(
                    base_channels,
                    base_channels,
                    wavelet_levels=wavelet_levels,
                    kernel_size=wavelet_kernel_size,
                )
                for _ in range(shallow_blocks)
            ]
        )
        deep: list[nn.Module] = [
            MWCBlock(
                base_channels,
                deep_channels,
                stride=2,
                wavelet_levels=wavelet_levels,
                kernel_size=wavelet_kernel_size,
            )
        ]
        deep.extend(
            MWCBlock(
                deep_channels,
                deep_channels,
                wavelet_levels=wavelet_levels,
                kernel_size=wavelet_kernel_size,
            )
            for _ in range(deep_blocks)
        )
        self.deep = nn.Sequential(*deep)
        self.kgfr = KGFR(temperature=kgfr_temperature)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(deep_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, return_feature_map: bool = False) -> dict[str, Any]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch,channels,length], got {tuple(x.shape)}")
        feature_map = self.deep(self.shallow(self.stem(x)))
        reweighted, kurtosis, channel_weights = self.kgfr(feature_map)
        features = self.pool(reweighted).squeeze(-1)
        output: dict[str, Any] = {
            "logits": self.classifier(features),
            "features": features,
            "kurtosis": kurtosis,
            "channel_weights": channel_weights,
        }
        if return_feature_map:
            output["feature_map"] = reweighted
        return output


class MWKGLoss(nn.Module):
    """Paper Eq. (12)-(14): CE plus progressive kurtosis maximization."""

    def __init__(self, total_epochs: int = 200, gamma: float = 0.5) -> None:
        super().__init__()
        if total_epochs < 1 or gamma <= 0:
            raise ValueError("total_epochs and gamma must be positive")
        self.total_epochs = total_epochs
        self.gamma = gamma

    def regularization_weight(self, epoch: int) -> float:
        progress = min(max(float(epoch), 0.0) / (self.total_epochs * self.gamma), 1.0)
        return math.cos(progress * math.pi / 2.0) ** 2

    def forward(self, outputs: dict[str, Tensor], target: Tensor, epoch: int) -> dict[str, Tensor | float]:
        classification = F.cross_entropy(outputs["logits"], target)
        kurtosis_loss = -outputs["kurtosis"].mean()
        weight = self.regularization_weight(epoch)
        total = classification + weight * kurtosis_loss
        return {
            "total": total,
            "classification": classification,
            "kurtosis": kurtosis_loss,
            "kurtosis_weight": weight,
        }


__all__ = [
    "HaarDWT1D",
    "DepthwiseSeparableConv1d",
    "MWConv1D",
    "MWCBlock",
    "KGFR",
    "MWKGNet",
    "MWKGLoss",
    "channel_kurtosis",
]
