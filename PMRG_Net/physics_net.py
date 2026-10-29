from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def finite(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Keep rare numerical spikes from interrupting training."""

    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


def safe_divide(num: torch.Tensor, den: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Stable division used by filter normalization and feature normalization."""

    return num / den.clamp_min(eps)


class VibrationAugment(nn.Module):
    """Training-only perturbations that simulate amplitude and phase variations."""

    def __init__(
        self,
        *,
        noise_std: float = 0.012,
        gain_range: tuple[float, float] = (0.80, 1.20),
        max_shift_ratio: float = 0.06,
        max_mask_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise_std = float(noise_std)
        self.gain_range = gain_range
        self.max_shift_ratio = float(max_shift_ratio)
        self.max_mask_ratio = float(max_mask_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, _, n = x.shape
        gain = torch.empty(b, 1, 1, device=x.device, dtype=x.dtype).uniform_(*self.gain_range)
        y = x * gain
        if self.max_shift_ratio > 0:
            max_shift = max(1, int(n * self.max_shift_ratio))
            shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=x.device)
            y = torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(y, shifts)], dim=0)
        if self.noise_std > 0:
            scale = y.detach().std(dim=-1, keepdim=True).clamp_min(EPS)
            y = y + torch.randn_like(y) * scale * self.noise_std
        if self.max_mask_ratio > 0 and n > 16:
            width = max(1, int(n * self.max_mask_ratio))
            starts = torch.randint(0, max(1, n - width), (b,), device=x.device)
            for i, start in enumerate(starts.tolist()):
                y[i, :, start:start + width] = 0
        return y


class LearnableGaborFilterBank(nn.Module):
    """Learnable Gabor bank with strict log-spaced center frequencies and Q-factor widths."""

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 24,
        kernel_size: int = 129,
        min_norm_freq: float = 0.025,
        max_norm_freq: float = 0.48,
        spacing_mode: Literal["log", "linear"] = "log",
        q_min: float = 2.0,
        q_max: float = 16.0,
        init_q: float = 7.0,
        max_log_gain: float = 1.0,
        quadrature: bool = True,
        output_mode: Literal["envelope", "filtered", "both"] = "envelope",
        share_filters_across_channels: bool = True,
        padding_mode: Literal["same", "valid"] = "same",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if spacing_mode not in {"log", "linear"}:
            raise ValueError("spacing_mode must be 'log' or 'linear'.")
        if min_norm_freq <= 0 or max_norm_freq <= min_norm_freq:
            raise ValueError("Require 0 < min_norm_freq < max_norm_freq.")
        self.in_channels = int(in_channels)
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.min_norm_freq = float(min_norm_freq)
        self.max_norm_freq = float(max_norm_freq)
        self.spacing_mode = spacing_mode
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.max_log_gain = float(max_log_gain)
        self.quadrature = bool(quadrature)
        self.output_mode = output_mode
        self.share_filters_across_channels = bool(share_filters_across_channels)
        self.padding_mode = padding_mode
        self.param_channels = 1 if self.share_filters_across_channels else self.in_channels
        self.register_buffer("time_index", torch.arange(kernel_size).float() - kernel_size // 2)

        if self.n_filters > 1:
            self.raw_gaps = nn.Parameter(torch.zeros(self.param_channels, self.n_filters - 1))
        else:
            self.raw_gaps = None
        q_pos = (float(init_q) - self.q_min) / max(self.q_max - self.q_min, EPS)
        self.raw_q = nn.Parameter(torch.full((self.param_channels, self.n_filters), math.log(q_pos / (1.0 - q_pos))))
        self.raw_phase = nn.Parameter(torch.zeros(self.param_channels, self.n_filters))
        self.raw_gain = nn.Parameter(torch.zeros(self.param_channels, self.n_filters))

    def centers_norm_to_nyquist(self) -> torch.Tensor:
        """
        返回严格递增的中心频率，
        并归一化到 Nyquist 频率范围 [0, 1]。

        注意：
            1 表示 Nyquist frequency
            实际数字频率 = 0.5 * normalized_frequency
        """

        if self.n_filters == 1:  # 如果只有一个滤波器
            # log spacing 时取几何中心
            value = math.sqrt(self.min_norm_freq * self.max_norm_freq) if self.spacing_mode == "log" else 0.5 * (self.min_norm_freq + self.max_norm_freq)
            # 返回 shape = [param_channels, 1]
            return torch.full((self.param_channels, 1), value, device=self.raw_q.device, dtype=self.raw_q.dtype)
        assert self.raw_gaps is not None  # 多滤波器情况下必须存在 raw_gaps
        weights = F.softmax(self.raw_gaps, dim=-1)  # softmax 后：所有 gap > 0, 所有 gap 总和 = 1
        if self.spacing_mode == "log":   # log spacing（推荐）
            low = math.log(self.min_norm_freq)
            high = math.log(self.max_norm_freq)
            gaps = (high - low) * weights
            first = torch.full((self.param_channels, 1), low, device=weights.device, dtype=weights.dtype) # 第一个中心频率固定为 low
            log_centers = torch.cat([first, first + torch.cumsum(gaps, dim=-1)], dim=-1)  # cumulative sum 保证频率严格递增
            return torch.exp(log_centers).clamp(self.min_norm_freq, self.max_norm_freq)  # 从 log-frequency 转回 frequency
        gaps = (self.max_norm_freq - self.min_norm_freq) * weights  # 在线性频率空间分配 gap
        first = torch.full((self.param_channels, 1), self.min_norm_freq, device=weights.device, dtype=weights.dtype)  # 第一个频率固定为最小频率
        centers = torch.cat([first, first + torch.cumsum(gaps, dim=-1)], dim=-1)  # cumulative sum 保证递增
        return centers.clamp(self.min_norm_freq, self.max_norm_freq)   # 防止数值越界

    def centers_cycles_per_sample(self) -> torch.Tensor:
        """将归一化 Nyquist 频率 [0,1] 转为数字频率 cycles/sample，其中 Nyquist=0.5。"""
        return 0.5 * self.centers_norm_to_nyquist()

    def q_factors(self) -> torch.Tensor:
        """用 sigmoid 将可学习参数 raw_q 映射到有界 Q 因子区间 [q_min, q_max]。"""
        return self.q_min + (self.q_max - self.q_min) * torch.sigmoid(self.raw_q)

    def sigmas(self) -> torch.Tensor:
        """根据 Gabor/Constant-Q 关系 σ=Q/(2πf) 计算时域高斯窗宽度（单位：samples）。"""
        cycles = self.centers_cycles_per_sample().clamp_min(EPS)
        return (self.q_factors() / (2.0 * math.pi * cycles)).clamp(2.0, float(self.kernel_size))

    def _base_filters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """生成归一化后的 cosine/sine Gabor 基滤波器。"""
        t = self.time_index.to(self.raw_q.device, self.raw_q.dtype)[None, None, :]  # 构造时间轴 [-K//2, ..., K//2]
        fc = self.centers_cycles_per_sample()[:, :, None]  # 中心频率 shape=[C,F,1]
        sigma = self.sigmas()[:, :, None]   # 高斯窗宽度 shape=[C,F,1]
        phase = math.pi * torch.tanh(self.raw_phase)[:, :, None]  # 相位限制到 [-π, π]
        gain = torch.exp(self.max_log_gain * torch.tanh(self.raw_gain))[:, :, None]  # 增益限制为有限正值
        window = torch.exp(-0.5 * (t / sigma.clamp_min(EPS)) ** 2) # Gaussian window: exp(-t²/(2σ²))
        h_cos = gain * window * torch.cos(2.0 * math.pi * fc * t + phase)    # cosine Gabor kernel
        h_cos = h_cos - h_cos.mean(dim=-1, keepdim=True)   # 去除 DC 分量
        h_cos = safe_divide(h_cos, h_cos.norm(dim=-1, keepdim=True))  # L2 归一化
        h_sin = None
        if self.quadrature:  # 若启用 quadrature 则生成 sine 核
            h_sin = gain * window * torch.sin(2.0 * math.pi * fc * t + phase)
            h_sin = h_sin - h_sin.mean(dim=-1, keepdim=True)
            h_sin = safe_divide(h_sin, h_sin.norm(dim=-1, keepdim=True))
        return finite(h_cos), finite(h_sin) if h_sin is not None else None

    def filters(self) -> torch.Tensor:
        """将 Gabor 基滤波器整理成 Conv1D 所需的 weight tensor。"""

        h_cos, h_sin = self._base_filters()
        if self.share_filters_across_channels:  # 若共享滤波器则复制到所有输入通道
            h_cos = h_cos.repeat(self.in_channels, 1, 1)
            if h_sin is not None:
                h_sin = h_sin.repeat(self.in_channels, 1, 1)
        if not self.quadrature:  # 不使用 quadrature 时仅返回 cosine 核
            return h_cos.reshape(self.in_channels * self.n_filters, 1, self.kernel_size)
        assert h_sin is not None
        return torch.stack([h_cos, h_sin], dim=2).reshape(self.in_channels * self.n_filters * 2, 1, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        b, c, _ = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}.")
        padding = self.kernel_size // 2 if self.padding_mode == "same" else 0
        y = F.conv1d(x, self.filters().to(x.device, x.dtype), padding=padding, groups=c)
        n_out = y.shape[-1]
        if self.quadrature:
            filtered = y.reshape(b, c, self.n_filters, 2, n_out)
            envelope = torch.sqrt((filtered ** 2).sum(dim=3).clamp_min(EPS))
        else:
            filtered = y.reshape(b, c, self.n_filters, n_out)
            envelope = filtered.abs()
        filtered = finite(filtered)
        envelope = finite(envelope)
        if self.output_mode == "filtered":
            return filtered
        if self.output_mode == "both":
            return {"filtered": filtered, "envelope": envelope}
        return envelope

    @torch.no_grad()
    def frequency_response(self, n_fft: int = 4096, fs_hz: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frequency response magnitude for interpretation figures."""

        h_cos, h_sin = self._base_filters()
        if self.share_filters_across_channels:
            h_cos = h_cos.repeat(self.in_channels, 1, 1)
            if h_sin is not None:
                h_sin = h_sin.repeat(self.in_channels, 1, 1)
        h_cos = h_cos.reshape(self.in_channels, self.n_filters, self.kernel_size)
        mag = torch.abs(torch.fft.rfft(h_cos, n=n_fft, dim=-1)) ** 2
        if h_sin is not None:
            h_sin = h_sin.reshape(self.in_channels, self.n_filters, self.kernel_size)
            mag = mag + torch.abs(torch.fft.rfft(h_sin, n=n_fft, dim=-1)) ** 2
        freqs = torch.linspace(0.0, fs_hz / 2.0, n_fft // 2 + 1, device=mag.device, dtype=mag.dtype)
        return freqs, torch.sqrt(mag.clamp_min(EPS))


class MultiScaleGaborFrontEnd(nn.Module):
    """Several log-spaced Gabor banks with different temporal resolutions."""

    def __init__(
        self,
        in_channels: int = 1,
        filters_per_scale: Sequence[int] = (8, 8, 8),
        kernel_sizes: Sequence[int] = (65, 129, 257),
        spacing_mode: Literal["log", "linear"] = "log",
        quadrature: bool = True,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if len(filters_per_scale) != len(kernel_sizes):
            raise ValueError("filters_per_scale and kernel_sizes must have the same length.")
        self.banks = nn.ModuleList([
            LearnableGaborFilterBank(
                in_channels=in_channels,
                n_filters=int(n_filters),
                kernel_size=int(kernel_size),
                spacing_mode=spacing_mode,
                quadrature=quadrature,
                output_mode="envelope",
            )
            for n_filters, kernel_size in zip(filters_per_scale, kernel_sizes)
        ])
        if not learnable:
            for p in self.parameters():
                p.requires_grad_(False)

    @property
    def n_filters(self) -> int:
        return sum(bank.n_filters for bank in self.banks)

    def centers_cycles_per_sample(self) -> torch.Tensor:
        return torch.cat([bank.centers_cycles_per_sample() for bank in self.banks], dim=1)

    def centers_norm_to_nyquist(self) -> torch.Tensor:
        return torch.cat([bank.centers_norm_to_nyquist() for bank in self.banks], dim=1)

    def sigmas(self) -> torch.Tensor:
        return torch.cat([bank.sigmas() for bank in self.banks], dim=1)

    def q_factors(self) -> torch.Tensor:
        return torch.cat([bank.q_factors() for bank in self.banks], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([bank(x) for bank in self.banks], dim=2)


class DilatedTemporalBlock(nn.Module):
    """
    时序空洞卷积残差块：
        先用 depthwise dilated convolution 提取时间依赖，
        再用 pointwise convolution 做通道混合，
        最后通过 residual connection 保持稳定训练。
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation), groups=channels, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.Dropout2d(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SEBlock(nn.Module):
    """Squeeze-and-excitation attention for channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x.mean(dim=(2, 3)))[:, :, None, None]


class ConvBlock(nn.Module):
    """Residual CNN block with stable group normalization."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


class StatsPool2d(nn.Module):
    """Mean/std/max pooling gives time-position-invariant vibration statistics."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        return torch.cat([flat.mean(-1), flat.std(-1), flat.amax(-1)], dim=1)


@dataclass(frozen=True)
class VibrationNetConfig:
    in_channels: int = 1
    num_classes: int = 3
    fs_hz: float = 25600.0
    emb_dim: int = 512
    front_end: Literal["multiscale_gabor", "single_gabor", "raw"] = "multiscale_gabor"
    spacing_mode: Literal["log", "linear"] = "log"
    quadrature: bool = True
    learnable_gabor: bool = True
    use_se: bool = True
    normalize_embedding: bool = False
    augment: bool = True
    filters_per_scale: Sequence[int] = (8, 8, 8)
    kernel_sizes: Sequence[int] = (65, 129, 257)


class MotorBearingGaborNet(nn.Module):
    """Vibration diagnosis model for source-condition training and target-condition generalization."""

    def __init__(self, cfg: VibrationNetConfig, name: str = "msgabor_cross_condition") -> None:
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.fs_hz = float(cfg.fs_hz)
        self.num_classes = int(cfg.num_classes)
        self.augment = VibrationAugment() if cfg.augment else nn.Identity()
        if cfg.front_end == "multiscale_gabor":
            self.gabor = MultiScaleGaborFrontEnd(
                in_channels=cfg.in_channels,
                filters_per_scale=cfg.filters_per_scale,
                kernel_sizes=cfg.kernel_sizes,
                spacing_mode=cfg.spacing_mode,
                quadrature=cfg.quadrature,
                learnable=cfg.learnable_gabor,
            )
        elif cfg.front_end == "single_gabor":
            self.gabor = MultiScaleGaborFrontEnd(
                in_channels=cfg.in_channels,
                filters_per_scale=(sum(cfg.filters_per_scale),),
                kernel_sizes=(cfg.kernel_sizes[len(cfg.kernel_sizes) // 2],),
                spacing_mode=cfg.spacing_mode,
                quadrature=cfg.quadrature,
                learnable=cfg.learnable_gabor,
            )
        else:
            self.gabor = None
        self.input_norm = nn.InstanceNorm2d(cfg.in_channels, affine=True)
        self.stem = ConvBlock(cfg.in_channels, 32, stride=1)
        self.temporal = nn.Sequential(DilatedTemporalBlock(32, 1), DilatedTemporalBlock(32, 2), DilatedTemporalBlock(32, 4))
        self.stage1 = ConvBlock(32, 64, stride=2)
        self.stage2 = ConvBlock(64, 96, stride=2)
        self.stage3 = ConvBlock(96, 128, stride=2)
        self.se = SEBlock(128) if cfg.use_se else nn.Identity()
        self.pool = StatsPool2d()
        self.embedding = nn.Sequential(nn.Linear(128 * 3, cfg.emb_dim), nn.LayerNorm(cfg.emb_dim), nn.GELU(), nn.Dropout(0.25))
        self.classifier = nn.Linear(cfg.emb_dim, cfg.num_classes)

    def _front_end(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        env = x.unsqueeze(2) if self.gabor is None else self.gabor(x)
        env = torch.log1p(env.abs())
        env = env - env.mean(dim=-1, keepdim=True)
        env = safe_divide(env, env.std(dim=-1, keepdim=True))
        return self.input_norm(env)

    def forward(self, x: torch.Tensor, return_feature_map: bool = False, **_: object) -> dict[str, torch.Tensor]:
        env = self._front_end(x)
        z = self.temporal(self.stem(env))
        z = self.stage1(z)
        z = self.stage2(z)
        feature_map = self.se(self.stage3(z))
        emb = self.embedding(self.pool(feature_map))
        if self.cfg.normalize_embedding:
            emb = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        out = {"logits": logits, "features": emb, "embedding": emb, "envelope": env}
        if return_feature_map:
            out["feature_map"] = feature_map
        return out


class GaborGeneralizationLoss(nn.Module):
    """Simple objective: label-smoothed CE plus optional scale decorrelation and impulse prior."""

    def __init__(
        self,
        *,
        lambda_decor: float = 1e-2,
        lambda_impulse: float = 1e-3,
        label_smoothing: float = 0.05,
        fault_labels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.lambda_decor = float(lambda_decor)
        self.lambda_impulse = float(lambda_impulse)
        self.label_smoothing = float(label_smoothing)
        self.fault_labels = tuple(int(x) for x in fault_labels) if fault_labels is not None else None

    def _decor_loss(self, envelope: torch.Tensor) -> torch.Tensor:
        _, _, n_freq, _ = envelope.shape
        z = envelope.mean(1)
        z = z - z.mean(-1, keepdim=True)
        z = F.normalize(z, p=2, dim=-1)
        corr = torch.matmul(z, z.transpose(1, 2))
        eye = torch.eye(n_freq, device=envelope.device, dtype=envelope.dtype)[None]
        return ((corr - eye) ** 2).mean()

    def _impulse_loss(self, envelope: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.fault_labels is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for label in self.fault_labels:
                mask |= labels == label
        if mask.sum() == 0:
            return envelope.new_tensor(0.0)
        env = envelope[mask]
        z = (env - env.mean(-1, keepdim=True)) / env.std(-1, keepdim=True).clamp_min(1e-6)
        return -(z ** 4).mean()

    def forward(self, outputs: Mapping[str, torch.Tensor], y: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        labels = y.long()
        loss_cls = F.cross_entropy(outputs["logits"], labels, label_smoothing=self.label_smoothing)
        loss_decor = self._decor_loss(outputs["envelope"])
        loss_impulse = self._impulse_loss(outputs["envelope"], labels)
        total = loss_cls + self.lambda_decor * loss_decor + self.lambda_impulse * loss_impulse
        return {"total": total, "cls": loss_cls.detach(), "decor": loss_decor.detach(), "impulse": loss_impulse.detach()}


SpeedInvariantVibrationNet = MotorBearingGaborNet
GeneralizationLoss = GaborGeneralizationLoss
PhysicsGuidedSpeedAwareNet = MotorBearingGaborNet
PhysicsGuidedFaultLoss = GaborGeneralizationLoss
