from __future__ import annotations

from typing import Dict, Optional, Sequence, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Dconformer: A denoising convolutional transformer with joint learning strategy for intelligent diagnosis of bearing faults
https://www.sciencedirect.com/science/article/pii/S0888327024000402
"""

# ============================================================
# Basic utilities
# ============================================================

def finite(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Replace NaN / Inf with finite values."""
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


# ============================================================
# Basic blocks
# ============================================================

class BConv1d(nn.Module):
    """
    Bconv in the paper:
        Conv1d + BatchNorm1d + LeakyReLU

    Paper uses Conv + BN + LeakyReLU in AMB and SDB.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation module.

    Paper Eq. (1)-(3):
        squeeze by global average pooling,
        excitation by two FC layers,
        scale input channels.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = self.pool(x).squeeze(-1)       # [B, C]
        w = self.fc(w).unsqueeze(-1)       # [B, C, 1]
        return x * w


# ============================================================
# AMB: Attention-guided Multi-Scale Branch
# ============================================================

class MSABlock1d(nn.Module):
    """
    Multi-Scale Attention block in AMB.

    According to Section 2.2 and Fig. 2:
        - multi-scale module
        - SE attention module
        - branches with 1x1 conv followed by kernels 1, 3, 5, 7
        - identity/shortcut branch
        - concatenate multi-scale branches
        - add shortcut
        - SE recalibration

    Input:
        x: [B, C, T]

    Output:
        y: [B, C, T]
    """

    def __init__(
        self,
        channels: int,
        branch_channels: int = 16,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        self.shortcut = BConv1d(
            channels,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                BConv1d(channels, branch_channels, kernel_size=1, padding=0),
                BConv1d(branch_channels, branch_channels, kernel_size=1),
            ),
            nn.Sequential(
                BConv1d(channels, branch_channels, kernel_size=1, padding=0),
                BConv1d(branch_channels, branch_channels, kernel_size=3),
            ),
            nn.Sequential(
                BConv1d(channels, branch_channels, kernel_size=1, padding=0),
                BConv1d(branch_channels, branch_channels, kernel_size=5),
            ),
            nn.Sequential(
                BConv1d(channels, branch_channels, kernel_size=1, padding=0),
                BConv1d(branch_channels, branch_channels, kernel_size=7),
            ),
        ])

        self.merge = nn.Sequential(
            nn.Conv1d(
                branch_channels * len(self.branches),
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.se = SEBlock1d(channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = torch.cat([branch(x) for branch in self.branches], dim=1)
        multi = self.merge(multi)

        y = multi + self.shortcut(x)
        y = self.se(y)

        return y


# ============================================================
# STB: Signal Transformer Branch
# ============================================================

class TransformerEncoderBlock1d(nn.Module):
    """
    Modified 1-D transformer encoder.

    Paper Section 2.3:
        - LayerNorm
        - multi-head self-attention
        - residual connection
        - MLP
        - residual connection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm transformer
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class SignalTransformerBranch(nn.Module):
    """
    Signal Transformer Branch.

    Paper Section 2.3:
        - split input signal into non-overlap 1 x 128 patches
        - linear projection
        - learnable class token
        - learnable positional embedding
        - stacked 1-D transformer encoders

    Input:
        x: [B, C, T]

    Output:
        cls_feature: [B, D]
        token_maps: list of [B, D, N+1]
    """

    def __init__(
        self,
        in_channels: int,
        win: int,
        emb_dim: int,
        patch_size: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.win = int(win)
        self.emb_dim = int(emb_dim)
        self.patch_size = int(patch_size)

        if self.win % self.patch_size != 0:
            raise ValueError(
                f"win={win} must be divisible by patch_size={patch_size}."
            )

        self.num_patches = self.win // self.patch_size

        # Linear projection of 1-D patches.
        self.patch_embed = nn.Conv1d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock1d(
                dim=emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]

        # [B, D, N]
        patches = self.patch_embed(x)

        # [B, N, D]
        tokens = patches.transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)

        # [B, N+1, D]
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        token_maps: List[torch.Tensor] = []

        for block in self.blocks:
            tokens = block(tokens)

            # store [B, D, N+1] for FAM
            token_maps.append(tokens.transpose(1, 2).contiguous())

        tokens = self.norm(tokens)

        cls_feature = tokens[:, 0]  # [B, D]

        return cls_feature, token_maps


# ============================================================
# SDB: Signal Denoising Branch
# ============================================================

class DenoisingSELayer(nn.Module):
    """
    Conv / ConvTranspose + SE + LeakyReLU block for SDB.
    """

    def __init__(
        self,
        module: nn.Module,
        channels: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            module,
            nn.BatchNorm1d(channels),
            SEBlock1d(channels, reduction=8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignalDenoisingBranch(nn.Module):
    """
    Signal Denoising Branch.

    Paper Section 2.4:
        - symmetric encoder-decoder
        - encoder uses Conv1d with stride 2
        - decoder uses ConvTranspose1d with stride 2
        - each layer followed by SE and LeakyReLU
        - output has same dimension as input signal

    Note:
        Paper states kernel size 1x1 and stride 2.
        In PyTorch ConvTranspose1d, kernel_size=2, stride=2 is used here
        to exactly recover temporal length more reliably.
    """

    def __init__(
        self,
        in_channels: int,
        n_stages: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.n_stages = int(n_stages)
        self.base_channels = int(base_channels)

        enc_channels: List[int] = [
            base_channels * (2 ** i)
            for i in range(n_stages)
        ]

        self.enc_channels = enc_channels

        encoder = []
        prev = in_channels

        for ch in enc_channels:
            conv = nn.Conv1d(
                prev,
                ch,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            )
            encoder.append(DenoisingSELayer(conv, ch))
            prev = ch

        self.encoder = nn.ModuleList(encoder)

        decoder = []
        rev_channels = list(reversed(enc_channels))

        prev = rev_channels[0]

        for idx, ch in enumerate(rev_channels[1:]):
            deconv = nn.ConvTranspose1d(
                prev,
                ch,
                kernel_size=2,
                stride=2,
                bias=False,
            )
            decoder.append(DenoisingSELayer(deconv, ch))
            prev = ch

        self.decoder = nn.ModuleList(decoder)

        self.final_deconv = nn.ConvTranspose1d(
            prev,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        original_len = x.shape[-1]

        feats: List[torch.Tensor] = []
        h = x

        for layer in self.encoder:
            h = layer(h)
            feats.append(h)

        for layer in self.decoder:
            h = layer(h)

        x_hat = self.final_deconv(h)

        # align length exactly
        if x_hat.shape[-1] > original_len:
            x_hat = x_hat[..., :original_len]
        elif x_hat.shape[-1] < original_len:
            pad = original_len - x_hat.shape[-1]
            x_hat = F.pad(x_hat, (0, pad))

        return x_hat, feats


# ============================================================
# FAM: Feature Alignment Module
# ============================================================

class FeatureAlignmentModule(nn.Module):
    """
    Feature Alignment Module.

    Paper Section 2.5:
        - 1x1 convolution to align channel dimension
        - GAP to align spatial dimension
        - element-wise summation into AMB

    This implementation converts arbitrary 1-D feature maps to:
        [B, out_channels, 1]

    Then it can be broadcast and added to AMB feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.align = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, T]
        y = self.align(feat)
        y = self.pool(y)
        return y


# ============================================================
# Dconformer model
# ============================================================

class Dconformer(nn.Module):
    """
    Dconformer reproduction.

    Compatible initialization:

        model_cls(
            win=...,
            num_classes=...,
            fs_hz=...,
            in_channels=...,
            emb_dim=...,
        )

    Inputs:
        x: [B, C, T]

    Outputs:
        {
            "logits":   [B, num_classes],
            "denoised": [B, C, T],
            "emb":      [B, emb_dim],
            "amb_emb":  [B, width],
            "stb_emb":  [B, emb_dim],
        }

    It does not return large intermediate feature maps by default.
    """

    def __init__(
        self,
        win: int,
        num_classes: int,
        fs_hz: float,
        in_channels: int,
        emb_dim: int,
        *,
        width: int = 64,
        n_msa_blocks: int = 3,
        msa_branch_channels: int = 16,
        patch_size: int = 128,
        transformer_depth: Optional[int] = None,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        sdb_base_channels: int = 32,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.win = int(win)
        self.num_classes = int(num_classes)
        self.fs_hz = float(fs_hz)
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)
        self.width = int(width)
        self.n_msa_blocks = int(n_msa_blocks)

        if transformer_depth is None:
            transformer_depth = n_msa_blocks

        # AMB stem
        self.amb_stem = BConv1d(
            in_channels,
            width,
            kernel_size=7,
        )

        self.amb_blocks = nn.ModuleList([
            MSABlock1d(
                channels=width,
                branch_channels=msa_branch_channels,
            )
            for _ in range(n_msa_blocks)
        ])

        # STB
        self.stb = SignalTransformerBranch(
            in_channels=in_channels,
            win=win,
            emb_dim=emb_dim,
            patch_size=patch_size,
            depth=transformer_depth,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
        )

        # SDB
        self.sdb = SignalDenoisingBranch(
            in_channels=in_channels,
            n_stages=n_msa_blocks,
            base_channels=sdb_base_channels,
        )

        # FAM: STB -> AMB
        self.fam_stb_to_amb = nn.ModuleList([
            FeatureAlignmentModule(
                in_channels=emb_dim,
                out_channels=width,
            )
            for _ in range(n_msa_blocks)
        ])

        # FAM: SDB -> AMB
        self.fam_sdb_to_amb = nn.ModuleList([
            FeatureAlignmentModule(
                in_channels=sdb_base_channels * (2 ** i),
                out_channels=width,
            )
            for i in range(n_msa_blocks)
        ])

        self.amb_pool = nn.AdaptiveAvgPool1d(1)

        self.stb_to_width = nn.Sequential(
            nn.Linear(emb_dim, width),
            nn.LayerNorm(width),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(width * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(classifier_dropout),
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, C, T], got {tuple(x.shape)}.")

        # STB branch
        stb_cls, stb_token_maps = self.stb(x)

        # SDB branch
        denoised, sdb_feats = self.sdb(x)

        # AMB branch with cross-cascaded FAM injections
        h = self.amb_stem(x)

        for i, block in enumerate(self.amb_blocks):
            context = 0.0

            if i < len(stb_token_maps):
                context = context + self.fam_stb_to_amb[i](stb_token_maps[i])

            if i < len(sdb_feats):
                context = context + self.fam_sdb_to_amb[i](sdb_feats[i])

            # broadcast context [B, C, 1] to [B, C, T]
            h = h + context
            h = block(h)

        amb_emb = self.amb_pool(h).squeeze(-1)       # [B, width]
        stb_emb = self.stb_to_width(stb_cls)         # [B, width]

        fused = torch.cat([amb_emb, stb_emb], dim=-1)
        emb = self.fusion_head(fused)
        logits = self.classifier(emb)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "denoised": denoised,
            "emb": emb,
            "amb_emb": amb_emb,
            "stb_emb": stb_cls,
        }

        if return_features:
            # Debug only. Do not store these in long-running training cache.
            out["amb_feature_map"] = h
            out["sdb_last_feature"] = sdb_feats[-1]
            out["stb_last_tokens"] = stb_token_maps[-1]

        return out





class FFTMSELoss(nn.Module):
    """
    FFT-MSE loss from Dconformer.

    Paper Eq. (11):
        MSE between FFT of denoised signal and FFT of clean/raw signal.

    Args:
        normalize:
            If True, remove mean and normalize amplitude before FFT.
            This is often more stable in fault diagnosis experiments.

        use_complex:
            If True, compute MSE on real and imaginary parts.
            If False, compute MSE on magnitude spectrum.
    """

    def __init__(
        self,
        normalize: bool = True,
        use_complex: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.use_complex = bool(use_complex)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x

        x = x - x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return x / std

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        if denoised.shape != clean.shape:
            raise ValueError(
                f"denoised shape {tuple(denoised.shape)} != clean shape {tuple(clean.shape)}"
            )

        denoised = self._norm(denoised)
        clean = self._norm(clean)

        Xd = torch.fft.rfft(denoised, dim=-1)
        Xc = torch.fft.rfft(clean, dim=-1)

        if self.use_complex:
            loss = F.mse_loss(Xd.real, Xc.real) + F.mse_loss(Xd.imag, Xc.imag)
        else:
            loss = F.mse_loss(Xd.abs(), Xc.abs())

        return finite(loss)


class DynamicWeightAverage:
    """
    Dynamic Weight Average, DWA.

    Paper Eq. (13)-(16):

        L = w_f * L_fft_mse + w_c * L_cross

        w_i(t) = 2 * exp(alpha_i(t-1) / psi)
                 / sum_j exp(alpha_j(t-1) / psi)

        alpha_i(t-1) = L_i(t-1) / L_i(t-2)

    This implementation updates weights at every call.
    For epoch-level DWA, call update only once per epoch.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.prev2: Optional[torch.Tensor] = None
        self.prev1: Optional[torch.Tensor] = None

    def weights(
        self,
        current_device: torch.device,
        current_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.prev1 is None or self.prev2 is None:
            return torch.ones(2, device=current_device, dtype=current_dtype)

        prev1 = self.prev1.to(device=current_device, dtype=current_dtype)
        prev2 = self.prev2.to(device=current_device, dtype=current_dtype)

        alpha = prev1 / (prev2 + self.eps)

        logits = alpha / self.temperature
        w = 2.0 * torch.softmax(logits, dim=0)

        return w

    def update(
        self,
        fft_loss: torch.Tensor,
        ce_loss: torch.Tensor,
    ) -> None:
        cur = torch.stack(
            [
                finite(fft_loss.detach()).float().cpu(),
                finite(ce_loss.detach()).float().cpu(),
            ],
            dim=0,
        )

        if self.prev1 is None:
            self.prev1 = cur
        elif self.prev2 is None:
            self.prev2 = self.prev1
            self.prev1 = cur
        else:
            self.prev2 = self.prev1
            self.prev1 = cur


class DconformerJointLoss(nn.Module):
    """
    Dconformer joint-learning loss.

    Compatible interface:

        forward(
            outputs: Dict[str, Tensor],
            y: Optional[Tensor] = None,
            **kwargs
        ) -> Dict[str, Tensor]

    Required outputs:
        outputs["logits"]
        outputs["denoised"]

    Required label:
        y / target / labels

    Required clean signal for denoising:
        kwargs["clean"]
        or kwargs["x_clean"]
        or kwargs["target_signal"]
        or kwargs["raw_signal"]

    If no clean signal is provided, only CE loss is used.

    Returns:
        {
            "total": total_loss,
            "ce": detached CE,
            "fft_mse": detached FFT-MSE,
            "w_ce": CE weight,
            "w_fft": FFT weight,
            ...
        }
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        use_dwa: bool = True,
        dwa_temperature: float = 2.0,
        fixed_ce_weight: float = 1.0,
        fixed_fft_weight: float = 1.0,
        fft_normalize: bool = True,
        fft_use_complex: bool = True,
    ) -> None:
        super().__init__()

        self.use_dwa = bool(use_dwa)

        self.ce = nn.CrossEntropyLoss(
            label_smoothing=float(label_smoothing)
        )

        self.fft_mse = FFTMSELoss(
            normalize=fft_normalize,
            use_complex=fft_use_complex,
        )

        self.dwa = DynamicWeightAverage(
            temperature=dwa_temperature
        )

        self.fixed_ce_weight = float(fixed_ce_weight)
        self.fixed_fft_weight = float(fixed_fft_weight)

    def _get_y(
        self,
        y: Optional[torch.Tensor],
        kwargs: Dict,
    ) -> torch.Tensor:
        if y is not None:
            return y

        for key in ("target", "labels", "label"):
            if key in kwargs and kwargs[key] is not None:
                return kwargs[key]

        raise ValueError(
            "DconformerJointLoss requires y, target, labels, or label."
        )

    def _get_clean_signal(
        self,
        kwargs: Dict,
    ) -> Optional[torch.Tensor]:
        for key in ("clean", "x_clean", "target_signal", "raw_signal", "x_raw"):
            value = kwargs.get(key, None)
            if value is not None:
                return value
        return None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        y = self._get_y(y, kwargs).long()

        if "logits" not in outputs:
            raise KeyError("DconformerJointLoss requires outputs['logits'].")

        logits = outputs["logits"]
        ce_loss = self.ce(logits, y)

        clean_signal = self._get_clean_signal(kwargs)

        has_denoise_target = (
            clean_signal is not None
            and "denoised" in outputs
            and outputs["denoised"] is not None
        )

        if has_denoise_target:
            fft_loss = self.fft_mse(
                outputs["denoised"],
                clean_signal,
            )
        else:
            fft_loss = torch.zeros(
                [],
                device=logits.device,
                dtype=logits.dtype,
            )

        if has_denoise_target:
            if self.use_dwa:
                w = self.dwa.weights(
                    current_device=logits.device,
                    current_dtype=logits.dtype,
                )
                w_fft = w[0]
                w_ce = w[1]
            else:
                w_fft = torch.tensor(
                    self.fixed_fft_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                w_ce = torch.tensor(
                    self.fixed_ce_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )

            total = w_fft * fft_loss + w_ce * ce_loss

            if self.training and self.use_dwa:
                self.dwa.update(fft_loss, ce_loss)

        else:
            w_fft = torch.zeros(
                [],
                device=logits.device,
                dtype=logits.dtype,
            )
            w_ce = torch.ones(
                [],
                device=logits.device,
                dtype=logits.dtype,
            )
            total = ce_loss

        logs: Dict[str, torch.Tensor] = {
            "ce": finite(ce_loss.detach()),
            "fft_mse": finite(fft_loss.detach()),
            "w_ce": finite(w_ce.detach()),
            "w_fft": finite(w_fft.detach()),
            "has_denoise_target": torch.tensor(
                float(has_denoise_target),
                device=logits.device,
            ),
            "total_raw": finite(total.detach()),
        }

        return {
            "total": finite(total),
            **logs,
        }
