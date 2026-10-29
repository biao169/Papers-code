# -*- coding: utf-8 -*-
"""IMAG-Net reconstructed from Cao et al., MSSP 257 (2026), 114586.

Paper: Interpretable multiscale algorithm unrolling network using learnable
Gabor filters for fault diagnosis.  The implementation follows Eqs. (1)-(32),
Figs. 6-7, and Table 1.

The paper does not report the Gabor/CSD/CNN kernel lengths or numerical ADMM
initializers.  They are exposed below instead of being presented as published
hyperparameters.  The default classifier kernel length (11) is the smallest
odd value that reproduces the paper's reported approximately 1.50 M parameters.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def soft_threshold(x: Tensor, threshold: Tensor | float) -> Tensor:
    """Element-wise soft thresholding used by Eqs. (13) and (19)."""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


class LearnableGaborFilterBank1D(nn.Module):
    """Paper Eqs. (1)-(4): ordered, learnable 1-D Gabor filter bank."""

    def __init__(
        self,
        num_filters: int = 8,
        kernel_size: int = 129,
        min_frequency: float = 0.01,
        max_frequency: float = 0.49,
        initial_sigma: float | None = None,
    ) -> None:
        super().__init__()
        if num_filters < 1:
            raise ValueError("num_filters must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if not 0.0 <= min_frequency < max_frequency <= 0.5:
            raise ValueError("frequencies must satisfy 0 <= min < max <= 0.5")
        self.num_filters = int(num_filters)
        self.kernel_size = int(kernel_size)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)

        frequencies = torch.linspace(min_frequency, max_frequency, num_filters)
        relative = (frequencies - min_frequency) / (max_frequency - min_frequency)
        relative = relative.clamp(1e-4, 1.0 - 1e-4)
        self.frequency_logits = nn.Parameter(torch.logit(relative))

        sigma = float(initial_sigma or kernel_size / 6.0)
        inverse_softplus = math.log(math.expm1(sigma))
        self.raw_sigma = nn.Parameter(torch.full((num_filters,), inverse_softplus))
        time = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        self.register_buffer("time", time)

    @property
    def center_frequencies(self) -> Tensor:
        span = self.max_frequency - self.min_frequency
        return self.min_frequency + span * torch.sigmoid(self.frequency_logits)

    @property
    def bandwidth_sigmas(self) -> Tensor:
        return F.softplus(self.raw_sigma).clamp_min(1e-3)

    def kernels(self) -> Tensor:
        time = self.time.unsqueeze(0)
        sigma = self.bandwidth_sigmas.unsqueeze(1)
        frequency = self.center_frequencies.unsqueeze(1)
        gaussian = torch.exp(-0.5 * (time / sigma).square())
        carrier = torch.cos(2.0 * math.pi * frequency * time)
        kernels = gaussian * carrier
        kernels = kernels - kernels.mean(dim=1, keepdim=True)
        kernels = kernels / kernels.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return kernels.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"Expected [batch,1,length], got {tuple(x.shape)}")
        # Paper p=floor(L/2): equal zero padding preserves T for odd L.
        return F.conv1d(x, self.kernels(), padding=self.kernel_size // 2)


class ConvolutionalSparseDictionary(nn.Module):
    """CSD W and W^T in Eqs. (8)-(20), with Table-1 T -> T/4 sizing."""

    def __init__(
        self,
        bands: int = 8,
        atoms: int = 64,
        kernel_size: int = 9,
        stride: int = 4,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.bands = int(bands)
        self.atoms = int(atoms)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(atoms, bands, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def analysis(self, z: Tensor) -> Tensor:
        """Apply W^T: [B,N,T] -> [B,M,T/4]."""
        return F.conv1d(z, self.weight, stride=self.stride, padding=self.padding)

    def synthesis(self, gamma: Tensor, output_length: int) -> Tensor:
        """Apply W: [B,M,T/4] -> [B,N,T]."""
        base_length = (
            (gamma.size(-1) - 1) * self.stride
            - 2 * self.padding
            + self.kernel_size
        )
        output_padding = int(output_length - base_length)
        if not 0 <= output_padding < self.stride:
            raise ValueError(
                f"Cannot synthesize length {output_length}; output_padding={output_padding}"
            )
        return F.conv_transpose1d(
            gamma,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
        )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MSCBlock(nn.Module):
    """One paper Fig. 7 ADMM iteration: MSSE -> ASO -> DRR -> CE."""

    def __init__(
        self,
        dictionary: ConvolutionalSparseDictionary,
        *,
        eta: float = 0.5,
        eta2: float = 0.05,
        lambda1: float = 1.0,
        lambda2: float = 0.01,
        lambda3: float = 0.01,
        rho: float = 1.0,
        learnable_parameters: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0,1]")
        if min(eta2, lambda1, lambda2, lambda3, rho) <= 0:
            raise ValueError("ADMM parameters must be positive")
        self.dictionary = dictionary
        eta_logit = math.log(eta / (1.0 - eta)) if eta < 1.0 else 12.0
        values = {
            "raw_eta": eta_logit,
            "raw_eta2": _inverse_softplus(eta2),
            "raw_lambda1": _inverse_softplus(lambda1),
            "raw_lambda2": _inverse_softplus(lambda2),
            "raw_lambda3": _inverse_softplus(lambda3),
            "raw_rho": _inverse_softplus(rho),
        }
        for name, value in values.items():
            tensor = torch.tensor(float(value))
            if learnable_parameters:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def positive_parameters(self) -> dict[str, Tensor]:
        return {
            "eta": torch.sigmoid(self.raw_eta),
            "eta2": F.softplus(self.raw_eta2),
            "lambda1": F.softplus(self.raw_lambda1),
            "lambda2": F.softplus(self.raw_lambda2),
            "lambda3": F.softplus(self.raw_lambda3),
            "rho": F.softplus(self.raw_rho),
        }

    def forward(
        self,
        x0: Tensor,
        z: Tensor,
        gamma: Tensor,
        u: Tensor,
        mu: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p = self.positive_parameters()
        eta, eta2 = p["eta"], p["eta2"]
        lambda1, lambda2, lambda3, rho = (
            p["lambda1"], p["lambda2"], p["lambda3"], p["rho"]
        )

        # MSSE, paper Eq. (13).
        w_gamma = self.dictionary.synthesis(gamma, x0.size(-1))
        h = eta / (1.0 + lambda1) * (x0 + lambda1 * w_gamma)
        z = soft_threshold((1.0 - eta) * z + h, eta * lambda2 / (1.0 + lambda1))

        # ASO, paper Eqs. (14), (16), (17).
        v = u - mu / rho
        residual = self.dictionary.synthesis(gamma, z.size(-1)) - z
        grad = lambda1 * self.dictionary.analysis(residual) + rho * (gamma - v)
        gamma = gamma - eta2 * grad

        # DRR and CE, paper Eqs. (19), (20).
        u = soft_threshold(gamma + mu / rho, lambda3 / rho)
        mu = mu + rho * (gamma - u)
        return z, gamma, u, mu


class ConvBlock(nn.Module):
    """Paper Sec. 3.4.3: two Conv-BN-ReLU layers followed by max pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.features(x))


class IMAGNet(nn.Module):
    """Paper IMAG-Net with Table-1 default tensor shapes."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        num_gabor_filters: int = 8,
        gabor_kernel_size: int = 129,
        num_atoms: int = 64,
        dictionary_kernel_size: int = 49,
        dictionary_stride: int = 4,
        unrolling_steps: int = 4,
        classifier_kernel_size: int = 11,
        adaptive_pool_size: int = 7,
        learnable_admm: bool = True,
        eta: float = 0.5,
        eta2: float = 0.05,
        lambda1: float = 1.0,
        lambda2: float = 0.01,
        lambda3: float = 0.01,
        rho: float = 1.0,
        name: str = "imag_net",
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("The paper architecture accepts one vibration channel")
        if num_classes < 2 or unrolling_steps < 1:
            raise ValueError("num_classes >= 2 and unrolling_steps >= 1 are required")
        self.name = name
        self.num_atoms = int(num_atoms)
        self.dictionary_stride = int(dictionary_stride)
        self.input_normalization_eps = 1e-6
        self.gabor = LearnableGaborFilterBank1D(
            num_filters=num_gabor_filters,
            kernel_size=gabor_kernel_size,
        )
        self.dictionary = ConvolutionalSparseDictionary(
            bands=num_gabor_filters,
            atoms=num_atoms,
            kernel_size=dictionary_kernel_size,
            stride=dictionary_stride,
        )
        block_kwargs = dict(
            eta=eta,
            eta2=eta2,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            rho=rho,
            learnable_parameters=learnable_admm,
        )
        # W is the common CSD in the paper; scalar optimizer parameters are stage-wise.
        self.msc_blocks = nn.ModuleList(
            [MSCBlock(self.dictionary, **block_kwargs) for _ in range(unrolling_steps)]
        )
        self.classifier_features = nn.Sequential(
            ConvBlock(num_atoms, 64, classifier_kernel_size),
            ConvBlock(64, 128, classifier_kernel_size),
            ConvBlock(128, 256, classifier_kernel_size),
        )
        self.pool = nn.AdaptiveAvgPool1d(adaptive_pool_size)
        self.classifier = nn.Linear(256 * adaptive_pool_size, num_classes)
        self._reset_classifier()

    def _reset_classifier(self) -> None:
        for module in self.classifier_features.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier.weight, 0.0, 0.01)
        nn.init.zeros_(self.classifier.bias)

    def normalize_input(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
            self.input_normalization_eps
        )
        return (x - mean) / std

    def forward(self, x: Tensor, return_intermediates: bool = False) -> dict[str, Any]:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"Expected [batch,1,length], got {tuple(x.shape)}")
        if x.size(-1) % self.dictionary_stride != 0:
            raise ValueError("Input length must be divisible by dictionary_stride")
        normalized = self.normalize_input(x)
        x0 = self.gabor(normalized)
        z = x0
        gamma = self.dictionary.analysis(torch.zeros_like(x0))
        u = torch.zeros_like(gamma)
        mu = torch.zeros_like(gamma)
        states: list[dict[str, Tensor]] = []
        for block in self.msc_blocks:
            z, gamma, u, mu = block(x0, z, gamma, u, mu)
            if return_intermediates:
                states.append({"z": z, "gamma": gamma, "u": u, "mu": mu})

        feature_map = self.classifier_features(gamma)
        features = self.pool(feature_map).flatten(1)
        output: dict[str, Any] = {
            "logits": self.classifier(features),
            "features": features,
            "gabor_responses": x0,
            "sparse_codes": gamma,
            "center_frequencies": self.gabor.center_frequencies,
        }
        if return_intermediates:
            output["states"] = states
            output["classifier_feature_map"] = feature_map
        return output

    def reconstruct_top_atoms(self, gamma: Tensor, top_n: int = 5) -> Tensor:
        """Paper Eq. (32): reconstruct using the globally strongest atoms."""
        if not 1 <= top_n <= gamma.size(1):
            raise ValueError("top_n must be between 1 and the number of atoms")
        strength = gamma.square().sum(dim=-1)
        indices = strength.topk(top_n, dim=1).indices
        mask = torch.zeros_like(strength).scatter_(1, indices, 1.0).unsqueeze(-1)
        return self.dictionary.synthesis(gamma * mask, gamma.size(-1) * self.dictionary_stride)

    @staticmethod
    def positive_sensitivity_importance(activation: Tensor, gradient: Tensor) -> Tensor:
        """Paper Eqs. (29)-(31), normalized over components."""
        if activation.shape != gradient.shape:
            raise ValueError("activation and gradient must have identical shapes")
        effect = (activation * gradient.clamp_min(0.0)).square().sum(dim=-1)
        return effect / effect.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "soft_threshold",
    "LearnableGaborFilterBank1D",
    "ConvolutionalSparseDictionary",
    "MSCBlock",
    "ConvBlock",
    "IMAGNet",
]
