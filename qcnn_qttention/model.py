# -*- coding: utf-8 -*-
"""Official QCNN reproduction adapted to the local dl_framework output format.

This file follows the network logic from:
``Paper1.1/models/QCNN_for_bearing_diagnosis-master/Model/QCNN.py`` and
``ConvQuadraticOperation.py``.

Only the final return format is adapted:
the official model returns ``softmax(out)``, while this reproduction returns a
dict with raw ``logits`` for ``torch.nn.functional.cross_entropy`` and ``prob``
for compatibility with the official behavior.

https://github.com/asdvfghg/QCNN_for_bearing_diagnosis
https://ieeexplore.ieee.org/abstract/document/10076833


"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


class ConvQuadraticOperation(nn.Module):
    """Official quadratic convolution operation.

    Formula:
        conv_r(x) * conv_g(x) + conv_b(x ** 2)

    The initialization matches the downloaded official implementation:
    ``weight_g=0``, ``weight_b=0``, ``bias_g=1``, ``bias_b=0`` and Kaiming init
    only for ``weight_r`` plus uniform init for ``bias_r``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = Parameter(torch.empty((out_channels, in_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty((out_channels, in_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty((out_channels, in_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_g", None)
            self.register_parameter("bias_b", None)

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        if self.bias_g is not None:
            nn.init.constant_(self.bias_g, 1)
        if self.bias_b is not None:
            nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self) -> None:
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_r = F.conv1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1)
        out_g = F.conv1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1)
        out_b = F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)
        return out_r * out_g + out_b


class OfficialQCNN(nn.Module):
    """QCNN builder following the official network topology.

    Official topology for input length 2048:
        ConvQuadratic(1,16,64,8,28) -> BN -> ReLU -> MaxPool
        ConvQuadratic(16,32,3,1,2) -> BN -> ReLU -> MaxPool
        ConvQuadratic(32,64,3,1,3) -> BN -> ReLU -> MaxPool
        ConvQuadratic(64,64,3,1,4) -> BN -> ReLU -> MaxPool
        ConvQuadratic(64,64,3,1,5) -> BN -> ReLU -> MaxPool
        ConvQuadratic(64,64,3,1,0) -> BN -> ReLU -> MaxPool
        Linear(192,100) -> ReLU -> Dropout(0.5) -> Linear(100,num_classes)

    ``num_classes`` defaults to 10 to match the official CWRU setting, but is
    configurable so the same architecture can train on PU/HUST subsets.
    """

    def __init__(
        self,
        win: int = 2048,
        num_classes: int = 10,
        in_channels: int = 1,
        emb_dim: int = 100,
        name: str = "official_qcnn",
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("Official QCNN expects one input channel.")
        self.name = name
        self.win = int(win)
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)

        self.cnn = nn.Sequential()
        self.cnn.add_module("Conv1D_1", ConvQuadraticOperation(1, 16, 64, 8, 28))
        self.cnn.add_module("BN_1", nn.BatchNorm1d(16))
        self.cnn.add_module("Relu_1", nn.ReLU())
        self.cnn.add_module("MAXPool_1", nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        flat_features = self._infer_flat_features(self.win)
        self.fc1 = nn.Linear(flat_features, 100)  #  self.emb_dim
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, self.num_classes)  #  self.emb_dim

    def __make_layerq(self, in_channels: int, out_channels: int, padding: int, nb_patch: int) -> None:
        self.cnn.add_module(f"Conv1D_{nb_patch}", ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module(f"BN_{nb_patch}", nn.BatchNorm1d(out_channels))
        self.cnn.add_module(f"ReLu_{nb_patch}", nn.ReLU())
        self.cnn.add_module(f"MAXPool_{nb_patch}", nn.MaxPool1d(2, 2))

    def _infer_flat_features(self, win: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, int(win))
            y = self.cnn(x)
        return int(y.numel())

    def forward(self, x: torch.Tensor, return_feature_map: bool = False, **_: object) -> dict[str, torch.Tensor]:
        feature_map = self.cnn(x)
        flat = feature_map.view(x.size(0), -1)
        features = self.relu1(self.fc1(flat))
        dropped = self.dp(features)
        logits = self.fc2(dropped)
        out = {
            "logits": logits,
            "prob": F.softmax(logits, dim=1),
            "features": features,
        }
        if return_feature_map:
            out["feature_map"] = feature_map
        return out


# Keep the previous local class name so existing training code keeps working,
# but make it a strict official-QCNN implementation now.
class QttentionQCNN(OfficialQCNN):
    pass

