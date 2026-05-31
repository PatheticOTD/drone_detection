"""RF-ResNet-1D — 1-D resnet classifier for IQ RF signals.

Architecture copied verbatim from ipynb notebooks/drone-rf-noisy-final.ipynb.
Input shape: (batch, 2, 16384) — I/Q signal.
Output: logits over nc classes (0-6: DJI, FutabaT14, FutabaT7, Graupner, Noise, Taranis, Turnigy).
"""

import torch.nn as nn
import torch.nn.functional as F


class _ResBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, downsample=None) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm1d(out_c), nn.ReLU(True),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
        )
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv(x) + identity)


class RFResNet1D(nn.Module):
    def __init__(self, in_ch: int = 2, nc: int = 7) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_block(64, 64)
        self.layer2 = self._make_block(64, 128, stride=2)
        self.layer3 = self._make_block(128, 256, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256, nc),
        )

    def _make_block(self, in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm1d(out_c),
            )
        return nn.Sequential(
            _ResBlock1D(in_c, out_c, stride, downsample),
            _ResBlock1D(out_c, out_c),
        )

    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))
