"""RadarResNet — Range-Doppler map classifier (Cars / Drones / People).

Architecture copied verbatim from ipynb notebooks/drone-radar-final.ipynb.
Input shape: (batch, 1, 11, 61)  — 11 range bins × 61 Doppler bins.
Output: logits over nc classes.
"""

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class RadarResNet(nn.Module):
    def __init__(self, nc: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
        )
        self.head = nn.Linear(256, nc)

    def forward(self, x):
        return self.head(self.net(x))
