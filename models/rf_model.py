"""RFMLP — MLP classifier for IQ RF signals.

Architecture copied verbatim from ipynb notebooks/drone-rf-noisy-final.ipynb.
Input shape: (batch, 2, 16384) — flattened internally to (batch, 32768).
Output: logits over nc classes (0-6: DJI, FutabaT14, FutabaT7, Graupner, Noise, Taranis, Turnigy).
"""

import torch.nn as nn


class RFMLP(nn.Module):
    def __init__(self, input_size: int = 32768, nc: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(256, nc),
        )

    def forward(self, x):
        return self.net(x)
