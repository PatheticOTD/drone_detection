"""Model backends and dataset classes for NN-powered sensor inference."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.radar_model import RadarResNet
from models.rf_model import RFMLP
from models.audio_model import AudioResNet


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class RadarDataset:
    """Loads Range-Doppler CSV files from `data/Real Doppler RAD-DAR database/`.

    Each CSV: 11 rows × 61 cols, no header, dB values.
    Preprocessing: min-max normalise to [0, 1], add channel dim → (1, 11, 61).
    """

    def __init__(self, path: str | Path) -> None:
        self.files = sorted(Path(path).rglob("*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No CSV files found under {path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import pandas as pd
        x = pd.read_csv(self.files[idx], header=None).values.astype(np.float32)
        xmin, xmax = x.min(), x.max()
        x = (x - xmin) / (xmax - xmin + 1e-8)
        return torch.tensor(x).unsqueeze(0)   # (1, 11, 61)


class AudioDataset:
    """Loads WAV files from `data/DroneAudioDataset/Binary_Drone_Audio/`.

    Preprocessing: mel-spectrogram (128 bins) → power_to_dB → z-score → (1, 128, 94).
    Returns (tensor, label) where label 1 = drone, 0 = no_drone.
    """

    SR = 16_000
    DURATION = 3.0
    N_MELS = 128
    N_FFT = 2048
    HOP = 512

    def __init__(self, path: str | Path) -> None:
        base = Path(path)
        drone_files = [(f, 1) for f in sorted((base / "yes_drone").glob("*.wav"))]
        bg_files    = [(f, 0) for f in sorted((base / "unknown").glob("*.wav"))]
        self._items = drone_files + bg_files
        if not self._items:
            raise FileNotFoundError(f"No WAV files found under {path}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        import librosa
        filepath, label = self._items[idx]
        y, _ = librosa.load(str(filepath), sr=self.SR, mono=True, duration=self.DURATION)
        # Pad to fixed length
        target_len = int(self.SR * self.DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        S = librosa.feature.melspectrogram(y=y, sr=self.SR, n_mels=self.N_MELS,
                                            n_fft=self.N_FFT, hop_length=self.HOP)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
        return torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0), label  # (1, 128, T)


class RFDataset:
    """Wraps the trimmed `dataset.pt` file.

    x_iq shape: (N, 2, 16384). Preprocessing: z-score per sample.
    noise_class_idx = 4 (confirmed: class_stats.csv row 4 = Noise).
    Returns (tensor, label_int).
    """

    NOISE_CLASS_IDX = 4

    def __init__(self, pt_path: str | Path) -> None:
        data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        self._x = data["x_iq"].float()                      # (N, 2, 16384)
        self._y = data["y"]

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int):
        x = self._x[idx].clone()                            # (2, 16384)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x, int(self._y[idx])


class VideoDataset:
    """Returns image file paths for YOLO inference."""

    def __init__(self, images_dir: str | Path) -> None:
        self.files = sorted(Path(images_dir).glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"No JPG files found under {images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        return str(self.files[idx])


# ---------------------------------------------------------------------------
# ModelBackend — wraps a model + cycling dataset iterator
# ---------------------------------------------------------------------------

class ModelBackend:
    def __init__(self, model, dataset, device: str = "cpu") -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self._iter = itertools.cycle(range(len(dataset)))

    def next_sample(self):
        return self.dataset[next(self._iter)]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def load_radar_backend(weights: str, data_path: str, device: str = "cpu") -> ModelBackend:
    dataset = RadarDataset(data_path)
    model = RadarResNet(nc=3).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return ModelBackend(model, dataset, device)


def load_audio_backend(weights: str, data_path: str, device: str = "cpu") -> ModelBackend:
    dataset = AudioDataset(data_path)
    model = AudioResNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return ModelBackend(model, dataset, device)


def load_rf_backend(weights: str, data_path: str, device: str = "cpu") -> ModelBackend:
    dataset = RFDataset(data_path)
    model = RFMLP(input_size=32768, nc=7).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return ModelBackend(model, dataset, device)


def load_video_backend(weights: str, data_path: str, device: str = "cpu") -> ModelBackend:
    from ultralytics import YOLO
    dataset = VideoDataset(data_path)
    model = YOLO(weights)
    return ModelBackend(model, dataset, device)
