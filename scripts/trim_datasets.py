"""One-time script to trim each dataset to 100 samples per class.

Run from the project root:
    python scripts/trim_datasets.py

WARNING: destructive — deletes files and overwrites dataset.pt in-place.
"""

import os
import random
import numpy as np
from pathlib import Path

SEED = 42
N = 100

BASE = Path(__file__).resolve().parent.parent


def trim_radar(base: Path = BASE / "data" / "Real Doppler RAD-DAR database", n: int = N, seed: int = SEED) -> None:
    random.seed(seed)
    for cls in ("Cars", "Drones", "People"):
        cls_dir = base / cls
        files = sorted(cls_dir.rglob("*.csv"))
        if len(files) <= n:
            print(f"  radar/{cls}: {len(files)} files — nothing to trim")
            continue
        keep = set(random.sample(files, n))
        removed = 0
        for f in files:
            if f not in keep:
                f.unlink()
                removed += 1
        print(f"  radar/{cls}: kept {n}, removed {removed}")


def trim_audio(base: Path = BASE / "data" / "DroneAudioDataset" / "Binary_Drone_Audio", n: int = N, seed: int = SEED) -> None:
    random.seed(seed)
    for folder, label in (("yes_drone", "drone"), ("unknown", "no_drone")):
        folder_path = base / folder
        files = sorted(folder_path.glob("*.wav"))
        if len(files) <= n:
            print(f"  audio/{folder}: {len(files)} files — nothing to trim")
            continue
        keep = set(random.sample(files, n))
        removed = 0
        for f in files:
            if f not in keep:
                f.unlink()
                removed += 1
        print(f"  audio/{folder}: kept {n}, removed {removed}")


def trim_rf(
    pt_path: Path = BASE / "data" / "Noisy Drone RF Signal Classification" / "dataset.pt",
    n: int = N,
    seed: int = SEED,
) -> None:
    import torch

    rng = np.random.default_rng(seed)
    print(f"  rf: loading {pt_path} (this may take a while for large files)…")
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    x = data["x_iq"]
    y = data["y"].numpy() if hasattr(data["y"], "numpy") else np.array(data["y"])
    snr = data.get("snr")

    keep_indices = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
        keep_indices.append(chosen)
        print(f"    class {int(cls)}: {len(idx)} → keeping {len(chosen)}")
    keep_indices = np.sort(np.concatenate(keep_indices))

    new_data = {
        "x_iq": x[keep_indices],
        "y": torch.tensor(y[keep_indices]),
    }
    if snr is not None:
        snr_np = snr.numpy() if hasattr(snr, "numpy") else np.array(snr)
        new_data["snr"] = torch.tensor(snr_np[keep_indices])

    torch.save(new_data, str(pt_path))
    print(f"  rf: saved trimmed dataset → shape {new_data['x_iq'].shape}, {pt_path.stat().st_size / 1e6:.1f} MB")


def trim_video(
    base: Path = BASE / "data" / "Airborne-Object-Detection-4-AOD4.yolov8" / "train",
    n: int = N,
    seed: int = SEED,
) -> None:
    random.seed(seed)
    images_dir = base / "images"
    labels_dir = base / "labels"

    images = sorted(images_dir.glob("*.jpg"))

    drone_imgs, bg_imgs = [], []
    for img in images:
        label_file = labels_dir / (img.stem + ".txt")
        if label_file.exists() and label_file.stat().st_size > 0:
            drone_imgs.append(img)
        else:
            bg_imgs.append(img)

    print(f"  video: drone={len(drone_imgs)}, background={len(bg_imgs)}")

    keep_drone = set(random.sample(drone_imgs, min(n, len(drone_imgs))))
    keep_bg = set(random.sample(bg_imgs, min(n, len(bg_imgs))))
    keep = keep_drone | keep_bg

    removed = 0
    for img in images:
        if img not in keep:
            img.unlink()
            label_file = labels_dir / (img.stem + ".txt")
            if label_file.exists():
                label_file.unlink()
            removed += 1

    print(f"  video: kept {len(keep)} images, removed {removed} pairs")


if __name__ == "__main__":
    print("=== Trimming radar dataset ===")
    trim_radar()

    print("\n=== Trimming audio dataset ===")
    trim_audio()

    print("\n=== Trimming RF dataset ===")
    trim_rf()

    print("\n=== Trimming video dataset ===")
    trim_video()

    print("\nDone.")
