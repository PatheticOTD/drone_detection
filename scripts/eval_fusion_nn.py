"""System-level evaluation for section 5.6 — REAL neural-network inference.

Unlike scripts/eval_fusion.py (pure simulation), this script runs the four
trained networks on their real labelled datasets and reports:

  * per-channel probability of correct detection (P_d) and false alarm (P_fa),
    measured against the true sample labels;
  * fused P_d / P_fa, estimated by Monte-Carlo over synchronised scenes that
    draw one real NN output per channel conditioned on the true scene state
    (the four datasets are independent, so a "scene" samples each channel from
    the pool matching the scene's ground-truth presence);
  * graceful degradation when channels are disabled;
  * per-cycle latency, measured as one real forward pass through all four
    networks.

Confidence per channel matches the production sensors:
  radar  -> softmax(logits)[1]        (class 1 = Drones)
  audio  -> softmax(logits)[1]        (class 1 = drone)
  rf     -> 1 - softmax(logits)[4]    (class 4 = Noise)
  video  -> max box confidence among drone-class (class 2) detections

Usage:
    python scripts/eval_fusion_nn.py [--cycles 500] [--cap 300] [--device cpu]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fusion.engine import FusionEngine
from sensors.base import SensorReading
from config import DETECTION_THRESHOLD, SENSOR_WEIGHTS
from models.loader import (
    RadarDataset,
    AudioDataset,
    RFDataset,
    VideoDataset,
)
from models.radar_model import RadarResNet
from models.audio_model import AudioResNet
from models.rf_model import RFResNet1D

CHANNELS = ("audio", "video", "radar", "rf")

# Weight / data paths (mirror main.py)
RADAR_W = "weights/radar_resnet_best.pt"
AUDIO_W = "weights/audio_resnet_best.pt"
RF_W = "weights/rf-resnet-1d_best.pt"
VIDEO_W = "weights/yolo8n-best.pt"

RADAR_D = "data/Real Doppler RAD-DAR database"
AUDIO_D = "data/DroneAudioDataset/Binary_Drone_Audio"
RF_D = "data/Noisy Drone RF Signal Classification/dataset.pt"
VIDEO_D = "data/Airborne-Object-Detection-4-AOD4.yolov8/train/images"

RF_NOISE_IDX = 4
VIDEO_DRONE_CLS = 2


# ---------------------------------------------------------------------------
# Per-channel confidence pools (real NN inference on labelled data)
# ---------------------------------------------------------------------------

def _subsample(idxs: List[int], cap: int, rng: random.Random) -> List[int]:
    if cap and len(idxs) > cap:
        return rng.sample(idxs, cap)
    return idxs


def collect_radar(device: str, cap: int, rng: random.Random) -> Tuple[List[float], List[float]]:
    ds = RadarDataset(RADAR_D)
    labels = [1 if any(p.lower() == "drones" for p in f.parts) else 0 for f in ds.files]
    model = RadarResNet(nc=3).to(device)
    model.load_state_dict(torch.load(RADAR_W, map_location=device, weights_only=True))
    model.eval()

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    pos = _subsample(pos, cap, rng)
    neg = _subsample(neg, cap, rng)

    present, absent = [], []
    with torch.no_grad():
        for bucket, target in ((pos, present), (neg, absent)):
            for i in bucket:
                x = ds[i].unsqueeze(0).to(device)
                p = torch.softmax(model(x), dim=1)[0]
                target.append(float(p[1]))
    return present, absent


def collect_audio(device: str, cap: int, rng: random.Random) -> Tuple[List[float], List[float]]:
    ds = AudioDataset(AUDIO_D)
    labels = [lbl for _, lbl in ds._items]
    model = AudioResNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(AUDIO_W, map_location=device, weights_only=True))
    model.eval()

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    pos = _subsample(pos, cap, rng)
    neg = _subsample(neg, cap, rng)

    present, absent = [], []
    with torch.no_grad():
        for bucket, target in ((pos, present), (neg, absent)):
            for i in bucket:
                x, _ = ds[i]
                p = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0]
                target.append(float(p[1]))
    return present, absent


def collect_rf(device: str, cap: int, rng: random.Random) -> Tuple[List[float], List[float]]:
    ds = RFDataset(RF_D)
    labels = [int(y) for y in ds._y]
    model = RFResNet1D(in_ch=2, nc=7).to(device)
    model.load_state_dict(torch.load(RF_W, map_location=device, weights_only=True))
    model.eval()

    pos = [i for i, l in enumerate(labels) if l != RF_NOISE_IDX]
    neg = [i for i, l in enumerate(labels) if l == RF_NOISE_IDX]
    pos = _subsample(pos, cap, rng)
    neg = _subsample(neg, cap, rng)

    present, absent = [], []
    with torch.no_grad():
        for bucket, target in ((pos, present), (neg, absent)):
            for i in bucket:
                x, _ = ds[i]
                p = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0]
                target.append(1.0 - float(p[RF_NOISE_IDX]))
    return present, absent


def _label_path(img_path: str) -> Path:
    p = Path(img_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def _img_has_drone(img_path: str) -> bool:
    lp = _label_path(img_path)
    if not lp.exists():
        return False
    for line in lp.read_text().splitlines():
        if line.strip() and int(line.split()[0]) == VIDEO_DRONE_CLS:
            return True
    return False


def _video_conf(model, img_path: str) -> float:
    """Max confidence among drone-class boxes (0 if none)."""
    r = model(img_path, verbose=False)[0]
    boxes = r.boxes
    if len(boxes) == 0:
        return 0.0
    confs = [float(c) for c, k in zip(boxes.conf.tolist(), boxes.cls.tolist())
             if int(k) == VIDEO_DRONE_CLS]
    return max(confs) if confs else 0.0


def collect_video(device: str, cap: int, rng: random.Random) -> Tuple[List[float], List[float]]:
    from ultralytics import YOLO
    ds = VideoDataset(VIDEO_D)
    model = YOLO(VIDEO_W)

    pos = [i for i, f in enumerate(ds.files) if _img_has_drone(f)]
    neg = [i for i, f in enumerate(ds.files) if not _img_has_drone(f)]
    pos = _subsample(pos, cap, rng)
    neg = _subsample(neg, cap, rng)

    present = [_video_conf(model, ds[i]) for i in pos]
    absent = [_video_conf(model, ds[i]) for i in neg]
    return present, absent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rate_above(values: List[float], thr: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v >= thr) / len(values)


def fused_scene_rates(pools: Dict[str, Tuple[List[float], List[float]]],
                      engine: FusionEngine, n_cycles: int,
                      disabled: tuple, rng: random.Random) -> Tuple[float, float]:
    """Monte-Carlo P_d / P_fa for the fused decision.

    Each cycle builds a synchronised scene: every active channel draws one
    real NN confidence from the pool matching the scene's true state.
    Disabled channels contribute a zero-confidence reading.
    """
    def run(present: bool) -> float:
        hits = 0
        for _ in range(n_cycles):
            readings = {}
            for ch in CHANNELS:
                if ch in disabled:
                    # Offline channel: excluded from fusion (weights renormalise
                    # over the active channels) — true graceful degradation.
                    readings[ch] = SensorReading(
                        sensor_type=ch, timestamp=time.time(),
                        confidence=0.0, detected=False,
                        metadata={"mode": "off"}, available=False,
                    )
                else:
                    pool = pools[ch][0] if present else pools[ch][1]
                    conf = rng.choice(pool) if pool else 0.0
                    readings[ch] = SensorReading(
                        sensor_type=ch, timestamp=time.time(),
                        confidence=conf, detected=conf > 0.5,
                        metadata={"mode": "nn"},
                    )
            if engine.fuse(readings).fused_confidence >= engine.threshold:
                hits += 1
        return hits / n_cycles

    return run(True), run(False)


def measure_latency(device: str, n: int = 100) -> dict:
    """Real per-cycle latency: one forward pass through all four networks."""
    from ultralytics import YOLO

    radar = RadarResNet(nc=3).to(device); radar.load_state_dict(
        torch.load(RADAR_W, map_location=device, weights_only=True)); radar.eval()
    audio = AudioResNet(in_channels=1, num_classes=2).to(device); audio.load_state_dict(
        torch.load(AUDIO_W, map_location=device, weights_only=True)); audio.eval()
    rf = RFResNet1D(in_ch=2, nc=7).to(device); rf.load_state_dict(
        torch.load(RF_W, map_location=device, weights_only=True)); rf.eval()
    yolo = YOLO(VIDEO_W)

    rds, ads, fds = RadarDataset(RADAR_D), AudioDataset(AUDIO_D), RFDataset(RF_D)
    vds = VideoDataset(VIDEO_D)
    img0 = vds[0]
    rx = rds[0].unsqueeze(0).to(device)
    ax = ads[0][0].unsqueeze(0).to(device)
    fx = fds[0][0].unsqueeze(0).to(device)

    # Warm-up (CUDA kernels / lazy init) so the first cold calls don't skew the mean
    with torch.no_grad():
        for _ in range(5):
            radar(rx); audio(ax); rf(fx); yolo(img0, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()

    lat = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            radar(rx); audio(ax); rf(fx)
            yolo(img0, verbose=False)
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(lat)
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()), "n": n}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    return f"{x*100:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=500)
    ap.add_argument("--cap", type=int, default=300,
                    help="Max samples per class per channel (bounds runtime)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    thr = DETECTION_THRESHOLD
    dev = args.device

    print(f"Устройство: {dev} | порог: {thr} | cap/класс: {args.cap}")
    print("Сбор выходов нейросетей по реальным размеченным данным…")

    pools: Dict[str, Tuple[List[float], List[float]]] = {}
    print("  radar…"); pools["radar"] = collect_radar(dev, args.cap, rng)
    print("  rf…");    pools["rf"] = collect_rf(dev, args.cap, rng)
    print("  video…"); pools["video"] = collect_video(dev, args.cap, rng)
    print("  audio…"); pools["audio"] = collect_audio(dev, args.cap, rng)

    # --- per-channel real P_d / P_fa ---
    print("\n=== 5.6.2 — Каналы (реальный NN-инференс) и объединённое решение ===")
    print(f"{'Канал':<14} {'N+':>5} {'N-':>5} {'P_d':>8} {'P_fa':>8}")
    print("-" * 44)
    for ch in CHANNELS:
        present, absent = pools[ch]
        print(f"{ch:<14} {len(present):>5} {len(absent):>5} "
              f"{fmt(rate_above(present, thr)):>8} {fmt(rate_above(absent, thr)):>8}")

    engine = FusionEngine()
    pd_f, pfa_f = fused_scene_rates(pools, engine, args.cycles, (), rng)
    print(f"{'объединённое':<14} {'—':>5} {'—':>5} {fmt(pd_f):>8} {fmt(pfa_f):>8}")
    print(f"(веса слияния: {SENSOR_WEIGHTS})")

    # --- degradation ---
    print("\n=== 5.6.3 — Деградация при отключении каналов (fused) ===")
    print(f"{'Активных':<9} {'Отключённые':<26} {'P_d':>8} {'P_fa':>8}")
    print("-" * 54)
    for disabled in [(), ("rf",), ("rf", "audio"), ("rf", "audio", "video")]:
        pd_d, pfa_d = fused_scene_rates(pools, FusionEngine(), args.cycles, disabled, rng)
        dis = ", ".join(disabled) if disabled else "—"
        print(f"{4 - len(disabled):<9} {dis:<26} {fmt(pd_d):>8} {fmt(pfa_d):>8}")

    # --- latency ---
    print("\n=== 5.6.4 — Задержка цикла (реальный инференс 4 сетей) ===")
    lat = measure_latency(dev)
    print(f"  Циклов:         {lat['n']}")
    print(f"  Среднее:        {lat['mean']:.1f} мс")
    print(f"  95-й проц.:     {lat['p95']:.1f} мс")
    print(f"  Максимум:       {lat['max']:.1f} мс")
    print(f"  Требование ≤2000 мс — {'выполнено' if lat['p95'] <= 2000 else 'НЕ ВЫПОЛНЕНО'}")


if __name__ == "__main__":
    main()
