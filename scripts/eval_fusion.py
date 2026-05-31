"""Measure system-level metrics for section 5.5.

Runs simulation cycles to estimate:
  - per-channel and fused probability of correct detection (P_d)
  - per-channel and fused probability of false alarm (P_fa)
  - graceful degradation when channels are disabled
  - per-cycle latency (mean and 95th percentile)

Usage:
    python scripts/eval_fusion.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fusion.engine import FusionEngine
from sensors.audio_sensor import AudioSensor
from sensors.ground_truth import SimulationGroundTruth
from sensors.radar_sensor import RadarSensor
from sensors.rf_sensor import RFSensor
from sensors.video_sensor import VideoSensor
from config import DETECTION_THRESHOLD


CHANNELS = ("audio", "video", "radar", "rf")


def build_sensors(present_probability: float = 1.0) -> Dict[str, object]:
    """Build all 4 sensors in simulation mode with a controllable ground truth."""
    gt = SimulationGroundTruth(
        present_probability=present_probability,
        spawn_chance=1.0 if present_probability > 0.5 else 0.0,
        despawn_chance=0.0 if present_probability > 0.5 else 1.0,
    )
    sensors = {
        "audio": AudioSensor(backend=None, ground_truth=gt),
        "video": VideoSensor(backend=None),
        "radar": RadarSensor(backend=None),
        "rf": RFSensor(backend=None, ground_truth=gt),
    }
    return sensors, gt


def run_cycles(n_cycles: int, drone_present: bool, disabled: tuple = ()) -> dict:
    """Run n_cycles simulation steps; return metrics."""
    sensors, gt = build_sensors(present_probability=1.0 if drone_present else 0.0)
    # Force initial state
    gt._state.drone_present = drone_present
    if drone_present:
        gt._state.x, gt._state.y, gt._state.z = 100.0, 100.0, 80.0
    engine = FusionEngine()

    per_channel_hits = {c: 0 for c in CHANNELS}
    fused_hits = 0
    latencies_ms: List[float] = []

    for _ in range(n_cycles):
        t0 = time.perf_counter()
        readings = {}
        for name, s in sensors.items():
            if name in disabled:
                # Disabled channel: zero confidence, marked not detected
                from sensors.base import SensorReading
                readings[name] = SensorReading(
                    sensor_type=name, timestamp=time.time(),
                    confidence=0.0, detected=False, metadata={"mode": "off"},
                )
            else:
                readings[name] = s.poll()
        result = engine.fuse(readings)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        for name, r in readings.items():
            if r.confidence >= DETECTION_THRESHOLD:
                per_channel_hits[name] += 1
        if result.fused_confidence >= DETECTION_THRESHOLD:
            fused_hits += 1

    arr = np.asarray(latencies_ms)
    return {
        "n": n_cycles,
        "drone_present": drone_present,
        "disabled": disabled,
        "per_channel_rate": {c: per_channel_hits[c] / n_cycles for c in CHANNELS},
        "fused_rate": fused_hits / n_cycles,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_max": float(arr.max()),
    }


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def print_table_2(pd_metrics: dict, pfa_metrics: dict) -> None:
    """5.5.2 — per-channel and fused P_d / P_fa."""
    print("\n=== 5.5.2 — Сравнение одиночных каналов и объединённого решения ===")
    print(f"{'Канал':<12} {'P_d':>8} {'P_fa':>8}")
    print("-" * 32)
    for c in CHANNELS:
        pd = pd_metrics["per_channel_rate"][c]
        pfa = pfa_metrics["per_channel_rate"][c]
        print(f"{c:<12} {fmt_pct(pd):>8} {fmt_pct(pfa):>8}")
    print(f"{'fused':<12} {fmt_pct(pd_metrics['fused_rate']):>8} "
          f"{fmt_pct(pfa_metrics['fused_rate']):>8}")


def print_table_3(rows: List[dict]) -> None:
    """5.5.3 — degradation as channels are removed."""
    print("\n=== 5.5.3 — Постепенная деградация при отключении каналов ===")
    print(f"{'Активных каналов':<18} {'Отключённые':<24} {'P_d':>8} {'P_fa':>8}")
    print("-" * 64)
    for row in rows:
        n_active = 4 - len(row["disabled"])
        dis = ", ".join(row["disabled"]) if row["disabled"] else "—"
        pd = row["pd"]["fused_rate"]
        pfa = row["pfa"]["fused_rate"]
        print(f"{n_active:<18} {dis:<24} {fmt_pct(pd):>8} {fmt_pct(pfa):>8}")


def print_table_4(metrics: dict) -> None:
    """5.5.4 — latency."""
    print("\n=== 5.5.4 — Задержка одного цикла обработки ===")
    print(f"  Циклов измерено:           {metrics['n']}")
    print(f"  Средняя задержка:          {metrics['latency_ms_mean']:.2f} мс")
    print(f"  95-й процентиль:           {metrics['latency_ms_p95']:.2f} мс")
    print(f"  Максимум:                  {metrics['latency_ms_max']:.2f} мс")
    print(f"  Требование (раздел 3.2.3): ≤ 2000 мс — {'выполнено' if metrics['latency_ms_p95'] <= 2000 else 'НЕ ВЫПОЛНЕНО'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=500,
                        help="Number of simulation cycles per scenario (default: 500)")
    args = parser.parse_args()

    print(f"Запуск {args.cycles} циклов на сценарий…")

    # 5.5.2 — Drone present (P_d) vs absent (P_fa), all channels on
    pd_full = run_cycles(args.cycles, drone_present=True, disabled=())
    pfa_full = run_cycles(args.cycles, drone_present=False, disabled=())
    print_table_2(pd_full, pfa_full)

    # 5.5.3 — degradation: remove one channel at a time, then two
    degradation = []
    for disabled in [(),
                     ("rf",),
                     ("rf", "audio"),
                     ("rf", "audio", "video")]:
        pd_row = run_cycles(args.cycles, drone_present=True, disabled=disabled)
        pfa_row = run_cycles(args.cycles, drone_present=False, disabled=disabled)
        degradation.append({"disabled": disabled, "pd": pd_row, "pfa": pfa_row})
    print_table_3(degradation)

    # 5.5.4 — latency (use the full-channel run)
    print_table_4(pd_full)


if __name__ == "__main__":
    main()
