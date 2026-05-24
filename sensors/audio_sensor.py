"""Audio sensor — multi-microphone array with trilateration.

Each microphone in AUDIO_ARRAY_POSITIONS measures an SNR derived from its
distance to the (simulated) drone.  Per-node SNRs are inverted into per-node
ranges via the inverse-square acoustic model and the array is trilaterated
to recover (x, y) — and (z) when at least one mic is elevated.

NN backend (when present) supplies drone/no-drone classification only;
geometry comes from the shared SimulationGroundTruth.
"""

from __future__ import annotations

import random
import math
from typing import List, Optional, TYPE_CHECKING

import torch

from sensors.base import BaseSensor, SensorReading, LocationEstimate
from sensors.ground_truth import SimulationGroundTruth
from fusion.localization import trilaterate, bearing_and_range_from_origin
from config import (
    SIMULATION_NOISE_LEVEL,
    SENSOR_POSITIONS,
    AUDIO_ARRAY_POSITIONS,
    AUDIO_SNR_AT_1M_DB,
    AUDIO_MAX_RANGE_M,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend

# Acoustic measurement noise, in dB, per mic.  Range-from-SNR is highly
# noise-sensitive (1 dB error ≈ 12% range error), so keep this tight to
# avoid LSQ trilateration blowing up.
_SNR_MEAS_NOISE_DB = 1.5
# Below this SNR a mic effectively hears nothing useful.
_SNR_DETECTION_FLOOR_DB = -10.0


class AudioSensor(BaseSensor):
    """Acoustic sensor backed by a microphone array.

    The array enables direction-finding: trilateration over per-mic ranges
    yields a real (x, y[, z]) position, not just a sphere around the
    sensor as a single-mic sensor produces.
    """

    DRONE_FREQ_BANDS = [200, 400, 800]

    def __init__(
        self,
        backend: Optional["ModelBackend"] = None,
        ground_truth: Optional[SimulationGroundTruth] = None,
    ) -> None:
        super().__init__("audio")
        self._backend = backend
        self._gt = ground_truth
        self._nodes = list(AUDIO_ARRAY_POSITIONS)

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        tensor, _ = self._backend.next_sample()              # (1, 128, T)
        with torch.no_grad():
            logits = self._backend.model(tensor.unsqueeze(0).to(self._backend.device))
            probs = torch.softmax(logits, dim=1)[0]          # (2,)

        drone_prob = float(probs[1])
        drone_active = drone_prob > 0.5

        node_readings = self._measure_array(drone_present=drone_active)
        peak_snr = max((n["snr_db"] for n in node_readings), default=0.0)
        dominant_freq = (random.choice(self.DRONE_FREQ_BANDS)
                         if drone_active else random.randint(50, 1500))

        return self._make_reading(
            drone_prob,
            {
                "drone_prob": round(drone_prob, 3),
                "no_drone_prob": round(float(probs[0]), 3),
                "dominant_freq_hz": dominant_freq,
                "snr_db": round(peak_snr, 1),
                "drone_signature_match": drone_active,
                "array_nodes": node_readings,
                "mode": "nn",
            },
            location=self._localize(node_readings),
        )

    def _poll_sim(self) -> SensorReading:
        gt_present = bool(self._gt.snapshot().drone_present) if self._gt else False
        node_readings = self._measure_array(drone_present=gt_present)
        peak_snr = max((n["snr_db"] for n in node_readings), default=0.0)

        if gt_present:
            base = random.uniform(0.55, 0.90)
            confidence = base + random.gauss(0, SIMULATION_NOISE_LEVEL)
        else:
            confidence = random.uniform(0.0, 0.30) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)

        dominant_freq = random.choice(self.DRONE_FREQ_BANDS) if gt_present else random.randint(50, 1500)

        return self._make_reading(
            confidence,
            {
                "dominant_freq_hz": dominant_freq,
                "snr_db": round(peak_snr, 1),
                "drone_signature_match": gt_present,
                "array_nodes": node_readings,
                "mode": "sim",
            },
            location=self._localize(node_readings),
        )

    # ------------------------------------------------------------------
    def _measure_array(self, drone_present: bool) -> List[dict]:
        """Produce one SNR + range reading per microphone."""
        gt = self._gt.snapshot() if self._gt else None
        readings: List[dict] = []

        for node in self._nodes:
            if drone_present and gt is not None and gt.drone_present:
                true_dist = max(1.0, math.sqrt(
                    (gt.x - node["x"]) ** 2
                    + (gt.y - node["y"]) ** 2
                    + (gt.z - node["z"]) ** 2
                ))
                # Inverse-square (amplitude): SNR(d) = SNR_1m - 20*log10(d)
                true_snr = AUDIO_SNR_AT_1M_DB - 20.0 * math.log10(true_dist)
                measured_snr = true_snr + random.gauss(0, _SNR_MEAS_NOISE_DB)
                if measured_snr > _SNR_DETECTION_FLOOR_DB:
                    est_dist = 10.0 ** ((AUDIO_SNR_AT_1M_DB - measured_snr) / 20.0)
                    est_dist = max(1.0, min(AUDIO_MAX_RANGE_M, est_dist))
                else:
                    est_dist = None
            else:
                measured_snr = random.uniform(-12.0, 2.0)
                est_dist = None

            readings.append({
                "node_id": node["node_id"],
                "x": node["x"], "y": node["y"], "z": node["z"],
                "snr_db": round(measured_snr, 1),
                "range_m": round(est_dist, 1) if est_dist is not None else None,
            })
        return readings

    def _localize(self, node_readings: List[dict]) -> Optional[LocationEstimate]:
        valid = [n for n in node_readings if n["range_m"] is not None]

        # Need at least 3 nodes for trilateration; fall back to range-only.
        if len(valid) < 3:
            if not valid:
                return None
            n = valid[0]
            pos = SENSOR_POSITIONS["audio"]
            return LocationEstimate(
                x=pos["x"], y=pos["y"], z=pos["z"],
                uncertainty_m=round(n["range_m"] * 0.5, 1),
                bearing_deg=None,
                range_m=n["range_m"],
                method="audio",
            )

        positions = [(n["x"], n["y"], n["z"]) for n in valid]
        ranges = [n["range_m"] for n in valid]
        result = trilaterate(positions, ranges)
        if result is None:
            return None

        bearing, horiz_range = bearing_and_range_from_origin(result.x, result.y)
        unc = max(result.uncertainty_m, result.z_uncertainty_m * 0.1 + result.uncertainty_m)
        return LocationEstimate(
            x=round(result.x, 1),
            y=round(result.y, 1),
            z=round(result.z, 1),
            uncertainty_m=round(unc, 1),
            bearing_deg=round(bearing, 1),
            range_m=round(horiz_range, 1),
            method="audio_array",
        )
