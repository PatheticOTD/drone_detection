"""RF sensor — multi-receiver array with RSSI trilateration.

Each receiver in RF_ARRAY_POSITIONS measures RSSI from its distance to the
(simulated) drone using a log-distance path-loss model.  Per-node ranges
feed into trilateration to recover (x, y) — and (z) when at least one RX
is elevated.

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
    RF_ARRAY_POSITIONS,
    RF_PATH_LOSS_EXPONENT,
    RF_RSSI_AT_1M_DBM,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend

# class_stats.csv: index 4 = Noise
_RF_NOISE_CLASS_IDX = 4

# Per-RX RSSI measurement noise (dB) and detection floor (dBm).  Path-loss
# range inversion amplifies dB error multiplicatively, so keep noise tight
# to avoid LSQ trilateration diverging.
_RSSI_MEAS_NOISE_DB = 1.5
_RSSI_NOISE_FLOOR_DBM = -95


class RFSensor(BaseSensor):
    """RF scanner sensor backed by a multi-receiver array.

    The array enables RSSI-trilateration: per-RX path-loss ranges combine
    into a real (x, y[, z]) position rather than just a sphere around a
    single receiver.
    """

    KNOWN_PROTOCOLS = ["DJI OcuSync", "FrSky ACCST", "Crossfire", "Wi-Fi 5 GHz", "LTE C2"]
    FREQ_BANDS_MHZ = [900, 2400, 5200, 5800]

    def __init__(
        self,
        backend: Optional["ModelBackend"] = None,
        ground_truth: Optional[SimulationGroundTruth] = None,
    ) -> None:
        super().__init__("rf")
        self._backend = backend
        self._gt = ground_truth
        self._nodes = list(RF_ARRAY_POSITIONS)

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        tensor, _ = self._backend.next_sample()
        with torch.no_grad():
            logits = self._backend.model(tensor.unsqueeze(0).to(self._backend.device))
            probs = torch.softmax(logits, dim=1)[0]

        noise_prob = float(probs[_RF_NOISE_CLASS_IDX])
        drone_prob = 1.0 - noise_prob
        signal_present = drone_prob > 0.5

        node_readings = self._measure_array(signal_present=signal_present)
        peak_rssi = max((n["rssi_dbm"] for n in node_readings), default=_RSSI_NOISE_FLOOR_DBM)
        protocol = random.choice(self.KNOWN_PROTOCOLS) if signal_present else None
        freq = (random.choice(self.FREQ_BANDS_MHZ) + random.randint(-20, 20)
                if signal_present else None)

        return self._make_reading(
            drone_prob,
            {
                "drone_prob": round(drone_prob, 3),
                "noise_prob": round(noise_prob, 3),
                "protocol": protocol,
                "frequency_mhz": freq,
                "rssi_dbm": int(peak_rssi),
                "signal_present": signal_present,
                "array_nodes": node_readings,
                "mode": "nn",
            },
            location=self._localize(node_readings),
        )

    def _poll_sim(self) -> SensorReading:
        gt_present = bool(self._gt.snapshot().drone_present) if self._gt else False
        node_readings = self._measure_array(signal_present=gt_present)
        peak_rssi = max((n["rssi_dbm"] for n in node_readings), default=_RSSI_NOISE_FLOOR_DBM)

        if gt_present:
            confidence = random.uniform(0.50, 0.88) + random.gauss(0, SIMULATION_NOISE_LEVEL)
            protocol = random.choice(self.KNOWN_PROTOCOLS)
            freq = random.choice(self.FREQ_BANDS_MHZ) + random.randint(-20, 20)
        else:
            confidence = random.uniform(0.0, 0.25) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)
            protocol = None
            freq = None

        return self._make_reading(
            confidence,
            {
                "protocol": protocol,
                "frequency_mhz": freq,
                "rssi_dbm": int(peak_rssi),
                "signal_present": gt_present,
                "array_nodes": node_readings,
                "mode": "sim",
            },
            location=self._localize(node_readings),
        )

    # ------------------------------------------------------------------
    def _measure_array(self, signal_present: bool) -> List[dict]:
        """Produce one RSSI + range reading per receiver."""
        gt = self._gt.snapshot() if self._gt else None
        readings: List[dict] = []

        for node in self._nodes:
            if signal_present and gt is not None and gt.drone_present:
                true_dist = max(1.0, math.sqrt(
                    (gt.x - node["x"]) ** 2
                    + (gt.y - node["y"]) ** 2
                    + (gt.z - node["z"]) ** 2
                ))
                # Log-distance path loss:
                #   RSSI(d) = RSSI_1m - 10 * n * log10(d)
                true_rssi = RF_RSSI_AT_1M_DBM - 10.0 * RF_PATH_LOSS_EXPONENT * math.log10(true_dist)
                measured_rssi = true_rssi + random.gauss(0, _RSSI_MEAS_NOISE_DB)
                if measured_rssi > _RSSI_NOISE_FLOOR_DBM:
                    exponent = (RF_RSSI_AT_1M_DBM - measured_rssi) / (10.0 * RF_PATH_LOSS_EXPONENT)
                    est_dist = 10.0 ** exponent
                    est_dist = max(1.0, min(2000.0, est_dist))
                else:
                    est_dist = None
            else:
                measured_rssi = random.randint(-100, -85)
                est_dist = None

            readings.append({
                "node_id": node["node_id"],
                "x": node["x"], "y": node["y"], "z": node["z"],
                "rssi_dbm": int(round(measured_rssi)),
                "range_m": round(est_dist, 1) if est_dist is not None else None,
            })
        return readings

    def _localize(self, node_readings: List[dict]) -> Optional[LocationEstimate]:
        valid = [n for n in node_readings if n["range_m"] is not None]

        if len(valid) < 3:
            if not valid:
                return None
            n = valid[0]
            pos = SENSOR_POSITIONS["rf"]
            return LocationEstimate(
                x=pos["x"], y=pos["y"], z=pos["z"],
                uncertainty_m=round(n["range_m"], 1),
                bearing_deg=None,
                range_m=n["range_m"],
                method="rf",
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
            method="rf_array",
        )
