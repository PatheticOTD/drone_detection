"""Sensor fusion engine — combines readings from multiple sensors using weighted averaging."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from sensors.base import SensorReading, LocationEstimate
from config import SENSOR_WEIGHTS, DETECTION_THRESHOLD


@dataclass
class FusionResult:
    """Result of fusing multiple sensor readings."""

    timestamp: float
    fused_confidence: float
    drone_detected: bool
    threat_level: str             # "none" | "low" | "medium" | "high"
    sensor_readings: Dict[str, SensorReading]
    weights_used: Dict[str, float]
    contributing_sensors: int
    location: Optional[LocationEstimate] = None

    def to_dict(self) -> dict:
        readings = {}
        for key, r in self.sensor_readings.items():
            loc = None
            if r.location is not None:
                loc = {
                    "x": r.location.x,
                    "y": r.location.y,
                    "z": r.location.z,
                    "uncertainty_m": r.location.uncertainty_m,
                    "bearing_deg": r.location.bearing_deg,
                    "range_m": r.location.range_m,
                    "method": r.location.method,
                }
            readings[key] = {
                "confidence": round(r.confidence, 3),
                "detected": r.detected,
                "metadata": r.metadata,
                "location": loc,
            }

        fused_loc = None
        if self.location is not None:
            fused_loc = {
                "x": self.location.x,
                "y": self.location.y,
                "z": self.location.z,
                "uncertainty_m": self.location.uncertainty_m,
                "bearing_deg": self.location.bearing_deg,
                "range_m": self.location.range_m,
                "method": self.location.method,
            }

        return {
            "timestamp": self.timestamp,
            "fused_confidence": round(self.fused_confidence, 3),
            "drone_detected": self.drone_detected,
            "threat_level": self.threat_level,
            "contributing_sensors": self.contributing_sensors,
            "weights": {k: round(v, 3) for k, v in self.weights_used.items()},
            "sensors": readings,
            "location": fused_loc,
        }


class FusionEngine:
    """Weighted sensor fusion.

    The fused confidence is a weighted average of individual sensor confidences:
        C_fused = Σ (w_i · c_i) / Σ w_i   (only over available sensors)

    Location fusion uses inverse-variance weighting (Kalman-style):
        w_i = confidence_i / uncertainty_i²
        pos_fused = Σ (w_i · pos_i) / Σ w_i
        uncertainty_fused = 1 / sqrt(Σ (confidence_i / uncertainty_i²))

    Only sensors that report a bearing (radar, video) contribute to the fused
    position.  Distance-only sensors (rf, audio) supply a range constraint that
    is preserved in individual readings but not used for XY fusion.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: Optional[float] = None) -> None:
        self.weights = dict(weights or SENSOR_WEIGHTS)
        self.threshold = threshold if threshold is not None else DETECTION_THRESHOLD
        self._history: List[FusionResult] = []

    # ------------------------------------------------------------------
    def fuse(self, readings: Dict[str, SensorReading]) -> FusionResult:
        """Fuse the latest readings and return a FusionResult."""

        total_weight = 0.0
        weighted_sum = 0.0
        used_weights: Dict[str, float] = {}

        for sensor_name, reading in readings.items():
            w = self.weights.get(sensor_name, 0.0)
            if w <= 0:
                continue
            # An offline/failed channel is excluded entirely so the remaining
            # weights renormalise over the active channels (graceful
            # degradation).  A channel that is online but sees nothing reports
            # confidence ~0 with available=True and still counts as evidence.
            if not reading.available:
                continue
            weighted_sum += w * reading.confidence
            total_weight += w
            used_weights[sensor_name] = w

        fused = weighted_sum / total_weight if total_weight > 0 else 0.0
        fused = max(0.0, min(1.0, fused))

        detected = fused >= self.threshold
        threat = self._classify_threat(fused)

        result = FusionResult(
            timestamp=time.time(),
            fused_confidence=fused,
            drone_detected=detected,
            threat_level=threat,
            sensor_readings=dict(readings),
            weights_used=used_weights,
            contributing_sensors=len(used_weights),
            location=self._fuse_locations(readings),
        )
        self._history.append(result)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        return result

    # ------------------------------------------------------------------
    def _fuse_locations(self, readings: Dict[str, SensorReading]) -> Optional[LocationEstimate]:
        """Fuse per-sensor location estimates into a single best-estimate position.

        Uses inverse-variance weighting scaled by sensor confidence:
            w_i = confidence_i / uncertainty_i^2

        Only estimates that include a bearing (i.e. have a real x/y derived
        from direction, not just the sensor's own position) are used for XY
        fusion.  This means radar and video contribute; rf and audio only
        provide a range hint stored in their individual readings.
        """
        directional: list[tuple[float, LocationEstimate]] = []  # (confidence, estimate)

        for reading in readings.values():
            loc = reading.location
            if loc is None or loc.bearing_deg is None or loc.uncertainty_m <= 0:
                continue
            directional.append((reading.confidence, loc))

        if not directional:
            return None

        # Inverse-variance weights scaled by confidence
        total_w = 0.0
        wx = wy = wz = 0.0
        sum_inv_var = 0.0

        for conf, est in directional:
            w = conf / (est.uncertainty_m ** 2)
            wx += w * est.x
            wy += w * est.y
            wz += w * est.z
            total_w += w
            sum_inv_var += conf / (est.uncertainty_m ** 2)

        if total_w <= 0:
            return None

        x_fused = wx / total_w
        y_fused = wy / total_w
        z_fused = wz / total_w

        # Combined uncertainty: 1 / sqrt(Σ w_i) — improves with each sensor added
        uncertainty_fused = max(1.0, 1.0 / math.sqrt(sum_inv_var))

        # Bearing and range of the fused position relative to origin
        bearing_fused = math.degrees(math.atan2(x_fused, y_fused)) % 360
        range_fused = math.hypot(x_fused, y_fused)

        return LocationEstimate(
            x=round(x_fused, 1),
            y=round(y_fused, 1),
            z=round(z_fused, 1),
            uncertainty_m=round(uncertainty_fused, 1),
            bearing_deg=round(bearing_fused, 1),
            range_m=round(range_fused, 1),
            method="fused",
        )

    # ------------------------------------------------------------------
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Hot-update sensor weights (must sum to 1.0)."""
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        self.weights.update(new_weights)

    def update_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = value

    # ------------------------------------------------------------------
    @property
    def history(self) -> List[FusionResult]:
        return list(self._history)

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_threat(confidence: float) -> str:
        if confidence < 0.3:
            return "none"
        if confidence < 0.55:
            return "low"
        if confidence < 0.75:
            return "medium"
        return "high"
