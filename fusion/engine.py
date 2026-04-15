"""Sensor fusion engine — combines readings from multiple sensors using weighted averaging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sensors.base import SensorReading
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

    def to_dict(self) -> dict:
        readings = {}
        for key, r in self.sensor_readings.items():
            readings[key] = {
                "confidence": round(r.confidence, 3),
                "detected": r.detected,
                "metadata": r.metadata,
            }
        return {
            "timestamp": self.timestamp,
            "fused_confidence": round(self.fused_confidence, 3),
            "drone_detected": self.drone_detected,
            "threat_level": self.threat_level,
            "contributing_sensors": self.contributing_sensors,
            "weights": {k: round(v, 3) for k, v in self.weights_used.items()},
            "sensors": readings,
        }


class FusionEngine:
    """Weighted sensor fusion.

    The fused confidence is a weighted average of individual sensor confidences:
        C_fused = Σ (w_i · c_i) / Σ w_i   (only over available sensors)
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
        )
        self._history.append(result)
        # Keep only last 500 entries
        if len(self._history) > 500:
            self._history = self._history[-500:]
        return result

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
