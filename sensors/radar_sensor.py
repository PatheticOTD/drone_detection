"""Radar sensor — real NN inference when a backend is supplied, simulation otherwise."""

import random
import math
from typing import Optional, TYPE_CHECKING

import torch

from sensors.base import BaseSensor, SensorReading, LocationEstimate
from config import (
    SIMULATION_DRONE_PRESENT_PROBABILITY,
    SIMULATION_NOISE_LEVEL,
    SENSOR_POSITIONS,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend


class RadarSensor(BaseSensor):
    """Radar sensor.

    Pass a ModelBackend to enable NN-powered inference on real Range-Doppler
    matrices (RadarResNet, 3 classes: Cars=0, Drones=1, People=2).
    Omit backend for pure simulation.
    """

    def __init__(self, backend: Optional["ModelBackend"] = None) -> None:
        super().__init__("radar")
        self._backend = backend
        self._tracking = False
        self._range_m = 0.0
        self._bearing_deg = 0.0
        self._altitude_m = 0.0

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        tensor = self._backend.next_sample()                # (1, 11, 61)
        with torch.no_grad():
            logits = self._backend.model(tensor.unsqueeze(0).to(self._backend.device))
            probs = torch.softmax(logits, dim=1)[0]         # (3,)

        drone_prob = float(probs[1])                        # class 1 = Drones

        self._tracking = drone_prob > 0.5
        if self._tracking:
            self._range_m = random.uniform(200, 2000)
            self._bearing_deg = random.uniform(0, 360)
            self._altitude_m = random.uniform(20, 300)
        else:
            self._range_m = 0.0
            self._bearing_deg = 0.0
            self._altitude_m = 0.0

        return self._make_reading(
            drone_prob,
            {
                "car_prob": round(float(probs[0]), 3),
                "drone_prob": round(drone_prob, 3),
                "people_prob": round(float(probs[2]), 3),
                "range_m": round(self._range_m, 1),
                "bearing_deg": round(self._bearing_deg, 1),
                "altitude_m": round(self._altitude_m, 1),
                "track_active": self._tracking,
                "mode": "nn",
            },
            location=self._estimate_location(),
        )

    def _poll_sim(self) -> SensorReading:
        if random.random() < 0.06:
            self._tracking = not self._tracking
        if not self._tracking and random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY * 0.25:
            self._tracking = True
            self._range_m = random.uniform(200, 3000)
            self._bearing_deg = random.uniform(0, 360)
            self._altitude_m = random.uniform(20, 400)

        if self._tracking:
            self._range_m = max(50, self._range_m + random.uniform(-80, 30))
            self._bearing_deg = (self._bearing_deg + random.uniform(-5, 5)) % 360
            self._altitude_m = max(5, self._altitude_m + random.uniform(-10, 10))
            rcs = random.uniform(0.001, 0.05)
            confidence = random.uniform(0.55, 0.92) + random.gauss(0, SIMULATION_NOISE_LEVEL)
        else:
            self._range_m = 0.0
            self._bearing_deg = 0.0
            self._altitude_m = 0.0
            rcs = 0.0
            confidence = random.uniform(0.0, 0.20) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)

        return self._make_reading(
            confidence,
            {
                "range_m": round(self._range_m, 1),
                "bearing_deg": round(self._bearing_deg, 1),
                "altitude_m": round(self._altitude_m, 1),
                "rcs_m2": round(rcs, 5),
                "track_active": self._tracking,
            },
            location=self._estimate_location(),
        )

    # ------------------------------------------------------------------
    def _estimate_location(self) -> Optional[LocationEstimate]:
        if not self._tracking:
            return None

        bearing_rad = math.radians(self._bearing_deg)
        pos = SENSOR_POSITIONS["radar"]
        x = pos["x"] + self._range_m * math.sin(bearing_rad)
        y = pos["y"] + self._range_m * math.cos(bearing_rad)
        z = self._altitude_m
        uncertainty = max(10.0, self._range_m * 0.02)

        return LocationEstimate(
            x=round(x, 1),
            y=round(y, 1),
            z=round(z, 1),
            uncertainty_m=round(uncertainty, 1),
            bearing_deg=round(self._bearing_deg, 1),
            range_m=round(self._range_m, 1),
            method="radar",
        )
