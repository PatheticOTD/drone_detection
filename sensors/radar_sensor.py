"""Mock radar sensor — simulates radar echo detection of small UAVs."""

import random
import math

from sensors.base import BaseSensor, SensorReading
from config import SIMULATION_DRONE_PRESENT_PROBABILITY, SIMULATION_NOISE_LEVEL


class RadarSensor(BaseSensor):
    """Simulated radar that reports range, bearing, altitude and RCS."""

    def __init__(self) -> None:
        super().__init__("radar")
        self._tracking = False
        self._range_m = 0.0
        self._bearing_deg = 0.0
        self._altitude_m = 0.0

    def poll(self) -> SensorReading:
        # Toggle track
        if random.random() < 0.06:
            self._tracking = not self._tracking
        if not self._tracking and random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY * 0.25:
            self._tracking = True
            self._range_m = random.uniform(200, 3000)
            self._bearing_deg = random.uniform(0, 360)
            self._altitude_m = random.uniform(20, 400)

        if self._tracking:
            # Drone approaches
            self._range_m = max(50, self._range_m + random.uniform(-80, 30))
            self._bearing_deg = (self._bearing_deg + random.uniform(-5, 5)) % 360
            self._altitude_m = max(5, self._altitude_m + random.uniform(-10, 10))
            rcs = random.uniform(0.001, 0.05)  # m² — small UAV cross-section
            confidence = random.uniform(0.55, 0.92) + random.gauss(0, SIMULATION_NOISE_LEVEL)
        else:
            self._range_m = 0
            self._bearing_deg = 0
            self._altitude_m = 0
            rcs = 0.0
            confidence = random.uniform(0.0, 0.20) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)

        return self._make_reading(confidence, {
            "range_m": round(self._range_m, 1),
            "bearing_deg": round(self._bearing_deg, 1),
            "altitude_m": round(self._altitude_m, 1),
            "rcs_m2": round(rcs, 5),
            "track_active": self._tracking,
        })
