"""Mock video sensor — simulates visual drone detection via a camera feed."""

import random
import math

from sensors.base import BaseSensor, SensorReading
from config import SIMULATION_DRONE_PRESENT_PROBABILITY, SIMULATION_NOISE_LEVEL


class VideoSensor(BaseSensor):
    """Simulated video/camera sensor with object-detection style output."""

    DRONE_CLASSES = ["quadcopter", "hexacopter", "fixed_wing_uav"]

    def __init__(self) -> None:
        super().__init__("video")
        self._drone_visible = False
        self._frames_since_change = 0

    def poll(self) -> SensorReading:
        self._frames_since_change += 1

        # Simulate drone appearing / disappearing
        if self._frames_since_change > random.randint(5, 20):
            if random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY:
                self._drone_visible = True
            else:
                self._drone_visible = random.random() < 0.3
            self._frames_since_change = 0

        if self._drone_visible:
            confidence = random.uniform(0.50, 0.95) + random.gauss(0, SIMULATION_NOISE_LEVEL)
            bbox = {
                "x": random.randint(100, 1800),
                "y": random.randint(100, 900),
                "w": random.randint(20, 120),
                "h": random.randint(20, 120),
            }
            detected_class = random.choice(self.DRONE_CLASSES)
        else:
            confidence = random.uniform(0.0, 0.25) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)
            bbox = None
            detected_class = None

        return self._make_reading(confidence, {
            "bbox": bbox,
            "detected_class": detected_class,
            "resolution": "1920x1080",
            "fps": 30,
        })
