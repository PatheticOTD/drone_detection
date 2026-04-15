"""Mock audio sensor — simulates acoustic drone detection."""

import random
import math
import time

from sensors.base import BaseSensor, SensorReading
from config import SIMULATION_DRONE_PRESENT_PROBABILITY, SIMULATION_NOISE_LEVEL


class AudioSensor(BaseSensor):
    """Simulated audio sensor that listens for propeller noise signatures."""

    DRONE_FREQ_BANDS = [200, 400, 800]  # Hz — typical multirotor blade-pass frequencies

    def __init__(self) -> None:
        super().__init__("audio")
        self._drone_active = False
        self._phase = random.uniform(0, 2 * math.pi)

    def poll(self) -> SensorReading:
        # Randomly toggle drone presence
        if random.random() < 0.05:
            self._drone_active = not self._drone_active

        if not self._drone_active and random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY * 0.3:
            self._drone_active = True
        elif self._drone_active and random.random() < 0.15:
            self._drone_active = False

        if self._drone_active:
            base = random.uniform(0.55, 0.90)
            # Simulate fluctuating signal strength
            self._phase += 0.3
            fluctuation = 0.08 * math.sin(self._phase)
            confidence = base + fluctuation + random.gauss(0, SIMULATION_NOISE_LEVEL)
        else:
            confidence = random.uniform(0.0, 0.30) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)

        dominant_freq = random.choice(self.DRONE_FREQ_BANDS) if self._drone_active else random.randint(50, 1500)
        snr = random.uniform(8, 25) if self._drone_active else random.uniform(-5, 5)

        return self._make_reading(confidence, {
            "dominant_freq_hz": dominant_freq,
            "snr_db": round(snr, 1),
            "drone_signature_match": self._drone_active,
        })
