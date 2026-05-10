"""Audio sensor — real NN inference when a backend is supplied, simulation otherwise."""

import random
import math
from typing import Optional, TYPE_CHECKING

import torch

from sensors.base import BaseSensor, SensorReading, LocationEstimate
from config import (
    SIMULATION_DRONE_PRESENT_PROBABILITY,
    SIMULATION_NOISE_LEVEL,
    SENSOR_POSITIONS,
    AUDIO_SNR_AT_1M_DB,
    AUDIO_MAX_RANGE_M,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend


class AudioSensor(BaseSensor):
    """Acoustic sensor.

    Pass a ModelBackend to enable NN-powered inference on mel-spectrograms
    (AudioResNet, 2 classes: no_drone=0, drone=1).
    Omit backend for pure simulation.
    """

    DRONE_FREQ_BANDS = [200, 400, 800]

    def __init__(self, backend: Optional["ModelBackend"] = None) -> None:
        super().__init__("audio")
        self._backend = backend
        self._drone_active = False
        self._phase = random.uniform(0, 2 * math.pi)

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        tensor, _ = self._backend.next_sample()              # (1, 128, T), int
        with torch.no_grad():
            logits = self._backend.model(tensor.unsqueeze(0).to(self._backend.device))
            probs = torch.softmax(logits, dim=1)[0]         # (2,)

        drone_prob = float(probs[1])                        # class 1 = drone
        self._drone_active = drone_prob > 0.5

        snr = random.uniform(8, 25) if self._drone_active else random.uniform(-5, 5)
        dominant_freq = (random.choice(self.DRONE_FREQ_BANDS)
                         if self._drone_active else random.randint(50, 1500))

        return self._make_reading(
            drone_prob,
            {
                "drone_prob": round(drone_prob, 3),
                "no_drone_prob": round(float(probs[0]), 3),
                "dominant_freq_hz": dominant_freq,
                "snr_db": round(snr, 1),
                "drone_signature_match": self._drone_active,
                "mode": "nn",
            },
            location=self._estimate_location(snr),
        )

    def _poll_sim(self) -> SensorReading:
        if random.random() < 0.05:
            self._drone_active = not self._drone_active

        if not self._drone_active and random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY * 0.3:
            self._drone_active = True
        elif self._drone_active and random.random() < 0.15:
            self._drone_active = False

        if self._drone_active:
            base = random.uniform(0.55, 0.90)
            self._phase += 0.3
            fluctuation = 0.08 * math.sin(self._phase)
            confidence = base + fluctuation + random.gauss(0, SIMULATION_NOISE_LEVEL)
        else:
            confidence = random.uniform(0.0, 0.30) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)

        dominant_freq = random.choice(self.DRONE_FREQ_BANDS) if self._drone_active else random.randint(50, 1500)
        snr = random.uniform(8, 25) if self._drone_active else random.uniform(-5, 5)

        return self._make_reading(
            confidence,
            {
                "dominant_freq_hz": dominant_freq,
                "snr_db": round(snr, 1),
                "drone_signature_match": self._drone_active,
            },
            location=self._estimate_location(snr),
        )

    # ------------------------------------------------------------------
    def _estimate_location(self, snr_db: float) -> Optional[LocationEstimate]:
        if not self._drone_active:
            return None

        distance = 1.0 * 10.0 ** ((AUDIO_SNR_AT_1M_DB - snr_db) / 20.0)
        distance = max(5.0, min(AUDIO_MAX_RANGE_M, distance))

        pos = SENSOR_POSITIONS["audio"]
        return LocationEstimate(
            x=pos["x"],
            y=pos["y"],
            z=pos["z"],
            uncertainty_m=round(distance * 0.5, 1),
            bearing_deg=None,
            range_m=round(distance, 1),
            method="audio",
        )
