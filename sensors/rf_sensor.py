"""RF sensor — real NN inference when a backend is supplied, simulation otherwise."""

import random
from typing import Optional, TYPE_CHECKING

import torch

from sensors.base import BaseSensor, SensorReading, LocationEstimate
from config import (
    SIMULATION_DRONE_PRESENT_PROBABILITY,
    SIMULATION_NOISE_LEVEL,
    SENSOR_POSITIONS,
    RF_PATH_LOSS_EXPONENT,
    RF_RSSI_AT_1M_DBM,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend

# class_stats.csv confirmed: index 4 = Noise
_RF_NOISE_CLASS_IDX = 4


class RFSensor(BaseSensor):
    """RF scanner sensor.

    Pass a ModelBackend to enable NN-powered inference on IQ signals
    (RFMLP, 7 classes; P(drone) = 1 - P(Noise at index 4)).
    Omit backend for pure simulation.
    """

    KNOWN_PROTOCOLS = ["DJI OcuSync", "FrSky ACCST", "Crossfire", "Wi-Fi 5 GHz", "LTE C2"]
    FREQ_BANDS_MHZ = [900, 2400, 5200, 5800]

    def __init__(self, backend: Optional["ModelBackend"] = None) -> None:
        super().__init__("rf")
        self._backend = backend
        self._signal_present = False

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        tensor, _ = self._backend.next_sample()              # (2, 16384), int
        with torch.no_grad():
            logits = self._backend.model(tensor.unsqueeze(0).to(self._backend.device))
            probs = torch.softmax(logits, dim=1)[0]         # (7,)

        noise_prob = float(probs[_RF_NOISE_CLASS_IDX])
        drone_prob = 1.0 - noise_prob                       # P(drone) = 1 - P(Noise)
        self._signal_present = drone_prob > 0.5

        rssi = random.randint(-75, -30) if self._signal_present else random.randint(-100, -80)
        protocol = random.choice(self.KNOWN_PROTOCOLS) if self._signal_present else None
        freq = (random.choice(self.FREQ_BANDS_MHZ) + random.randint(-20, 20)
                if self._signal_present else None)

        return self._make_reading(
            drone_prob,
            {
                "drone_prob": round(drone_prob, 3),
                "noise_prob": round(noise_prob, 3),
                "protocol": protocol,
                "frequency_mhz": freq,
                "rssi_dbm": rssi,
                "signal_present": self._signal_present,
                "mode": "nn",
            },
            location=self._estimate_location(rssi),
        )

    def _poll_sim(self) -> SensorReading:
        if random.random() < 0.07:
            self._signal_present = not self._signal_present
        if not self._signal_present and random.random() < SIMULATION_DRONE_PRESENT_PROBABILITY * 0.3:
            self._signal_present = True

        if self._signal_present:
            confidence = random.uniform(0.50, 0.88) + random.gauss(0, SIMULATION_NOISE_LEVEL)
            protocol = random.choice(self.KNOWN_PROTOCOLS)
            freq = random.choice(self.FREQ_BANDS_MHZ) + random.randint(-20, 20)
            rssi = random.randint(-75, -30)
        else:
            confidence = random.uniform(0.0, 0.25) + random.gauss(0, SIMULATION_NOISE_LEVEL * 0.5)
            protocol = None
            freq = None
            rssi = random.randint(-100, -80)

        return self._make_reading(
            confidence,
            {
                "protocol": protocol,
                "frequency_mhz": freq,
                "rssi_dbm": rssi,
                "signal_present": self._signal_present,
            },
            location=self._estimate_location(rssi),
        )

    # ------------------------------------------------------------------
    def _estimate_location(self, rssi_dbm: int) -> Optional[LocationEstimate]:
        if not self._signal_present:
            return None

        exponent = (RF_RSSI_AT_1M_DBM - rssi_dbm) / (10.0 * RF_PATH_LOSS_EXPONENT)
        distance = 10.0 ** exponent
        distance = max(5.0, min(2000.0, distance))

        pos = SENSOR_POSITIONS["rf"]
        return LocationEstimate(
            x=pos["x"],
            y=pos["y"],
            z=pos["z"],
            uncertainty_m=round(distance, 1),
            bearing_deg=None,
            range_m=round(distance, 1),
            method="rf",
        )
