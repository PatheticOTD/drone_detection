"""Mock RF scanner sensor — simulates detection of drone control-link signals."""

import random

from sensors.base import BaseSensor, SensorReading
from config import SIMULATION_DRONE_PRESENT_PROBABILITY, SIMULATION_NOISE_LEVEL


class RFSensor(BaseSensor):
    """Simulated RF scanner that looks for known drone control/video downlink protocols."""

    KNOWN_PROTOCOLS = ["DJI OcuSync", "FrSky ACCST", "Crossfire", "Wi-Fi 5 GHz", "LTE C2"]
    FREQ_BANDS_MHZ = [900, 2400, 5200, 5800]

    def __init__(self) -> None:
        super().__init__("rf")
        self._signal_present = False

    def poll(self) -> SensorReading:
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

        return self._make_reading(confidence, {
            "protocol": protocol,
            "frequency_mhz": freq,
            "rssi_dbm": rssi,
            "signal_present": self._signal_present,
        })
