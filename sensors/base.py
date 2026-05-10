"""Base class for all sensors."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocationEstimate:
    """Estimated 3-D drone position in a local East-North-Up (ENU) frame.

    The origin is the centre of the sensor cluster.
    x = metres East, y = metres North, z = metres above ground.
    uncertainty_m is the 1-sigma spatial uncertainty radius.
    """

    x: float
    y: float
    z: float
    uncertainty_m: float
    bearing_deg: Optional[float]   # azimuth 0=North, 90=East; None if unknown
    range_m: Optional[float]       # slant range to target; None if unknown
    method: str                    # "radar" | "video" | "rf" | "audio"


@dataclass
class SensorReading:
    """A single reading produced by a sensor."""

    sensor_type: str
    timestamp: float
    confidence: float          # 0.0 – 1.0 probability that a drone is present
    detected: bool
    metadata: dict = field(default_factory=dict)
    location: Optional[LocationEstimate] = None


class BaseSensor(abc.ABC):
    """Abstract sensor interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._last_reading: Optional[SensorReading] = None

    @abc.abstractmethod
    def poll(self) -> SensorReading:
        """Take a new reading and return it."""

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    def _make_reading(
        self,
        confidence: float,
        metadata: dict | None = None,
        location: Optional[LocationEstimate] = None,
    ) -> SensorReading:
        confidence = max(0.0, min(1.0, confidence))
        reading = SensorReading(
            sensor_type=self.name,
            timestamp=time.time(),
            confidence=confidence,
            detected=confidence > 0.5,
            metadata=metadata or {},
            location=location,
        )
        self._last_reading = reading
        return reading
