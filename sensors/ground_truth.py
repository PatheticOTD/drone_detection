"""Shared simulated ground-truth drone state.

Multi-node sensor arrays (audio mics, RF receivers) need a *consistent*
underlying drone position to drive their per-node range measurements,
otherwise trilateration over independently-randomised nodes is meaningless.

A single SimulationGroundTruth instance is created in main() and passed
to each array-aware sensor.  In NN mode the model still owns drone/no-drone
classification — the ground-truth supplies only the geometry that the model
cannot infer.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GroundTruthState:
    drone_present: bool
    x: float
    y: float
    z: float


class SimulationGroundTruth:
    """Thread-safe simulator that owns the 'true' drone position."""

    def __init__(
        self,
        present_probability: float = 0.5,
        update_interval_s: float = 1.0,
        area_radius_m: float = 300.0,
        altitude_range_m: Tuple[float, float] = (30.0, 200.0),
        spawn_chance: float = 0.15,
        despawn_chance: float = 0.05,
    ) -> None:
        self._present_probability = present_probability
        self._update_interval_s = update_interval_s
        self._area_radius_m = area_radius_m
        self._alt_min, self._alt_max = altitude_range_m
        self._spawn_chance = spawn_chance
        self._despawn_chance = despawn_chance

        self._lock = threading.Lock()
        self._last_evolve = 0.0
        self._state = GroundTruthState(False, 0.0, 0.0, 0.0)

    def _evolve_locked(self) -> None:
        now = time.time()
        if now - self._last_evolve < self._update_interval_s:
            return
        self._last_evolve = now

        if not self._state.drone_present:
            if random.random() < self._spawn_chance:
                bearing = random.uniform(0.0, 2.0 * math.pi)
                r = random.uniform(60.0, self._area_radius_m)
                self._state = GroundTruthState(
                    drone_present=True,
                    x=r * math.sin(bearing),
                    y=r * math.cos(bearing),
                    z=random.uniform(self._alt_min, self._alt_max),
                )
        else:
            if random.random() < self._despawn_chance:
                self._state = GroundTruthState(False, 0.0, 0.0, 0.0)
            else:
                self._state.x += random.uniform(-12.0, 12.0)
                self._state.y += random.uniform(-12.0, 12.0)
                self._state.z = max(
                    self._alt_min,
                    min(self._alt_max, self._state.z + random.uniform(-4.0, 4.0)),
                )

    def snapshot(self) -> GroundTruthState:
        with self._lock:
            self._evolve_locked()
            return GroundTruthState(
                self._state.drone_present,
                self._state.x,
                self._state.y,
                self._state.z,
            )

    def distance_to(self, x: float, y: float, z: float) -> Optional[float]:
        s = self.snapshot()
        if not s.drone_present:
            return None
        return math.sqrt((s.x - x) ** 2 + (s.y - y) ** 2 + (s.z - z) ** 2)
