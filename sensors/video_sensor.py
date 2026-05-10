"""Video sensor — real NN inference when a backend is supplied, simulation otherwise."""

import random
import math
from typing import Optional, TYPE_CHECKING

from sensors.base import BaseSensor, SensorReading, LocationEstimate
from config import (
    SIMULATION_DRONE_PRESENT_PROBABILITY,
    SIMULATION_NOISE_LEVEL,
    SENSOR_POSITIONS,
    CAMERA_CONFIG,
)

if TYPE_CHECKING:
    from models.loader import ModelBackend


class VideoSensor(BaseSensor):
    """Camera/video sensor.

    Pass a ModelBackend (wrapping YOLOv8) to enable real object-detection
    inference on dataset images.  Omit backend for pure simulation.
    """

    DRONE_CLASSES = ["quadcopter", "hexacopter", "fixed_wing_uav"]

    def __init__(self, backend: Optional["ModelBackend"] = None) -> None:
        super().__init__("video")
        self._backend = backend
        self._drone_visible = False
        self._frames_since_change = 0

    def poll(self) -> SensorReading:
        if self._backend is not None:
            return self._poll_nn()
        return self._poll_sim()

    # ------------------------------------------------------------------
    def _poll_nn(self) -> SensorReading:
        img_path = self._backend.next_sample()              # str path
        results = self._backend.model(img_path, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            confidence = 0.0
            bbox = None
        else:
            best_idx = int(boxes.conf.argmax())
            best = boxes[best_idx]
            confidence = float(best.conf)
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            bbox = {
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
            }

        return self._make_reading(
            confidence,
            {
                "bbox": bbox,
                "num_detections": len(boxes),
                "resolution": "1920x1080",
                "fps": 30,
                "mode": "nn",
            },
            location=self._estimate_location(bbox),
        )

    def _poll_sim(self) -> SensorReading:
        self._frames_since_change += 1

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

        return self._make_reading(
            confidence,
            {
                "bbox": bbox,
                "detected_class": detected_class,
                "resolution": "1920x1080",
                "fps": 30,
            },
            location=self._estimate_location(bbox),
        )

    # ------------------------------------------------------------------
    def _estimate_location(self, bbox: Optional[dict]) -> Optional[LocationEstimate]:
        if bbox is None:
            return None

        cam = CAMERA_CONFIG
        res_w = cam["resolution_w"]
        res_h = cam["resolution_h"]

        cx = bbox["x"] + bbox["w"] / 2.0
        cy = bbox["y"] + bbox["h"] / 2.0

        angle_h = (cx - res_w / 2.0) / res_w * cam["fov_h_deg"]
        angle_v = (res_h / 2.0 - cy) / res_h * cam["fov_v_deg"]

        bearing_deg = (cam["azimuth_deg"] + angle_h) % 360
        elevation_deg = cam["elevation_deg"] + angle_v

        bbox_size_px = max(bbox["w"], bbox["h"])
        angular_size_deg = bbox_size_px / max(res_w, res_h) * max(cam["fov_h_deg"], cam["fov_v_deg"])
        angular_size_rad = math.radians(max(angular_size_deg, 0.01))
        slant_range = cam["known_drone_size_m"] / math.tan(angular_size_rad)
        slant_range = max(10.0, min(3000.0, slant_range))

        bearing_rad = math.radians(bearing_deg)
        elevation_rad = math.radians(elevation_deg)
        horiz_range = slant_range * math.cos(elevation_rad)

        pos = SENSOR_POSITIONS["video"]
        x = pos["x"] + horiz_range * math.sin(bearing_rad)
        y = pos["y"] + horiz_range * math.cos(bearing_rad)
        z = max(0.0, pos["z"] + slant_range * math.sin(elevation_rad))

        uncertainty = max(50.0, slant_range * 0.20)

        return LocationEstimate(
            x=round(x, 1),
            y=round(y, 1),
            z=round(z, 1),
            uncertainty_m=round(uncertainty, 1),
            bearing_deg=round(bearing_deg, 1),
            range_m=round(slant_range, 1),
            method="video",
        )
