"""Drone Detection System — main entry point.

Starts sensors (NN-backed when weights are available, simulated otherwise),
fusion engine, and the web dashboard.
"""

import sys
import threading
import logging

import torch

from sensors.audio_sensor import AudioSensor
from sensors.video_sensor import VideoSensor
from sensors.radar_sensor import RadarSensor
from sensors.rf_sensor import RFSensor
from sensors.ground_truth import SimulationGroundTruth
from fusion.engine import FusionEngine
from dashboard.app import set_fusion_engine, broadcast_update, run_dashboard
from config import SENSOR_POLL_INTERVAL, DASHBOARD_HOST, DASHBOARD_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drone_detection")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _try_load(loader_fn, *args, **kwargs):
    """Call loader_fn(*args) and return None on any error (graceful fallback)."""
    try:
        return loader_fn(*args, **kwargs)
    except Exception as exc:
        log.warning("Backend unavailable — falling back to simulation: %s", exc)
        return None


def _build_sensors(ground_truth: SimulationGroundTruth) -> dict:
    from models.loader import (
        load_radar_backend,
        load_audio_backend,
        load_rf_backend,
        load_video_backend,
    )

    log.info("Loading NN backends (device=%s)…", DEVICE)

    radar_backend = _try_load(
        load_radar_backend,
        "weights/radar_resnet_best.pt",
        "data/Real Doppler RAD-DAR database",
        DEVICE,
    )
    audio_backend = _try_load(
        load_audio_backend,
        "weights/audio_resnet_best.pt",
        "data/DroneAudioDataset/Binary_Drone_Audio",
        DEVICE,
    )
    rf_backend = _try_load(
        load_rf_backend,
        "weights/rf-resnet-1d_best.pt",
        "data/Noisy Drone RF Signal Classification/dataset.pt",
        DEVICE,
    )
    video_backend = _try_load(
        load_video_backend,
        "weights/yolov8n.pt",
        "data/Airborne-Object-Detection-4-AOD4.yolov8/train/images",
        DEVICE,
    )

    for name, backend in [("radar", radar_backend), ("audio", audio_backend),
                           ("rf", rf_backend), ("video", video_backend)]:
        mode = "nn" if backend is not None else "simulation"
        log.info("  %-6s → %s", name, mode)

    return {
        "audio": AudioSensor(backend=audio_backend, ground_truth=ground_truth),
        "video": VideoSensor(backend=video_backend),
        "radar": RadarSensor(backend=radar_backend),
        "rf":    RFSensor(backend=rf_backend, ground_truth=ground_truth),
    }


def sensor_loop(engine: FusionEngine, sensors: dict, stop_event: threading.Event) -> None:
    """Continuously poll sensors, fuse readings and broadcast to the dashboard."""
    log.info("Sensor loop started — polling every %.1fs", SENSOR_POLL_INTERVAL)
    while not stop_event.is_set():
        readings = {}
        for name, sensor in sensors.items():
            readings[name] = sensor.poll()

        result = engine.fuse(readings)

        status = "DRONE DETECTED" if result.drone_detected else "clear"
        log.info(
            "fused=%.3f  threat=%-6s  [audio=%.2f video=%.2f radar=%.2f rf=%.2f]  %s",
            result.fused_confidence,
            result.threat_level,
            readings["audio"].confidence,
            readings["video"].confidence,
            readings["radar"].confidence,
            readings["rf"].confidence,
            status,
        )

        broadcast_update(result.to_dict())
        stop_event.wait(SENSOR_POLL_INTERVAL)


def main() -> None:
    log.info("=== Drone Detection System ===")

    ground_truth = SimulationGroundTruth()
    sensors = _build_sensors(ground_truth)
    engine = FusionEngine()
    set_fusion_engine(engine)

    stop_event = threading.Event()
    sensor_thread = threading.Thread(target=sensor_loop, args=(engine, sensors, stop_event), daemon=True)
    sensor_thread.start()

    log.info("Starting web dashboard at http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)
    try:
        run_dashboard(DASHBOARD_HOST, DASHBOARD_PORT)
    except KeyboardInterrupt:
        log.info("Shutting down…")
        stop_event.set()
        sensor_thread.join(timeout=3)
        sys.exit(0)


if __name__ == "__main__":
    main()
