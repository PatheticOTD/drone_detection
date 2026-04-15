"""Drone Detection System — main entry point.

Starts mock sensors, fusion engine and the web dashboard.
"""

import sys
import time
import threading
import logging

from sensors.audio_sensor import AudioSensor
from sensors.video_sensor import VideoSensor
from sensors.radar_sensor import RadarSensor
from sensors.rf_sensor import RFSensor
from fusion.engine import FusionEngine
from dashboard.app import set_fusion_engine, broadcast_update, run_dashboard
from config import SENSOR_POLL_INTERVAL, DASHBOARD_HOST, DASHBOARD_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drone_detection")


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
    log.info("Initialising mock sensors…")

    sensors = {
        "audio": AudioSensor(),
        "video": VideoSensor(),
        "radar": RadarSensor(),
        "rf": RFSensor(),
    }

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
