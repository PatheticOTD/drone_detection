"""Flask + SocketIO web dashboard for the drone detection system."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

if TYPE_CHECKING:
    from fusion.engine import FusionEngine

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Will be set by main.py before the server starts
_fusion_engine: FusionEngine | None = None


def set_fusion_engine(engine: FusionEngine) -> None:
    global _fusion_engine
    _fusion_engine = engine


# ---- Routes ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/weights", methods=["GET"])
def get_weights():
    if _fusion_engine is None:
        return jsonify({"error": "Engine not ready"}), 503
    return jsonify(_fusion_engine.weights)


@app.route("/api/weights", methods=["POST"])
def update_weights():
    if _fusion_engine is None:
        return jsonify({"error": "Engine not ready"}), 503
    data = request.get_json(force=True)
    weights = {}
    for key in ("audio", "video", "radar", "rf"):
        val = data.get(key)
        if val is None:
            return jsonify({"error": f"Missing weight for '{key}'"}), 400
        try:
            weights[key] = float(val)
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid value for '{key}'"}), 400
    try:
        _fusion_engine.update_weights(weights)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "weights": _fusion_engine.weights})


@app.route("/api/threshold", methods=["POST"])
def update_threshold():
    if _fusion_engine is None:
        return jsonify({"error": "Engine not ready"}), 503
    data = request.get_json(force=True)
    try:
        value = float(data.get("threshold", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid threshold value"}), 400
    try:
        _fusion_engine.update_threshold(value)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "threshold": _fusion_engine.threshold})


@app.route("/api/history", methods=["GET"])
def get_history():
    if _fusion_engine is None:
        return jsonify({"error": "Engine not ready"}), 503
    limit = request.args.get("limit", 50, type=int)
    history = _fusion_engine.history[-limit:]
    return jsonify([r.to_dict() for r in history])


# ---- SocketIO helpers ----

def broadcast_update(fusion_result_dict: dict) -> None:
    """Called from the sensor loop to push data to all connected clients."""
    socketio.emit("sensor_update", fusion_result_dict)


def run_dashboard(host: str, port: int) -> None:
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
