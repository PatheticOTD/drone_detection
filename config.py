"""Configuration for drone detection system."""

# Sensor fusion weights (must sum to 1.0)
SENSOR_WEIGHTS = {
    "audio": 0.20,
    "video": 0.35,
    "radar": 0.30,
    "rf": 0.15,
}

# Detection threshold (0.0–1.0). Fused confidence above this = drone detected
DETECTION_THRESHOLD = 0.55

# Sensor polling interval in seconds
SENSOR_POLL_INTERVAL = 1.0

# Web dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000

# Simulation settings
SIMULATION_DRONE_PRESENT_PROBABILITY = 0.4   # chance a simulated drone appears
SIMULATION_NOISE_LEVEL = 0.15                 # noise added to sensor readings
