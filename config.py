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
SIMULATION_DRONE_PRESENT_PROBABILITY = 0.4   
SIMULATION_NOISE_LEVEL = 0.15


SENSOR_POSITIONS = {
    "radar": {"x":  0.0, "y":  0.0, "z": 2.0},
    "video": {"x":  5.0, "y":  0.0, "z": 3.0},
    "audio": {"x": -5.0, "y":  0.0, "z": 1.0},
    "rf":    {"x":  0.0, "y":  5.0, "z": 2.0},
}


AUDIO_ARRAY_POSITIONS = [
    {"node_id": "audio_0", "x":  -60.0, "y":   0.0, "z": 1.5},
    {"node_id": "audio_1", "x":   60.0, "y":   0.0, "z": 1.5},
    {"node_id": "audio_2", "x":    0.0, "y":  60.0, "z": 1.5},
    {"node_id": "audio_3", "x":    0.0, "y": -60.0, "z": 1.5},
]


RF_ARRAY_POSITIONS = [
    {"node_id": "rf_0", "x": -100.0, "y":  -60.0, "z": 3.0},
    {"node_id": "rf_1", "x":  100.0, "y":  -60.0, "z": 3.0},
    {"node_id": "rf_2", "x":    0.0, "y":  110.0, "z": 3.0},
    {"node_id": "rf_3", "x":    0.0, "y":    0.0, "z": 3.0},
]

# Camera parameters for video-based location estimation
CAMERA_CONFIG = {
    "fov_h_deg":          60.0,   
    "fov_v_deg":          45.0,   
    "azimuth_deg":         0.0,   
    "elevation_deg":      10.0,   
    "resolution_w":      1920,    
    "resolution_h":      1080,    
    "known_drone_size_m": 0.5,    
}

# RF path-loss model: d = 10^((RSSI_1m - RSSI) / (10 * n))
RF_PATH_LOSS_EXPONENT = 2.7     
RF_RSSI_AT_1M_DBM     = -30     
# Acoustic distance model (inverse-square law in amplitude):
#   d = d_ref * 10^((SNR_ref - SNR) / 20)
AUDIO_SNR_AT_1M_DB = 40.0       
AUDIO_MAX_RANGE_M  = 500.0      
