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

# ---------------------------------------------------------------------------
# Location estimation configuration
# ---------------------------------------------------------------------------

# Sensor positions in local ENU frame (metres).
# Origin is the centre of the sensor cluster; x=East, y=North, z=Up.
# Used by single-node sensors (radar, video) and as the fallback centre
# for the audio / RF arrays.
SENSOR_POSITIONS = {
    "radar": {"x":  0.0, "y":  0.0, "z": 2.0},
    "video": {"x":  5.0, "y":  0.0, "z": 3.0},
    "audio": {"x": -5.0, "y":  0.0, "z": 1.0},
    "rf":    {"x":  0.0, "y":  5.0, "z": 2.0},
}

# ---------------------------------------------------------------------------
# Multi-node sensor arrays
# ---------------------------------------------------------------------------
# Audio microphone array — 3+ spatially-separated mics enable trilateration
# of the acoustic source (azimuth + horizontal range).  Coplanar mics cannot
# recover the source altitude; the elevation column in the result is the
# array altitude, with large vertical uncertainty.
AUDIO_ARRAY_POSITIONS = [
    {"node_id": "audio_0", "x":  -60.0, "y":   0.0, "z": 1.5},
    {"node_id": "audio_1", "x":   60.0, "y":   0.0, "z": 1.5},
    {"node_id": "audio_2", "x":    0.0, "y":  60.0, "z": 1.5},
    {"node_id": "audio_3", "x":    0.0, "y": -60.0, "z": 1.5},
]

# RF receiver array — 3+ synchronised RX nodes enable RSSI-trilateration.
# (Real systems use TDOA between GPS-disciplined RX; here we use RSSI
# range estimates plus least-squares localisation, which is the simpler
# but lower-accuracy variant.)  Nodes are kept coplanar — recovering Z
# from noisy RSSI is unstable and would just add error.
RF_ARRAY_POSITIONS = [
    {"node_id": "rf_0", "x": -100.0, "y":  -60.0, "z": 3.0},
    {"node_id": "rf_1", "x":  100.0, "y":  -60.0, "z": 3.0},
    {"node_id": "rf_2", "x":    0.0, "y":  110.0, "z": 3.0},
    {"node_id": "rf_3", "x":    0.0, "y":    0.0, "z": 3.0},
]

# Camera parameters for video-based location estimation
CAMERA_CONFIG = {
    "fov_h_deg":          60.0,   # horizontal field of view (degrees)
    "fov_v_deg":          45.0,   # vertical field of view (degrees)
    "azimuth_deg":         0.0,   # direction camera faces (0=North, 90=East)
    "elevation_deg":      10.0,   # tilt above horizontal (degrees)
    "resolution_w":      1920,    # frame width (pixels)
    "resolution_h":      1080,    # frame height (pixels)
    "known_drone_size_m": 0.5,    # approximate max drone dimension (metres)
}

# RF path-loss model: d = 10^((RSSI_1m - RSSI) / (10 * n))
RF_PATH_LOSS_EXPONENT = 2.7     # 2.0 = free space; 2.7–3.5 = urban/obstructed
RF_RSSI_AT_1M_DBM     = -30     # reference RSSI measured at 1 m (dBm)

# Acoustic distance model (inverse-square law in amplitude):
#   d = d_ref * 10^((SNR_ref - SNR) / 20)
AUDIO_SNR_AT_1M_DB = 40.0       # reference SNR at 1 m distance (dB)
AUDIO_MAX_RANGE_M  = 500.0      # maximum detectable acoustic range (metres)
