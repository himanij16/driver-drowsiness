"""Central configuration. Tune values here, not inside modules."""

# ---- Camera ----
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720          # 720p, 30 fps (Chapter 4)

# ---- Calibration (driver should be alert, seated normally, looking ahead) ----
CALIB_SEC = 45

# ---- Eye / mouth / head ----
EAR_DEFAULT_OPEN = 0.28               # used until calibration finishes
EAR_CLOSED_RATIO = 0.75               # eye "closed" if EAR < ratio * calibrated open EAR
MAR_YAWN_THRESH = 0.60                # mouth aspect ratio for a yawn
YAWN_MIN_SEC = 1.2
NOD_PITCH_DEG = 15                    # head pitch deviation from baseline that counts as a nod
BLINK_MAX_SEC = 0.5                   # closures longer than this are micro-sleeps, not blinks
PERCLOS_WINDOW_SEC = 60
EVENT_WINDOW_SEC = 120                # window for yawn / nod frequency (events per minute)

# ---- rPPG / respiration ----
RPPG_WINDOW_SEC = 15
RPPG_MIN_SPAN_SEC = 10
RPPG_MIN_QUALITY = 0.25               # spectral peak quality gate
RESP_WINDOW_SEC = 40
RESP_MIN_SPAN_SEC = 20

# ---- Audio ----
AUDIO_SR = 16000                      # >= 16 kHz (Chapter 4)
AUDIO_SEGMENT_SEC = 5.0
AUDIO_SILENCE_DB = -55.0              # below this: no music detected -> neutral index

# ---- BLE ----
BLE_DEVICE_NAME = None                # e.g. "Mi Smart Band" (None = first HR-service device)
BLE_SCAN_SEC = 10
BLE_STALE_SEC = 5

# ---- Fusion (Equation 5.1). Start from literature-informed guesses, then refine via ablation ----
WEIGHTS = {
    "perclos": 0.30,   # w1
    "yawn":    0.10,   # w2
    "nod":     0.15,   # w3
    "posture": 0.10,   # w4
    "audio":   0.10,   # w5
    "hrv":     0.25,   # w6
}
# Normalisation: feature value that maps to 1.0 (saturation point)
NORM = {"perclos": 0.30, "yawn": 3.0, "nod": 4.0, "hrv": 0.50}
EMA_ALPHA = 0.25
WARN_THRESHOLD = 0.35
ALARM_THRESHOLD = 0.50
# Safety overrides (independent of the weighted score)
OVERRIDE_PERCLOS = 0.40
OVERRIDE_CLOSED_SEC = 2.0

WARN_COOLDOWN = 20
ALARM_COOLDOWN = 4
