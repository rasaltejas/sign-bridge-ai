"""
SignBridge AI - Central Configuration
All class IDs, Marathi character mappings, paths, and thresholds live here.
Change the CHARACTER_MAP values to customize which Marathi letter maps to which posture.
"""

import os

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so scripts work from any CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "signbridge_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "training_metrics.txt")

# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
# MediaPipe returns 21 landmarks, each with (x, y, z) = 63 raw values.
# We normalize relative to the wrist (landmark 0) and scale by the max
# distance, giving 63 normalized features. We keep all 63.
FEATURE_DIM = 63
NUM_LANDMARKS = 21

# ---------------------------------------------------------------------------
# Class IDs and Marathi character mapping
# ---------------------------------------------------------------------------
# These 26 IDs correspond to the 26 custom hand postures (inspired by ASL
# hand positions). The Marathi characters below are an EDITABLE EXAMPLE
# MAPPING - they are NOT an official sign-language standard.
# Change the values on the right to whatever Marathi letters you decide.

CHARACTER_MAP = {
    "P01": "अ",
    "P02": "आ",
    "P03": "इ",
    "P04": "ई",
    "P05": "उ",
    "P06": "ऊ",
    "P07": "ए",
    "P08": "ऐ",
    "P09": "ओ",
    "P10": "औ",
    "P11": "क",
    "P12": "ख",
    "P13": "ग",
    "P14": "घ",
    "P15": "च",
    "P16": "छ",
    "P17": "ज",
    "P18": "झ",
    "P19": "ट",
    "P20": "ठ",
    "P21": "ड",
    "P22": "ढ",
    "P23": "ण",
    "P24": "त",
    "P25": "थ",
    "P26": "द",
}

# Ordered list of class IDs (this ordering is used everywhere)
CLASS_IDS = list(CHARACTER_MAP.keys())

# ---------------------------------------------------------------------------
# Runtime thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60      # below this -> "Unknown"
SMOOTHING_WINDOW = 5             # majority vote over last N predictions
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
SAMPLES_PER_CLASS_TARGET = 150   # recommended; you can stop earlier