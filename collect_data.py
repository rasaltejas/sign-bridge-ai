"""
SignBridge AI - Stage 3: Data Collection
Collects normalized hand-landmark vectors for a chosen class ID.

Usage:
    python collect_data.py --class P01
    python collect_data.py --class P01 --samples 150
Press 'q' to stop early (samples collected so far are still saved).
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

from config import (
    RAW_DATA_PATH,
    DATA_DIR,
    FEATURE_DIM,
    CHARACTER_MAP,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
)

# ---------------------------------------------------------------------------
# Landmark normalization
# ---------------------------------------------------------------------------
def normalize_landmarks(hand_landmarks):
    """
    Convert a MediaPipe hand_landmarks object into a 63-dim numpy vector.

    Steps:
    1. Read 21 (x, y, z) points -> shape (21, 3)
    2. Subtract wrist (landmark 0) so the vector is position-independent
    3. Divide by the max absolute value so it is scale-independent
    Returns None if the vector is degenerate (all zeros).
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )  # (21, 3)

    # Position invariance: move wrist to origin
    pts = pts - pts[0]

    # Scale invariance: normalize by the largest magnitude
    max_val = np.max(np.abs(pts))
    if max_val < 1e-6:
        return None
    pts = pts / max_val

    return pts.flatten()  # (63,)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def ensure_csv_header(path):
    """Create the CSV with a header if it does not yet exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(FEATURE_DIM)] + ["label"]
            writer.writerow(header)


def append_sample(path, features, label):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [label])


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
def collect(class_id, target_samples):
    if class_id not in CHARACTER_MAP:
        print(f"ERROR: '{class_id}' is not a valid class ID.")
        print("Valid IDs:", ", ".join(CHARACTER_MAP.keys()))
        sys.exit(1)

    marathi_char = CHARACTER_MAP[class_id]
    ensure_csv_header(RAW_DATA_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check camera permissions or close other apps.")
        sys.exit(1)

    collected = 0
    print(f"Collecting class {class_id} ({marathi_char}) - target {target_samples} samples")
    print("Press 'q' to stop early. Press SPACE to capture the current frame.")

    last_capture_time = 0.0
    capture_interval = 0.05  # at most 20 samples per second

    try:
        while collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_present = bool(results.multi_hand_landmarks)

            if hand_present:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

            # Auto-capture when a hand is present (throttled)
            now = time.time()
            if hand_present and (now - last_capture_time) > capture_interval:
                vec = normalize_landmarks(results.multi_hand_landmarks[0])
                if vec is not None:
                    append_sample(RAW_DATA_PATH, vec, class_id)
                    collected += 1
                    last_capture_time = now

            # Overlay info
            status = "HAND DETECTED" if hand_present else "NO HAND"
            color = (0, 255, 0) if hand_present else (0, 0, 255)
            cv2.putText(frame, f"Class: {class_id} ({marathi_char})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {collected}/{target_samples}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("SignBridge AI - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and hand_present:
                vec = normalize_landmarks(results.multi_hand_landmarks[0])
                if vec is not None:
                    append_sample(RAW_DATA_PATH, vec, class_id)
                    collected += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print(f"\nDone. Collected {collected} samples for {class_id}. Saved to {RAW_DATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_id", required=True,
                        help="Class ID, e.g. P01")
    parser.add_argument("--samples", type=int, default=150,
                        help="Number of samples to collect (default 150)")
    args = parser.parse_args()
    collect(args.class_id.upper(), args.samples)