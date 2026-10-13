"""
SignBridge AI - Stage 5: Real-Time Prediction (local webcam)
Uses cv2.VideoCapture(0). This will NOT work on Hugging Face Spaces
because the server has no webcam. Use app.py for the deployed version.

Usage:
    python predict_webcam.py
Press 'q' to quit.
"""

import os
import sys
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import joblib

from config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    CHARACTER_MAP,
    FEATURE_DIM,
    CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
)


def normalize_landmarks(hand_landmarks):
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    pts = pts - pts[0]
    max_val = np.max(np.abs(pts))
    if max_val < 1e-6:
        return None
    pts = pts / max_val
    return pts.flatten()


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        print("ERROR: Model files not found.")
        print(f"Expected: {MODEL_PATH}")
        print(f"Expected: {LABEL_ENCODER_PATH}")
        print("Run train_model.py first.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    history = deque(maxlen=SMOOTHING_WINDOW)
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            predicted_char = "?"
            confidence = 0.0
            hand_present = bool(results.multi_hand_landmarks)

            if hand_present:
                for hl in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS
                    )

                vec = normalize_landmarks(results.multi_hand_landmarks[0])
                if vec is not None:
                    proba = clf.predict_proba([vec])[0]
                    idx = int(np.argmax(proba))
                    confidence = float(proba[idx])
                    class_id = le.inverse_transform([idx])[0]

                    history.append(class_id if confidence >= CONFIDENCE_THRESHOLD else "UNKNOWN")

                    # Majority vote smoothing
                    if len(history) > 0:
                        smoothed, _ = Counter(history).most_common(1)[0]
                        if smoothed == "UNKNOWN":
                            predicted_char = "Unknown"
                        else:
                            predicted_char = CHARACTER_MAP.get(smoothed, smoothed)
            else:
                history.clear()

            # Overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Prediction: {predicted_char}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not hand_present:
                cv2.putText(frame, "No hand detected", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("SignBridge AI - Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Camera released.")


if __name__ == "__main__":
    main()