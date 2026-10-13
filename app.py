"""
SignBridge AI - Stage 6: Streamlit UI
Works BOTH locally and on Hugging Face Spaces because it uses
st.camera_input(), which reads from the USER'S BROWSER camera.

Run locally:
    streamlit run app.py
"""

import csv
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    RAW_DATA_PATH,
    PROJECT_ROOT,
    CHARACTER_MAP,
    FEATURE_DIM,
    CONFIDENCE_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
)
from training_data import export_training_csv, merge_training_csv, read_training_csv

# MediaPipe imports are slow; cache the model loaders
def model_artifact_version():
    try:
        return tuple(
            (os.stat(path).st_mtime_ns, os.path.getsize(path))
            for path in (MODEL_PATH, LABEL_ENCODER_PATH)
        )
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_classifier(artifact_version):
    import joblib
    if artifact_version is None:
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)


@st.cache_resource(show_spinner=False)
def load_mediapipe():
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,          # single image from camera_input
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return hands, mp


def normalize_landmarks(hand_landmarks):
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    pts = pts - pts[0]
    max_val = np.max(np.abs(pts))
    if max_val < 1e-6:
        return None
    return (pts / max_val).flatten()


def get_training_sample_counts():
    if not os.path.exists(RAW_DATA_PATH):
        return {}

    samples = read_training_csv(RAW_DATA_PATH)
    return samples["label"].value_counts().to_dict()


def save_training_sample(features, class_id):
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    file_exists = os.path.exists(RAW_DATA_PATH) and os.path.getsize(RAW_DATA_PATH) > 0
    with open(RAW_DATA_PATH, "a", newline="", encoding="utf-8") as sample_file:
        writer = csv.writer(sample_file)
        if not file_exists:
            writer.writerow([f"f{i}" for i in range(FEATURE_DIM)] + ["label"])
        writer.writerow(list(features) + [class_id])


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SignBridge AI", page_icon="🤟", layout="centered")

st.title("🤟 SignBridge AI — Marathi Hand Gesture Recognition")
st.caption("Version 1 — static hand posture → Marathi character (custom mapping)")

st.warning(
    "**Disclaimer:** This app recognizes a *custom* posture-to-character mapping "
    "created for this project. It is not an official ASL or Indian Sign Language standard."
)

with st.expander("Gesture reference chart", expanded=True):
    st.image(
        os.path.join(PROJECT_ROOT, "postures.jpeg"),
        caption="Reference image supplied for learning the hand shapes.",
        use_column_width=True,
    )
    st.caption(
        "This chart is a visual reference only. SignBridge uses its own P01–P26 "
        "character mapping, so the chart labels may not match the character "
        "assigned to each class in this app."
    )

with st.expander("How to use", expanded=True):
    st.markdown(
        """
        1. Allow camera access when your browser asks.
        2. Click **Take Photo** and hold one of the trained hand postures.
        3. Keep the full hand inside frame; the 21 landmark points and connections are drawn to help you position the hand correctly.
        4. The highest-scoring Marathi character and its score appear below.
        5. Low scores are shown with a warning; the app still displays the best match.
        """
    )

# Load model
clf, le = load_classifier(model_artifact_version())
hands, mp = load_mediapipe()

model_missing = clf is None or le is None
try:
    sample_counts = get_training_sample_counts()
except (OSError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not read training data: {error}")
    st.stop()

saved_class_id = st.session_state.pop("last_saved_training_class", None)
if saved_class_id is not None:
    st.success(
        f"Saved one sample for {saved_class_id}. "
        f"The CSV now contains {sample_counts.get(saved_class_id, 0)} "
        f"{saved_class_id} samples."
    )
import_result = st.session_state.pop("last_imported_training_samples", None)
if import_result is not None:
    added, imported_counts = import_result
    st.success(
        f"Imported {added} new training samples; exact duplicates were skipped. "
        f"Current totals: "
        + ", ".join(
            f"{class_id}: {sample_counts.get(class_id, 0)}"
            for class_id in sorted(imported_counts)
        )
    )
st.caption(f"Training sample CSV: `{RAW_DATA_PATH}`")

if model_missing:
    st.warning(
        "No trained model is available yet. Collect samples for at least two "
        "different postures below, then train the model."
    )
else:
    low_sample_classes = {
        class_id: int(sample_counts.get(class_id, 0))
        for class_id in le.classes_
        if sample_counts.get(class_id, 0) < 10
    }
    if low_sample_classes:
        st.warning(
            "This model is trained with very few examples for some classes "
            f"({len(low_sample_classes)} classes have fewer than 10). "
            "Predictions are provisional and may be wrong. Collect more varied "
            "samples below, then retrain."
        )

st.caption(f"Model files: `{MODEL_PATH}` and `{LABEL_ENCODER_PATH}`.")

with st.expander(
    "Collect more posture samples and train" if not model_missing
    else "Collect real posture samples and train",
    expanded=model_missing,
):
        st.markdown(
            "Choose a class, hold that posture in view, take a photo, and save it. "
            "Capture different examples with small changes in hand position, "
            "distance, and lighting. Aim for at least 30 varied real photos per "
            "class; two samples per class are not enough for reliable recognition."
        )

        if sample_counts:
            counts_frame = pd.DataFrame(
                [
                    {
                        "Class": class_id,
                        "Marathi character": CHARACTER_MAP[class_id],
                        "Samples": int(sample_counts.get(class_id, 0)),
                    }
                    for class_id in CHARACTER_MAP
                    if sample_counts.get(class_id, 0)
                ]
            )
            st.dataframe(counts_frame, hide_index=True, use_container_width=True)
        else:
            st.info("No real training samples collected yet.")

        st.markdown("#### Transfer training samples between cloud and computer")
        st.markdown(
            "Download the full training CSV from the app where you collected "
            "samples, then upload that file in the app where you want to add "
            "them. Import keeps existing samples and skips exact duplicates."
        )
        st.download_button(
            "Download all training samples (CSV)",
            data=export_training_csv(RAW_DATA_PATH),
            file_name="signbridge_training_samples.csv",
            mime="text/csv",
            key="download_training_samples",
        )
        imported_file = st.file_uploader(
            "Choose a SignBridge training CSV to import",
            type=["csv"],
            key="upload_training_samples",
        )
        if imported_file is not None and st.button(
            "Import training samples",
            key="import_training_samples",
        ):
            try:
                added, imported_counts = merge_training_csv(
                    imported_file.getvalue(),
                    RAW_DATA_PATH,
                )
            except (
                OSError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                ValueError,
            ) as error:
                st.error(f"Could not import training samples: {error}")
            else:
                st.session_state["last_imported_training_samples"] = (
                    added,
                    imported_counts,
                )
                st.rerun()

        class_id = st.selectbox(
            "Posture class",
            list(CHARACTER_MAP),
            format_func=lambda key: f"{key} — {CHARACTER_MAP[key]}",
        )
        sample_image = st.camera_input(
            f"Capture posture {class_id} ({CHARACTER_MAP[class_id]})",
            key=f"training_camera_{class_id}",
        )
        if sample_image is not None and st.button("Save this training sample"):
            image_bytes = sample_image.getvalue()
            sample_hash = hashlib.sha256(image_bytes).hexdigest()
            saved_samples = st.session_state.setdefault("saved_training_samples", set())
            if (class_id, sample_hash) in saved_samples:
                st.warning("This exact photo was already saved for this posture.")
            else:
                frame = np.array(Image.open(sample_image).convert("RGB"))
                sample_results = hands.process(frame)
                if not sample_results.multi_hand_landmarks:
                    st.warning("No hand detected. Retake the photo with your hand clearly visible.")
                else:
                    for hand_landmarks in sample_results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )
                    features = normalize_landmarks(sample_results.multi_hand_landmarks[0])
                    if features is None:
                        st.warning("The detected hand landmarks were degenerate. Please retake the photo.")
                    else:
                        save_training_sample(features, class_id)
                        saved_samples.add((class_id, sample_hash))
                        st.session_state["last_saved_training_class"] = class_id
                        st.rerun()

        collected_classes = [
            key for key in CHARACTER_MAP
            if sample_counts.get(key, 0) > 0
        ]
        classes_ready = [
            key for key in collected_classes
            if sample_counts.get(key, 0) >= 2
        ]
        can_train = len(collected_classes) >= 2 and len(classes_ready) == len(collected_classes)
        if not can_train:
            st.caption(
                "Collect at least 2 samples for each class you have started, "
                "and collect samples for at least 2 classes."
            )
        elif min(sample_counts[key] for key in classes_ready) < 10:
            st.warning(
                "Training is enabled, but the model is unlikely to generalize "
                "well until each class has more varied samples."
            )

        if st.button("Train / retrain model", disabled=not can_train):
            result = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "train_model.py")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                st.error("Model training failed. Review the output below.")
                st.code(result.stderr or result.stdout)
            else:
                st.success("Model trained successfully. Reloading the app with the new model.")
                st.code(result.stdout)
                st.rerun()

if model_missing:
    st.stop()

# Camera input
img_file = st.camera_input("Show a hand posture to the camera")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    frame = np.array(image)

    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.image(frame, caption="Captured frame with 21 hand landmarks", use_column_width=True)

    if results.multi_hand_landmarks:
        vec = normalize_landmarks(results.multi_hand_landmarks[0])
        if vec is None:
            st.warning("Hand detected but landmarks were degenerate. Try again.")
        else:
            proba = clf.predict_proba([vec])[0]
            idx = int(np.argmax(proba))
            confidence = float(proba[idx])
            class_id = le.inverse_transform([idx])[0]
            marathi = CHARACTER_MAP.get(class_id, class_id)

            with col2:
                st.metric(
                    "Highest-scoring character",
                    marathi,
                    f"{confidence*100:.1f}% score",
                )
                st.write(f"Class ID: `{class_id}`")
                st.progress(min(confidence, 1.0))
                if confidence < CONFIDENCE_THRESHOLD:
                    st.info(
                        "This is the model's highest-scoring match, but its score "
                        "is low. Treat the prediction as uncertain and try again."
                    )

                # Show top-3 (best-effort, not calibrated probabilities)
                order = np.argsort(proba)[::-1][:3]
                top3 = pd.DataFrame({
                    "Class": [le.inverse_transform([i])[0] for i in order],
                    "Char": [CHARACTER_MAP.get(le.inverse_transform([i])[0], "?") for i in order],
                    "Score": [f"{proba[i]*100:.1f}%" for i in order],
                })
                st.caption("Top-3 raw scores (not calibrated probabilities):")
                st.table(top3)
    else:
        st.warning("No hand detected. Move your hand into the frame and retake the photo.")

st.divider()

with st.expander("Supported characters (custom mapping)"):
    rows = [{"Class ID": k, "Marathi character": v} for k, v in CHARACTER_MAP.items()]
    st.table(pd.DataFrame(rows))
    st.caption(
        "Marathi has more than 26 characters. Version 1 supports only the "
        "26 characters explicitly listed above. Recognition works only for "
        "classes that have actually been collected and trained."
    )

with st.expander("Limitations"):
    st.markdown(
        """
        - Only static postures are recognized (no motion or word signs).
        - Accuracy depends on lighting, hand distance, and how much training data you collected.
        - Browser camera input is snapshot-based here, not continuous video.
        - The confidence score is a model score, not a calibrated probability.
        """
    )

st.caption("SignBridge AI — university prototype. Built with MediaPipe + scikit-learn + Streamlit.")