"""Gradio entry point for the Hugging Face Space deployment."""

import os
from functools import lru_cache

import gradio as gr
import joblib
import mediapipe as mp
import numpy as np
import spaces
from PIL import Image

from config import (
    CHARACTER_MAP,
    CONFIDENCE_THRESHOLD,
    LABEL_ENCODER_PATH,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    PROJECT_ROOT,
)


def normalize_landmarks(hand_landmarks):
    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    )
    points -= points[0]
    scale = np.max(np.abs(points))
    if scale < 1e-6:
        return None
    return (points / scale).flatten()


@lru_cache(maxsize=1)
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            "The trained classifier is missing. Add the model artifacts to the Space."
        )

    classifier = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return classifier, label_encoder, hands


@spaces.GPU(duration=15)
def predict(image):
    if image is None:
        return "Take a photo of a hand posture first.", None

    classifier, label_encoder, hands = load_resources()
    rgb_image = np.asarray(Image.fromarray(image).convert("RGB"))
    result = hands.process(rgb_image)
    if not result.multi_hand_landmarks:
        return "No hand detected. Retake the photo with your hand clearly visible.", None

    features = normalize_landmarks(result.multi_hand_landmarks[0])
    if features is None:
        return "Hand landmarks were degenerate. Please retake the photo.", None

    scores = classifier.predict_proba([features])[0]
    order = np.argsort(scores)[::-1][:5]
    rows = []
    for index in order:
        class_id = str(label_encoder.inverse_transform([int(index)])[0])
        score = float(scores[index])
        rows.append(
            [
                class_id,
                CHARACTER_MAP.get(class_id, class_id),
                f"{score * 100:.1f}%",
            ]
        )

    best_class, best_character, best_score = rows[0]
    message = (
        f"Highest-scoring match: {best_character} ({best_class}) — {best_score} score."
    )
    if float(scores[order[0]]) < CONFIDENCE_THRESHOLD:
        message += " Low score: this prediction is uncertain."
    return message, rows


reference_image = os.path.join(PROJECT_ROOT, "assets", "posture-reference.png")
with gr.Blocks(title="SignBridge AI") as demo:
    gr.Markdown(
        "# 🤟 SignBridge AI — Marathi Hand Gesture Recognition\n"
        "Static hand posture to Marathi character using this project's custom "
        "mapping; it is not an official sign-language standard."
    )
    with gr.Accordion("Gesture reference chart", open=False):
        gr.Image(
            value=reference_image,
            label="Reference hand postures",
            interactive=False,
        )
        gr.Markdown(
            "Visual guide only. The chart labels may not match SignBridge's "
            "custom P01–P26 class mapping."
        )

    with gr.Accordion("Model limitations", open=True):
        gr.Markdown(
            "This model has only a few training examples per posture. Predictions "
            "are provisional and may be wrong. Scores are model scores, not "
            "calibrated probabilities."
        )

    camera = gr.Image(
        sources=["webcam"],
        type="numpy",
        label="Take a photo of a hand posture",
    )
    predict_button = gr.Button("Predict character", variant="primary")
    prediction = gr.Markdown()
    top_matches = gr.Dataframe(
        headers=["Class ID", "Marathi character", "Score"],
        datatype=["str", "str", "str"],
        label="Top-scoring matches",
        interactive=False,
    )
    predict_button.click(
        fn=predict,
        inputs=camera,
        outputs=[prediction, top_matches],
    )


if __name__ == "__main__":
    demo.launch()
