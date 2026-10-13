---
title: SignBridge AI
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: hf_app.py
pinned: false
---

# SignBridge AI — Marathi Hand Gesture Recognition (Version 1)

Real-time recognition of 26 static hand postures, each mapped to a Marathi
character (Devanagari). The posture set is *inspired by* ASL hand positions,
but the posture-to-character mapping is **custom** and **not** an official
ASL or Indian Sign Language standard.

## Stack
- Python 3.10 / 3.11
- OpenCV, MediaPipe (21 hand landmarks)
- NumPy, scikit-learn (RandomForest)
- Streamlit (UI + Hugging Face deployment)
- Joblib (model persistence)

## Project layout

The Streamlit UI includes `assets/posture-reference.png` as a visual gesture
guide. Its labels are for reference; the project uses the custom mapping in
`config.py`, which may not match the chart.

The Hugging Face Space uses `hf_app.py` as its Gradio entry point. Run
`streamlit run app.py` for the local Streamlit interface.
The deployed app requests ZeroGPU only while running a prediction.

## Train with real posture samples

The app cannot recognize gestures until it has a model trained on real posture
samples. When the model is missing, open **Collect real posture samples and
train** in the Streamlit app. Choose a class, capture a hand photo, and save it.
Repeat with different photos for at least two classes (at least 2 samples per
class to enable training; 30 or more varied samples per class are recommended),
then click **Train / retrain model**. With fewer than 5 samples per class, the
model is trained on all samples and no holdout accuracy is reported; predictions
from such a sparse model are provisional and may be incorrect.

Alternatively, collect samples using the local webcam scripts and train from
PowerShell:

```powershell
.\.venv\Scripts\python.exe collect_data.py --class P01 --samples 150
.\.venv\Scripts\python.exe collect_data.py --class P02 --samples 150
.\.venv\Scripts\python.exe train_model.py
```

Repeat data collection for each supported posture, performing the posture
consistently while collecting. The app only recognizes classes present in the
collected training data.