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

Image-based recognition of 26 static hand postures, each mapped to a Marathi
character (Devanagari). The posture set is *inspired by* ASL hand positions,
but the posture-to-character mapping is **custom** and **not** an official
ASL or Indian Sign Language standard.

## Stack
- Python 3.10 / 3.11
- OpenCV, MediaPipe (21 hand landmarks)
- NumPy, scikit-learn (RandomForest)
- Streamlit (local UI) and Gradio (Hugging Face Space UI)
- Joblib (model persistence)

## Project layout

Both app UIs show `postures.jpeg` from the project root as a visual gesture
guide. Its labels are for reference; the project uses the custom mapping in
`config.py`, which may not match the chart.

The Hugging Face Space uses `hf_app.py` as its Gradio entry point. Inference
uses CPU-based MediaPipe and scikit-learn, but this Space uses ZeroGPU hardware
and requires a `@spaces.GPU` function for startup. Hugging Face may queue each
prediction for GPU allocation even though this model does not use the GPU, so
the Space can be less predictable for a live presentation. The Streamlit app
does not request a GPU and can be deployed to Streamlit Community Cloud instead.

## Deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io/> using the GitHub account that can
   access `rasaltejas/sign-bridge-ai`.
2. Choose **Create app** → **Yup, I have an app**.
3. Select repository `rasaltejas/sign-bridge-ai`, branch `main`, and file path
   `app.py`.
4. In **Advanced settings**, select Python **3.11**, then deploy.

Community Cloud installs the dependencies from the root `requirements.txt`.
The root `packages.txt` supplies the Linux graphics libraries required by
MediaPipe/OpenCV.
After it finishes building, open the app URL and allow browser camera access.
The app uses `st.camera_input()` and runs inference on CPU, so no ZeroGPU quota
or Hugging Face GPU queue is involved.

For local development, run `streamlit run app.py` from the project root.

## Train with real posture samples

The app cannot recognize gestures until it has a model trained on real posture
samples. When the model is missing, open **Collect real posture samples and
train** in the Streamlit app. Choose a class, capture a hand photo, and save it.
Repeat with different photos for at least two classes (at least 2 samples per
class to enable training; 30 or more varied samples per class are recommended),
then click **Train / retrain model**. With fewer than 5 samples per class, the
model is trained on all samples and no holdout accuracy is reported. The current
bundled model has only 2–3 samples per class, so predictions are provisional
and may be incorrect; collect more varied real samples and retrain before
presenting recognition accuracy as validated. The app also uses a more
conservative confidence threshold so uncertain matches are labeled as low
confidence instead of being presented as strong answers.

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


<img width="935" height="1600" alt="postures" src="https://github.com/user-attachments/assets/d338eef2-8b34-4d1f-9e62-92a69b0a7c12" />

