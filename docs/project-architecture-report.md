# SignBridge AI: Project Architecture and File Guide

**Prepared:** October 26, 2026  
**Purpose:** Explain how the project files fit together, which terms and
techniques the project uses, and what reasonable alternatives exist.

## 1. Project at a glance

SignBridge AI is a prototype that recognizes a **static hand posture** and maps
its predicted class to a Devanagari character. It does not recognize motion,
words, or an official sign-language alphabet. The posture-to-character mapping
is custom and is defined in `config.py`.

The main prediction path is:

```text
Browser camera photo                         Local webcam video
        |                                             |
        v                                             v
     app.py                                  predict_webcam.py
        |                                             |
        +-------------------+-------------------------+
                            v
                    MediaPipe Hands
             detects 21 hand landmarks (x, y, z)
                            |
                            v
               Normalize to 63 numeric features
        wrist-relative coordinates, scaled to a common range
                            |
                            v
           Random Forest model + LabelEncoder
                 files in models/
                            |
                            v
              P01–P26 class -> config.py
                -> Devanagari character
```

The training-data path is:

```text
app.py camera capture                 collect_data.py webcam
         |                                      |
         +---- normalize landmarks -------------+
                          |
                          v
                  data/landmarks.csv
                          ^
                          | validated download / import
                  training_data.py
                          |
                          v
                    train_model.py
             model + encoder + metrics in models/
```

`app.py` can also launch `train_model.py` from its training section. The
standalone `predict_webcam.py` loads the same model and encoder, but uses
continuous local webcam frames rather than a browser photo.

## 2. File-by-file guide

The inventory below covers the project files present in this checkout, including
the local training CSV and the existing untracked presentation report. Runtime
environments, caches, and Git internals are generated or machine-specific and
are not treated as application source files.

### Application and Python modules

| File | Role | How it connects |
|---|---|---|
| [`app.py`](../app.py) | Main Streamlit application and current browser-facing UI. Displays the posture reference, captures photos, detects landmarks, predicts characters, collects examples, imports/exports sample CSVs, and starts model training. | Reads constants from `config.py`; uses CSV helpers in `training_data.py`; reads the model and encoder from `models/`; reads and writes `data/landmarks.csv`; launches `train_model.py`. |
| [`config.py`](../config.py) | Shared configuration: project paths, 63-feature width, 26 class IDs and character mappings, MediaPipe settings, confidence threshold, and sample target. | Imported by the app, collector, training-data helpers, trainer, and webcam predictor. It is the single source of truth for class-to-character mappings and common paths. |
| [`collect_data.py`](../collect_data.py) | Optional command-line collector using a computer-attached webcam and OpenCV. Captures detected hand landmarks for a selected class and appends them to the CSV. | Uses `config.py`; writes the same `data/landmarks.csv` that `train_model.py` reads and that `app.py` can update. |
| [`training_data.py`](../training_data.py) | Shared schema validation and CSV transfer helpers. Checks the feature columns, labels, numeric values, and finite values; supports download/export and import/merge with exact-duplicate removal. | Used by `app.py` for CSV transfer and sample-count reading. Its `FEATURE_COLUMNS` schema matches the collector and trainer’s 63 features plus `label`. |
| [`train_model.py`](../train_model.py) | Command-line model-training program. Reads feature columns and labels, encodes labels, trains a Random Forest, optionally evaluates a stratified holdout, then writes model, encoder, and metrics. | Reads `data/landmarks.csv`; uses paths and mapping from `config.py`; writes the three primary artifacts under `models/`; can be started manually or by `app.py`. |
| [`predict_webcam.py`](../predict_webcam.py) | Optional local, continuous webcam prediction program. Draws detected hand landmarks, predicts a class, smooths recent predictions, and displays the mapped character. | Loads the same model and label encoder as `app.py`; uses `config.py` for paths, mapping, detection settings, confidence threshold, and smoothing window. It does not run on a remote server without access to a webcam. |
| [`tests/test_training_data.py`](../tests/test_training_data.py) | Unit tests for training-data transfer helpers and the feature-schema width. Checks duplicate handling, invalid labels, non-finite values, and schema width. | Imports `FEATURE_COLUMNS` and `merge_training_csv` from `training_data.py`, and `FEATURE_DIM` from `config.py`. It does not test camera capture, model accuracy, or the Streamlit UI. |

### Data, models, and reference images

| File | Role | How it connects |
|---|---|---|
| [`data/landmarks.csv`](../data/landmarks.csv) | Local training examples: one row per sample, with `f0`–`f62` and a `label` such as `P04`. | Written by the app or collector; read by the validator and trainer. It is ignored by Git, so it is machine-local unless transferred using the app’s CSV export/import. The local file currently has 92 valid samples across 26 labels. |
| [`data/.gitkeep`](../data/.gitkeep) | Empty placeholder that lets Git retain the otherwise-empty `data/` directory. | The actual dataset is ignored; this placeholder keeps the directory available in a fresh checkout. |
| [`models/signbridge_model.joblib`](../models/signbridge_model.joblib) | Serialized fitted scikit-learn Random Forest classifier. | Written by `train_model.py`; loaded by `app.py` and `predict_webcam.py`. It must stay paired with the label encoder created during the same training run. |
| [`models/label_encoder.joblib`](../models/label_encoder.joblib) | Serialized `LabelEncoder` mapping between string class IDs (`P01`, etc.) and the integer class indices used by the classifier. | Written with the model by `train_model.py`; used at prediction time to turn the classifier’s integer result back into a class ID, which `config.py` then maps to a character. |
| [`models/training_metrics.txt`](../models/training_metrics.txt) | Text report from the last recorded training run: sample counts, evaluation status, and metric placeholders/results. | Written by `train_model.py`; provides context for the model artifact but does not itself affect predictions. Its current content records an 87-sample training run, so it predates the present 92-row local CSV. |
| [`models/.gitkeep`](../models/.gitkeep) | Empty placeholder for the model directory. | Keeps `models/` in source control even when generated model files are absent. |
| [`postures.jpeg`](../postures.jpeg) | Main posture reference image displayed by the Streamlit app. | Read directly by `app.py` using a path based on the project root. Its chart is a visual reference and does not define the app’s custom mapping. |
| [`assets/posture-reference.png`](../assets/posture-reference.png) | Additional posture reference image asset. | It is present in the repository, but the current Python application code does not reference it. The active app uses `postures.jpeg`. |

### Documentation, dependencies, and editor support

| File | Role | How it connects |
|---|---|---|
| [`README.md`](../README.md) | Setup, deployment, collection, and usage instructions. | Describes the project to developers and users. Some deployment metadata is stale: it names a Gradio `hf_app.py` entry point, but that file is absent from this checkout and the active application is `app.py` (Streamlit). Confirm the intended deployment before relying on those Hugging Face instructions. |
| [`docs/prediction-accuracy-report.md`](prediction-accuracy-report.md) | Earlier narrative about model reliability and evaluation limits. | Complements the metrics file, but it describes the earlier 87-sample state and should be read as historical context rather than a report of the currently repaired CSV. |
| [`docs/prediction-accuracy-report.pdf`](prediction-accuracy-report.pdf) | PDF counterpart for the prediction-accuracy report. | A presentation/distribution format of the accuracy discussion; it is not consumed by the application. |
| [`docs/presentation-report.md`](presentation-report.md) | Existing local presentation/implementation notes. | This file is currently untracked in Git. It discusses UI/deployment history and is separate from the runtime pipeline; its deployment notes should be reconciled with the current repository before presenting them as current. |
| [`docs/testing.md`](testing.md) | Reserved testing documentation file. | It is empty in this checkout and currently adds no test instructions beyond the README and the unit test file. |
| [`docs/project-architecture-report.md`](project-architecture-report.md) | This file. | Explains the architecture, file relationships, techniques, alternatives, and current limitations. |
| [`requirements.txt`](../requirements.txt) | Pinned Python dependencies for the project environment. | Supplies Python libraries used by the application, computer vision, model training, and UI. It includes Gradio and `spaces` as well as Streamlit, although no `hf_app.py` is present in this checkout. |
| [`packages.txt`](../packages.txt) | System-level Linux package list for deployment environments. | Supplies `libgl1` and `libglib2.0-0`, native graphics libraries commonly required by OpenCV/MediaPipe. It complements, rather than replaces, `requirements.txt`. |
| [`.gitignore`](../.gitignore) | Excludes virtual environments, bytecode, editor settings, the local dataset, and the accuracy-report PDF from Git. | Explains why the local CSV does not appear in normal Git changes and why machine-specific/generated files are not part of a fresh clone. |
| [`.vscode/settings.json](../.vscode/settings.json) | Local VS Code Python interpreter and analysis settings. | Points editor analysis at the project virtual environment and project root. It does not alter runtime behavior. |
| [`typings/mediapipe/__init__.pyi`](../typings/mediapipe/__init__.pyi) | Partial Python type stub for MediaPipe APIs used by the project. | Helps editor/type analysis understand the `Hands` and landmark-drawing APIs; it is not imported as the runtime MediaPipe implementation. |

## 3. Terms, techniques, and alternatives

| Term or technique | Meaning and use here | Alternative and trade-off |
|---|---|---|
| **Hand landmark** | A detected hand key point. MediaPipe Hands returns 21 points, each with `x`, `y`, and `z`; together they make 63 numeric values. | A model could classify the entire image directly (for example, a convolutional neural network). That avoids a separate landmark feature stage but generally needs more images, compute, and careful training. |
| **Feature vector** | The fixed-length numeric representation given to the classifier: `f0` through `f62`. | Raw pixels, image embeddings, joint angles, distances between selected points, or a hybrid of geometry and image data are other representations. Handcrafted geometry can be smaller but may discard useful information. |
| **Wrist-relative normalization** | Subtracts the wrist point from every landmark, reducing dependence on where the hand appears in the image. | Keep absolute image coordinates, center using the hand bounding box, or use learned image features. Absolute positions may encode camera framing; bounding-box normalization has different scale and rotation behavior. |
| **Max-absolute-value scaling** | Divides the wrist-relative coordinates by the largest absolute coordinate, making a sample’s largest magnitude equal to 1. This is the implemented scale-normalization step. | Standard scaling (mean and standard deviation), min-max scaling, palm-width normalization, or no scaling are possible. Each makes different assumptions; train and inference must use exactly the same transformation. |
| **MediaPipe Hands** | Prebuilt hand detector/landmark estimator; used on camera frames to produce the 21 points. | Other hand-pose estimators or a custom model are possible. A custom detector may better fit a domain but costs more data and engineering. |
| **Random Forest** | An ensemble of decision trees; this project trains 400 trees and uses balanced-sub-sample class weighting to reduce the effect of unequal class counts. | SVMs can work well on small feature datasets but need tuning; gradient-boosted trees may improve tabular performance but add tuning complexity; a neural network can learn richer patterns but needs substantially more representative data. No alternative is automatically more accurate for this dataset. |
| **Label encoding** | Converts text class IDs into integer targets for scikit-learn and reverses the mapping after prediction. | An explicit, versioned class-to-index table or a scikit-learn pipeline could make mapping and preprocessing more self-contained. The current classifier must always be used with its matching saved encoder. |
| **Stratified holdout split** | Reserves part of the dataset for evaluation while attempting to preserve class proportions in train and test. The code uses an 80/20 split with a fixed seed only when every class has at least five samples. | A separate session-independent test set or stratified cross-validation can give stronger evidence. Grouping by capture session is especially useful to avoid near-duplicate frames appearing in both train and test. With very few examples, no split can create reliable evidence from missing variation. |
| **Accuracy, classification report, confusion matrix** | Accuracy is the overall fraction correct; per-class precision/recall and the confusion matrix help show which classes are confused. | Macro F1, balanced accuracy, confidence calibration, and session-separated evaluation can add useful detail, especially with imbalanced classes. A single overall accuracy can hide weak classes. |
| **Raw `predict_proba` score / confidence threshold** | The app and webcam script show the highest Random Forest score; `CONFIDENCE_THRESHOLD` is 0.70. The Streamlit app warns on a low score, and the webcam path can put low-score predictions into its `UNKNOWN` smoothing history. The score is not a calibrated probability of correctness. | Calibrate scores on representative validation data, use a validated reject option, or show top-k guesses without implying certainty. A threshold should be selected from measured errors, not treated as universally reliable. |
| **Majority-vote smoothing** | The webcam script votes over a short deque of recent predicted classes to reduce frame-to-frame flicker. | Exponential smoothing, a hidden Markov model, temporal neural networks, or no smoothing are alternatives. Smoothing can delay a genuine posture change. The browser app processes a single photo, so it does not use this temporal technique. |
| **CSV storage and transfer** | A simple row-oriented, human-readable format used for collection, validation, export, and import. The importer keeps existing rows and skips exact duplicates. | SQLite is better for concurrent updates and richer metadata; Parquet is more compact/efficient for larger analytic datasets; a managed database supports centralized multi-user access. Those options add infrastructure and are unnecessary for this prototype’s current small dataset. |
| **Browser snapshot vs. continuous webcam** | Streamlit `st.camera_input()` provides a user-authorized browser photo; `predict_webcam.py` uses OpenCV to read repeated frames from a local webcam. | A custom browser video component/WebRTC stream could provide continuous remote video but adds frontend, transport, and latency complexity. A desktop GUI could avoid a browser but is less convenient to deploy remotely. |
| **Cached resources** | `app.py` caches the MediaPipe instance and model resources to avoid expensive reloads on Streamlit reruns. Model cache identity includes artifact modification time and file size. | Load on every rerun (simpler but slower), or manage explicit application-level lifecycle/versioning. Cache invalidation must remain correct when artifacts change. |

## 4. Current data and model status

The local `data/landmarks.csv` currently contains **92 samples** across 26
labels. The class counts are imbalanced:

| Classes | Samples per class |
|---|---:|
| P01 | 11 |
| P02 | 15 |
| P03 | 7 |
| P04 | 10 |
| P05 | 7 |
| P06–P26 | 2 each |

The dataset was validated against the training-data schema after correcting
five records that had tab delimiters. The current CSV is local and Git-ignored.
The existing model artifacts and `models/training_metrics.txt` describe an
earlier 87-sample run; the CSV repair does **not** retrain the model. Therefore,
the bundled model should not be described as trained on all 92 current samples.

The metrics file reports that holdout evaluation was **not performed** because
some classes had fewer than five samples. This is not a measured accuracy of
zero or 100%; it means there is no holdout accuracy result for that run. Even
after retraining, two examples per class are too few to establish generalization
to new people, lighting, cameras, or hand positions.

## 5. Important consistency notes

1. **Two normalization implementations:** `app.py`, `collect_data.py`, and
   `predict_webcam.py` each implement the same wrist-centering and max-absolute
   scaling logic. They currently agree in intent, but separate copies can drift.
   A shared normalization module would reduce that risk. The key requirement is
   that collection, training inputs, and inference all use the same feature
   definition.
2. **Deployment documentation mismatch:** The active application in this
   checkout is `app.py` using Streamlit. The README front matter and deployment
   prose refer to Gradio and `hf_app.py`, but `hf_app.py` is not in the file
   inventory. Treat those instructions as stale until the entry point and
   hosting target are confirmed.
3. **Reference image mismatch:** `app.py` reads the root-level `postures.jpeg`.
   `assets/posture-reference.png` is present but not referenced by active code.
4. **Historical metrics:** `docs/prediction-accuracy-report.md` and the saved
   metrics document an earlier 87-sample state. Current local data has 92
   samples, but the model files were not regenerated as part of the CSV repair.
5. **Custom labels:** P01–P26 and their Devanagari character mapping are
   project-specific. The visual reference and labels should not be represented
   as an official sign-language standard.

## 6. Suggested next steps

1. Collect many more varied samples for every class; prioritize the classes
   with only two examples. The configured target is 150 per class, while the
   UI recommends at least 30 varied examples as an initial practical target.
2. Retrain only after reviewing the collected labels and counts, then verify
   that the model, encoder, and metrics were generated together from the
   intended CSV.
3. Reserve independent data by person or recording session for evaluation.
   Report per-class results and a confusion matrix, not only an aggregate score.
4. Consolidate landmark normalization into a shared function (or add a test
   asserting the implementations remain equivalent) before changing feature
   preprocessing.
5. Reconcile README deployment instructions with the actual supported entry
   point, and either populate `docs/testing.md` or remove it if it is not needed.
6. Decide whether the unused Gradio/`spaces` dependencies and
   `assets/posture-reference.png` are retained for a future deployment or can
   be removed in a separate cleanup.

## 7. Plain-language summary

The camera image is not classified directly. MediaPipe first marks the 21
landmarks of one hand. The program moves those points relative to the wrist and
scales their coordinates, then gives the resulting 63 numbers to a saved
Random Forest. A saved label encoder converts the model's numeric class back to
P01–P26, and `config.py` converts that class ID to the displayed Devanagari
character. Training examples flow through the CSV into the trainer; the model
and its matching encoder then flow back into either prediction interface.

This design is a relatively lightweight, interpretable prototype compared with
training an image-recognition network end to end. Its main trade-off is that
recognition quality depends on reliable hand landmarks, consistent feature
normalization, correctly labeled samples, and enough varied data. The current
class imbalance and stale model artifacts mean that predictions should remain
provisional until retraining and independent evaluation are completed.
