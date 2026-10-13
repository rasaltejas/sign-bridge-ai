"""
SignBridge AI - Stage 4: Model Training
Loads data/landmarks.csv, trains a scikit-learn classifier, saves model + label encoder + metrics.

Usage:
    python train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    RAW_DATA_PATH,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    MODELS_DIR,
    FEATURE_DIM,
    CHARACTER_MAP,
)

RANDOM_SEED = 42
MIN_SAMPLES_FOR_VALIDATION = 5


def load_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Data file not found at {RAW_DATA_PATH}")
        print("Run collect_data.py first for at least two classes.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH)

    feature_cols = [f"f{i}" for i in range(FEATURE_DIM)]
    missing = [c for c in feature_cols + ["label"] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        sys.exit(1)

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # Drop rows with NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        dropped = (~mask).sum()
        print(f"Warning: dropping {dropped} rows with NaN/Inf.")
        X, y = X[mask], y[mask]

    return X, y


def main():
    X, y = load_dataset()

    unique_labels = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(unique_labels)} classes.")
    print("Classes:", unique_labels)

    if len(unique_labels) < 2:
        print("ERROR: Need at least 2 classes to train a classifier.")
        sys.exit(1)

    # Warn if any class has too few samples
    counts = pd.Series(y).value_counts()
    print("\nSamples per class:")
    print(counts.to_string())

    if counts.min() < 2:
        insufficient = counts[counts < 2]
        print("\nERROR: Each class must have at least 2 real samples:")
        print(insufficient.to_string())
        sys.exit(1)

    low = counts[counts < 15]
    if not low.empty:
        print("\nWARNING: These classes have fewer than 15 samples and may train poorly:")
        print(low.to_string())

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    evaluate_model = counts.min() >= MIN_SAMPLES_FOR_VALIDATION
    if evaluate_model:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
        )
        print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
    else:
        X_train, y_train = X, y_enc
        X_test = np.empty((0, FEATURE_DIM), dtype=np.float32)
        y_test = np.array([], dtype=y_enc.dtype)
        print(
            "\nNot enough samples per class for a reliable holdout evaluation; "
            "training on all collected samples."
        )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    if evaluate_model:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

        present_labels = sorted(set(y_test_labels) | set(y_pred_labels))
        report = classification_report(
            y_test_labels, y_pred_labels, labels=present_labels, zero_division=0
        )
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=present_labels)
        accuracy_text = f"{acc:.4f}"

        print(f"\nTest accuracy: {acc:.4f}")
        print("\nClassification report:")
        print(report)
        print("\nConfusion matrix (rows=true, cols=pred):")
        print("Labels:", present_labels)
        print(cm)
    else:
        acc = None
        report = "Not evaluated: fewer than 5 samples in at least one class."
        present_labels = []
        cm = np.empty((0, 0), dtype=int)
        accuracy_text = "Not evaluated (fewer than 5 samples per class)"
        print(f"\n{accuracy_text}.")

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    with open(METRICS_PATH, "w") as f:
        f.write("SignBridge AI - Training Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Classes: {len(unique_labels)}\n")
        f.write(f"Class list: {unique_labels}\n")
        f.write(f"Test accuracy: {accuracy_text}\n\n")
        f.write("Samples per class:\n")
        f.write(counts.to_string() + "\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(f"Labels: {present_labels}\n")
        f.write(np.array2string(cm) + "\n")

    print(f"\nSaved model to        {MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved metrics to      {METRICS_PATH}")


if __name__ == "__main__":
    main()