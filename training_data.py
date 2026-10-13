"""Validation and transfer helpers for collected hand-landmark samples."""

import io
import os

import numpy as np
import pandas as pd

from config import CHARACTER_MAP, FEATURE_DIM


FEATURE_COLUMNS = [f"f{i}" for i in range(FEATURE_DIM)]
TRAINING_COLUMNS = FEATURE_COLUMNS + ["label"]


def empty_training_dataframe():
    return pd.DataFrame(columns=TRAINING_COLUMNS)


def validate_training_dataframe(samples):
    if list(samples.columns) != TRAINING_COLUMNS:
        raise ValueError(
            "CSV columns do not match the SignBridge training format "
            f"(expected f0–f{FEATURE_DIM - 1} and label)."
        )

    validated = samples.copy()
    if validated["label"].isna().any() or not validated["label"].isin(CHARACTER_MAP).all():
        raise ValueError("CSV contains a missing or unsupported posture label.")

    try:
        validated[FEATURE_COLUMNS] = validated[FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("CSV landmark features must all be numeric.") from error

    if not np.isfinite(validated[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("CSV landmark features cannot contain NaN or infinity.")

    return validated


def read_training_csv(source):
    return validate_training_dataframe(pd.read_csv(source))


def export_training_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        samples = read_training_csv(path)
    else:
        samples = empty_training_dataframe()
    return samples.to_csv(index=False).encode("utf-8")


def merge_training_csv(uploaded_bytes, path):
    imported = read_training_csv(io.BytesIO(uploaded_bytes))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        existing = read_training_csv(path)
    else:
        existing = empty_training_dataframe()

    combined = pd.concat(
        [
            existing.assign(_source=0),
            imported.assign(_source=1),
        ],
        ignore_index=True,
    )
    unique = combined.drop_duplicates(subset=TRAINING_COLUMNS, keep="first")
    new_samples = unique.loc[unique["_source"] == 1, TRAINING_COLUMNS]

    if not new_samples.empty:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_samples.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path) or os.path.getsize(path) == 0,
            index=False,
        )

    merged = pd.concat([existing, new_samples], ignore_index=True)
    counts = {
        str(class_id): int(count)
        for class_id, count in merged["label"].value_counts().items()
    }
    return len(new_samples), counts
