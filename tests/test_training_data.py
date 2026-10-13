import os
import tempfile
import unittest

import pandas as pd

from config import FEATURE_DIM
from training_data import FEATURE_COLUMNS, merge_training_csv


class TrainingDataTransferTests(unittest.TestCase):
    def make_samples(self, labels):
        rows = []
        for index, label in enumerate(labels):
            row = {column: float(index) for column in FEATURE_COLUMNS}
            row["label"] = label
            rows.append(row)
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label"])

    def test_import_merges_and_skips_exact_duplicates(self):
        initial = self.make_samples(["P01"])
        incoming = self.make_samples(["P01", "P02"])
        incoming = pd.concat([incoming, incoming.iloc[[1]]], ignore_index=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "landmarks.csv")
            initial.to_csv(csv_path, index=False)
            added, counts = merge_training_csv(
                incoming.to_csv(index=False).encode("utf-8"),
                csv_path,
            )
            self.assertEqual(added, 1)
            self.assertEqual(counts, {"P01": 1, "P02": 1})

            added_again, counts_again = merge_training_csv(
                incoming.to_csv(index=False).encode("utf-8"),
                csv_path,
            )
            self.assertEqual(added_again, 0)
            self.assertEqual(counts_again, counts)
            self.assertEqual(len(pd.read_csv(csv_path)), 2)

    def test_import_rejects_unsupported_labels(self):
        invalid = self.make_samples(["P99"])
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "data", "landmarks.csv")
            with self.assertRaisesRegex(ValueError, "unsupported posture label"):
                merge_training_csv(invalid.to_csv(index=False).encode("utf-8"), csv_path)
            self.assertFalse(os.path.exists(csv_path))

    def test_import_rejects_non_finite_features(self):
        invalid = self.make_samples(["P01"])
        invalid.loc[0, "f0"] = float("inf")
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "landmarks.csv")
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                merge_training_csv(invalid.to_csv(index=False).encode("utf-8"), csv_path)
            self.assertFalse(os.path.exists(csv_path))

    def test_feature_schema_has_expected_width(self):
        self.assertEqual(len(FEATURE_COLUMNS), FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
