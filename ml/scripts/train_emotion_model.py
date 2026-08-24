"""Train the speaker-aware CREMA-D emotion detection pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ml.src.models.emotion_model import ARTIFACT_DIR, DATASET_DIR, train_emotion_model


def main() -> None:
    metrics = train_emotion_model(DATASET_DIR, ARTIFACT_DIR, seed=42)
    summary = {
        "accuracy": metrics["evaluation"]["accuracy"],
        "macro_f1": metrics["evaluation"]["macro_f1"],
        "weighted_f1": metrics["evaluation"]["weighted_f1"],
        "model_path": metrics["model_path"],
        "train_files": metrics["dataset"]["train_files"],
        "validation_files": metrics["dataset"]["validation_files"],
        "test_files": metrics["dataset"]["test_files"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
