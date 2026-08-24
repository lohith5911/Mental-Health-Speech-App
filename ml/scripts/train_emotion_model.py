"""Train the speaker-aware CREMA-D emotion detection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ml.src.models.emotion_model import (
    ARTIFACT_DIR,
    DATASET_DIR,
    VERSION_2_ARTIFACT_DIR,
    train_emotion_model,
    train_emotion_model_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CREMA-D emotion detection models.")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Which model version to train.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Optional explicit artifact directory.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or (ARTIFACT_DIR if args.version == "v1" else VERSION_2_ARTIFACT_DIR)
    trainer = train_emotion_model if args.version == "v1" else train_emotion_model_v2
    metrics = trainer(DATASET_DIR, artifact_dir, seed=42)
    summary = {
        "version": metrics["version"],
        "feature_dimension": metrics["feature_dimension"],
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
