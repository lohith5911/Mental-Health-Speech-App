"""Evaluate the saved V4 emotion model on the held-out CREMA-D test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.models.emotion_model import (
    DATASET_DIR,
    EMOTION_MAP,
    build_speaker_aware_split,
    list_audio_files,
    parse_emotion_from_filename,
    parse_speaker_id,
)
from ml.src.models.v4_emotion_model import get_v4_predictor

EVALUATION_SEED = 42
EMOTION_CLASSES = tuple(EMOTION_MAP.values())
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "models" / "v4" / "evaluation.json"


def verify_speaker_disjointness(split: dict[str, Sequence[Path]]) -> None:
    """Fail evaluation if the held-out speakers overlap training speakers."""
    train_speakers = {parse_speaker_id(path.name) for path in split["train"]}
    validation_speakers = {parse_speaker_id(path.name) for path in split["validation"]}
    test_speakers = {parse_speaker_id(path.name) for path in split["test"]}
    if train_speakers & test_speakers or validation_speakers & test_speakers:
        raise ValueError("Evaluation split is not speaker-independent.")


def calculate_evaluation_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Calculate deterministic overall, per-class, and confusion-matrix metrics."""
    labels = list(EMOTION_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            emotion: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, emotion in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_evaluation_payload(
    metrics: dict[str, Any],
    *,
    test_files: int,
    test_speakers: int,
    split_seed: int = EVALUATION_SEED,
) -> dict[str, Any]:
    """Build the persisted evaluation schema with explicit reproducibility metadata."""
    return {
        "model_version": "v4",
        "dataset": "CREMA-D",
        "evaluation_protocol": "speaker-independent",
        "split_seed": split_seed,
        "test_files": test_files,
        "test_speakers": test_speakers,
        "classes": list(EMOTION_CLASSES),
        "speaker_separation_verified": True,
        "metrics": {
            key: metrics[key]
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            )
        },
        "per_class": metrics["per_class"],
        "test_support": {
            emotion: metrics["per_class"][emotion]["support"]
            for emotion in EMOTION_CLASSES
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }


def evaluate_v4(
    data_dir: Path = DATASET_DIR,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Evaluate saved V4 predictions on the exact held-out seed-42 test split."""
    split = build_speaker_aware_split(list_audio_files(data_dir), seed=EVALUATION_SEED)
    verify_speaker_disjointness(split)

    predictor = get_v4_predictor()
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    prediction_probabilities: list[dict[str, float]] = []
    confidences: list[float] = []
    speaker_ids: list[str] = []
    for file_path in split["test"]:
        emotion_code = parse_emotion_from_filename(file_path.name)
        if emotion_code is None:
            raise ValueError(f"Unable to parse emotion label from {file_path.name}.")
        prediction = predictor.predict(file_path)
        true_labels.append(EMOTION_MAP[emotion_code])
        predicted_labels.append(str(prediction["emotion"]))
        prediction_probabilities.append(dict(prediction["probabilities"]))
        confidences.append(float(prediction["confidence"]))
        speaker_ids.append(parse_speaker_id(file_path.name))

    metrics = calculate_evaluation_metrics(true_labels, predicted_labels)
    test_speakers = len({parse_speaker_id(path.name) for path in split["test"]})
    payload = build_evaluation_payload(
        metrics,
        test_files=len(split["test"]),
        test_speakers=test_speakers,
    )
    payload["prediction_summary"] = {
        "probability_records": len(prediction_probabilities),
        "confidence_records": len(confidences),
        "speaker_records": len(speaker_ids),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    payload = evaluate_v4()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
