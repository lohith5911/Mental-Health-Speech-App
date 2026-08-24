"""Speaker-aware CREMA-D emotion training and inference pipeline."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.src.features.features import extract_mfcc_stats
from ml.src.preprocessing.audio import load_audio

DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "crema-d" / "AudioWAV"
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "emotion_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

EMOTION_ORDER = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def parse_speaker_id(filename: str | Path) -> str:
    """Return the CREMA-D speaker identifier from the filename."""
    stem = Path(filename).stem
    if not stem:
        raise ValueError(f"Invalid filename for speaker parsing: {filename}")
    return stem.split("_")[0]


def parse_emotion_from_filename(filename: str | Path) -> str | None:
    """Parse the emotion code from a CREMA-D filename."""
    stem = Path(filename).stem
    if not stem:
        return None

    parts = stem.split("_")
    if len(parts) < 3:
        return None

    code = parts[-2].upper()
    if code in EMOTION_MAP:
        return code

    if parts[-1].upper() in EMOTION_MAP:
        return parts[-1].upper()

    return None


def list_audio_files(data_dir: Path) -> list[Path]:
    """Return the discovered WAV files for the dataset."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return sorted(path for path in data_dir.rglob("*.wav") if path.is_file())


def build_speaker_aware_split(files: list[Path], seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15) -> dict[str, list[Path]]:
    """Split by speaker ID so each speaker appears in only one split."""
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Validation and test ratios must be between 0 and 1.")

    speaker_to_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        speaker_to_files[parse_speaker_id(file_path.name)].append(file_path)

    speakers = sorted(speaker_to_files)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    total_speakers = len(speakers)
    train_end = math.floor(total_speakers * 0.70)
    val_end = train_end + math.floor(total_speakers * 0.15)

    train_speakers = speakers[:train_end]
    validation_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]

    split = {
        "train": [path for speaker in train_speakers for path in speaker_to_files[speaker]],
        "validation": [path for speaker in validation_speakers for path in speaker_to_files[speaker]],
        "test": [path for speaker in test_speakers for path in speaker_to_files[speaker]],
    }
    return split


def load_feature_matrix(files: list[Path]) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract MFCC statistics from audio files and return arrays plus metadata."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    corrupted: list[str] = []

    for file_path in files:
        emotion = parse_emotion_from_filename(file_path.name)
        if emotion is None:
            corrupted.append(str(file_path.relative_to(DATASET_DIR)))
            continue

        try:
            waveform, sample_rate = load_audio(file_path)
            feature_vector = extract_mfcc_stats(waveform, sample_rate=sample_rate, n_mfcc=13)
            features.append(feature_vector)
            labels.append(emotion)
        except Exception:
            corrupted.append(str(file_path.relative_to(DATASET_DIR)))

    if not features:
        raise ValueError("No valid audio features were extracted from the dataset.")

    feature_matrix = np.vstack(features).astype(np.float32)
    return feature_matrix, labels, corrupted


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Compute evaluation metrics for the speaker-independent test set."""
    labels = EMOTION_ORDER
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[EMOTION_MAP[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_emotion = {}
    for idx, label in enumerate(labels):
        per_emotion[label] = {
            "precision": report[EMOTION_MAP[label]]["precision"],
            "recall": report[EMOTION_MAP[label]]["recall"],
            "f1": report[EMOTION_MAP[label]]["f1-score"],
            "support": report[EMOTION_MAP[label]]["support"],
        }

    return {
        "accuracy": float(accuracy),
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_emotion": per_emotion,
        "label_order": labels,
    }


def train_emotion_model(data_dir: Path = DATASET_DIR, artifact_dir: Path = ARTIFACT_DIR, seed: int = 42) -> dict[str, Any]:
    """Train a speaker-aware SVM pipeline and save the model artifact."""
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    train_X, train_y, train_corrupt = load_feature_matrix(split["train"])
    val_X, val_y, val_corrupt = load_feature_matrix(split["validation"])
    test_X, test_y, test_corrupt = load_feature_matrix(split["test"])

    labels = sorted(set(train_y + val_y + test_y))
    label_index = {label: idx for idx, label in enumerate(labels)}
    label_mapping = {label: EMOTION_MAP[label] for label in labels}

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=5.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(train_X, train_y)

    val_pred = model.predict(val_X)
    test_pred = model.predict(test_X)
    metrics = compute_classification_metrics(test_y, test_pred)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "pipeline": model,
        "label_mapping": label_mapping,
        "feature_config": {"n_mfcc": 13, "feature_dim": train_X.shape[1], "target_sr": 16000},
        "emotion_order": EMOTION_ORDER,
        "speakers": {
            "train": len({parse_speaker_id(path.name) for path in split["train"]}),
            "validation": len({parse_speaker_id(path.name) for path in split["validation"]}),
            "test": len({parse_speaker_id(path.name) for path in split["test"]}),
        },
        "corrupted": train_corrupt + val_corrupt + test_corrupt,
    }

    joblib.dump(artifact_payload, artifact_dir / "emotion_pipeline.joblib")

    metrics_payload = {
        "dataset": {
            "total_wav_files": len(all_files),
            "train_files": len(split["train"]),
            "validation_files": len(split["validation"]),
            "test_files": len(split["test"]),
            "corrupted_files": len(train_corrupt + val_corrupt + test_corrupt),
            "speaker_count": len({parse_speaker_id(path.name) for path in all_files}),
        },
        "split": {
            "train_speakers": len({parse_speaker_id(path.name) for path in split["train"]}),
            "validation_speakers": len({parse_speaker_id(path.name) for path in split["validation"]}),
            "test_speakers": len({parse_speaker_id(path.name) for path in split["test"]}),
        },
        "evaluation": metrics,
        "model_path": str(artifact_dir / "emotion_pipeline.joblib"),
        "validation_prediction_examples": {
            "accuracy": float(accuracy_score(val_y, val_pred)),
            "n_validation": len(val_y),
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    return metrics_payload


def predict_emotion_from_file(audio_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Predict an emotion and confidence for a single WAV file."""
    model_bundle = joblib.load(model_path)
    pipeline: Pipeline = model_bundle["pipeline"]
    label_mapping: dict[str, str] = model_bundle["label_mapping"]

    waveform, sample_rate = load_audio(audio_path)
    feature_vector = extract_mfcc_stats(waveform, sample_rate=sample_rate, n_mfcc=13)
    probabilities = pipeline.predict_proba(feature_vector.reshape(1, -1))[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = pipeline.classes_[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_emotion": predicted_label,
        "emotion_name": label_mapping.get(predicted_label, EMOTION_MAP.get(predicted_label, predicted_label)),
        "confidence": confidence,
        "probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(pipeline.classes_)},
    }
