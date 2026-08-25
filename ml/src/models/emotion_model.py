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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.src.features.features import extract_acoustic_features_stats, extract_mfcc_delta_stats, extract_mfcc_stats
from ml.src.preprocessing.audio import load_audio

DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "crema-d" / "AudioWAV"
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "models"
VERSION_2_ARTIFACT_DIR = ARTIFACT_DIR / "v2"
VERSION_3_ARTIFACT_DIR = ARTIFACT_DIR / "v3"
MODEL_PATH = ARTIFACT_DIR / "emotion_classifier.pkl"
VERSION_2_MODEL_PATH = VERSION_2_ARTIFACT_DIR / "emotion_classifier.pkl"
VERSION_3_MODEL_PATH = VERSION_3_ARTIFACT_DIR / "emotion_classifier.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
VERSION_2_SCALER_PATH = VERSION_2_ARTIFACT_DIR / "scaler.pkl"
VERSION_3_SCALER_PATH = VERSION_3_ARTIFACT_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
VERSION_2_LABEL_ENCODER_PATH = VERSION_2_ARTIFACT_DIR / "label_encoder.pkl"
VERSION_3_LABEL_ENCODER_PATH = VERSION_3_ARTIFACT_DIR / "label_encoder.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
VERSION_2_METRICS_PATH = VERSION_2_ARTIFACT_DIR / "metrics.json"
VERSION_3_METRICS_PATH = VERSION_3_ARTIFACT_DIR / "metrics.json"

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
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be between 0 and 1 and sum to less than 1.")

    speaker_to_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        speaker_to_files[parse_speaker_id(file_path.name)].append(file_path)

    speakers = sorted(speaker_to_files)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    total_speakers = len(speakers)
    train_end = math.floor(total_speakers * (1 - val_ratio - test_ratio))
    val_end = train_end + math.floor(total_speakers * val_ratio)

    train_speakers = speakers[:train_end]
    validation_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]

    split = {
        "train": [path for speaker in train_speakers for path in speaker_to_files[speaker]],
        "validation": [path for speaker in validation_speakers for path in speaker_to_files[speaker]],
        "test": [path for speaker in test_speakers for path in speaker_to_files[speaker]],
    }
    return split


def load_cremad_dataset(data_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load CREMA-D audio features and emotion codes as machine-learning arrays."""
    files = list_audio_files(data_dir)
    features, labels, corrupted = load_feature_matrix(files, data_dir=data_dir)
    if corrupted:
        raise ValueError(f"Unable to load {len(corrupted)} CREMA-D audio files.")
    return features, np.asarray(labels)


def load_feature_matrix(
    files: list[Path],
    data_dir: Path = DATASET_DIR,
    feature_extractor: Any = extract_mfcc_stats,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract fixed-size audio features from files and return arrays plus metadata."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    corrupted: list[str] = []

    for file_path in files:
        emotion = parse_emotion_from_filename(file_path.name)
        if emotion is None:
            corrupted.append(str(file_path.relative_to(data_dir)))
            continue

        try:
            waveform, sample_rate = load_audio(file_path)
            feature_vector = feature_extractor(waveform, sample_rate=sample_rate, n_mfcc=13)
            if feature_vector.shape[0] == 0 or not np.isfinite(feature_vector).all():
                raise ValueError(f"Feature vector for {file_path} is invalid")
            features.append(feature_vector)
            labels.append(emotion)
        except Exception:
            corrupted.append(str(file_path.relative_to(data_dir)))

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
    """Train the original baseline speaker-aware scaled SVM and save separate model artifacts."""
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    train_X, train_y, train_corrupt = load_feature_matrix(split["train"], feature_extractor=extract_mfcc_stats)
    val_X, val_y, val_corrupt = load_feature_matrix(split["validation"], feature_extractor=extract_mfcc_stats)
    test_X, test_y, test_corrupt = load_feature_matrix(split["test"], feature_extractor=extract_mfcc_stats)

    label_encoder = LabelEncoder()
    train_y_encoded = label_encoder.fit_transform(train_y)
    val_y_encoded = label_encoder.transform(val_y)
    test_y_encoded = label_encoder.transform(test_y)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    classifier = SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )
    classifier.fit(train_X_scaled, train_y_encoded)

    val_pred = classifier.predict(val_X_scaled)
    test_pred = classifier.predict(test_X_scaled)
    metrics = compute_classification_metrics(test_y, label_encoder.inverse_transform(test_pred).tolist())

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / "emotion_classifier.pkl")
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(label_encoder, artifact_dir / "label_encoder.pkl")

    metrics_payload = {
        "version": "v1",
        "feature_type": "mfcc_stats",
        "feature_dimension": int(train_X.shape[1]),
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
        "model_path": str(artifact_dir / "emotion_classifier.pkl"),
        "validation_prediction_examples": {
            "accuracy": float(accuracy_score(val_y_encoded, val_pred)),
            "n_validation": len(val_y_encoded),
        },
        "saved_files": {
            "classifier": str(artifact_dir / "emotion_classifier.pkl"),
            "scaler": str(artifact_dir / "scaler.pkl"),
            "label_encoder": str(artifact_dir / "label_encoder.pkl"),
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def train_emotion_model_v2(data_dir: Path = DATASET_DIR, artifact_dir: Path = VERSION_2_ARTIFACT_DIR, seed: int = 42) -> dict[str, Any]:
    """Train an improved speaker-aware scaled SVM using MFCC, delta, and delta-delta features."""
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    train_X, train_y, train_corrupt = load_feature_matrix(split["train"], feature_extractor=extract_mfcc_delta_stats)
    val_X, val_y, val_corrupt = load_feature_matrix(split["validation"], feature_extractor=extract_mfcc_delta_stats)
    test_X, test_y, test_corrupt = load_feature_matrix(split["test"], feature_extractor=extract_mfcc_delta_stats)

    label_encoder = LabelEncoder()
    train_y_encoded = label_encoder.fit_transform(train_y)
    val_y_encoded = label_encoder.transform(val_y)
    test_y_encoded = label_encoder.transform(test_y)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    classifier = SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )
    classifier.fit(train_X_scaled, train_y_encoded)

    val_pred = classifier.predict(val_X_scaled)
    test_pred = classifier.predict(test_X_scaled)
    metrics = compute_classification_metrics(test_y, label_encoder.inverse_transform(test_pred).tolist())

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / "emotion_classifier.pkl")
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(label_encoder, artifact_dir / "label_encoder.pkl")

    metrics_payload = {
        "version": "v2",
        "feature_type": "mfcc_delta_delta_stats",
        "feature_dimension": int(train_X.shape[1]),
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
        "model_path": str(artifact_dir / "emotion_classifier.pkl"),
        "validation_prediction_examples": {
            "accuracy": float(accuracy_score(val_y_encoded, val_pred)),
            "n_validation": len(val_y_encoded),
        },
        "saved_files": {
            "classifier": str(artifact_dir / "emotion_classifier.pkl"),
            "scaler": str(artifact_dir / "scaler.pkl"),
            "label_encoder": str(artifact_dir / "label_encoder.pkl"),
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def train_emotion_model_v3(data_dir: Path = DATASET_DIR, artifact_dir: Path = VERSION_3_ARTIFACT_DIR, seed: int = 42) -> dict[str, Any]:
    """Train a richer speaker-aware acoustic SVM while preserving the V1/V2 split and artifacts."""
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    train_X, train_y, train_corrupt = load_feature_matrix(split["train"], feature_extractor=extract_acoustic_features_stats)
    val_X, val_y, val_corrupt = load_feature_matrix(split["validation"], feature_extractor=extract_acoustic_features_stats)
    test_X, test_y, test_corrupt = load_feature_matrix(split["test"], feature_extractor=extract_acoustic_features_stats)

    label_encoder = LabelEncoder()
    train_y_encoded = label_encoder.fit_transform(train_y)
    val_y_encoded = label_encoder.transform(val_y)
    test_y_encoded = label_encoder.transform(test_y)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    classifier = SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )
    classifier.fit(train_X_scaled, train_y_encoded)

    val_pred = classifier.predict(val_X_scaled)
    test_pred = classifier.predict(test_X_scaled)
    metrics = compute_classification_metrics(test_y, label_encoder.inverse_transform(test_pred).tolist())

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / "emotion_classifier.pkl")
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(label_encoder, artifact_dir / "label_encoder.pkl")

    metrics_payload = {
        "version": "v3",
        "feature_type": "mfcc_delta_delta_plus_acoustic_stats",
        "feature_dimension": int(train_X.shape[1]),
        "feature_config": {
            "n_mfcc": 13,
            "summary_statistics": ["mean", "std", "min", "max"],
            "base_features": ["mfcc", "delta_mfcc", "delta_delta_mfcc"],
            "extra_features": [
                "zero_crossing_rate",
                "rms_energy",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "chroma",
            ],
        },
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
        "model_path": str(artifact_dir / "emotion_classifier.pkl"),
        "validation_prediction_examples": {
            "accuracy": float(accuracy_score(val_y_encoded, val_pred)),
            "n_validation": len(val_y_encoded),
        },
        "saved_files": {
            "classifier": str(artifact_dir / "emotion_classifier.pkl"),
            "scaler": str(artifact_dir / "scaler.pkl"),
            "label_encoder": str(artifact_dir / "label_encoder.pkl"),
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def predict_emotion_from_file(audio_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Predict an emotion and confidence for a single WAV file using the correct feature extractor for the saved model."""
    classifier = joblib.load(model_path)
    artifact_dir = Path(model_path).parent
    scaler: StandardScaler = joblib.load(artifact_dir / "scaler.pkl")
    label_encoder: LabelEncoder = joblib.load(artifact_dir / "label_encoder.pkl")

    waveform, sample_rate = load_audio(audio_path)
    feature_dim = int(scaler.n_features_in_)
    if feature_dim == 52:
        feature_extractor = extract_mfcc_stats
    elif feature_dim == 78:
        feature_extractor = extract_mfcc_delta_stats
    else:
        feature_extractor = extract_acoustic_features_stats
    feature_vector = feature_extractor(waveform, sample_rate=sample_rate, n_mfcc=13)
    scaled_features = scaler.transform(feature_vector.reshape(1, -1))
    probabilities = classifier.predict_proba(scaled_features)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_encoded_label = int(classifier.classes_[predicted_index])
    predicted_label = str(label_encoder.inverse_transform([predicted_encoded_label])[0])
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_emotion": predicted_label,
        "emotion_name": EMOTION_MAP.get(predicted_label, predicted_label),
        "confidence": confidence,
        "probabilities": {
            str(label_encoder.inverse_transform([int(encoded_label)])[0]): float(probabilities[idx])
            for idx, encoded_label in enumerate(classifier.classes_)
        },
    }
