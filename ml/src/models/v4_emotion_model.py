"""Frozen pretrained wav2vec 2.0 speech representation for CREMA-D emotion classification."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from ml.src.models.emotion_model import (
    ARTIFACT_DIR,
    DATASET_DIR,
    EMOTION_MAP,
    EMOTION_ORDER,
    build_speaker_aware_split,
    compute_classification_metrics,
    list_audio_files,
    parse_emotion_from_filename,
    parse_speaker_id,
)
from ml.src.preprocessing.audio import load_audio

V4_ARTIFACT_DIR = ARTIFACT_DIR / "v4"
V4_CONFIG_PATH = V4_ARTIFACT_DIR / "config.json"
V4_MODEL_PATH = V4_ARTIFACT_DIR / "model.pt"
V4_LABEL_ENCODER_PATH = V4_ARTIFACT_DIR / "label_encoder.pkl"
V4_METRICS_PATH = V4_ARTIFACT_DIR / "metrics.json"

DEFAULT_WAV2VEC_MODEL = "facebook/wav2vec2-base"

V4_CONFIG = {
    "model_name": DEFAULT_WAV2VEC_MODEL,
    "batch_size": 16,
    "epochs": 12,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "patience": 4,
    "seed": 42,
    "embedding_dim": 768,
    "dropout": 0.2,
}


def set_reproducible_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_wav2vec(model_name: str = DEFAULT_WAV2VEC_MODEL) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model, torch.device]:
    """Load the selected wav2vec 2.0 encoder and the matching feature extractor on the active device."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    device = _get_device()
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return feature_extractor, model, device


def _encode_waveform_batch(
    waveforms: list[np.ndarray],
    sample_rate: int,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    """Encode a list of 1D waveforms into pooled wav2vec embeddings."""
    if not waveforms:
        raise ValueError("Waveform batch is empty.")

    inputs = feature_extractor(
        [np.asarray(waveform, dtype=np.float32) for waveform in waveforms],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

    return pooled.cpu().numpy().astype(np.float32)


def compute_wav2vec_embeddings(
    audio_path: str | Path,
    model_name: str = DEFAULT_WAV2VEC_MODEL,
    feature_extractor: Wav2Vec2FeatureExtractor | None = None,
    model: Wav2Vec2Model | None = None,
) -> np.ndarray:
    """Return the pooled wav2vec embedding for a single audio file."""
    waveform, sample_rate = load_audio(audio_path)
    if feature_extractor is None or model is None:
        feature_extractor, model, device = load_pretrained_wav2vec(model_name)
    else:
        device = _get_device()
        model.to(device)

    embeddings = _encode_waveform_batch([waveform], sample_rate, feature_extractor, model, device)
    return embeddings[0]


class Wav2Vec2EmotionClassifier(nn.Module):
    """A small linear classifier head trained on frozen wav2vec embeddings."""

    def __init__(self, embedding_dim: int = V4_CONFIG["embedding_dim"], n_classes: int = len(EMOTION_MAP), dropout: float = V4_CONFIG["dropout"]):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)


def _collect_embeddings_for_split(
    file_paths: list[Path],
    feature_extractor: Wav2Vec2FeatureExtractor,
    wav2vec_model: Wav2Vec2Model,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Compute wav2vec embeddings for a split and return labels for each sample."""
    embeddings: list[np.ndarray] = []
    labels: list[str] = []

    for file_path in file_paths:
        emotion = parse_emotion_from_filename(file_path.name)
        if emotion is None:
            continue
        waveform, sample_rate = load_audio(file_path)
        embedding = _encode_waveform_batch([waveform], sample_rate, feature_extractor, wav2vec_model, device)[0]
        embeddings.append(embedding)
        labels.append(emotion)

    if not embeddings:
        raise ValueError("No valid wav2vec embeddings could be extracted.")

    return np.vstack(embeddings).astype(np.float32), labels


def train_v4_model(data_dir: Path = DATASET_DIR, artifact_dir: Path = V4_ARTIFACT_DIR, seed: int = 42) -> dict[str, Any]:
    """Train a frozen wav2vec 2.0 classifier with the same speaker-aware split used by V1-V3."""
    start_time = time.time()
    set_reproducible_seed(seed)
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    feature_extractor, wav2vec_model, device = load_pretrained_wav2vec(DEFAULT_WAV2VEC_MODEL)

    train_X, train_y = _collect_embeddings_for_split(split["train"], feature_extractor, wav2vec_model, device)
    val_X, val_y = _collect_embeddings_for_split(split["validation"], feature_extractor, wav2vec_model, device)
    test_X, test_y = _collect_embeddings_for_split(split["test"], feature_extractor, wav2vec_model, device)

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(EMOTION_MAP))
    train_y_encoded = label_encoder.transform(train_y)
    val_y_encoded = label_encoder.transform(val_y)
    test_y_encoded = label_encoder.transform(test_y)

    classifier = Wav2Vec2EmotionClassifier(embedding_dim=train_X.shape[1], n_classes=len(label_encoder.classes_))
    classifier.to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=V4_CONFIG["learning_rate"], weight_decay=V4_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(classifier.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_history: list[dict[str, Any]] = []

    train_X_tensor = torch.tensor(train_X, dtype=torch.float32, device=device)
    val_X_tensor = torch.tensor(val_X, dtype=torch.float32, device=device)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(train_y_encoded, dtype=torch.long, device=device)
    val_y_tensor = torch.tensor(val_y_encoded, dtype=torch.long, device=device)

    for epoch in range(1, V4_CONFIG["epochs"] + 1):
        classifier.train()
        permutation = torch.randperm(train_X_tensor.size(0))
        epoch_losses: list[float] = []

        for batch_start in range(0, train_X_tensor.size(0), V4_CONFIG["batch_size"]):
            batch_indices = permutation[batch_start:batch_start + V4_CONFIG["batch_size"]]
            batch_x = train_X_tensor[batch_indices]
            batch_y = train_y_tensor[batch_indices]

            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_X_tensor)
            val_pred_indices = val_logits.argmax(dim=1).cpu().numpy()
            val_pred_labels = label_encoder.inverse_transform(val_pred_indices).tolist()
            val_metrics = compute_classification_metrics(val_y, val_pred_labels)
            val_f1 = float(val_metrics["macro_f1"])

        train_history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "validation_accuracy": float(val_metrics["accuracy"]),
            "validation_macro_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(classifier.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= V4_CONFIG["patience"]:
                break

    classifier.load_state_dict(best_state)
    classifier.eval()

    with torch.no_grad():
        test_logits = classifier(test_X_tensor)
        test_pred_indices = test_logits.argmax(dim=1).cpu().numpy()
        pred_labels = label_encoder.inverse_transform(test_pred_indices).tolist()
        test_metrics = compute_classification_metrics(test_y, pred_labels)

    confusion = confusion_matrix(test_y, pred_labels, labels=EMOTION_ORDER)
    metrics_payload = {
        "version": "v4",
        "feature_type": "frozen_wav2vec2_base",
        "pretrained_model": DEFAULT_WAV2VEC_MODEL,
        "embedding_dimension": int(train_X.shape[1]),
        "model_size_mb": 360.0,
        "dataset": {
            "total_wav_files": len(all_files),
            "train_files": len(split["train"]),
            "validation_files": len(split["validation"]),
            "test_files": len(split["test"]),
            "speaker_count": len({parse_speaker_id(path.name) for path in all_files}),
        },
        "split": {
            "train_speakers": len({parse_speaker_id(path.name) for path in split["train"]}),
            "validation_speakers": len({parse_speaker_id(path.name) for path in split["validation"]}),
            "test_speakers": len({parse_speaker_id(path.name) for path in split["test"]}),
        },
        "training": {
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_val_f1),
            "history": train_history,
            "patience": V4_CONFIG["patience"],
            "learning_rate": V4_CONFIG["learning_rate"],
            "weight_decay": V4_CONFIG["weight_decay"],
            "device": str(device),
            "training_time_seconds": round(time.time() - start_time, 2),
        },
        "evaluation": {
            **test_metrics,
            "confusion_matrix": confusion.tolist(),
        },
        "model_path": str(artifact_dir / "model.pt"),
        "saved_files": {
            "model": str(artifact_dir / "model.pt"),
            "config": str(artifact_dir / "config.json"),
            "label_encoder": str(artifact_dir / "label_encoder.pkl"),
            "metrics": str(artifact_dir / "metrics.json"),
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **V4_CONFIG,
        "label_order": label_encoder.classes_.tolist(),
        "n_classes": len(label_encoder.classes_),
    }
    V4_CONFIG_PATH.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    joblib.dump(label_encoder, V4_LABEL_ENCODER_PATH)
    torch.save({"state_dict": classifier.state_dict(), "config": config_payload}, V4_MODEL_PATH)
    V4_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def load_saved_v4_model(model_path: str | Path = V4_MODEL_PATH, config_path: str | Path = V4_CONFIG_PATH) -> tuple[Wav2Vec2EmotionClassifier, dict[str, Any], torch.device]:
    """Load a saved frozen wav2vec head and config for inference."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = Wav2Vec2EmotionClassifier(
        embedding_dim=int(config["embedding_dim"]),
        n_classes=int(config["n_classes"]),
        dropout=float(config["dropout"]),
    )
    device = _get_device()
    payload = torch.load(model_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config, device


def predict_emotion_from_file_v4(
    audio_path: str | Path,
    model_path: str | Path = V4_MODEL_PATH,
    config_path: str | Path = V4_CONFIG_PATH,
    label_encoder_path: str | Path = V4_LABEL_ENCODER_PATH,
) -> dict[str, Any]:
    """Predict emotion and confidence for a single WAV file using the saved frozen wav2vec classifier."""
    model, config, device = load_saved_v4_model(model_path=model_path, config_path=config_path)
    label_encoder = joblib.load(label_encoder_path)
    waveform, sample_rate = load_audio(audio_path)

    feature_extractor, wav2vec_model, _ = load_pretrained_wav2vec(config["model_name"])
    wav2vec_model.to(device)
    wav2vec_model.eval()

    embedding = _encode_waveform_batch([waveform], sample_rate, feature_extractor, wav2vec_model, device)[0]
    tensor = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = str(label_encoder.inverse_transform([predicted_index])[0])
        confidence = float(probabilities[predicted_index].item())

    return {
        "predicted_emotion": predicted_label,
        "emotion_name": EMOTION_MAP.get(predicted_label, predicted_label),
        "confidence": confidence,
        "probabilities": {
            str(label_encoder.inverse_transform([int(idx)])[0]): float(probabilities[idx].item())
            for idx in range(len(probabilities))
        },
    }
