"""Small CNN speech emotion model trained on log-Mel spectrograms."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.src.models.emotion_model import (
    DATASET_DIR,
    EMOTION_MAP,
    EMOTION_ORDER,
    build_speaker_aware_split,
    compute_classification_metrics,
    list_audio_files,
    load_audio,
    parse_emotion_from_filename,
    parse_speaker_id,
)

CNN_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "models" / "cnn_v1"
CNN_CONFIG_PATH = CNN_ARTIFACT_DIR / "config.json"
CNN_MODEL_PATH = CNN_ARTIFACT_DIR / "model.pt"
CNN_LABEL_ENCODER_PATH = CNN_ARTIFACT_DIR / "label_encoder.pkl"
CNN_METRICS_PATH = CNN_ARTIFACT_DIR / "metrics.json"

SMALL_CNN_CONFIG = {
    "n_mels": 40,
    "n_frames": 128,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 25,
    "patience": 5,
    "dropout": 0.3,
    "seed": 42,
}


def set_reproducible_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = SMALL_CNN_CONFIG["n_mels"],
    n_frames: int = SMALL_CNN_CONFIG["n_frames"],
) -> np.ndarray:
    """Compute a normalized log-Mel spectrogram with fixed output dimensions."""
    waveform = np.nan_to_num(np.asarray(waveform, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=512,
        hop_length=160,
        win_length=400,
        fmin=50.0,
        fmax=min(float(sample_rate) / 2.0, 8000.0),
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    if log_mel.shape[1] < n_frames:
        pad_width = n_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    elif log_mel.shape[1] > n_frames:
        log_mel = log_mel[:, :n_frames]

    return log_mel.astype(np.float32)


class MelSpectrogramDataset(Dataset):
    """Dataset that yields fixed-size log-Mel spectrograms and emotion labels."""

    def __init__(self, file_paths: list[Path], label_encoder: LabelEncoder, config: dict[str, Any]):
        self.file_paths = file_paths
        self.label_encoder = label_encoder
        self.config = config
        self.labels = []
        self.samples: list[tuple[str, str]] = []

        for file_path in self.file_paths:
            emotion = parse_emotion_from_filename(file_path.name)
            if emotion is None:
                continue
            self.samples.append((str(file_path), emotion))
            self.labels.append(emotion)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, emotion = self.samples[index]
        waveform, sample_rate = load_audio(file_path)
        spectrogram = compute_log_mel_spectrogram(
            waveform,
            sample_rate=sample_rate,
            n_mels=self.config["n_mels"],
            n_frames=self.config["n_frames"],
        )
        tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
        label_idx = int(self.label_encoder.transform([emotion])[0])
        return tensor, label_idx


class CNNEmotionModel(nn.Module):
    """Compact CNN for fixed-size Mel spectrograms."""

    def __init__(self, n_mels: int = SMALL_CNN_CONFIG["n_mels"], n_frames: int = SMALL_CNN_CONFIG["n_frames"], n_classes: int = 6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, n_classes),
        )
        self.n_mels = n_mels
        self.n_frames = n_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_model(model: CNNEmotionModel, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    all_labels: list[str] = []
    all_preds: list[str] = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch_y.numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)

    encoded_labels = np.asarray(all_labels)
    encoded_preds = np.asarray(all_preds)

    label_encoder = dataloader.dataset.label_encoder
    true_labels = label_encoder.inverse_transform(encoded_labels).tolist()
    pred_labels = label_encoder.inverse_transform(encoded_preds).tolist()

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=EMOTION_ORDER, average="macro")
    precision = f1_score(true_labels, pred_labels, labels=EMOTION_ORDER, average="macro")
    recall = f1_score(true_labels, pred_labels, labels=EMOTION_ORDER, average="macro")
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
    }
    return encoded_labels, encoded_preds, metrics


def train_cnn_model(data_dir: Path = DATASET_DIR, artifact_dir: Path = CNN_ARTIFACT_DIR, seed: int = 42) -> dict[str, Any]:
    """Train the CNN on the same speaker-independent split used for V1/V2/V3."""
    set_reproducible_seed(seed)
    all_files = list_audio_files(data_dir)
    split = build_speaker_aware_split(all_files, seed=seed)

    label_encoder = LabelEncoder()
    all_emotions = sorted(EMOTION_MAP)
    label_encoder.fit(all_emotions)

    train_dataset = MelSpectrogramDataset(split["train"], label_encoder=label_encoder, config=SMALL_CNN_CONFIG)
    val_dataset = MelSpectrogramDataset(split["validation"], label_encoder=label_encoder, config=SMALL_CNN_CONFIG)
    test_dataset = MelSpectrogramDataset(split["test"], label_encoder=label_encoder, config=SMALL_CNN_CONFIG)

    train_loader = DataLoader(train_dataset, batch_size=SMALL_CNN_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=SMALL_CNN_CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=SMALL_CNN_CONFIG["batch_size"], shuffle=False)

    model = CNNEmotionModel(n_classes=len(label_encoder.classes_))
    device = _get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=SMALL_CNN_CONFIG["learning_rate"])

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_history: list[dict[str, Any]] = []

    for epoch in range(1, SMALL_CNN_CONFIG["epochs"] + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        _, _, val_metrics = _evaluate_model(model, val_loader, device)
        training_history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= SMALL_CNN_CONFIG["patience"]:
                break

    model.load_state_dict(best_state)
    model.to(device)

    _, _, test_metrics = _evaluate_model(model, test_loader, device)
    test_true = []
    test_pred = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
            test_true.extend(batch_y.numpy())
            test_pred.extend(batch_pred)

    true_labels = label_encoder.inverse_transform(np.asarray(test_true)).tolist()
    pred_labels = label_encoder.inverse_transform(np.asarray(test_pred)).tolist()
    cm = confusion_matrix(true_labels, pred_labels, labels=EMOTION_ORDER)

    test_eval = compute_classification_metrics(true_labels, pred_labels)
    metrics_payload = {
        "version": "cnn_v1",
        "feature_type": "log_mel_spectrogram",
        "feature_dimension": {
            "n_mels": SMALL_CNN_CONFIG["n_mels"],
            "n_frames": SMALL_CNN_CONFIG["n_frames"],
            "channels": 1,
        },
        "model_architecture": {
            "type": "CNN",
            "layers": [
                "Conv2d(1,16,3,pad=1)",
                "ReLU",
                "MaxPool(2)",
                "Conv2d(16,32,3,pad=1)",
                "ReLU",
                "MaxPool(2)",
                "Conv2d(32,64,3,pad=1)",
                "ReLU",
                "AdaptiveAvgPool",
                "Dropout(0.3)",
                "Linear(64,6)",
            ],
        },
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
            "history": training_history,
            "patience": SMALL_CNN_CONFIG["patience"],
            "learning_rate": SMALL_CNN_CONFIG["learning_rate"],
            "device": str(device),
            "training_time_seconds": 0.0,
        },
        "evaluation": {
            **test_eval,
            "confusion_matrix": cm.tolist(),
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
        **SMALL_CNN_CONFIG,
        "input_shape": [1, SMALL_CNN_CONFIG["n_mels"], SMALL_CNN_CONFIG["n_frames"]],
        "n_classes": len(label_encoder.classes_),
        "label_order": label_encoder.classes_.tolist(),
    }
    CNN_CONFIG_PATH.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    joblib.dump(label_encoder, CNN_LABEL_ENCODER_PATH)
    torch.save(model.state_dict(), CNN_MODEL_PATH)
    CNN_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def load_saved_cnn_model(model_path: str | Path = CNN_MODEL_PATH, config_path: str | Path = CNN_CONFIG_PATH) -> tuple[CNNEmotionModel, dict[str, Any], torch.device]:
    """Load a saved CNN model and config for inference."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = CNNEmotionModel(
        n_mels=int(config["n_mels"]),
        n_frames=int(config["n_frames"]),
        n_classes=int(config["n_classes"]),
    )
    device = _get_device()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, device


def predict_emotion_from_file_cnn(
    audio_path: str | Path,
    model_path: str | Path = CNN_MODEL_PATH,
    config_path: str | Path = CNN_CONFIG_PATH,
    label_encoder_path: str | Path = CNN_LABEL_ENCODER_PATH,
) -> dict[str, Any]:
    """Predict emotion and confidence for a single WAV file using the saved CNN model."""
    model, config, device = load_saved_cnn_model(model_path=model_path, config_path=config_path)
    label_encoder = joblib.load(label_encoder_path)

    waveform, sample_rate = load_audio(audio_path)
    spectrogram = compute_log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_mels=int(config["n_mels"]),
        n_frames=int(config["n_frames"]),
    )

    tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(device).float()
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
