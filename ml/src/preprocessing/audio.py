"""Reusable audio preprocessing helpers for future CREMA-D speech emotion work."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(file_path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load and preprocess an audio file to a mono waveform at a consistent sample rate."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = sf.read(path, always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform, sample_rate


def preprocess_audio(file_path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Return a normalized mono waveform suitable for future speech emotion feature extraction."""
    return load_audio(file_path, target_sr=target_sr)
