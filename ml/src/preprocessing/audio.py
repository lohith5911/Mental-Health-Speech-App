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

    try:
        header = path.read_bytes()[:128]
        if b"git-lfs.github.com/spec/v1" in header:
            raise ValueError(f"Git LFS pointer file detected: {path}")
    except OSError:
        pass

    try:
        waveform, sample_rate = sf.read(path, always_2d=False)
    except Exception as exc:
        raise ValueError(f"Unreadable audio file: {path}") from exc

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio file contains non-finite samples: {path}")

    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform, sample_rate


def preprocess_audio(file_path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Return a normalized mono waveform suitable for future speech emotion feature extraction."""
    return load_audio(file_path, target_sr=target_sr)
