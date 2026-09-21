"""Scalar acoustic measurements for normalized speech waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ml.src.preprocessing.audio import load_audio


@dataclass(frozen=True)
class AcousticFeatures:
    """Signal-level measurements; these values do not provide clinical interpretation."""

    duration_seconds: float
    rms_mean: float
    rms_std: float
    zcr_mean: float
    zcr_std: float
    pitch_mean_hz: float | None
    pitch_std_hz: float | None
    pitch_range_hz: float | None
    silence_ratio: float
    speaking_rate_proxy: float | None
    is_silent: bool
    is_noisy: bool


def _validate_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Waveform must be a one-dimensional array.")
    if values.size == 0:
        raise ValueError("Waveform is empty.")
    if not np.isfinite(values).all():
        raise ValueError("Waveform contains non-finite samples.")
    return values


def _frame_values(
    waveform: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    return np.asarray(rms, dtype=np.float64), np.asarray(zcr, dtype=np.float64)


def _pitch_values(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int,
    pitch_min_hz: float,
    pitch_max_hz: float,
    voiced_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pitch_frame_length = max(1024, int(np.ceil(2 * sample_rate / pitch_min_hz)) + 1)
    padded_length = max(waveform.size, pitch_frame_length)
    padded = np.pad(waveform, (0, padded_length - waveform.size))
    pitches, voiced_flags, voiced_probabilities = librosa.pyin(
        padded,
        fmin=pitch_min_hz,
        fmax=pitch_max_hz,
        sr=sample_rate,
        frame_length=pitch_frame_length,
        hop_length=hop_length,
        fill_na=np.nan,
    )
    pitches = np.asarray(pitches, dtype=np.float64)
    voiced_flags = np.asarray(voiced_flags, dtype=bool)
    voiced_probabilities = np.asarray(voiced_probabilities, dtype=np.float64)
    valid = (
        voiced_flags
        & (voiced_probabilities >= voiced_threshold)
        & np.isfinite(pitches)
        & (pitches >= pitch_min_hz)
        & (pitches <= pitch_max_hz)
    )
    return pitches, valid


def _optional_pitch_statistics(pitches: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None, float | None]:
    reliable = pitches[valid]
    if reliable.size < 2:
        return None, None, None
    return (
        float(np.mean(reliable)),
        float(np.std(reliable)),
        float(np.max(reliable) - np.min(reliable)),
    )


def extract_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 400,
    hop_length: int = 160,
    silence_threshold_db: float = -40.0,
    pitch_min_hz: float = 50.0,
    pitch_max_hz: float = 500.0,
) -> AcousticFeatures:
    """Extract finite scalar acoustic measurements from a waveform."""
    values = _validate_waveform(waveform, sample_rate)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Frame length and hop length must be positive.")
    if not 0 < pitch_min_hz < pitch_max_hz:
        raise ValueError("Pitch bounds must be positive and ordered.")

    duration_seconds = float(values.size / sample_rate)
    rms, zcr = _frame_values(values, sample_rate, frame_length, hop_length)
    rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
    zcr = np.clip(np.nan_to_num(zcr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    threshold_amplitude = 10.0 ** (silence_threshold_db / 20.0)
    silent_frames = rms <= threshold_amplitude
    silence_ratio = float(np.mean(silent_frames))
    is_silent = bool(np.all(silent_frames))

    if is_silent:
        return AcousticFeatures(
            duration_seconds=duration_seconds,
            rms_mean=0.0,
            rms_std=0.0,
            zcr_mean=0.0,
            zcr_std=0.0,
            pitch_mean_hz=None,
            pitch_std_hz=None,
            pitch_range_hz=None,
            silence_ratio=1.0,
            speaking_rate_proxy=None,
            is_silent=True,
            is_noisy=False,
        )

    pitches, valid_pitch = _pitch_values(
        values,
        sample_rate,
        hop_length,
        pitch_min_hz,
        pitch_max_hz,
        voiced_threshold=0.5,
    )
    pitch_mean, pitch_std, pitch_range = _optional_pitch_statistics(pitches, valid_pitch)

    voiced_frames = np.zeros(rms.size, dtype=bool)
    voiced_frames[: min(voiced_frames.size, valid_pitch.size)] = valid_pitch[: voiced_frames.size]
    onset_count = int(np.count_nonzero(voiced_frames & ~np.concatenate(([False], voiced_frames[:-1]))))
    speaking_rate_proxy = (
        float(onset_count / duration_seconds)
        if duration_seconds >= 0.1 and np.count_nonzero(voiced_frames) >= 2
        else None
    )

    is_noisy = bool(
        pitch_mean is None
        and rms_mean > 0.2
        and zcr_mean > 0.35
    )
    return AcousticFeatures(
        duration_seconds=duration_seconds,
        rms_mean=rms_mean,
        rms_std=rms_std,
        zcr_mean=zcr_mean,
        zcr_std=zcr_std,
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        pitch_range_hz=pitch_range,
        silence_ratio=float(np.clip(silence_ratio, 0.0, 1.0)),
        speaking_rate_proxy=speaking_rate_proxy,
        is_silent=False,
        is_noisy=is_noisy,
    )


def extract_acoustic_features_from_file(audio_path: str | Path) -> AcousticFeatures:
    """Load audio through the shared V4 preprocessing path before extraction."""
    waveform, sample_rate = load_audio(audio_path)
    return extract_acoustic_features(waveform, sample_rate)
