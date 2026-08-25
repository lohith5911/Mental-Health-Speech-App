"""Feature extraction utilities for speech emotion recognition work."""

from __future__ import annotations

import numpy as np


def extract_mfcc(waveform: np.ndarray, sample_rate: int, n_mfcc: int = 13):
    """Compute MFCCs from a waveform."""
    import librosa

    waveform = np.nan_to_num(np.asarray(waveform, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)


def _summarize_feature_matrix(feature_matrix: np.ndarray, include_min_max: bool = False) -> np.ndarray:
    """Return mean and standard deviation statistics for each coefficient."""
    feature_matrix = np.nan_to_num(np.asarray(feature_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)

    mean = np.mean(feature_matrix, axis=1)
    std = np.std(feature_matrix, axis=1)
    if not include_min_max:
        return np.concatenate([mean, std]).astype(np.float32)

    minimum = np.min(feature_matrix, axis=1)
    maximum = np.max(feature_matrix, axis=1)
    return np.concatenate([mean, std, minimum, maximum]).astype(np.float32)


def extract_mfcc_stats(waveform: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Return the original fixed-length MFCC feature vector using mean/std statistics per coefficient."""
    mfcc = extract_mfcc(waveform, sample_rate=sample_rate, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    delta = np.mean(np.diff(mfcc, axis=1), axis=1) if mfcc.shape[1] > 1 else np.zeros_like(mean)
    delta_std = np.std(np.diff(mfcc, axis=1), axis=1) if mfcc.shape[1] > 2 else np.zeros_like(mean)
    features = np.concatenate([mean, std, delta, delta_std]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_mfcc_delta_stats(waveform: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Return MFCC + delta + delta-delta summary features as a fixed-size 78-dimensional vector."""
    import librosa

    mfcc = extract_mfcc(waveform, sample_rate=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate(
        [
            _summarize_feature_matrix(mfcc),
            _summarize_feature_matrix(delta),
            _summarize_feature_matrix(delta_delta),
        ]
    )
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_mel_spectrogram(waveform: np.ndarray, sample_rate: int):
    """Compute a mel-scaled spectrogram."""
    import librosa

    return librosa.feature.melspectrogram(y=waveform, sr=sample_rate)


def extract_spectral_centroid(waveform: np.ndarray, sample_rate: int):
    """Compute the spectral centroid."""
    import librosa

    return librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)


def extract_zero_crossing_rate(waveform: np.ndarray):
    """Compute zero-crossing rate."""
    import librosa

    return librosa.feature.zero_crossing_rate(waveform)


def extract_rms_energy(waveform: np.ndarray):
    """Compute RMS energy."""
    import librosa

    return librosa.feature.rms(y=waveform)


def extract_chroma(waveform: np.ndarray, sample_rate: int):
    """Compute chroma features."""
    import librosa

    return librosa.feature.chroma_cqt(y=waveform, sr=sample_rate)


def extract_spectral_bandwidth(waveform: np.ndarray, sample_rate: int):
    """Compute spectral bandwidth."""
    import librosa

    return librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)


def extract_spectral_rolloff(waveform: np.ndarray, sample_rate: int):
    """Compute spectral rolloff."""
    import librosa

    return librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)


def extract_acoustic_features_stats(waveform: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Create a fixed-size acoustic representation using MFCCs plus richer low-level spectral features."""
    import librosa

    waveform = np.nan_to_num(np.asarray(waveform, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mfcc = extract_mfcc(waveform, sample_rate=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    feature_blocks = [
        _summarize_feature_matrix(mfcc),
        _summarize_feature_matrix(delta),
        _summarize_feature_matrix(delta_delta),
        _summarize_feature_matrix(extract_zero_crossing_rate(waveform), include_min_max=True),
        _summarize_feature_matrix(extract_rms_energy(waveform), include_min_max=True),
        _summarize_feature_matrix(extract_spectral_centroid(waveform, sample_rate=sample_rate), include_min_max=True),
        _summarize_feature_matrix(extract_spectral_bandwidth(waveform, sample_rate=sample_rate), include_min_max=True),
        _summarize_feature_matrix(extract_spectral_rolloff(waveform, sample_rate=sample_rate), include_min_max=True),
        _summarize_feature_matrix(extract_chroma(waveform, sample_rate=sample_rate), include_min_max=True),
    ]

    features = np.concatenate(feature_blocks).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
