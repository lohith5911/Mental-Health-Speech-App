"""Feature extraction utilities for future speech emotion recognition work."""

from __future__ import annotations

import numpy as np


def extract_mfcc(waveform: np.ndarray, sample_rate: int, n_mfcc: int = 13):
    """Compute MFCCs from a waveform."""
    import librosa

    return librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)


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
