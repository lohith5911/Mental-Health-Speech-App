from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.src.features.acoustic_features import extract_acoustic_features, extract_acoustic_features_from_file


SAMPLE_RATE = 16000


def sine_wave(frequency: float = 220.0, duration: float = 1.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    timeline = np.arange(int(duration * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)


def test_sine_wave_has_expected_duration_pitch_and_finite_metrics() -> None:
    features = extract_acoustic_features(sine_wave(), SAMPLE_RATE)

    assert features.duration_seconds == pytest.approx(1.0)
    assert np.isfinite(features.rms_mean)
    assert np.isfinite(features.rms_std)
    assert np.isfinite(features.zcr_mean)
    assert np.isfinite(features.zcr_std)
    assert features.pitch_mean_hz == pytest.approx(220.0, abs=5.0)
    assert features.silence_ratio == pytest.approx(0.0)
    assert features.speaking_rate_proxy is not None


def test_silent_waveform_returns_explicit_silent_measurements() -> None:
    features = extract_acoustic_features(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    assert features.rms_mean == 0.0
    assert features.rms_std == 0.0
    assert features.zcr_mean == 0.0
    assert features.zcr_std == 0.0
    assert features.silence_ratio == 1.0
    assert features.pitch_mean_hz is None
    assert features.pitch_std_hz is None
    assert features.pitch_range_hz is None
    assert features.speaking_rate_proxy is None
    assert features.is_silent is True


def test_white_noise_is_finite_and_bounded() -> None:
    random = np.random.default_rng(42)
    features = extract_acoustic_features((0.5 * random.standard_normal(SAMPLE_RATE)).astype(np.float32), SAMPLE_RATE)

    for value in (features.rms_mean, features.rms_std, features.zcr_mean, features.zcr_std):
        assert np.isfinite(value)
    assert 0.0 <= features.zcr_mean <= 1.0
    assert 0.0 <= features.silence_ratio <= 1.0
    assert features.speaking_rate_proxy is None or features.speaking_rate_proxy >= 0.0


def test_direct_waveform_rejects_invalid_shapes_values_and_rates() -> None:
    with pytest.raises(ValueError):
        extract_acoustic_features(np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError):
        extract_acoustic_features(np.array([np.nan], dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError):
        extract_acoustic_features(np.ones(10, dtype=np.float32), 0)
    with pytest.raises(ValueError):
        extract_acoustic_features(np.array([], dtype=np.float32), SAMPLE_RATE)


def test_very_short_waveform_returns_stable_nullable_measurements() -> None:
    features = extract_acoustic_features(sine_wave(duration=0.01), SAMPLE_RATE)

    assert features.duration_seconds == pytest.approx(0.01)
    assert features.pitch_mean_hz is None
    assert features.speaking_rate_proxy is None
    assert 0.0 <= features.silence_ratio <= 1.0


def test_file_extraction_resamples_and_downmixes_without_changing_duration(tmp_path: Path) -> None:
    source_rate = 8000
    source = np.column_stack((sine_wave(220.0, 1.0, source_rate), sine_wave(220.0, 1.0, source_rate)))
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, source, source_rate)

    features = extract_acoustic_features_from_file(audio_path)

    assert features.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert features.pitch_mean_hz == pytest.approx(220.0, abs=8.0)
    assert np.isfinite(features.rms_mean)


def test_file_extraction_rejects_empty_and_malformed_files(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.wav"
    sf.write(empty_path, np.array([], dtype=np.float32), SAMPLE_RATE)
    malformed_path = tmp_path / "malformed.wav"
    malformed_path.write_bytes(b"not audio")

    with pytest.raises(ValueError):
        extract_acoustic_features_from_file(empty_path)
    with pytest.raises(ValueError):
        extract_acoustic_features_from_file(malformed_path)
