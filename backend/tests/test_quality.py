from types import SimpleNamespace

from app.services.quality import (
    LOW_RMS_THRESHOLD,
    MIN_DURATION_SECONDS,
    SILENCE_RATIO_THRESHOLD,
    assess_audio_quality,
)


def measurements(**overrides):
    values = {
        "duration_seconds": 2.0,
        "rms_mean": 0.1,
        "silence_ratio": 0.1,
        "is_silent": False,
        "is_noisy": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_quality_is_usable_for_clear_signal():
    assert assess_audio_quality(measurements()).status == "usable"


def test_quality_marks_low_signal():
    result = assess_audio_quality(measurements(rms_mean=LOW_RMS_THRESHOLD - 0.001))
    assert result.status == "low_signal"
    assert "rms_below_threshold" in result.reasons


def test_quality_marks_mostly_silent():
    result = assess_audio_quality(measurements(silence_ratio=SILENCE_RATIO_THRESHOLD))
    assert result.status == "mostly_silent"
    assert "silence_ratio_above_threshold" in result.reasons


def test_quality_marks_noisy_signal():
    result = assess_audio_quality(measurements(is_noisy=True))
    assert result.status == "noisy_signal"
    assert result.reasons == ["noise_indicator_detected"]


def test_quality_marks_insufficient_audio():
    result = assess_audio_quality(measurements(duration_seconds=MIN_DURATION_SECONDS - 0.01))
    assert result.status == "insufficient_audio"
    assert "duration_below_minimum" in result.reasons


def test_quality_collects_multiple_signal_reasons():
    result = assess_audio_quality(
        measurements(
            duration_seconds=0.2,
            rms_mean=0.001,
            silence_ratio=0.9,
            is_silent=True,
            is_noisy=True,
        )
    )
    assert result.status == "insufficient_audio"
    assert len(result.reasons) == 4