"""Deterministic signal-quality assessment for speech check-ins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


MIN_DURATION_SECONDS = 0.5
SILENCE_RATIO_THRESHOLD = 0.85
LOW_RMS_THRESHOLD = 0.01


class AcousticMeasurements(Protocol):
    duration_seconds: float
    rms_mean: float
    silence_ratio: float
    is_silent: bool
    is_noisy: bool


@dataclass(frozen=True)
class QualityAssessment:
    status: str
    reasons: list[str]


def assess_audio_quality(features: AcousticMeasurements) -> QualityAssessment:
    """Classify audio usability from signal measurements only."""
    reasons: list[str] = []

    if features.duration_seconds < MIN_DURATION_SECONDS:
        reasons.append("duration_below_minimum")
    if features.is_silent:
        reasons.append("audio_marked_silent")
    if features.silence_ratio >= SILENCE_RATIO_THRESHOLD:
        reasons.append("silence_ratio_above_threshold")
    if features.is_noisy:
        reasons.append("noise_indicator_detected")
    if features.rms_mean < LOW_RMS_THRESHOLD and not features.is_silent:
        reasons.append("rms_below_threshold")

    if features.duration_seconds < MIN_DURATION_SECONDS:
        status = "insufficient_audio"
    elif features.is_silent or features.silence_ratio >= SILENCE_RATIO_THRESHOLD:
        status = "mostly_silent"
    elif features.is_noisy:
        status = "noisy_signal"
    elif features.rms_mean < LOW_RMS_THRESHOLD:
        status = "low_signal"
    else:
        status = "usable"

    return QualityAssessment(status=status, reasons=reasons)