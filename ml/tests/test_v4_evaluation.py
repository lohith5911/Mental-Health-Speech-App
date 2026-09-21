from __future__ import annotations

from pathlib import Path

import pytest

from ml.scripts.evaluate_v4 import (
    EMOTION_CLASSES,
    build_evaluation_payload,
    calculate_evaluation_metrics,
    verify_speaker_disjointness,
)


def test_metric_calculation_has_expected_overall_values() -> None:
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    metrics = calculate_evaluation_metrics(labels, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_metric_calculation_uses_fixed_class_order_and_per_class_support() -> None:
    true_labels = ["angry", "angry", "disgust", "fear", "happy", "neutral", "sad"]
    predicted_labels = ["angry", "sad", "disgust", "fear", "neutral", "neutral", "happy"]

    metrics = calculate_evaluation_metrics(true_labels, predicted_labels)

    assert EMOTION_CLASSES == ("angry", "disgust", "fear", "happy", "neutral", "sad")
    assert list(metrics["per_class"]) == list(EMOTION_CLASSES)
    assert metrics["per_class"]["angry"]["support"] == 2
    assert metrics["per_class"]["disgust"]["support"] == 1
    assert len(metrics["confusion_matrix"]) == 6
    assert all(len(row) == 6 for row in metrics["confusion_matrix"])


def test_speaker_disjointness_accepts_disjoint_splits() -> None:
    split = {
        "train": [Path("1001_DFA_ANG_XX.wav")],
        "validation": [Path("1002_DFA_ANG_XX.wav")],
        "test": [Path("1003_DFA_ANG_XX.wav")],
    }

    verify_speaker_disjointness(split)


def test_speaker_disjointness_rejects_train_test_overlap() -> None:
    split = {
        "train": [Path("1001_DFA_ANG_XX.wav")],
        "validation": [Path("1002_DFA_ANG_XX.wav")],
        "test": [Path("1001_DFA_DIS_XX.wav")],
    }

    with pytest.raises(ValueError, match="not speaker-independent"):
        verify_speaker_disjointness(split)


def test_speaker_disjointness_rejects_validation_test_overlap() -> None:
    split = {
        "train": [Path("1001_DFA_ANG_XX.wav")],
        "validation": [Path("1002_DFA_ANG_XX.wav")],
        "test": [Path("1002_DFA_DIS_XX.wav")],
    }

    with pytest.raises(ValueError, match="not speaker-independent"):
        verify_speaker_disjointness(split)


def test_evaluation_payload_schema() -> None:
    metrics = calculate_evaluation_metrics(EMOTION_CLASSES, EMOTION_CLASSES)
    payload = build_evaluation_payload(metrics, test_files=6, test_speakers=2)

    assert payload["model_version"] == "v4"
    assert payload["dataset"] == "CREMA-D"
    assert payload["evaluation_protocol"] == "speaker-independent"
    assert payload["split_seed"] == 42
    assert payload["test_files"] == 6
    assert payload["test_speakers"] == 2
    assert payload["classes"] == list(EMOTION_CLASSES)
    assert payload["speaker_separation_verified"] is True
    assert list(payload["test_support"]) == list(EMOTION_CLASSES)
    assert set(payload["metrics"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    }
