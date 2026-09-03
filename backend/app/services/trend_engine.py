from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad")
DEFAULT_WINDOW_SIZE = 7
MAX_WINDOW_SIZE = 30
PERSISTENCE_WINDOW_SIZE = 5
PERSISTENCE_MINIMUM_COUNT = 3
PERSISTENCE_DELTA = 0.10
STABLE_THRESHOLD = 0.10
SIGNIFICANT_CHANGE_THRESHOLD = 0.20


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return row.get(key)


def _fallback_vector(emotion: Any) -> dict[str, float]:
    """Use a one-hot winning-label vector for legacy rows without probabilities."""
    vector = {label: 0.0 for label in EMOTIONS}
    if emotion in vector:
        vector[emotion] = 1.0
    return vector


def _row_vector(row: Mapping[str, Any]) -> dict[str, float]:
    raw_probabilities = _row_value(row, "probabilities")
    if raw_probabilities:
        try:
            probabilities = json.loads(raw_probabilities) if isinstance(raw_probabilities, str) else raw_probabilities
            if set(probabilities) == set(EMOTIONS):
                values = {label: float(probabilities[label]) for label in EMOTIONS}
                total = sum(values.values())
                if (
                    all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values())
                    and total > 0.0
                ):
                    return {label: values[label] / total for label in EMOTIONS}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # Pre-M5.1 records have no probability JSON, so use their winning label only.
    return _fallback_vector(_row_value(row, "emotion"))


def _average_vectors(vectors: Sequence[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {label: 0.0 for label in EMOTIONS}

    count = len(vectors)
    averages = {label: sum(vector[label] for vector in vectors) / count for label in EMOTIONS}
    total = sum(averages.values())
    if total == 0.0:
        return averages
    return {label: averages[label] / total for label in EMOTIONS}


def calculate_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return _average_vectors([_row_vector(row) for row in rows])


def calculate_recent(rows: Sequence[Mapping[str, Any]], window_size: int = DEFAULT_WINDOW_SIZE) -> dict[str, float]:
    return _average_vectors([_row_vector(row) for row in rows[:window_size]])


def classify_change(change_score: float) -> str:
    if change_score < STABLE_THRESHOLD:
        return "stable"
    if change_score < SIGNIFICANT_CHANGE_THRESHOLD:
        return "change_detected"
    return "significant_change"


def find_persistent_emotions(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, float],
) -> list[str]:
    recent_rows = rows[:PERSISTENCE_WINDOW_SIZE]
    if len(recent_rows) < PERSISTENCE_MINIMUM_COUNT:
        return []

    vectors = [_row_vector(row) for row in recent_rows]
    return [
        label
        for label in EMOTIONS
        if sum(vector[label] - baseline[label] >= PERSISTENCE_DELTA for vector in vectors)
        >= PERSISTENCE_MINIMUM_COUNT
    ]


def build_trend_insight(rows: Sequence[Mapping[str, Any]], window_size: int = DEFAULT_WINDOW_SIZE) -> dict[str, Any]:
    materialized_rows = list(rows)
    baseline = calculate_baseline(materialized_rows)
    recent_rows = materialized_rows[:window_size]
    recent = calculate_recent(materialized_rows, window_size)
    change = {label: recent[label] - baseline[label] for label in EMOTIONS}
    change_score = 0.5 * sum(abs(value) for value in change.values())

    return {
        "status": "ok",
        "sample_size": len(materialized_rows),
        "window_size": window_size,
        "baseline": baseline,
        "recent": recent,
        "change": change,
        "change_score": change_score,
        "trend": classify_change(change_score),
        "persistent_emotions": find_persistent_emotions(recent_rows, baseline),
    }
