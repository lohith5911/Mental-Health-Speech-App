from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.services.trend_engine import (
    EMOTIONS,
    build_trend_insight,
    calculate_baseline,
    calculate_recent,
    classify_change,
    find_persistent_emotions,
)

client = TestClient(main.app)


def vector(**overrides: float) -> dict[str, float]:
    values = {emotion: 0.0 for emotion in EMOTIONS}
    values.update(overrides)
    return values


def row(probabilities: dict[str, float] | None = None, emotion: str = "neutral") -> dict[str, object]:
    return {
        "emotion": emotion,
        "probabilities": json.dumps(probabilities) if probabilities is not None else None,
    }


def insert_rows(database_path: Path, rows: list[dict[str, object]]) -> None:
    main.init_db()
    with sqlite3.connect(database_path) as connection:
        for index, check_in in enumerate(rows):
            connection.execute(
                """
                INSERT INTO check_ins (created_at, emotion, confidence, duration_seconds, probabilities)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"2026-09-{index + 1:02d}T12:00:00+00:00",
                    check_in["emotion"],
                    0.8,
                    10,
                    check_in["probabilities"],
                ),
            )
        connection.commit()


def test_zero_check_ins_returns_empty_valid_response(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        response = client.get("/api/insights/trends")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["sample_size"] == 0
    assert body["window_size"] == 7
    assert body["baseline"] == vector()
    assert body["recent"] == vector()
    assert body["change"] == vector()
    assert body["change_score"] == 0.0
    assert body["trend"] == "stable"
    assert body["persistent_emotions"] == []


def test_one_check_in_uses_all_available_data(tmp_path: Path) -> None:
    check_in = row(vector(happy=0.7, neutral=0.3), emotion="happy")
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        insert_rows(tmp_path / "checkins.db", [check_in])
        body = client.get("/api/insights/trends").json()

    assert body["sample_size"] == 1
    assert body["baseline"] == vector(happy=0.7, neutral=0.3)
    assert body["recent"] == body["baseline"]
    assert body["change_score"] == 0.0


def test_fewer_than_seven_check_ins_use_all_rows(tmp_path: Path) -> None:
    rows = [row(vector(neutral=1.0)) for _ in range(3)]
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        insert_rows(tmp_path / "checkins.db", rows)
        body = client.get("/api/insights/trends").json()

    assert body["sample_size"] == 3
    assert body["window_size"] == 7
    assert body["recent"] == vector(neutral=1.0)


def test_exactly_seven_check_ins_use_default_window(tmp_path: Path) -> None:
    rows = [row(vector(happy=1.0)) for _ in range(7)]
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        insert_rows(tmp_path / "checkins.db", rows)
        body = client.get("/api/insights/trends").json()

    assert body["sample_size"] == 7
    assert body["recent"] == vector(happy=1.0)


def test_more_than_seven_check_ins_limit_recent_window(tmp_path: Path) -> None:
    rows = [row(vector(happy=1.0)) for _ in range(3)] + [row(vector(sad=1.0)) for _ in range(7)]
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        insert_rows(tmp_path / "checkins.db", rows)
        body = client.get("/api/insights/trends?window_size=7").json()

    assert body["sample_size"] == 10
    assert body["recent"] == vector(sad=1.0)
    assert body["baseline"] == vector(happy=0.3, sad=0.7)


def test_baseline_averages_probability_vectors_with_all_emotions() -> None:
    baseline = calculate_baseline([
        row(vector(happy=0.8, neutral=0.2)),
        row(vector(happy=0.2, neutral=0.8)),
    ])

    assert set(baseline) == set(EMOTIONS)
    assert baseline == vector(happy=0.5, neutral=0.5)


def test_recent_distribution_respects_custom_window() -> None:
    rows = [row(vector(happy=1.0)), row(vector(sad=1.0)), row(vector(neutral=1.0))]

    assert calculate_recent(rows, window_size=2) == vector(happy=0.5, sad=0.5)


def test_change_and_total_variation_score() -> None:
    result = build_trend_insight([
        row(vector(happy=1.0)),
        row(vector(sad=1.0)),
    ], window_size=1)

    assert result["change"] == vector(happy=0.5, sad=-0.5)
    assert result["change_score"] == 0.5


def test_trend_classification_thresholds() -> None:
    assert classify_change(0.09) == "stable"
    assert classify_change(0.10) == "change_detected"
    assert classify_change(0.19) == "change_detected"
    assert classify_change(0.20) == "significant_change"


def test_persistent_emotion_requires_three_of_five_recent_rows() -> None:
    baseline = vector(neutral=1.0)
    rows = [row(vector(sad=0.2, neutral=0.8)) for _ in range(3)]
    rows.extend(row(vector(neutral=1.0)) for _ in range(2))

    assert find_persistent_emotions(rows, baseline) == ["sad"]


def test_persistent_emotion_is_not_reported_for_two_rows() -> None:
    baseline = vector(neutral=1.0)
    rows = [row(vector(sad=0.2, neutral=0.8)) for _ in range(2)]
    rows.extend(row(vector(neutral=1.0)) for _ in range(3))

    assert find_persistent_emotions(rows, baseline) == []


def test_legacy_null_probabilities_fall_back_to_winning_label() -> None:
    baseline = calculate_baseline([row(None, emotion="sad"), row(None, emotion="happy")])

    assert baseline == vector(sad=0.5, happy=0.5)


def test_invalid_window_sizes_are_rejected(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        below_minimum = client.get("/api/insights/trends?window_size=0")
        above_maximum = client.get("/api/insights/trends?window_size=31")

    assert below_minimum.status_code == 422
    assert above_maximum.status_code == 422


def test_custom_maximum_window_size_is_accepted(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        response = client.get("/api/insights/trends?window_size=30")

    assert response.status_code == 200
    assert response.json()["window_size"] == 30
