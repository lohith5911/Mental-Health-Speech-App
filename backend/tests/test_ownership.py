from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

VALID_FEATURES = {
    "duration_seconds": 1.0,
    "rms_mean": 0.3,
    "rms_std": 0.1,
    "zcr_mean": 0.05,
    "zcr_std": 0.02,
    "pitch_mean_hz": 220.0,
    "pitch_std_hz": 5.0,
    "pitch_range_hz": 15.0,
    "silence_ratio": 0.1,
    "speaking_rate_proxy": 2.0,
    "is_silent": False,
    "is_noisy": False,
}
VALID_PROBABILITIES = {
    "angry": 0.01,
    "disgust": 0.01,
    "fear": 0.02,
    "happy": 0.10,
    "neutral": 0.80,
    "sad": 0.06,
}


def register(email: str) -> tuple[dict[str, str], int]:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "display_name": email.split("@")[0], "password": "correct horse battery staple"},
    )
    assert response.status_code == 201
    body = response.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]["id"]


def create_check_in(headers: dict[str, str], emotion: str = "happy"):
    return client.post(
        "/api/check-ins",
        json={
            "emotion": emotion,
            "confidence": 0.8,
            "duration_seconds": 10,
            "user_id": 999,
            "owner_id": 999,
            "account_id": 999,
        },
        headers=headers,
    )


def test_check_in_endpoints_require_authentication(tmp_path: Path):
    with patch.object(main, "DATABASE_PATH", tmp_path / "ownership.db"):
        main.init_db()
        assert client.post(
            "/api/check-ins", json={"emotion": "happy", "confidence": 0.8}
        ).status_code == 401
        assert client.get("/api/check-ins").status_code == 401
        assert client.get("/api/check-ins/1").status_code == 401
        assert client.post(
            "/api/check-ins/analyze-and-save", files={"file": ("sample.wav", b"audio", "audio/wav")}
        ).status_code == 401
        assert client.get("/api/insights/trends").status_code == 401


def test_users_can_only_create_list_and_read_their_own_check_ins(tmp_path: Path):
    database_path = tmp_path / "ownership.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        headers_a, user_a = register("owner-a@example.com")
        headers_b, user_b = register("owner-b@example.com")
        created_a = create_check_in(headers_a, "happy")
        created_b = create_check_in(headers_b, "sad")

        assert created_a.status_code == created_b.status_code == 200
        id_a = created_a.json()["id"]
        id_b = created_b.json()["id"]
        assert client.get("/api/check-ins", headers=headers_a).json()[0]["emotion"] == "happy"
        assert client.get("/api/check-ins", headers=headers_b).json()[0]["emotion"] == "sad"
        assert client.get(f"/api/check-ins/{id_b}", headers=headers_a).status_code == 404
        assert client.get(f"/api/check-ins/{id_a}", headers=headers_b).status_code == 404

        with sqlite3.connect(database_path) as connection:
            owners = connection.execute(
                "SELECT id, user_id FROM check_ins ORDER BY id"
            ).fetchall()
        assert owners == [(id_a, user_a), (id_b, user_b)]


def test_client_cannot_override_jwt_ownership(tmp_path: Path):
    database_path = tmp_path / "ownership.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        headers_a, user_a = register("tamper-a@example.com")
        headers_b, user_b = register("tamper-b@example.com")
        response = create_check_in(headers_a)

        assert response.status_code == 200
        with sqlite3.connect(database_path) as connection:
            owner = connection.execute(
                "SELECT user_id FROM check_ins WHERE id = ?", (response.json()["id"],)
            ).fetchone()[0]
        assert owner == user_a
        assert owner != user_b


def test_legacy_records_remain_unassigned_and_hidden(tmp_path: Path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE check_ins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration_seconds INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO check_ins (created_at, emotion, confidence, duration_seconds) VALUES (?, ?, ?, ?)",
            ("2026-09-01T12:00:00+00:00", "sad", 0.6, 12),
        )
        connection.commit()

    with patch.object(main, "DATABASE_PATH", database_path):
        main.init_db()
        headers, _ = register("legacy-reader@example.com")
        assert client.get("/api/check-ins", headers=headers).json() == []
        assert client.get("/api/check-ins/1", headers=headers).status_code == 404
        trends = client.get("/api/insights/trends", headers=headers)

        with sqlite3.connect(database_path) as connection:
            row = connection.execute("SELECT user_id FROM check_ins WHERE id = 1").fetchone()
        assert row == (None,)
        assert trends.status_code == 200
        assert trends.json()["sample_size"] == 0


def test_ownership_migration_is_repeatable_and_indexed(tmp_path: Path):
    database_path = tmp_path / "migration.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        main.init_db()
        main.init_db()

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(check_ins)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(check_ins)")}
    assert {"user_id", "idempotency_scope_key"}.issubset(columns)
    assert "idx_check_ins_user_id" in indexes
    assert "idx_check_ins_idempotency_scope_key" in indexes


def test_analyze_and_save_is_user_owned_and_scoped_by_user(tmp_path: Path):
    database_path = tmp_path / "analyze.db"
    prediction = {
        "emotion": "neutral",
        "confidence": 0.8,
        "model_version": "v4",
        "probabilities": VALID_PROBABILITIES,
    }
    with patch.object(main, "DATABASE_PATH", database_path), patch.object(
        main,
        "_process_uploaded_audio",
        new=AsyncMock(return_value=(prediction, VALID_FEATURES)),
    ):
        headers_a, user_a = register("analyze-a@example.com")
        headers_b, user_b = register("analyze-b@example.com")

        def submit(headers: dict[str, str], key: str):
            return client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                headers={**headers, "Idempotency-Key": key},
            )

        first_a = submit(headers_a, "same-key")
        repeated_a = submit(headers_a, "same-key")
        first_b = submit(headers_b, "same-key")

        assert first_a.status_code == repeated_a.status_code == first_b.status_code == 200
        assert first_a.json()["id"] == repeated_a.json()["id"]
        assert first_a.json()["id"] != first_b.json()["id"]

        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                "SELECT user_id, idempotency_key, idempotency_scope_key FROM check_ins ORDER BY id"
            ).fetchall()
        assert rows == [
            (user_a, None, f"{user_a}:same-key"),
            (user_b, None, f"{user_b}:same-key"),
        ]


def test_trends_only_include_authenticated_users(tmp_path: Path):
    database_path = tmp_path / "trends.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        headers_a, user_a = register("trend-a@example.com")
        headers_b, user_b = register("trend-b@example.com")
        with sqlite3.connect(database_path) as connection:
            for user_id, emotion in ((user_a, "happy"), (user_b, "sad"), (None, "angry")):
                connection.execute(
                    """
                    INSERT INTO check_ins (
                        created_at, emotion, confidence, duration_seconds, probabilities, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-09-01T12:00:00+00:00",
                        emotion,
                        0.8,
                        10,
                        json.dumps({emotion: 1.0} if emotion in {"happy", "sad", "angry"} else VALID_PROBABILITIES),
                        user_id,
                    ),
                )
            connection.commit()

        trends_a = client.get("/api/insights/trends", headers=headers_a).json()
        trends_b = client.get("/api/insights/trends", headers=headers_b).json()

    assert trends_a["sample_size"] == 1
    assert trends_b["sample_size"] == 1
    assert trends_a["baseline"]["happy"] == 1.0
    assert trends_b["baseline"]["sad"] == 1.0
