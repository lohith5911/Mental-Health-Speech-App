from datetime import datetime, timedelta, timezone
import sqlite3
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient

from app import main
from app.auth import JWT_ALGORITHM, JWT_ISSUER, JWT_SECRET, hash_password

client = TestClient(main.app)


VALID_USER = {
    "email": "Person@Example.com",
    "display_name": "  Test Person  ",
    "password": "correct horse battery staple",
}


def register(database_path, payload=None):
    with patch.object(main, "DATABASE_PATH", database_path):
        return client.post("/api/auth/register", json=payload or VALID_USER)


def test_registers_user_returns_safe_profile_and_token(tmp_path):
    database_path = tmp_path / "auth.db"

    response = register(database_path)

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"] == {
        "id": 1,
        "email": "person@example.com",
        "display_name": "Test Person",
        "created_at": body["user"]["created_at"],
    }
    assert "password_hash" not in body

    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT email, display_name, password_hash FROM users").fetchone()

    assert row[0:2] == ("person@example.com", "Test Person")
    assert row[2] != VALID_USER["password"]
    assert row[2].startswith("$argon2")


def test_register_rejects_duplicate_normalized_email(tmp_path):
    database_path = tmp_path / "auth.db"

    assert register(database_path).status_code == 201
    duplicate = register(database_path, {**VALID_USER, "email": " person@example.com "})

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "An account with that email already exists."


def test_register_rejects_invalid_input(tmp_path):
    database_path = tmp_path / "auth.db"

    invalid_email = register(database_path, {**VALID_USER, "email": "not-an-email"})
    short_password = register(database_path, {**VALID_USER, "password": "short"})
    blank_name = register(database_path, {**VALID_USER, "display_name": "   "})

    assert invalid_email.status_code == 422
    assert short_password.status_code == 422
    assert blank_name.status_code == 422


def test_login_returns_token_for_correct_credentials(tmp_path):
    database_path = tmp_path / "auth.db"
    register(database_path)

    with patch.object(main, "DATABASE_PATH", database_path):
        response = client.post(
            "/api/auth/login",
            json={"email": " PERSON@example.com ", "password": VALID_USER["password"]},
        )

    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_rejects_wrong_password_and_unknown_email_generically(tmp_path):
    database_path = tmp_path / "auth.db"
    register(database_path)

    with patch.object(main, "DATABASE_PATH", database_path):
        wrong_password = client.post(
            "/api/auth/login",
            json={"email": VALID_USER["email"], "password": "wrong password"},
        )
        unknown_email = client.post(
            "/api/auth/login",
            json={"email": "unknown@example.com", "password": "wrong password"},
        )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json()["detail"] == "Invalid email or password."


def test_me_returns_authenticated_user_without_password_hash(tmp_path):
    database_path = tmp_path / "auth.db"
    response = register(database_path)
    token = response.json()["access_token"]

    with patch.object(main, "DATABASE_PATH", database_path):
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert me.status_code == 200
    assert me.json()["email"] == "person@example.com"
    assert me.json()["display_name"] == "Test Person"
    assert "password_hash" not in me.json()


def test_me_rejects_missing_and_invalid_tokens(tmp_path):
    database_path = tmp_path / "auth.db"
    register(database_path)

    with patch.object(main, "DATABASE_PATH", database_path):
        missing = client.get("/api/auth/me")
        invalid = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid"})

    assert missing.status_code == invalid.status_code == 401


def test_me_rejects_expired_token(tmp_path):
    database_path = tmp_path / "auth.db"
    register(database_path)
    expired = jwt.encode(
        {
            "sub": "1",
            "type": "access",
            "iss": JWT_ISSUER,
            "iat": datetime.now(timezone.utc) - timedelta(hours=1),
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    with patch.object(main, "DATABASE_PATH", database_path):
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401


def test_me_rejects_token_for_nonexistent_user(tmp_path):
    database_path = tmp_path / "auth.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        main.init_db()
    token = jwt.encode(
        {"sub": "999", "type": "access", "iss": JWT_ISSUER, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    with patch.object(main, "DATABASE_PATH", database_path):
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


def test_init_db_preserves_legacy_check_ins(tmp_path):
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

    with sqlite3.connect(database_path) as connection:
        check_in = connection.execute("SELECT id, emotion, confidence, duration_seconds FROM check_ins").fetchone()
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    assert check_in == (1, "sad", 0.6, 12)
    assert user_count == 0


def test_init_db_is_repeatable(tmp_path):
    database_path = tmp_path / "repeatable.db"
    with patch.object(main, "DATABASE_PATH", database_path):
        main.init_db()
        main.init_db()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM check_ins").fetchone()[0] == 0


def test_hash_password_helper_uses_argon2():
    encoded = hash_password("password-value")
    assert encoded.startswith("$argon2")
    assert encoded != "password-value"
