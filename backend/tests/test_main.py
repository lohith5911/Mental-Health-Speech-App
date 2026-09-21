from __future__ import annotations

import subprocess
import sqlite3
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app import main
from ml.src.features.acoustic_features import AcousticFeatures


client = TestClient(main.app)

VALID_PROBABILITIES = {
    "angry": 0.01,
    "disgust": 0.01,
    "fear": 0.02,
    "happy": 0.10,
    "neutral": 0.80,
    "sad": 0.06,
}

VALID_ACOUSTIC_FEATURES = {
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


def _auth_headers(email: str = "tester@example.com") -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "display_name": "Test User",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _write_wav(path: Path) -> None:
    sample_rate = 16000
    waveform = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sample_rate, endpoint=False)).astype(np.float32)
    sf.write(path, waveform, sample_rate)


def test_analyze_emotion_wav_returns_prediction(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)

    with wav_path.open("rb") as audio:
        response = client.post("/api/analyze-emotion", files={"file": (wav_path.name, audio, "audio/wav")})

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] in {"angry", "disgust", "fear", "happy", "neutral", "sad"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_analyze_emotion_webm_returns_prediction(tmp_path: Path) -> None:
    source_wav = tmp_path / "source.wav"
    webm_path = tmp_path / "sample.webm"
    _write_wav(source_wav)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source_wav), "-c:a", "libopus", str(webm_path)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0

    with webm_path.open("rb") as audio:
        response = client.post("/api/analyze-emotion", files={"file": (webm_path.name, audio, "audio/webm")})

    assert response.status_code == 200
    assert response.json()["emotion"] in {"angry", "disgust", "fear", "happy", "neutral", "sad"}


def test_analyze_emotion_rejects_invalid_audio() -> None:
    response = client.post(
        "/api/analyze-emotion",
        files={"file": ("broken.wav", b"not audio", "audio/wav")},
    )

    assert response.status_code == 422
    assert "Invalid audio file" in response.json()["detail"]


def test_analyze_emotion_reports_missing_models(tmp_path: Path) -> None:
    with patch.object(main, "V4_ARTIFACT_PATHS", (tmp_path / "missing-model.pt",)):
        response = client.post(
            "/api/analyze-emotion",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 500
    assert "missing" in response.json()["detail"].lower()
def test_analyze_emotion_reports_prediction_failure(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)

    with patch.object(main, "predict_v4_emotion", side_effect=RuntimeError("model error")):
        with wav_path.open("rb") as audio:
            response = client.post("/api/analyze-emotion", files={"file": (wav_path.name, audio, "audio/wav")})

    assert response.status_code == 500
    assert response.json()["detail"] == "Emotion prediction failed."


def test_analyze_emotion_adds_acoustic_features_from_converted_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    prediction = {
        "emotion": "neutral",
        "confidence": 0.8,
        "model_version": "v4",
        "probabilities": VALID_PROBABILITIES,
    }

    with patch.object(main, "predict_v4_emotion", return_value=prediction) as predict, patch.object(
        main, "extract_acoustic_features_from_file", return_value=AcousticFeatures(**VALID_ACOUSTIC_FEATURES)
    ) as extract:
        with wav_path.open("rb") as audio:
            response = client.post("/api/analyze-emotion", files={"file": (wav_path.name, audio, "audio/wav")})

    assert response.status_code == 200
    assert response.json() == {**prediction, "acoustic_features": VALID_ACOUSTIC_FEATURES}
    assert predict.call_args.args[0] == extract.call_args.args[0]


def test_create_check_in_persists_record(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "happy", "confidence": 0.91, "duration_seconds": 42},
            headers=headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] == "happy"
    assert body["confidence"] == 0.91
    assert body["duration_seconds"] == 42
    assert body["id"] > 0


def test_old_format_check_in_still_succeeds(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "happy", "confidence": 0.8, "duration_seconds": 10},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["model_version"] is None
    assert response.json()["probabilities"] is None
    assert response.json()["acoustic_features"] is None


def test_v4_probabilities_are_persisted_and_returned(tmp_path: Path) -> None:
    payload = {
        "emotion": "neutral",
        "confidence": 0.8,
        "duration_seconds": 10,
        "model_version": "v4",
        "probabilities": VALID_PROBABILITIES,
        "acoustic_features": VALID_ACOUSTIC_FEATURES,
    }
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        created = client.post("/api/check-ins", json=payload, headers=headers)
        fetched = client.get(f"/api/check-ins/{created.json()['id']}", headers=headers)

    assert created.status_code == 200
    assert fetched.status_code == 200
    assert fetched.json()["model_version"] == "v4"
    assert fetched.json()["probabilities"] == VALID_PROBABILITIES
    assert fetched.json()["acoustic_features"] == VALID_ACOUSTIC_FEATURES


def test_invalid_acoustic_features_are_rejected(tmp_path: Path) -> None:
    invalid_features = {**VALID_ACOUSTIC_FEATURES, "silence_ratio": 1.5}
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.8, "acoustic_features": invalid_features},
            headers=headers,
        )

    assert response.status_code == 422


def test_invalid_probability_values_are_rejected(tmp_path: Path) -> None:
    probabilities = {**VALID_PROBABILITIES, "happy": 1.1}
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.8, "probabilities": probabilities},
            headers=headers,
        )

    assert response.status_code == 422


def test_missing_emotion_probability_is_rejected(tmp_path: Path) -> None:
    probabilities = {key: value for key, value in VALID_PROBABILITIES.items() if key != "sad"}
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.8, "probabilities": probabilities},
            headers=headers,
        )

    assert response.status_code == 422


def test_probability_sum_outside_tolerance_is_rejected(tmp_path: Path) -> None:
    probabilities = {**VALID_PROBABILITIES, "happy": 0.30}
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.8, "probabilities": probabilities},
            headers=headers,
        )

    assert response.status_code == 422


def test_existing_database_records_survive_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "checkins.db"
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
    with patch.object(main, "DATABASE_PATH", database_path):
        main.init_db()
        headers = _auth_headers()
        response = client.get("/api/check-ins", headers=headers)

    assert response.status_code == 200
    assert response.json() == []


def test_list_check_ins_returns_created_records(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        headers = _auth_headers()
        client.post("/api/check-ins", json={"emotion": "happy", "confidence": 0.85, "duration_seconds": 12}, headers=headers)
        client.post("/api/check-ins", json={"emotion": "sad", "confidence": 0.63, "duration_seconds": 30}, headers=headers)
        response = client.get("/api/check-ins", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [item["emotion"] for item in body] == ["sad", "happy"]


def test_get_check_in_by_id_remains_compatible(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        created = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.74, "duration_seconds": 18},
            headers=headers,
        ).json()
        response = client.get(f"/api/check-ins/{created['id']}", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_check_in_by_id_returns_record(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        headers = _auth_headers()
        created = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.74, "duration_seconds": 18},
            headers=headers,
        ).json()
        response = client.get(f"/api/check-ins/{created['id']}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["emotion"] == "neutral"


def test_create_check_in_rejects_invalid_values(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "unknown", "confidence": 1.2, "duration_seconds": 0},
            headers=headers,
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    error_text = "\n".join(item.get("msg", "") for item in detail)
    assert "emotion" in error_text.lower() or "confidence" in error_text.lower()


def _mock_prediction():
    return {
        "emotion": "neutral",
        "confidence": 0.8,
        "model_version": "v4",
        "probabilities": VALID_PROBABILITIES,
    }


def test_analyze_and_save_persists_one_complete_record(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    features = AcousticFeatures(**{**VALID_ACOUSTIC_FEATURES, "duration_seconds": 1.0})

    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"), patch.object(
        main, "predict_v4_emotion", return_value=_mock_prediction()
    ), patch.object(main, "extract_acoustic_features_from_file", return_value=features):
        headers = _auth_headers()
        with wav_path.open("rb") as audio:
            response = client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": (wav_path.name, audio, "audio/wav")},
                data={"client_duration_seconds": "1"},
                headers={**headers, "Idempotency-Key": "request-1"},
            )
        records = client.get("/api/check-ins", headers=headers).json()

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == records[0]["id"]
    assert body["model_version"] == "v4"
    assert body["quality"]["status"] == "usable"
    assert body["acoustic_features"] == VALID_ACOUSTIC_FEATURES
    assert body["duration_seconds"] == 1
    assert len(records) == 1


def test_analyze_and_save_supports_webm(tmp_path: Path) -> None:
    source_wav = tmp_path / "source.wav"
    webm_path = tmp_path / "sample.webm"
    _write_wav(source_wav)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source_wav), "-c:a", "libopus", str(webm_path)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0

    with patch.object(main, "predict_v4_emotion", return_value=_mock_prediction()), patch.object(
        main, "extract_acoustic_features_from_file", return_value=AcousticFeatures(**VALID_ACOUSTIC_FEATURES)
    ), patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        with webm_path.open("rb") as audio:
            response = client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": (webm_path.name, audio, "audio/webm")},
                headers=headers,
            )

    assert response.status_code == 200


def test_analyze_and_save_rejects_unsupported_format(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        headers = _auth_headers()
        response = client.post(
            "/api/check-ins/analyze-and-save",
            files={"file": ("sample.flac", b"audio", "audio/flac")},
            headers=headers,
        )
    assert response.status_code == 415


def test_analyze_and_save_does_not_persist_when_prediction_fails(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    database_path = tmp_path / "checkins.db"
    with patch.object(main, "DATABASE_PATH", database_path), patch.object(
        main, "predict_v4_emotion", side_effect=RuntimeError("model error")
    ):
        headers = _auth_headers()
        with wav_path.open("rb") as audio:
            response = client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": (wav_path.name, audio, "audio/wav")},
                headers=headers,
            )
        assert response.status_code == 500
        assert client.get("/api/check-ins", headers=headers).json() == []


def test_analyze_and_save_does_not_persist_when_extraction_fails(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"), patch.object(
        main, "predict_v4_emotion", return_value=_mock_prediction()
    ), patch.object(main, "extract_acoustic_features_from_file", side_effect=RuntimeError("features")):
        headers = _auth_headers()
        with wav_path.open("rb") as audio:
            response = client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": (wav_path.name, audio, "audio/wav")},
                headers=headers,
            )
        assert response.status_code == 500
        assert client.get("/api/check-ins", headers=headers).json() == []


def test_analyze_and_save_idempotency_and_different_keys(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    database_path = tmp_path / "checkins.db"
    with patch.object(main, "DATABASE_PATH", database_path), patch.object(
        main, "predict_v4_emotion", return_value=_mock_prediction()
    ), patch.object(main, "extract_acoustic_features_from_file", return_value=AcousticFeatures(**VALID_ACOUSTIC_FEATURES)):
        headers = _auth_headers()
        def submit(key: str):
            with wav_path.open("rb") as audio:
                return client.post(
                    "/api/check-ins/analyze-and-save",
                    files={"file": (wav_path.name, audio, "audio/wav")},
                    headers={**headers, "Idempotency-Key": key},
                )

        first = submit("same-key")
        repeated = submit("same-key")
        separate = submit("different-key")
        records = client.get("/api/check-ins", headers=headers).json()

    assert first.status_code == repeated.status_code == separate.status_code == 200
    assert first.json()["id"] == repeated.json()["id"]
    assert separate.json()["id"] != first.json()["id"]
    assert len(records) == 2


def test_analyze_and_save_records_duration_mismatch_reason(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"), patch.object(
        main, "predict_v4_emotion", return_value=_mock_prediction()
    ), patch.object(main, "extract_acoustic_features_from_file", return_value=AcousticFeatures(**VALID_ACOUSTIC_FEATURES)):
        headers = _auth_headers()
        with wav_path.open("rb") as audio:
            response = client.post(
                "/api/check-ins/analyze-and-save",
                files={"file": (wav_path.name, audio, "audio/wav")},
                data={"client_duration_seconds": "30"},
                headers=headers,
            )

    assert response.status_code == 200
    assert "frontend_duration_mismatch" in response.json()["quality"]["reasons"]
