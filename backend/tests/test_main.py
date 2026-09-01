from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


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


def test_create_check_in_persists_record(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "happy", "confidence": 0.91, "duration_seconds": 42},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] == "happy"
    assert body["confidence"] == 0.91
    assert body["duration_seconds"] == 42
    assert body["id"] > 0


def test_list_check_ins_returns_created_records(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        client.post("/api/check-ins", json={"emotion": "happy", "confidence": 0.85, "duration_seconds": 12})
        client.post("/api/check-ins", json={"emotion": "sad", "confidence": 0.63, "duration_seconds": 30})
        response = client.get("/api/check-ins")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [item["emotion"] for item in body] == ["sad", "happy"]


def test_get_check_in_by_id_returns_record(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        created = client.post(
            "/api/check-ins",
            json={"emotion": "neutral", "confidence": 0.74, "duration_seconds": 18},
        ).json()
        response = client.get(f"/api/check-ins/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["emotion"] == "neutral"


def test_create_check_in_rejects_invalid_values(tmp_path: Path) -> None:
    with patch.object(main, "DATABASE_PATH", tmp_path / "checkins.db"):
        main.init_db()
        response = client.post(
            "/api/check-ins",
            json={"emotion": "unknown", "confidence": 1.2, "duration_seconds": 0},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    error_text = "\n".join(item.get("msg", "") for item in detail)
    assert "emotion" in error_text.lower() or "confidence" in error_text.lower()
