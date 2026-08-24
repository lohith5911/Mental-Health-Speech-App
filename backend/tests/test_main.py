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
    with patch.object(main, "MODEL_PATH", tmp_path / "missing-classifier.pkl"):
        response = client.post(
            "/api/analyze-emotion",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 503
    assert "missing" in response.json()["detail"].lower()


def test_analyze_emotion_reports_prediction_failure(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)

    with patch.object(main, "predict_emotion_from_file", side_effect=RuntimeError("model error")):
        with wav_path.open("rb") as audio:
            response = client.post("/api/analyze-emotion", files={"file": (wav_path.name, audio, "audio/wav")})

    assert response.status_code == 500
    assert response.json()["detail"] == "Emotion prediction failed."
