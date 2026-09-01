import shutil
import sqlite3
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.v4_emotion_model import V4_ARTIFACT_PATHS, predict_v4_emotion

DATABASE_DIR = Path(__file__).resolve().parent.parent / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "checkins.db"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
V4_ARTIFACT_DIR = MODEL_DIR / "v4"
VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}

ALLOWED_AUDIO_MIME_TYPES = {
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/aac",
    "audio/x-m4a",
    "audio/m4a",
}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".webm"}

app = FastAPI(
    title="AI Mental Health Screening API",
    version="0.1.0",
    description="Backend foundation for the mental health screening application.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running",
    }


class CheckInCreate(BaseModel):
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    duration_seconds: int = Field(default=0, ge=0)

    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_EMOTIONS:
            raise ValueError("Emotion must be one of: angry, disgust, fear, happy, neutral, sad.")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        numeric_value = float(value)
        if numeric_value < 0.0 or numeric_value > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0 inclusive.")
        return numeric_value


def init_db() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS check_ins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration_seconds INTEGER NOT NULL
            )
            """
        )
        connection.commit()


def _serialize_check_in(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "emotion": row["emotion"],
        "confidence": float(row["confidence"]),
        "duration_seconds": int(row["duration_seconds"]),
    }


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/api/check-ins")
def create_check_in(payload: CheckInCreate):
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            INSERT INTO check_ins (created_at, emotion, confidence, duration_seconds)
            VALUES (?, ?, ?, ?)
            """,
            (created_at, payload.emotion, payload.confidence, payload.duration_seconds),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM check_ins WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Unable to persist check-in.")

    return _serialize_check_in(row)


@app.get("/api/check-ins")
def list_check_ins():
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM check_ins ORDER BY created_at DESC, id DESC"
        ).fetchall()

    return [_serialize_check_in(row) for row in rows]


@app.get("/api/check-ins/{check_in_id}")
def get_check_in(check_in_id: int):
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM check_ins WHERE id = ?",
            (check_in_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Check-in not found.")

    return _serialize_check_in(row)


def _convert_to_wav(source_path: Path, wav_path: Path) -> None:
    """Convert WebM audio to a mono 16 kHz WAV using the managed FFmpeg binary."""
    if source_path.suffix.lower() == ".wav":
        shutil.copyfile(source_path, wav_path)
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("WebM audio support requires imageio-ffmpeg.") from exc

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not wav_path.is_file():
        raise ValueError("Unable to decode the uploaded WebM audio.")


@app.post("/api/analyze-emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """Predict one emotion from an uploaded WAV or WebM recording."""
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No audio file was provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only .wav and .webm audio files are supported.")

    missing_models = [path.name for path in V4_ARTIFACT_PATHS if not path.is_file()]
    if missing_models:
        raise HTTPException(status_code=500, detail=f"V4 model files are missing: {', '.join(missing_models)}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        with tempfile.TemporaryDirectory(prefix="emotion-") as temp_dir:
            temp_path = Path(temp_dir) / f"upload{suffix}"
            wav_path = Path(temp_dir) / "converted.wav"
            temp_path.write_bytes(contents)
            try:
                _convert_to_wav(temp_path, wav_path)
            except (ValueError, OSError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid audio file: {exc}") from exc

            try:
                prediction = predict_v4_emotion(wav_path)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid audio file: {exc}") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Emotion prediction failed.") from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Emotion prediction failed.") from exc

    return prediction
