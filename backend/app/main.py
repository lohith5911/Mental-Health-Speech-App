import shutil
import os
import sqlite3
import sys
import subprocess
import tempfile
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.v4_emotion_model import V4_ARTIFACT_PATHS, predict_v4_emotion
from ml.src.features.acoustic_features import extract_acoustic_features_from_file
from app.services.trend_engine import DEFAULT_WINDOW_SIZE, MAX_WINDOW_SIZE, build_trend_insight
from app.services.quality import QualityAssessment, assess_audio_quality
from app.auth import create_access_token, get_current_user, hash_password, verify_password

DATABASE_DIR = Path(__file__).resolve().parent.parent / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "checkins.db"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
V4_ARTIFACT_DIR = MODEL_DIR / "v4"
VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}
MAX_UPLOAD_BYTES = int(os.getenv("MINDTRACE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def configured_cors_origins() -> list[str]:
    configured = os.getenv("MINDTRACE_CORS_ORIGINS", "").strip()
    if not configured:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins or "*" in origins:
        raise RuntimeError("MINDTRACE_CORS_ORIGINS must contain explicit origins; '*' is not allowed with credentials.")
    return origins

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
ALLOWED_AUDIO_SUFFIXES = {".wav", ".webm", ".mp4", ".ogg"}

app = FastAPI(
    title="AI Mental Health Screening API",
    version="0.1.0",
    description="Backend foundation for the mental health screening application.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins(),
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
    model_version: str | None = None
    probabilities: dict[str, float] | None = None
    acoustic_features: "AcousticFeaturesPayload | None" = None

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

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Model version must be a non-empty string.")
        return normalized

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        if set(value) != VALID_EMOTIONS:
            raise ValueError("Probabilities must contain exactly the six supported emotions.")
        normalized = {emotion: float(value[emotion]) for emotion in VALID_EMOTIONS}
        if any(
            not math.isfinite(probability) or probability < 0.0 or probability > 1.0
            for probability in normalized.values()
        ):
            raise ValueError("Probability values must be between 0.0 and 1.0 inclusive.")
        if abs(sum(normalized.values()) - 1.0) > 0.01:
            raise ValueError("Probabilities must sum to approximately 1.0.")
        return normalized


class AcousticFeaturesPayload(BaseModel):
    """Validated scalar measurements from the M6.1 acoustic extractor."""

    duration_seconds: float = Field(ge=0.0)
    rms_mean: float = Field(ge=0.0)
    rms_std: float = Field(ge=0.0)
    zcr_mean: float = Field(ge=0.0, le=1.0)
    zcr_std: float = Field(ge=0.0, le=1.0)
    pitch_mean_hz: float | None = Field(default=None, ge=0.0)
    pitch_std_hz: float | None = Field(default=None, ge=0.0)
    pitch_range_hz: float | None = Field(default=None, ge=0.0)
    silence_ratio: float = Field(ge=0.0, le=1.0)
    speaking_rate_proxy: float | None = Field(default=None, ge=0.0)
    is_silent: bool
    is_noisy: bool

    @field_validator(
        "duration_seconds",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "pitch_mean_hz",
        "pitch_std_hz",
        "pitch_range_hz",
        "silence_ratio",
        "speaking_rate_proxy",
    )
    @classmethod
    def validate_finite_measurement(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Acoustic feature values must be finite.")
        return value


CheckInCreate.model_rebuild()


class CheckInQuality(BaseModel):
    status: str
    reasons: list[str]


class AnalyzeAndSaveResponse(BaseModel):
    id: int
    created_at: str
    emotion: str
    confidence: float
    model_version: str
    probabilities: dict[str, float]
    acoustic_features: AcousticFeaturesPayload
    quality: CheckInQuality
    duration_seconds: int


class UserRegister(BaseModel):
    email: str
    display_name: str
    password: str = Field(..., min_length=8, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Enter a valid email address.")
        local_part, domain = normalized.rsplit("@", 1)
        if not local_part or "." not in domain or domain.startswith(".") or domain.endswith("."):
            raise ValueError("Enter a valid email address.")
        return normalized

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 100:
            raise ValueError("Display name must be between 1 and 100 characters.")
        return normalized


class UserLogin(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


def _serialize_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


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
                duration_seconds INTEGER NOT NULL,
                model_version TEXT,
                probabilities TEXT,
                acoustic_features TEXT,
                quality TEXT,
                idempotency_key TEXT UNIQUE,
                user_id INTEGER,
                idempotency_scope_key TEXT
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(check_ins)")}
        if "model_version" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN model_version TEXT")
        if "probabilities" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN probabilities TEXT")
        if "acoustic_features" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN acoustic_features TEXT")
        if "quality" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN quality TEXT")
        if "idempotency_key" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN idempotency_key TEXT")
        if "user_id" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN user_id INTEGER")
        if "idempotency_scope_key" not in columns:
            connection.execute("ALTER TABLE check_ins ADD COLUMN idempotency_scope_key TEXT")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_check_ins_idempotency_key "
            "ON check_ins(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_check_ins_user_id ON check_ins(user_id)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_check_ins_idempotency_scope_key "
            "ON check_ins(idempotency_scope_key) WHERE idempotency_scope_key IS NOT NULL"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def _serialize_check_in(row: sqlite3.Row, *, include_quality: bool = False) -> dict:
    probabilities: dict[str, Any] | None = None
    if row["probabilities"] is not None:
        probabilities = json.loads(row["probabilities"])
    acoustic_features: dict[str, Any] | None = None
    if row["acoustic_features"] is not None:
        acoustic_features = json.loads(row["acoustic_features"])
    serialized = {
        "id": row["id"],
        "created_at": row["created_at"],
        "emotion": row["emotion"],
        "confidence": float(row["confidence"]),
        "duration_seconds": int(row["duration_seconds"]),
        "model_version": row["model_version"],
        "probabilities": probabilities,
        "acoustic_features": acoustic_features,
    }
    if include_quality or row["quality"] is not None:
        serialized["quality"] = json.loads(row["quality"]) if row["quality"] is not None else None
    return serialized


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/api/auth/register", response_model=AuthResponse, status_code=201)
def register_user(payload: UserRegister):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(str(DATABASE_PATH)) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                """
                INSERT INTO users (email, display_name, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (payload.email, payload.display_name, hash_password(payload.password), now, now),
            )
            connection.commit()
            row = connection.execute(
                "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with that email already exists.") from None
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail="Unable to create account.") from exc

    if row is None:
        raise HTTPException(status_code=500, detail="Unable to create account.")
    user = _serialize_user(row)
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "user": user}


@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(payload: UserLogin):
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, email, display_name, password_hash, created_at FROM users WHERE email = ?",
            (payload.email,),
        ).fetchone()

    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.", headers={"WWW-Authenticate": "Bearer"})

    user = _serialize_user(row)
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=UserResponse)
def get_authenticated_user(current_user: dict[str, Any] = Depends(get_current_user)):
    return current_user


@app.post("/api/check-ins")
def create_check_in(payload: CheckInCreate, current_user: dict[str, Any] = Depends(get_current_user)):
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            INSERT INTO check_ins (
                created_at, emotion, confidence, duration_seconds, model_version,
                probabilities, acoustic_features, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                payload.emotion,
                payload.confidence,
                payload.duration_seconds,
                payload.model_version,
                json.dumps(payload.probabilities, sort_keys=True) if payload.probabilities is not None else None,
                json.dumps(payload.acoustic_features.model_dump(), sort_keys=True)
                if payload.acoustic_features is not None
                else None,
                current_user["id"],
            ),
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
def list_check_ins(current_user: dict[str, Any] = Depends(get_current_user)):
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM check_ins WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (current_user["id"],),
        ).fetchall()

    return [_serialize_check_in(row) for row in rows]


@app.get("/api/insights/trends")
def get_trends(
    window_size: int = Query(default=DEFAULT_WINDOW_SIZE, ge=1, le=MAX_WINDOW_SIZE),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT emotion, probabilities FROM check_ins "
            "WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (current_user["id"],),
        ).fetchall()

    return build_trend_insight(rows, window_size=window_size)


def _convert_to_wav(source_path: Path, wav_path: Path) -> None:
    """Convert supported non-WAV audio to a mono 16 kHz WAV."""
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
        raise ValueError("Unable to decode the uploaded audio.")


async def _process_uploaded_audio(file: UploadFile) -> tuple[dict[str, Any], dict[str, Any]]:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No audio file was provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only .wav, .webm, .mp4, and .ogg audio files are supported.")

    missing_models = [path.name for path in V4_ARTIFACT_PATHS if not path.is_file()]
    if missing_models:
        raise HTTPException(status_code=500, detail=f"V4 model files are missing: {', '.join(missing_models)}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Audio upload exceeds the {limit_mb:g} MiB limit.")

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
            try:
                acoustic_features = asdict(extract_acoustic_features_from_file(wav_path))
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid audio file: {exc}") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Acoustic feature extraction failed.") from exc
            return prediction, acoustic_features
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Emotion prediction failed.") from exc


@app.post("/api/analyze-emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """Predict one emotion from an uploaded WAV or WebM recording."""
    prediction, acoustic_features = await _process_uploaded_audio(file)
    return {**prediction, "acoustic_features": acoustic_features}


@app.post("/api/check-ins/analyze-and-save", response_model=AnalyzeAndSaveResponse)
async def analyze_and_save_check_in(
    file: UploadFile = File(...),
    client_duration_seconds: float | None = Form(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Analyze one recording and persist its complete result exactly once per key."""
    init_db()
    normalized_key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None
    scoped_key = f'{current_user["id"]}:{normalized_key}' if normalized_key else None
    if scoped_key:
        with sqlite3.connect(str(DATABASE_PATH)) as connection:
            connection.row_factory = sqlite3.Row
            existing = connection.execute(
                "SELECT * FROM check_ins WHERE idempotency_scope_key = ?", (scoped_key,)
            ).fetchone()
        if existing is not None:
            return _serialize_check_in(existing, include_quality=True)

    prediction, acoustic_features = await _process_uploaded_audio(file)
    validated_features = AcousticFeaturesPayload.model_validate(acoustic_features)
    assessment: QualityAssessment = assess_audio_quality(validated_features)
    reasons = list(assessment.reasons)
    if client_duration_seconds is not None:
        if not math.isfinite(client_duration_seconds) or client_duration_seconds < 0:
            raise HTTPException(status_code=422, detail="Client duration must be a finite non-negative number.")
        if abs(client_duration_seconds - validated_features.duration_seconds) > 2.0:
            reasons.append("frontend_duration_mismatch")
    quality = CheckInQuality(status=assessment.status, reasons=reasons)
    payload = CheckInCreate(
        emotion=prediction["emotion"],
        confidence=prediction["confidence"],
        duration_seconds=max(0, int(round(validated_features.duration_seconds))),
        model_version=prediction.get("model_version", "v4"),
        probabilities=prediction.get("probabilities"),
        acoustic_features=validated_features,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(str(DATABASE_PATH)) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                """
                INSERT INTO check_ins (
                    created_at, emotion, confidence, duration_seconds, model_version,
                    probabilities, acoustic_features, quality, idempotency_key,
                    user_id, idempotency_scope_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    payload.emotion,
                    payload.confidence,
                    payload.duration_seconds,
                    payload.model_version,
                    json.dumps(payload.probabilities, sort_keys=True),
                    json.dumps(payload.acoustic_features.model_dump(), sort_keys=True),
                    json.dumps(quality.model_dump(), sort_keys=True),
                    None,
                    current_user["id"],
                    scoped_key,
                ),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM check_ins WHERE id = ?", (cursor.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:
        if scoped_key:
            with sqlite3.connect(str(DATABASE_PATH)) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    "SELECT * FROM check_ins WHERE idempotency_scope_key = ?", (scoped_key,)
                ).fetchone()
            if row is not None:
                return _serialize_check_in(row, include_quality=True)
        raise HTTPException(status_code=500, detail="Unable to persist check-in.")
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail="Unable to persist check-in.") from exc

    if row is None:
        raise HTTPException(status_code=500, detail="Unable to persist check-in.")
    return _serialize_check_in(row, include_quality=True)


@app.get("/api/check-ins/{check_in_id}")
def get_check_in(check_in_id: int, current_user: dict[str, Any] = Depends(get_current_user)):
    init_db()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM check_ins WHERE id = ? AND user_id = ?",
            (check_in_id, current_user["id"]),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Check-in not found.")

    return _serialize_check_in(row)
