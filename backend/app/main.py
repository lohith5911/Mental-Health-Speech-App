import shutil
import sys
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.v4_emotion_model import V4_ARTIFACT_PATHS, predict_v4_emotion

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
V4_ARTIFACT_DIR = MODEL_DIR / "v4"

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


@app.post("/api/check-ins")
async def upload_check_in(file: UploadFile = File(...)):
    if file is None or file.filename in (None, ""):
        raise HTTPException(status_code=400, detail="No audio file was provided.")

    raw_content_type = (file.content_type or "").split(";", 1)[0].lower()
    if raw_content_type and raw_content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. Please upload a common browser audio file such as "
                "webm, mp4, ogg, wav, or mp3."
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    original_name = Path(file.filename).name or "audio"
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", ".", "-"} else "_" for ch in original_name).strip()
    if not safe_name:
        safe_name = "audio"

    check_in_id = uuid4().hex
    saved_filename = f"{check_in_id}_{safe_name}"
    save_path = UPLOAD_DIR / saved_filename
    save_path.write_bytes(contents)

    return {
        "check_in_id": check_in_id,
        "status": "uploaded",
        "filename": saved_filename,
    }


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
