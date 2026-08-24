from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
