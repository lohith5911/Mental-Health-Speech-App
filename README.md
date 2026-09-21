# MindTrace AI

MindTrace AI is a speech-based daily check-in application for emotional self-monitoring. A user records a short voice check-in, receives speech-derived emotion probabilities, acoustic measurements, and signal-quality information, and can review personal history and longitudinal trends.

The application is non-diagnostic. It does not diagnose depression or any other mental-health condition, and its results are not a substitute for professional care.

## Features

- Authenticated registration, login, logout, and account-specific check-ins
- Browser microphone recording with permission and recording-error handling
- V4 six-class speech emotion inference
- Acoustic speech measurements and signal-quality assessment
- Results with confidence and emotion probabilities
- History and longitudinal trend views
- SQLite persistence for the current prototype
- Speaker-independent M9 evaluation artifact for the V4 model

## Architecture and stack

- `frontend/`: React 19, TypeScript, Vite, React Router, and Oxlint
- `backend/`: FastAPI, Pydantic, SQLite, JWT authentication, password hashing, and FFmpeg-backed audio conversion
- `ml/`: PyTorch, Hugging Face Transformers, librosa, and the frozen wav2vec2 V4 classifier
- `ml/models/v4/`: V4 model head, configuration, label encoder, metrics, and evaluation artifact

The browser sends authenticated requests to FastAPI. The backend validates the request, converts supported audio to mono 16 kHz WAV in a temporary directory, runs cached V4 inference, extracts acoustic features, assesses signal quality, and persists the result for the authenticated user.

## Local development

### Prerequisites

- Python 3.12 or compatible Python version supported by the project environments
- Node.js and npm
- The V4 artifacts in `ml/models/v4/`
- Access to the Hugging Face `facebook/wav2vec2-base` weights, either through an existing local cache or network access on first inference

### Backend

Use the backend environment and install its declared dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

The backend creates `backend/data/checkins.db` automatically. The local development JWT fallback is intentionally development-only.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

The frontend uses `http://127.0.0.1:8000` by default. To point it at another backend, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Only non-secret frontend configuration belongs in Vite variables. Never place JWT secrets, passwords, or API keys in `VITE_*` variables.

### ML environment and tests

The ML requirements are separate from the backend environment:

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
ml/.venv/bin/python -m pytest ml/tests -q
```

The raw CREMA-D dataset belongs under `ml/data/raw/crema-d/AudioWAV/` and is ignored by Git. Training is not required to run the application; the saved V4 artifacts are used for inference.

## Configuration

### Local development

Local defaults are intentionally convenient:

- `MINDTRACE_ENV` defaults to `development`.
- `MINDTRACE_JWT_SECRET` may be omitted and then uses a clearly development-only fallback.
- `MINDTRACE_CORS_ORIGINS` may be omitted and allows `http://localhost:5173` and `http://127.0.0.1:5173`.
- `MINDTRACE_MAX_UPLOAD_BYTES` defaults to `10485760` (10 MiB).

### Production or deployment

Set an unpredictable secret and explicit frontend origin before starting the backend:

```bash
export MINDTRACE_ENV=production
export MINDTRACE_JWT_SECRET='replace-with-a-long-random-secret'
export MINDTRACE_CORS_ORIGINS='https://your-frontend.example'
export MINDTRACE_MAX_UPLOAD_BYTES=10485760
```

`MINDTRACE_ENV=production` without `MINDTRACE_JWT_SECRET` fails at startup rather than silently using the development fallback. CORS accepts a comma-separated list of explicit origins; wildcard `*` is rejected because authenticated requests use credentials.

The frontend deployment must set `VITE_API_BASE_URL` to the deployed backend URL at build time.

## V4 model and evaluation

Production inference uses V4 and the six classes `angry`, `disgust`, `fear`, `happy`, `neutral`, and `sad`. The application requires these files in `ml/models/v4/`:

- `model.pt`
- `config.json`
- `label_encoder.pkl`
- `metrics.json`
- `evaluation.json` for the M9 evaluation record

V4 uses a frozen `facebook/wav2vec2-base` encoder. Its weights are not bundled in this repository and must be available from the Hugging Face cache or downloaded in the runtime environment. The application does not train a model during startup.

M9 evaluation is speaker-independent on CREMA-D with seed 42 and 1,229 test files. Recorded metrics include accuracy `0.4955`, macro precision `0.5222`, macro recall `0.4914`, and macro F1 `0.4936`. These metrics describe the evaluation dataset and should not be presented as clinical accuracy.

## Demo workflow

1. Open the frontend and create an account.
2. Log in and confirm the dashboard is account-specific.
3. Open Daily Check-In and allow microphone access.
4. Record approximately 30–60 seconds in a quiet environment.
5. Preview the recording and continue to analysis.
6. Review emotion, confidence, probabilities, acoustic measurements, and quality status.
7. Open Results, History, and trends.
8. Open Profile and Resources.
9. Log out and confirm protected pages require authentication.

## Authentication and data persistence

Passwords are hashed on the backend. Access tokens are short-lived JWTs sent in the `Authorization: Bearer` header and stored in browser session storage. The backend is authoritative for user identity and ownership; authenticated queries filter check-ins by `user_id`.

SQLite is appropriate for the current single-instance prototype. `backend/data/checkins.db` is local runtime data and is ignored by Git. A deployment needs persistent storage and a deliberate backup and migration strategy. Temporary uploaded audio is removed after processing; the persisted check-in stores derived results, not the temporary upload.

## Privacy, safety, and limitations

- Do not use the application as a diagnosis, emergency service, or treatment tool.
- Do not upload speech recordings containing information you do not have permission to process.
- Use a unique password and sign out on shared devices.
- Browser microphone access requires a permitted localhost or HTTPS context.
- Results can be affected by language, speaking style, recording hardware, background noise, and model limitations.
- The current CORS, SQLite, local filesystem, and runtime model-cache setup requires hardening before public deployment.
- Production deployments should use a secret manager, explicit CORS origins, persistent storage, upload monitoring, and a documented model-cache strategy.

## Validation commands

Run the project-local checks from the repository root:

```bash
cd frontend && npm run build && npm run lint
cd ../backend && .venv/bin/python -m pytest -q
cd .. && ml/.venv/bin/python -m pytest ml/tests -q
```

The backend and ML test suites include authentication, ownership, quality, speech pipeline, V4, and M9 evaluation coverage.
