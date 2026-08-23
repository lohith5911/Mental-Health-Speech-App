# Backend Foundation

This backend provides the initial FastAPI application for the mental health screening project.

## Setup

From the project root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Health endpoint

```bash
curl http://127.0.0.1:8000/api/health
```

This project intentionally contains only the foundation required for Milestone 1.
