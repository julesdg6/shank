import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

app = FastAPI(title='SHANK API')

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
UPLOADS_DIR = DATA_DIR / 'uploads'
TASKS_DIR = DATA_DIR / 'tasks'

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def _write_task(task: dict) -> None:
    _ensure_dirs()
    # Parse through uuid.UUID to guarantee a safe, canonical filename.
    safe_task_id = str(uuid.UUID(task['task_id']))
    task_file = TASKS_DIR / f'{safe_task_id}.json'
    task_file.write_text(json.dumps(task, indent=2))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get('/')
def read_root():
    return {'status': 'online', 'service': 'SHANK API'}


# ---------------------------------------------------------------------------
# Upload audio file
# ---------------------------------------------------------------------------

@app.post('/tasks/upload', status_code=202)
async def upload_audio(file: UploadFile = File(...)):
    """Accept an audio file (MP3, WAV, FLAC) and queue it for analysis."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ''
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    task_id = str(uuid.uuid4())
    _ensure_dirs()

    # Enforce size limit before loading into memory
    if file.size is not None and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB',
        )
    # Read at most MAX_UPLOAD_SIZE + 1 bytes so we can detect oversize content
    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB',
        )

    # Save the uploaded file using the internally generated task_id as the filename
    upload_path = UPLOADS_DIR / f"{task_id}{suffix}"
    upload_path.write_bytes(content)

    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': file.filename,
        'file_path': str(upload_path),
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    _write_task(task)

    return JSONResponse(status_code=202, content={'task_id': task_id, 'status': 'pending'})


# ---------------------------------------------------------------------------
# Submit YouTube URL
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def must_be_youtube(cls, v: str) -> str:
        if not (
            v.startswith('https://www.youtube.com/')
            or v.startswith('https://youtu.be/')
        ):
            raise ValueError('url must be a YouTube HTTPS URL')
        return v


@app.post('/tasks/url', status_code=202)
def submit_url(body: URLRequest):
    """Accept a YouTube URL and queue it for analysis."""
    task_id = str(uuid.uuid4())

    task = {
        'task_id': task_id,
        'type': 'url',
        'source': body.url,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    _write_task(task)

    return JSONResponse(status_code=202, content={'task_id': task_id, 'status': 'pending'})


# ---------------------------------------------------------------------------
# Get task status
# ---------------------------------------------------------------------------

@app.get('/tasks/{task_id}')
def get_task(task_id: str):
    """Return the current status of a queued task."""
    # Parse the task_id as a UUID; this validates and produces a canonical,
    # safe string that cannot contain path-traversal characters.
    try:
        safe_task_id = str(uuid.UUID(task_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='Task not found')
    _ensure_dirs()
    task_file = TASKS_DIR / f'{safe_task_id}.json'
    if not task_file.exists():
        raise HTTPException(status_code=404, detail='Task not found')
    return json.loads(task_file.read_text())
