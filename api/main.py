import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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


def _safe_task_file(task_id: str) -> Path:
    try:
        safe_task_id = str(uuid.UUID(task_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='Task not found')
    _ensure_dirs()
    return TASKS_DIR / f'{safe_task_id}.json'


def _load_task(task_id: str) -> dict:
    task_file = _safe_task_file(task_id)
    if not task_file.exists():
        raise HTTPException(status_code=404, detail='Task not found')
    try:
        return json.loads(task_file.read_text())
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail='Task file is unreadable')


def _resolve_data_path(path_value: str) -> Path | None:
    base_dir = DATA_DIR.resolve()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = DATA_DIR / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


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


@app.get('/tasks/completed')
def list_completed_tasks():
    """Return all tasks with status='done', sorted by completion time desc."""
    _ensure_dirs()
    completed_tasks = []
    for task_file in TASKS_DIR.glob('*.json'):
        try:
            task = json.loads(task_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if task.get('status') == 'done':
            completed_tasks.append(task)

    completed_tasks.sort(key=lambda task: task.get('completed_at') or '', reverse=True)
    return {'tasks': completed_tasks}


# ---------------------------------------------------------------------------
# Get task status
# ---------------------------------------------------------------------------

@app.get('/tasks/{task_id}')
def get_task(task_id: str):
    """Return the current status of a queued task."""
    return _load_task(task_id)


def _mt3_track(task: dict, track_name: str) -> dict | None:
    mt3_data = task.get('mt3')
    if not isinstance(mt3_data, dict):
        return None
    if track_name == 'full_mix':
        track = mt3_data.get('full_mix')
        return track if isinstance(track, dict) else None
    stems = mt3_data.get('stems')
    if isinstance(stems, dict):
        track = stems.get(track_name)
        return track if isinstance(track, dict) else None
    return None


def _task_artifacts(task: dict) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}

    normalized_path = task.get('normalized_path')
    if isinstance(normalized_path, str) and normalized_path:
        resolved = _resolve_data_path(normalized_path)
        if resolved is not None:
            artifacts['normalized_wav'] = resolved

    stems = task.get('stems')
    if isinstance(stems, dict):
        for stem_name, stem_path in stems.items():
            if not isinstance(stem_name, str) or not isinstance(stem_path, str) or not stem_path:
                continue
            resolved = _resolve_data_path(stem_path)
            if resolved is not None:
                artifacts[f'stem_{stem_name}_wav'] = resolved

    mt3_data = task.get('mt3')
    if isinstance(mt3_data, dict):
        full_mix = mt3_data.get('full_mix')
        if isinstance(full_mix, dict):
            midi_path = full_mix.get('midi_path')
            if isinstance(midi_path, str) and midi_path:
                resolved = _resolve_data_path(midi_path)
                if resolved is not None:
                    artifacts['midi'] = resolved

            notes_path = full_mix.get('notes_path')
            if isinstance(notes_path, str) and notes_path:
                resolved = _resolve_data_path(notes_path)
                if resolved is not None:
                    artifacts['notes_json'] = resolved

        stems = mt3_data.get('stems')
        if isinstance(stems, dict):
            for stem_name, stem_data in stems.items():
                if not isinstance(stem_name, str) or not isinstance(stem_data, dict):
                    continue
                midi_path = stem_data.get('midi_path')
                if not isinstance(midi_path, str) or not midi_path:
                    continue
                resolved = _resolve_data_path(midi_path)
                if resolved is None:
                    continue
                artifacts[f'stem_{stem_name}_midi'] = resolved

    return artifacts


@app.get('/tasks/{task_id}/artifacts')
def list_task_artifacts(task_id: str):
    task = _load_task(task_id)
    artifacts = _task_artifacts(task)
    return {'artifacts': sorted(artifacts.keys())}


@app.get('/tasks/{task_id}/artifacts/{artifact_name}')
def download_task_artifact(task_id: str, artifact_name: str):
    task = _load_task(task_id)
    artifacts = _task_artifacts(task)
    artifact = artifacts.get(artifact_name)
    if artifact is None:
        raise HTTPException(status_code=404, detail='Artifact not found')
    return FileResponse(path=artifact, filename=artifact.name)


@app.get('/tasks/{task_id}/mt3/midi/{track_name}')
def download_mt3_midi(task_id: str, track_name: str):
    """Download an MT3 MIDI artifact for full mix or a specific stem."""
    task = _load_task(task_id)
    track = _mt3_track(task, track_name)
    midi_path = track.get('midi_path') if isinstance(track, dict) else None
    if not isinstance(midi_path, str) or not midi_path:
        raise HTTPException(status_code=404, detail='MT3 MIDI not found')
    resolved = _resolve_data_path(midi_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail='MT3 MIDI not found')
    return FileResponse(path=resolved, media_type='audio/midi', filename=resolved.name)


@app.get('/tasks/{task_id}/mt3/notes/{track_name}')
def get_mt3_notes(task_id: str, track_name: str):
    """Return note metadata JSON for full mix or a specific stem."""
    task = _load_task(task_id)
    track = _mt3_track(task, track_name)
    notes_path = track.get('notes_path') if isinstance(track, dict) else None
    if not isinstance(notes_path, str) or not notes_path:
        raise HTTPException(status_code=404, detail='MT3 note metadata not found')
    resolved = _resolve_data_path(notes_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail='MT3 note metadata not found')
    try:
        return json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail='MT3 note metadata is unreadable')


# ---------------------------------------------------------------------------
# Static UI — mount last so API routes take precedence
# ---------------------------------------------------------------------------

_UI_DIR = Path(__file__).parent / 'ui'
if _UI_DIR.is_dir():
    app.mount('/ui', StaticFiles(directory=str(_UI_DIR), html=True), name='ui')
