import json
import logging
import os
import shutil
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

app = FastAPI(title='SHANK API')

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
UPLOADS_DIR = DATA_DIR / 'uploads'
TASKS_DIR = DATA_DIR / 'tasks'

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}
ALLOWED_REQUESTED_TYPES = {'melody'}
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB


def _get_media_type_quality(accept_header: str, media_type: str) -> float:
    """Return the highest q-value that makes ``media_type`` acceptable.

    The parser handles exact media types plus ``type/*`` and ``*/*`` wildcards.
    Invalid entries are ignored. When no matching entry exists, this returns 0.0.
    """
    wanted_type, sep, wanted_subtype = media_type.lower().partition('/')
    if sep != '/' or not wanted_type or not wanted_subtype:
        return 0.0
    best_q = 0.0
    for raw_part in accept_header.split(','):
        part = raw_part.strip()
        if not part:
            continue
        media_range, *params = [item.strip() for item in part.split(';') if item.strip()]
        range_type, sep, range_subtype = media_range.lower().partition('/')
        if sep != '/' or not range_type or not range_subtype:
            continue
        if range_type not in (wanted_type, '*'):
            continue
        if range_subtype not in (wanted_subtype, '*'):
            continue

        q_value = 1.0
        for param in params:
            key, sep, value = param.partition('=')
            if key.strip().lower() != 'q' or sep != '=':
                continue
            try:
                q_value = float(value.strip())
            except ValueError:
                q_value = 0.0
            break
        best_q = max(best_q, q_value)
    return best_q


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


async def _queue_audio_task(file: UploadFile, *, requested_type: str | None = None) -> dict:
    if requested_type is not None and requested_type not in ALLOWED_REQUESTED_TYPES:
        raise HTTPException(status_code=400, detail='Unsupported requested_type')

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
    if requested_type is not None:
        task['requested_type'] = requested_type
    _write_task(task)

    return {'task_id': task_id, 'status': 'pending'}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

_UI_DIR = Path(__file__).parent / 'ui'


@app.get('/')
def read_root(request: Request):
    """Serve the dashboard for browser-style requests and JSON for API clients.

    When HTML and JSON are equally acceptable, prefer HTML so the bare root path
    behaves as the product landing page in browsers. If the Accept header is
    missing or does not express a preference for either HTML or JSON, default to
    the dashboard for the same reason.
    """
    accept_header = request.headers.get('accept', '')
    if accept_header.strip():
        html_quality = _get_media_type_quality(accept_header, 'text/html')
        json_quality = _get_media_type_quality(accept_header, 'application/json')
    else:
        html_quality = 1.0
        json_quality = 0.0
    if html_quality == 0 and json_quality == 0:
        html_quality = 1.0
    accepts_html = html_quality > 0 and html_quality >= json_quality
    index_file = _UI_DIR / 'index.html'
    if accepts_html:
        if index_file.is_file():
            return FileResponse(index_file, media_type='text/html')
        log.warning('Dashboard HTML requested at / but %s is missing', index_file)
    return {'status': 'online', 'service': 'SHANK API'}


# ---------------------------------------------------------------------------
# Upload audio file
# ---------------------------------------------------------------------------

@app.post('/tasks/upload', status_code=202)
async def upload_audio(file: UploadFile = File(...)):
    """Accept an audio file (MP3, WAV, FLAC) and queue it for analysis."""
    return JSONResponse(status_code=202, content=await _queue_audio_task(file))


@app.post('/tasks/melody', status_code=202)
async def submit_melody(file: UploadFile = File(...)):
    """Accept an audio file and queue a melody-focused analysis task."""
    return JSONResponse(status_code=202, content=await _queue_audio_task(file, requested_type='melody'))


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

    structured_results = task.get('results')
    if isinstance(structured_results, dict):
        structured_files = {
            'results_task_json': structured_results.get('task_json'),
            'results_analysis_json': structured_results.get('analysis_json'),
            'results_mt3_json': structured_results.get('mt3_json'),
            'results_artifacts_json': structured_results.get('artifacts_json'),
        }
        for artifact_name, artifact_path in structured_files.items():
            if not isinstance(artifact_path, str) or not artifact_path:
                continue
            resolved = _resolve_data_path(artifact_path)
            if resolved is not None:
                artifacts[artifact_name] = resolved

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


@app.get('/tasks/{task_id}/chords')
def get_task_chords(task_id: str):
    """Return the chord detection results for a completed task.

    The response mirrors the ``chords`` field of the task JSON and includes
    ``segments`` (each with ``symbol``, ``root``, ``quality``, ``confidence``,
    ``start_seconds``, ``end_seconds``) and a flat ``progression`` list.
    """
    task = _load_task(task_id)
    chords = task.get('chords')
    if not isinstance(chords, dict):
        raise HTTPException(status_code=404, detail='Chord data not available for this task')
    return chords


# ---------------------------------------------------------------------------
# Worker status
# ---------------------------------------------------------------------------


@app.get('/worker/status')
def get_worker_status():
    """Return the current health status of the background worker process.

    The worker writes a heartbeat timestamp to *DATA_DIR/.worker_heartbeat*
    at the start of every poll cycle.  This endpoint reads that file and
    reports whether the heartbeat is recent enough to consider the worker
    alive.
    """
    heartbeat_file = DATA_DIR / '.worker_heartbeat'
    stale_threshold = int(os.getenv('POLL_INTERVAL', '10')) * 3 + 30

    try:
        raw = heartbeat_file.read_text().strip()
        last_beat = datetime.fromisoformat(raw)
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_beat).total_seconds()
        online = age_seconds <= stale_threshold
        return {
            'status': 'online' if online else 'offline',
            'last_heartbeat': raw,
            'age_seconds': round(age_seconds, 1),
            'stale_threshold_seconds': stale_threshold,
        }
    except FileNotFoundError:
        return {
            'status': 'offline',
            'last_heartbeat': None,
            'age_seconds': None,
            'stale_threshold_seconds': stale_threshold,
        }
    except Exception as exc:
        log.warning('Failed to read worker heartbeat: %s', exc)
        return {
            'status': 'unknown',
            'last_heartbeat': None,
            'age_seconds': None,
            'stale_threshold_seconds': stale_threshold,
        }


# ---------------------------------------------------------------------------
# Stem backend status
# ---------------------------------------------------------------------------

@app.get('/stem-backend/status')
def get_stem_backend_status():
    """Return the configured stem separation backend and its health status."""
    configured_backend = os.getenv('STEM_BACKEND', 'auto').strip().lower()
    ace_step_url = os.getenv('ACE_STEP_API_URL', '').strip().rstrip('/')
    ace_step_key = os.getenv('ACE_STEP_API_KEY', '').strip()
    demucs_model = os.getenv('DEMUCS_MODEL', 'htdemucs').strip() or 'htdemucs'
    demucs_device = os.getenv('DEMUCS_DEVICE', 'cpu').strip() or 'cpu'

    # Check Ace-Step reachability with a short timeout.
    ace_step_healthy = False
    if ace_step_url:
        try:
            req = urllib.request.Request(ace_step_url)
            if ace_step_key:
                req.add_header('Authorization', f'Bearer {ace_step_key}')
            with urllib.request.urlopen(req, timeout=3):
                ace_step_healthy = True
        except Exception as exc:
            log.debug('Ace-Step health check failed: %s', exc)
            ace_step_healthy = False

    demucs_available = shutil.which('demucs') is not None

    # Determine the effective active backend.
    if configured_backend == 'none':
        active_backend = 'none'
    elif configured_backend == 'acestep':
        active_backend = 'acestep' if (ace_step_url and ace_step_healthy) else 'none'
    elif configured_backend == 'demucs':
        active_backend = 'demucs' if demucs_available else 'none'
    else:  # auto
        if ace_step_url and ace_step_healthy:
            active_backend = 'acestep'
        elif demucs_available:
            active_backend = 'demucs'
        else:
            active_backend = 'none'

    return {
        'configured_backend': configured_backend,
        'active_backend': active_backend,
        'acestep': {
            'configured': bool(ace_step_url),
            'url': ace_step_url or None,
            'healthy': ace_step_healthy,
        },
        'demucs': {
            'available': demucs_available,
            'model': demucs_model,
            'device': demucs_device,
        },
    }


# ---------------------------------------------------------------------------
# Static UI — mount last so API routes take precedence
# ---------------------------------------------------------------------------

if _UI_DIR.is_dir():
    app.mount('/ui', StaticFiles(directory=str(_UI_DIR), html=True), name='ui')
