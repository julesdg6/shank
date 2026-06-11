import json
import logging
import os
import subprocess
import shutil
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

app = FastAPI(title='SHANK API')

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
UPLOADS_DIR = DATA_DIR / 'uploads'
TASKS_DIR = DATA_DIR / 'tasks'
DEFAULT_SEPARATOR_MODEL_DIR = Path(os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/srv/shank/models/separator'))

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}
ALLOWED_REQUESTED_TYPES = {'melody'}
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

_MODEL_DOWNLOAD_LOCK = threading.Lock()
_MODEL_DOWNLOAD_STATE: dict[str, Any] = {
    'is_downloading': False,
    'status': 'idle',
    'status_message': '',
    'progress_percent': 0,
    'six_stems': False,
    'model_dir': str(DEFAULT_SEPARATOR_MODEL_DIR),
    'started_at': None,
    'completed_at': None,
    'return_code': None,
    'error': None,
    'output_tail': [],
    'pid': None,
    'process': None,
}

_MODEL_SPECS = {
    'htdemucs_ft.yaml': 400,
    'htdemucs_6s.yaml': 530,
}


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


def _models_payload(model_dir: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for model_name in _MODEL_SPECS:
        path = model_dir / model_name
        exists = path.is_file()
        payload[model_name] = {
            'exists': exists,
            'size_bytes': path.stat().st_size if exists else 0,
        }
    return payload


def _disk_free_gb(path: Path) -> float | None:
    candidate = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(candidate)
    except FileNotFoundError:
        return None
    return round(usage.free / (1024 ** 3), 2)


def _is_dir_writable(path: Path) -> bool:
    """Return whether the target directory is writable, creating it if needed."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f'.write-test-{uuid.uuid4().hex}'
        probe.write_text('ok')
        probe.unlink()
        return True
    except OSError:
        return False


def _snapshot_model_download_status() -> dict[str, Any]:
    with _MODEL_DOWNLOAD_LOCK:
        state = dict(_MODEL_DOWNLOAD_STATE)
    model_dir = Path(state.get('model_dir') or DEFAULT_SEPARATOR_MODEL_DIR)
    models = _models_payload(model_dir)
    four_stem_ready = bool(models['htdemucs_ft.yaml']['exists'])
    six_stem_ready = bool(models['htdemucs_6s.yaml']['exists'])
    wants_six_stems = bool(state.get('six_stems')) or six_stem_ready
    estimated_total_mb = 530 if wants_six_stems else 400
    progress = int(state.get('progress_percent') or 0)
    if four_stem_ready and not state.get('is_downloading') and state.get('status') != 'failed':
        progress = 100
    downloaded_mb = int(round((progress / 100) * estimated_total_mb))
    status = state.get('status') or 'idle'
    if status == 'idle':
        status = 'ready' if four_stem_ready else 'not_found'
    warning = None
    free_gb = _disk_free_gb(model_dir)
    if free_gb is not None and free_gb < 1.0:
        warning = f'Low disk space: only {free_gb} GB available.'

    return {
        'status': status,
        'models_ready': four_stem_ready,
        'six_stem_ready': six_stem_ready,
        'is_downloading': bool(state.get('is_downloading')),
        'progress_percent': max(0, min(100, progress)),
        'downloaded_mb': downloaded_mb,
        'estimated_total_mb': estimated_total_mb,
        'status_message': state.get('status_message') or '',
        'error': state.get('error'),
        'model_dir': str(model_dir),
        'models': models,
        'available_disk_gb': free_gb,
        'warning': warning,
        'output_tail': list(state.get('output_tail') or []),
        'started_at': state.get('started_at'),
        'completed_at': state.get('completed_at'),
        'return_code': state.get('return_code'),
        'pid': state.get('pid'),
    }


def _run_model_download(cmd: list[str], cwd: Path, total_steps: int) -> None:
    process: subprocess.Popen[str] | None = None
    completed_steps = 0
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with _MODEL_DOWNLOAD_LOCK:
            _MODEL_DOWNLOAD_STATE['process'] = process
            _MODEL_DOWNLOAD_STATE['pid'] = process.pid
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line:
                    with _MODEL_DOWNLOAD_LOCK:
                        tail = list(_MODEL_DOWNLOAD_STATE.get('output_tail') or [])
                        tail.append(line)
                        _MODEL_DOWNLOAD_STATE['output_tail'] = tail[-30:]
                        _MODEL_DOWNLOAD_STATE['status_message'] = line
                if 'ready.' in line:
                    completed_steps += 1
                    progress = int((completed_steps / max(total_steps, 1)) * 100)
                    with _MODEL_DOWNLOAD_LOCK:
                        _MODEL_DOWNLOAD_STATE['progress_percent'] = min(99, progress)
        return_code = process.wait()
        with _MODEL_DOWNLOAD_LOCK:
            _MODEL_DOWNLOAD_STATE['return_code'] = return_code
            _MODEL_DOWNLOAD_STATE['is_downloading'] = False
            _MODEL_DOWNLOAD_STATE['process'] = None
            _MODEL_DOWNLOAD_STATE['pid'] = None
            _MODEL_DOWNLOAD_STATE['completed_at'] = datetime.now(timezone.utc).isoformat()
            if return_code == 0:
                _MODEL_DOWNLOAD_STATE['status'] = 'completed'
                _MODEL_DOWNLOAD_STATE['status_message'] = 'Models downloaded successfully.'
                _MODEL_DOWNLOAD_STATE['progress_percent'] = 100
                _MODEL_DOWNLOAD_STATE['error'] = None
            elif _MODEL_DOWNLOAD_STATE.get('status') == 'cancelling':
                _MODEL_DOWNLOAD_STATE['status'] = 'cancelled'
                _MODEL_DOWNLOAD_STATE['status_message'] = 'Download cancelled.'
            else:
                _MODEL_DOWNLOAD_STATE['status'] = 'failed'
                _MODEL_DOWNLOAD_STATE['error'] = f'Model download failed with exit code {return_code}.'
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive fallback
        with _MODEL_DOWNLOAD_LOCK:
            _MODEL_DOWNLOAD_STATE['is_downloading'] = False
            _MODEL_DOWNLOAD_STATE['process'] = None
            _MODEL_DOWNLOAD_STATE['pid'] = None
            _MODEL_DOWNLOAD_STATE['status'] = 'failed'
            _MODEL_DOWNLOAD_STATE['status_message'] = 'Model download failed.'
            _MODEL_DOWNLOAD_STATE['error'] = str(exc)
            _MODEL_DOWNLOAD_STATE['completed_at'] = datetime.now(timezone.utc).isoformat()
        log.exception('Model download failed: %s', exc)


def _start_model_download(six_stems: bool, model_dir: str | None) -> dict[str, Any]:
    runtime_dir = Path('/srv/shank')
    if not runtime_dir.is_dir():
        runtime_dir = Path(__file__).resolve().parents[1]

    if model_dir:
        raise HTTPException(status_code=400, detail='Custom model_dir is not supported by this endpoint')
    resolved_model_dir = DEFAULT_SEPARATOR_MODEL_DIR.resolve()

    with _MODEL_DOWNLOAD_LOCK:
        already_downloading = bool(_MODEL_DOWNLOAD_STATE.get('is_downloading'))
    if already_downloading:
        return {
            'started': False,
            'message': 'A model download is already in progress.',
            **_snapshot_model_download_status(),
        }
    with _MODEL_DOWNLOAD_LOCK:
        _MODEL_DOWNLOAD_STATE.update({
            'is_downloading': True,
            'status': 'downloading',
            'status_message': 'Starting model download...',
            'progress_percent': 0,
            'six_stems': bool(six_stems),
            'model_dir': str(resolved_model_dir),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'completed_at': None,
            'return_code': None,
            'error': None,
            'output_tail': [],
            'pid': None,
            'process': None,
        })

    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'download_stem_models.py'
    cmd = ['python3', str(script_path), '--model-dir', str(resolved_model_dir)]
    if six_stems:
        cmd.append('--6stems')
    total_steps = 2 if six_stems else 1
    threading.Thread(
        target=_run_model_download,
        args=(cmd, runtime_dir, total_steps),
        daemon=True,
    ).start()

    return {'started': True, **_snapshot_model_download_status()}


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


async def _queue_audio_task(
    file: UploadFile,
    *,
    requested_type: str | None = None,
    enable_mt3: bool | None = None,
) -> dict:
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

    task: dict[str, Any] = {
        'task_id': task_id,
        'type': 'upload',
        'source': file.filename,
        'file_path': str(upload_path),
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    if requested_type is not None:
        task['requested_type'] = requested_type
    if enable_mt3 is not None:
        task['enable_mt3'] = enable_mt3
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
async def upload_audio(
    file: UploadFile = File(...),
    enable_mt3: bool | None = Form(default=None),
):
    """Accept an audio file (MP3, WAV, FLAC) and queue it for analysis."""
    return JSONResponse(status_code=202, content=await _queue_audio_task(file, enable_mt3=enable_mt3))


@app.post('/tasks/melody', status_code=202)
async def submit_melody(
    file: UploadFile = File(...),
    enable_mt3: bool = Form(True),
):
    """Accept an audio file and queue a melody-focused analysis task."""
    return JSONResponse(
        status_code=202,
        content=await _queue_audio_task(file, requested_type='melody', enable_mt3=enable_mt3),
    )


# ---------------------------------------------------------------------------
# Submit YouTube URL
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    url: str
    enable_mt3: bool | None = None

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

    task: dict[str, Any] = {
        'task_id': task_id,
        'type': 'url',
        'source': body.url,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    if body.enable_mt3 is not None:
        task['enable_mt3'] = body.enable_mt3
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
            'beatgrid_json': structured_results.get('beatgrid_json'),
            'waveform_beats_png': structured_results.get('waveform_beats_png'),
            'tempo_curve_png': structured_results.get('tempo_curve_png'),
            'beatgraph_png': structured_results.get('beatgraph_png'),
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


@app.get('/tasks/{task_id}/beatgrid')
def get_task_beatgrid(task_id: str):
    """Return the beat grid and beat detection metadata for a completed task.

    The response contains:

    * ``beatgrid`` – beat grid with ``bpm``, ``first_beat_seconds``, and a
      ``beats`` list.  Each beat entry has an ``index`` and ``time``
      (seconds).  Variable-tempo grids additionally carry a ``local_bpm``
      per beat and a top-level ``mode`` of ``'variable_tempo'``.
    * ``beat_detection`` – detection metadata including the ``engine`` used
      (``'librosa'``, ``'madmom'``, or ``'mixxx'``), ``mode``,
      ``first_beat_seconds``, ``beat_count``, and ``confidence`` (0–1 or
      ``null`` when unavailable).
    """
    task = _load_task(task_id)
    beatgrid = task.get('beatgrid')
    if not isinstance(beatgrid, dict):
        raise HTTPException(status_code=404, detail='Beatgrid data not available for this task')
    result: dict[str, Any] = {'beatgrid': beatgrid}
    beat_detection = task.get('beat_detection')
    if isinstance(beat_detection, dict):
        result['beat_detection'] = beat_detection
    return result


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
        # Ensure timezone-aware for comparison: assume UTC if naive
        if last_beat.tzinfo is None:
            last_beat = last_beat.replace(tzinfo=timezone.utc)
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


@app.get('/doctor')
def get_doctor_status():
    """Return a consolidated deployment health snapshot."""
    worker_status = get_worker_status()
    stem_backend_status = get_stem_backend_status()
    transcription_status = get_transcription_status()
    models_status = _snapshot_model_download_status()

    ffmpeg_path = shutil.which('ffmpeg')
    yt_dlp_path = shutil.which('yt-dlp')
    model_entries = models_status.get('models', {})
    found_models = [name for name, details in model_entries.items() if details.get('exists')]
    missing_models = [name for name, details in model_entries.items() if not details.get('exists')]
    free_disk_gb = _disk_free_gb(DATA_DIR)

    return {
        'api': {'ok': True, 'service': 'SHANK API'},
        'worker': worker_status,
        'ffmpeg': {'available': ffmpeg_path is not None, 'path': ffmpeg_path},
        'yt_dlp': {'available': yt_dlp_path is not None, 'path': yt_dlp_path},
        'stem_backend': stem_backend_status,
        'models': {
            'model_dir': models_status.get('model_dir'),
            'models_ready': bool(models_status.get('models_ready')),
            'found': found_models,
            'missing': missing_models,
        },
        'transcription': transcription_status,
        'data_dir': {
            'path': str(DATA_DIR),
            'writable': _is_dir_writable(DATA_DIR),
        },
        'disk': {'free_gb': free_disk_gb},
    }


@app.get('/transcription/status')
def get_transcription_status():
    """Return MT3 transcription availability and current backend configuration."""
    backend = os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip() or 'basic_pitch'
    mt3_enabled = os.getenv('MT3_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
    service_url = os.getenv('MT3_SERVICE_URL', '').strip().rstrip('/')
    return {
        'backend': backend,
        'mt3_enabled': mt3_enabled,
        'service_configured': bool(service_url),
        'service_url': service_url or None,
        'available': mt3_enabled and bool(service_url),
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


@app.get('/mt3/status')
def get_mt3_status():
    enabled = os.getenv('MT3_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
    backend = os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip().lower() or 'basic_pitch'
    service_url = os.getenv('MT3_SERVICE_URL', '').strip()
    state = 'available'
    reason = 'ok'
    reason_detail = 'MT3 is available.'
    if not enabled:
        state = 'unavailable'
        reason = 'mt3_disabled'
        reason_detail = 'MT3 is disabled by configuration (MT3_ENABLED=false).'
    elif backend == 'disabled':
        state = 'unavailable'
        reason = 'backend_disabled'
        reason_detail = 'Transcription backend is disabled.'
    elif not service_url:
        state = 'unavailable'
        reason = 'service_unconfigured'
        reason_detail = 'MT3 service URL is not configured.'
    available = state == 'available'

    return {
        'available': available,
        'state': state,
        'reason': reason,
        'reason_detail': reason_detail,
        'enabled': enabled,
        'backend': backend,
        'service_url': service_url or None,
        'message': reason_detail,
    }


@app.get('/api/models/status')
def get_models_status():
    """Return separator model availability and download status."""
    return _snapshot_model_download_status()


@app.post('/api/models/download')
def download_models_endpoint(six_stems: bool = False, model_dir: str | None = None):
    """Start downloading audio-separator models in the background."""
    return _start_model_download(six_stems=six_stems, model_dir=model_dir)


@app.post('/api/models/cancel')
def cancel_models_download_endpoint():
    """Cancel a currently running model download process."""
    process_to_cancel: subprocess.Popen[str] | None = None
    no_active_download = False
    with _MODEL_DOWNLOAD_LOCK:
        if not _MODEL_DOWNLOAD_STATE.get('is_downloading'):
            no_active_download = True
        else:
            process_to_cancel = _MODEL_DOWNLOAD_STATE.get('process')
            _MODEL_DOWNLOAD_STATE['status'] = 'cancelling'
            _MODEL_DOWNLOAD_STATE['status_message'] = 'Cancelling download...'
    if no_active_download:
        return {'cancelled': False, 'message': 'No active download.', **_snapshot_model_download_status()}
    if process_to_cancel is not None:
        process_to_cancel.terminate()
    return {'cancelled': True, **_snapshot_model_download_status()}


# ---------------------------------------------------------------------------
# Static UI — mount last so API routes take precedence
# ---------------------------------------------------------------------------

if _UI_DIR.is_dir():
    app.mount('/ui', StaticFiles(directory=str(_UI_DIR), html=True), name='ui')
