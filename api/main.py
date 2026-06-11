import json
import importlib.util
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
VALID_STEM_BACKENDS = frozenset({'auto', 'disabled', 'audio_separator', 'demucs', 'acestep'})
VALID_STEM_DEVICES = frozenset({'auto', 'cpu', 'cuda'})
VALID_STEM_MODES = frozenset({'4_stem', '6_stem'})
VALID_REPROCESS_SETTINGS = frozenset({
    'use_current_replace',
    'use_current_archive',
    'reuse_original_replace',
    'reuse_original_archive',
})

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
_MODEL_CONFIG_MIN_BYTES = 1024
_MODEL_WEIGHT_MIN_BYTES = 5 * 1024 * 1024
_MODEL_WEIGHT_EXTENSIONS = ('.ckpt', '.pt', '.pth', '.bin', '.safetensors')
_MODEL_CONFIG_EXTENSIONS = ('.yaml', '.yml')


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
    has_weight_files = any(
        path.is_file()
        and path.suffix.lower() in _MODEL_WEIGHT_EXTENSIONS
        and path.stat().st_size >= _MODEL_WEIGHT_MIN_BYTES
        for path in model_dir.iterdir()
    ) if model_dir.is_dir() else False
    payload: dict[str, dict[str, Any]] = {}
    for model_name in _MODEL_SPECS:
        path = model_dir / model_name
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        looks_like_config_only = (
            exists
            and path.suffix.lower() in _MODEL_CONFIG_EXTENSIONS
            and size_bytes < _MODEL_CONFIG_MIN_BYTES
        )
        ready = exists and (not looks_like_config_only or has_weight_files)
        payload[model_name] = {
            'exists': exists,
            'size_bytes': size_bytes,
            'ready': ready,
            'config_only': bool(looks_like_config_only and not has_weight_files),
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
    four_stem_ready = bool(models['htdemucs_ft.yaml']['ready'])
    six_stem_ready = bool(models['htdemucs_6s.yaml']['ready'])
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
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'download_stem_models.py'
    if not script_path.is_file():
        raise HTTPException(
            status_code=500,
            detail='Stem model download script is missing from the Docker image.',
        )

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


def _env_flag(name: str, default: str = 'false') -> bool:
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def _normalize_stem_backend(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace('-', '_')
    if normalized == 'none':
        normalized = 'disabled'
    if normalized not in VALID_STEM_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stem_backend '{value}'. Valid values: {sorted(VALID_STEM_BACKENDS)}",
        )
    return normalized


def _normalize_stem_device(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ('gpu', 'cuda/gpu'):
        normalized = 'cuda'
    if normalized not in VALID_STEM_DEVICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stem_device '{value}'. Valid values: {sorted(VALID_STEM_DEVICES)}",
        )
    return normalized


def _normalize_stem_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace('-', '_')
    if normalized in ('4stem', 'four_stem'):
        normalized = '4_stem'
    if normalized in ('6stem', 'six_stem'):
        normalized = '6_stem'
    if normalized not in VALID_STEM_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stem_mode '{value}'. Valid values: {sorted(VALID_STEM_MODES)}",
        )
    return normalized


def _normalize_reprocess_setting(value: str | None) -> str:
    if value is None:
        return 'use_current_replace'
    normalized = value.strip().lower().replace('-', '_')
    if normalized not in VALID_REPROCESS_SETTINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reprocess_mode '{value}'. Valid values: {sorted(VALID_REPROCESS_SETTINGS)}",
        )
    return normalized


def _infer_stem_mode_from_model(model_name: str | None) -> str:
    if isinstance(model_name, str) and '6s' in model_name.lower():
        return '6_stem'
    return '4_stem'


def _cuda_available() -> bool:
    nvidia_visible = os.getenv('NVIDIA_VISIBLE_DEVICES', '').strip().lower()
    if nvidia_visible and nvidia_visible != 'none':
        return True
    cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '').strip()
    if cuda_visible and cuda_visible != '-1':
        return True
    return shutil.which('nvidia-smi') is not None


def _analysis_defaults() -> dict[str, Any]:
    configured_backend = _normalize_stem_backend(os.getenv('STEM_BACKEND', 'auto')) or 'auto'
    default_model = os.getenv('AUDIO_SEPARATOR_MODEL', 'htdemucs_ft.yaml').strip() or 'htdemucs_ft.yaml'
    default_mode = _infer_stem_mode_from_model(default_model)
    return {
        'midi_enabled': _env_flag('MT3_ENABLED', 'false'),
        'midi_backend': os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip().lower() or 'basic_pitch',
        'stem_backend': configured_backend,
        'stem_model': default_model,
        'stem_device': _normalize_stem_device(os.getenv('AUDIO_SEPARATOR_DEVICE', 'cpu')) or 'cpu',
        'stem_mode': default_mode,
    }


def _resolve_reprocess_settings(
    reprocess_mode: str | None,
    *,
    preserve_existing: bool | None = None,
) -> dict[str, bool]:
    if reprocess_mode is None and preserve_existing is not None:
        normalized = 'use_current_archive' if preserve_existing else 'use_current_replace'
    else:
        normalized = _normalize_reprocess_setting(reprocess_mode)
    return {
        'use_current_settings': normalized.startswith('use_current_'),
        'replace_existing': normalized.endswith('replace'),
        'archive_previous': normalized.endswith('archive'),
    }


def _task_analysis_inputs(task: dict[str, Any]) -> dict[str, Any]:
    analysis_config = task.get('analysis_config')
    midi_config = analysis_config.get('midi') if isinstance(analysis_config, dict) else None
    stems_config = analysis_config.get('stems') if isinstance(analysis_config, dict) else None
    reprocess_config = analysis_config.get('reprocess') if isinstance(analysis_config, dict) else None
    return {
        'enable_mt3': (
            task.get('enable_mt3')
            if isinstance(task.get('enable_mt3'), bool)
            else (midi_config.get('enabled') if isinstance(midi_config, dict) else None)
        ),
        'stem_backend': task.get('stem_backend') or (stems_config.get('backend') if isinstance(stems_config, dict) else None),
        'stem_model': task.get('stem_model') or (stems_config.get('model') if isinstance(stems_config, dict) else None),
        'stem_device': task.get('stem_device') or (stems_config.get('device') if isinstance(stems_config, dict) else None),
        'stem_mode': task.get('stem_mode') or (stems_config.get('mode') if isinstance(stems_config, dict) else None),
        'reprocess_mode': (
            task.get('reprocess_mode')
            or (
                'reuse_original_archive'
                if isinstance(reprocess_config, dict)
                and not reprocess_config.get('use_current_settings', True)
                and reprocess_config.get('archive_previous')
                else 'reuse_original_replace'
                if isinstance(reprocess_config, dict)
                and not reprocess_config.get('use_current_settings', True)
                else 'use_current_archive'
                if isinstance(reprocess_config, dict) and reprocess_config.get('archive_previous')
                else 'use_current_replace'
            )
        ),
    }


def _build_analysis_config(
    *,
    enable_mt3: bool | None = None,
    stem_backend: str | None = None,
    stem_model: str | None = None,
    stem_device: str | None = None,
    stem_mode: str | None = None,
    reprocess_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = _analysis_defaults()
    backend_status = get_stem_backend_status()
    models_status = _snapshot_model_download_status()
    models = models_status.get('models', {})

    resolved_backend = _normalize_stem_backend(stem_backend) or defaults['stem_backend']
    if resolved_backend == 'auto':
        resolved_backend = backend_status.get('active_backend') or 'disabled'
    resolved_mode = _normalize_stem_mode(stem_mode)
    resolved_model = stem_model.strip() if isinstance(stem_model, str) and stem_model.strip() else defaults['stem_model']
    if resolved_mode is None:
        resolved_mode = _infer_stem_mode_from_model(resolved_model)
    elif not stem_model:
        candidate = 'htdemucs_6s.yaml' if resolved_mode == '6_stem' else 'htdemucs_ft.yaml'
        if candidate in models:
            resolved_model = candidate
    resolved_device = _normalize_stem_device(stem_device)
    if resolved_device in (None, 'auto'):
        if resolved_backend == 'demucs':
            resolved_device = _normalize_stem_device(os.getenv('DEMUCS_DEVICE', 'cpu')) or 'cpu'
        else:
            resolved_device = defaults['stem_device']
    reprocess_settings = _resolve_reprocess_settings(reprocess_mode)
    midi_enabled = defaults['midi_enabled'] if enable_mt3 is None else enable_mt3

    warnings: list[str] = []
    if resolved_backend == 'audio_separator':
        model_state = models.get(resolved_model)
        if not isinstance(model_state, dict):
            warnings.append(f'Audio Separator model {resolved_model} is not present in the model directory.')
        elif not model_state.get('ready'):
            warnings.append(f'Audio Separator model {resolved_model} is not ready yet.')
    if resolved_backend == 'demucs' and not backend_status.get('demucs', {}).get('available'):
        warnings.append('Demucs is not currently available.')
    if resolved_backend == 'acestep' and not backend_status.get('acestep', {}).get('configured'):
        warnings.append('Ace-Step is not configured.')
    if resolved_backend in ('disabled', 'none'):
        warnings.append('Stem separation is disabled.')
        resolved_backend = 'disabled'

    analysis_config: dict[str, Any] = {
        'midi': {
            'enabled': bool(midi_enabled),
            'backend': defaults['midi_backend'],
        },
        'stems': {
            'enabled': resolved_backend != 'disabled',
            'backend': resolved_backend,
            'model': resolved_model if resolved_backend == 'audio_separator' else None,
            'device': resolved_device,
            'mode': resolved_mode,
        },
        'reprocess': reprocess_settings,
    }
    if warnings:
        analysis_config['warnings'] = warnings

    task_fields = {
        'enable_mt3': bool(midi_enabled),
        'stem_backend': resolved_backend,
        'stem_model': resolved_model,
        'stem_device': resolved_device,
        'stem_mode': resolved_mode,
        'reprocess_mode': _normalize_reprocess_setting(reprocess_mode),
        'analysis_config': analysis_config,
    }
    return analysis_config, task_fields


def _analysis_settings_payload() -> dict[str, Any]:
    defaults = _analysis_defaults()
    stem_status = get_stem_backend_status()
    midi_status = get_mt3_status()
    models_status = _snapshot_model_download_status()
    warnings: list[str] = []
    if isinstance(models_status.get('warning'), str) and models_status.get('warning'):
        warnings.append(models_status['warning'])
    output_tail = models_status.get('output_tail')
    warnings.extend(output_tail[-3:] if isinstance(output_tail, list) else [])

    model_entries = []
    for name, details in models_status.get('models', {}).items():
        if not isinstance(details, dict):
            continue
        model_entries.append({
            'name': name,
            'mode': _infer_stem_mode_from_model(name),
            'available': bool(details.get('exists')),
            'ready': bool(details.get('ready')),
            'config_only': bool(details.get('config_only')),
        })

    return {
        'defaults': {
            'midi': {
                'selection': 'auto',
                'enabled': defaults['midi_enabled'],
                'backend': defaults['midi_backend'],
            },
            'stems': {
                'backend': defaults['stem_backend'],
                'model': defaults['stem_model'],
                'device': 'auto',
                'mode': defaults['stem_mode'],
                'active_backend': stem_status.get('active_backend'),
            },
            'reprocess': {
                'mode': 'use_current_replace',
                'replace_existing': True,
                'archive_previous': False,
                'use_current_settings': True,
            },
        },
        'midi': midi_status,
        'stem_backends': {
            'configured_backend': stem_status.get('configured_backend'),
            'active_backend': stem_status.get('active_backend'),
            'audio_separator': {
                **stem_status.get('audio_separator', {}),
                'status': (
                    'available, model ready'
                    if stem_status.get('audio_separator', {}).get('available')
                    and stem_status.get('audio_separator', {}).get('model_ready')
                    else 'available, model missing'
                    if stem_status.get('audio_separator', {}).get('available')
                    else 'unavailable'
                ),
            },
            'demucs': {
                **stem_status.get('demucs', {}),
                'status': 'available' if stem_status.get('demucs', {}).get('available') else 'unavailable',
            },
            'acestep': {
                **stem_status.get('acestep', {}),
                'status': (
                    'available'
                    if stem_status.get('acestep', {}).get('healthy')
                    else 'not configured'
                    if not stem_status.get('acestep', {}).get('configured')
                    else 'configured but unhealthy'
                ),
            },
        },
        'available_models': model_entries,
        'devices': {
            'default': 'auto',
            'available': ['auto', 'cpu', *(['cuda'] if _cuda_available() else [])],
            'cuda_available': _cuda_available(),
        },
        'warnings': warnings,
    }


def _archive_task_snapshot(task: dict[str, Any]) -> str | None:
    task_id = task.get('task_id')
    if not isinstance(task_id, str) or not task_id:
        return None
    archive_dir = DATA_DIR / 'results_archives' / task_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    archive_path = archive_dir / archive_name
    archive_path.write_text(json.dumps(task, indent=2))
    return str(archive_path)


async def _queue_audio_task(
    file: UploadFile,
    *,
    requested_type: str | None = None,
    enable_mt3: bool | None = None,
    stem_backend: str | None = None,
    stem_model: str | None = None,
    stem_device: str | None = None,
    stem_mode: str | None = None,
    reprocess_mode: str | None = None,
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

    _, analysis_fields = _build_analysis_config(
        enable_mt3=enable_mt3,
        stem_backend=stem_backend,
        stem_model=stem_model,
        stem_device=stem_device,
        stem_mode=stem_mode,
        reprocess_mode=reprocess_mode,
    )

    task: dict[str, Any] = {
        'task_id': task_id,
        'type': 'upload',
        'source': file.filename,
        'file_path': str(upload_path),
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        **analysis_fields,
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
async def upload_audio(
    file: UploadFile = File(...),
    enable_mt3: bool | None = Form(default=None),
    stem_backend: str | None = Form(default=None),
    stem_model: str | None = Form(default=None),
    stem_device: str | None = Form(default=None),
    stem_mode: str | None = Form(default=None),
):
    """Accept an audio file (MP3, WAV, FLAC) and queue it for analysis."""
    return JSONResponse(
        status_code=202,
        content=await _queue_audio_task(
            file,
            enable_mt3=enable_mt3,
            stem_backend=stem_backend,
            stem_model=stem_model,
            stem_device=stem_device,
            stem_mode=stem_mode,
        ),
    )


@app.post('/tasks/melody', status_code=202)
async def submit_melody(
    file: UploadFile = File(...),
    enable_mt3: bool = Form(True),
    stem_backend: str | None = Form(default=None),
    stem_model: str | None = Form(default=None),
    stem_device: str | None = Form(default=None),
    stem_mode: str | None = Form(default=None),
):
    """Accept an audio file and queue a melody-focused analysis task."""
    return JSONResponse(
        status_code=202,
        content=await _queue_audio_task(
            file,
            requested_type='melody',
            enable_mt3=enable_mt3,
            stem_backend=stem_backend,
            stem_model=stem_model,
            stem_device=stem_device,
            stem_mode=stem_mode,
        ),
    )


# ---------------------------------------------------------------------------
# Submit YouTube URL
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    url: str
    enable_mt3: bool | None = None
    stem_backend: str | None = None
    stem_model: str | None = None
    stem_device: str | None = None
    stem_mode: str | None = None

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
    _, analysis_fields = _build_analysis_config(
        enable_mt3=body.enable_mt3,
        stem_backend=body.stem_backend,
        stem_model=body.stem_model,
        stem_device=body.stem_device,
        stem_mode=body.stem_mode,
    )

    task: dict[str, Any] = {
        'task_id': task_id,
        'type': 'url',
        'source': body.url,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        **analysis_fields,
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
# Reprocess task
# ---------------------------------------------------------------------------

_VALID_REPROCESS_MODES = frozenset({
    'all',
    'audio_analysis',
    'stems',
    'midi',
    'metadata',
    'ai_prompts',
})


class ReprocessRequest(BaseModel):
    mode: str = 'all'
    preserve_existing: bool | None = None
    enable_mt3: bool | None = None
    stem_backend: str | None = None
    stem_model: str | None = None
    stem_device: str | None = None
    stem_mode: str | None = None
    reprocess_mode: str | None = None


@app.post('/tasks/{task_id}/reprocess', status_code=202)
def reprocess_task(task_id: str, body: ReprocessRequest):
    """Requeue a task using either current settings or the original task snapshot."""
    if body.mode not in _VALID_REPROCESS_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{body.mode}'. Valid modes: {sorted(_VALID_REPROCESS_MODES)}",
        )

    original = _load_task(task_id)
    task_type = original.get('type')
    source = original.get('source')

    if not source:
        raise HTTPException(status_code=400, detail='Original task has no source to reprocess')
    if task_type not in ('url', 'upload'):
        raise HTTPException(status_code=400, detail=f"Cannot reprocess task of type '{task_type}'")

    reprocess_settings = _resolve_reprocess_settings(
        body.reprocess_mode,
        preserve_existing=body.preserve_existing,
    )
    normalized_reprocess_mode = (
        _normalize_reprocess_setting(body.reprocess_mode)
        if body.reprocess_mode is not None
        else 'use_current_archive'
        if body.preserve_existing is True
        else 'use_current_replace'
    )
    original_inputs = _task_analysis_inputs(original)
    current_analysis_inputs = {
        'enable_mt3': body.enable_mt3,
        'stem_backend': body.stem_backend,
        'stem_model': body.stem_model,
        'stem_device': body.stem_device,
        'stem_mode': body.stem_mode,
        'reprocess_mode': body.reprocess_mode,
    }
    selected_inputs = current_analysis_inputs if reprocess_settings['use_current_settings'] else original_inputs
    selected_enable_mt3 = selected_inputs.get('enable_mt3')
    selected_stem_backend = selected_inputs.get('stem_backend')
    selected_stem_model = selected_inputs.get('stem_model')
    selected_stem_device = selected_inputs.get('stem_device')
    selected_stem_mode = selected_inputs.get('stem_mode')
    _, analysis_fields = _build_analysis_config(
        enable_mt3=selected_enable_mt3 if isinstance(selected_enable_mt3, bool) else None,
        stem_backend=selected_stem_backend if isinstance(selected_stem_backend, str) else None,
        stem_model=selected_stem_model if isinstance(selected_stem_model, str) else None,
        stem_device=selected_stem_device if isinstance(selected_stem_device, str) else None,
        stem_mode=selected_stem_mode if isinstance(selected_stem_mode, str) else None,
        reprocess_mode=normalized_reprocess_mode,
    )
    analysis_fields['analysis_config']['reprocess'] = reprocess_settings
    analysis_fields['reprocess_mode'] = normalized_reprocess_mode

    archive_path = _archive_task_snapshot(original) if reprocess_settings['archive_previous'] else None

    reset_task: dict[str, Any] = {
        'task_id': task_id,
        'type': task_type,
        'source': source,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'reprocess_target': body.mode,
        'reprocess_count': int(original.get('reprocess_count') or 0) + 1,
        **analysis_fields,
    }
    if archive_path:
        reset_task['archived_analysis'] = archive_path

    if task_type == 'upload':
        file_path = original.get('file_path')
        if file_path:
            reset_task['file_path'] = file_path
        requested_type = original.get('requested_type')
        if requested_type:
            reset_task['requested_type'] = requested_type

    if task_type == 'url' and isinstance(original.get('youtube'), dict):
        reset_task['youtube'] = original['youtube']
        reset_task['source_type'] = original.get('source_type')

    _write_task(reset_task)

    return JSONResponse(
        status_code=202,
        content={
            'task_id': task_id,
            'source_task_id': task_id,
            'status': 'pending',
            'archived_analysis': archive_path,
        },
    )


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
            'structure_json': structured_results.get('structure_json'),
            'waveform_beats_png': structured_results.get('waveform_beats_png'),
            'tempo_curve_png': structured_results.get('tempo_curve_png'),
            'beatgraph_png': structured_results.get('beatgraph_png'),
            'results_mt3_json': structured_results.get('mt3_json'),
            'lyrics_json': structured_results.get('lyrics_json'),
            'credits_json': structured_results.get('credits_json'),
            'song_metadata_json': structured_results.get('song_metadata_json'),
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


@app.get('/analysis/settings')
def get_analysis_settings():
    """Return page-level analysis defaults, availability, and warnings."""
    return _analysis_settings_payload()


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
    audio_separator_model = os.getenv('AUDIO_SEPARATOR_MODEL', 'htdemucs_ft.yaml').strip() or 'htdemucs_ft.yaml'
    audio_separator_model_dir = Path(os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/srv/shank/models/separator'))
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

    audio_separator_available = importlib.util.find_spec('audio_separator') is not None
    audio_separator_models = _models_payload(audio_separator_model_dir)
    audio_separator_model_status = audio_separator_models.get(
        audio_separator_model,
        {'exists': False, 'size_bytes': 0, 'ready': False, 'config_only': False},
    )
    audio_separator_ready = bool(audio_separator_model_status['ready'])
    demucs_available = shutil.which('demucs') is not None

    # Determine the effective active backend.
    if configured_backend == 'none':
        active_backend = 'none'
    elif configured_backend == 'acestep':
        active_backend = 'acestep' if (ace_step_url and ace_step_healthy) else 'none'
    elif configured_backend == 'audio_separator':
        active_backend = 'audio_separator' if audio_separator_available and audio_separator_ready else 'none'
    elif configured_backend == 'demucs':
        active_backend = 'demucs' if demucs_available else 'none'
    else:  # auto
        if ace_step_url and ace_step_healthy:
            active_backend = 'acestep'
        elif audio_separator_available and audio_separator_ready:
            active_backend = 'audio_separator'
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
        'audio_separator': {
            'available': audio_separator_available,
            'model': audio_separator_model,
            'model_dir': str(audio_separator_model_dir),
            'model_exists': bool(audio_separator_model_status['exists']),
            'model_ready': audio_separator_ready,
            'config_only': bool(audio_separator_model_status['config_only']),
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
    
    # Map backend names for display
    backend_display = {
        'basic_pitch': 'Basic Pitch',
        'mt3': 'MT3',
        'omnizart': 'Omnizart',
        'disabled': 'disabled'
    }.get(backend, backend)
    
    # Generic transcription wording unless backend is specifically MT3
    if backend == 'mt3':
        reason_detail = 'MT3 is available.'
    else:
        reason_detail = f'MIDI transcription is available (backend: {backend_display}).'
    
    if not enabled:
        state = 'unavailable'
        reason = 'transcription_disabled'
        reason_detail = 'MIDI transcription is disabled by configuration (MT3_ENABLED=false).'
    elif backend == 'disabled':
        state = 'unavailable'
        reason = 'backend_disabled'
        reason_detail = 'MIDI transcription backend is disabled.'
    elif not service_url:
        state = 'unavailable'
        reason = 'service_unconfigured'
        if backend == 'mt3':
            reason_detail = 'MT3 service URL is not configured.'
        else:
            reason_detail = f'MIDI transcription service URL is not configured (backend: {backend_display}).'
    available = state == 'available'

    return {
        'available': available,
        'state': state,
        'reason': reason,
        'reason_detail': reason_detail,
        'enabled': enabled,
        'backend': backend,
        'backend_display': backend_display,
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
