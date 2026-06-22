import json
import importlib.util
import logging
import os
import re
import struct
import subprocess
import shutil
import threading
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

app = FastAPI(title='SHANK API')

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
UPLOADS_DIR = DATA_DIR / 'uploads'
TASKS_DIR = DATA_DIR / 'tasks'
RESULTS_DIR = DATA_DIR / 'results'
DEFAULT_SEPARATOR_MODEL_DIR = Path(os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/srv/shank/models/separator'))

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}
ALLOWED_REQUESTED_TYPES = {'melody'}
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
VALID_STEM_BACKENDS = frozenset({'auto', 'disabled', 'audio_separator', 'demucs', 'acestep'})
VALID_STEM_DEVICES = frozenset({'auto', 'cpu', 'cuda'})
VALID_STEM_MODES = frozenset({'4_stem', '6_stem'})
VALID_LOOP_BAR_LENGTHS = frozenset({1, 2, 4, 8, 16})
LOOP_BEATS_PER_BAR = 4
MIDI_TICKS_PER_BEAT = 480
CHORD_BASE_MIDI_NOTE = 48
BASS_MELODY_SPLIT_MIDI = 60
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


def _loop_track_slug(task: dict[str, Any]) -> str:
    source = task.get('source')
    if isinstance(source, str) and source.strip():
        base_name = Path(source.strip()).stem or 'track'
    else:
        base_name = f"track_{str(task.get('task_id') or '')[:8] or 'unknown'}"
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', base_name).strip('_').lower()
    return slug or 'track'


def _validated_task_id(task_id: str) -> str:
    try:
        return str(uuid.UUID(task_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='Task not found')


def _loop_key_slug(task: dict[str, Any]) -> str:
    key = task.get('key')
    if not isinstance(key, str) or not key.strip():
        return 'unk'
    normalized = key.strip().replace(' minor', 'min').replace(' major', 'maj').replace(' ', '')
    return re.sub(r'[^a-zA-Z0-9#b_-]+', '', normalized) or 'unk'


def _task_beat_times(task: dict[str, Any]) -> list[float]:
    analysis_raw = task.get('analysis')
    analysis: dict[str, Any] = analysis_raw if isinstance(analysis_raw, dict) else {}
    full_mix_raw = analysis.get('full_mix')
    full_mix: dict[str, Any] = full_mix_raw if isinstance(full_mix_raw, dict) else {}
    beatgrid_raw = full_mix.get('beatgrid')
    beatgrid = beatgrid_raw if isinstance(beatgrid_raw, dict) else task.get('beatgrid')

    beat_times: list[float] = []
    beat_rows = beatgrid.get('beats') if isinstance(beatgrid, dict) else None
    if isinstance(beat_rows, list):
        for beat in beat_rows:
            value = beat.get('time') if isinstance(beat, dict) else beat
            if isinstance(value, (int, float)) and float(value) >= 0:
                beat_times.append(float(value))
    if beat_times:
        return sorted(set(beat_times))

    legacy_beats = full_mix.get('beats')
    if isinstance(legacy_beats, list):
        for beat in legacy_beats:
            if isinstance(beat, (int, float)) and float(beat) >= 0:
                beat_times.append(float(beat))
    if beat_times:
        return sorted(set(beat_times))

    bpm_raw = full_mix.get('bpm')
    bpm = bpm_raw if isinstance(bpm_raw, (int, float)) else task.get('bpm')
    duration = full_mix.get('duration_seconds')
    if not isinstance(duration, (int, float)):
        duration = task.get('duration_seconds')
    if isinstance(bpm, (int, float)) and isinstance(duration, (int, float)) and float(bpm) > 0 and float(duration) > 0:
        interval = 60.0 / float(bpm)
        current = 0.0
        while current <= float(duration) + interval:
            beat_times.append(round(current, 6))
            current += interval
    return beat_times


def _task_bar_starts(task: dict[str, Any], beats_per_bar: int = LOOP_BEATS_PER_BAR) -> list[float]:
    beat_times = _task_beat_times(task)
    bar_starts = [beat_times[idx] for idx in range(0, len(beat_times), beats_per_bar)]
    return bar_starts


def _loop_time_range(task: dict[str, Any], start_bar: int, bars: int, beats_per_bar: int = LOOP_BEATS_PER_BAR) -> tuple[float, float, list[float]]:
    bar_starts = _task_bar_starts(task, beats_per_bar=beats_per_bar)
    if start_bar < 1:
        raise HTTPException(status_code=400, detail='start_bar must be >= 1')
    if bars not in VALID_LOOP_BAR_LENGTHS:
        raise HTTPException(status_code=400, detail=f'bars must be one of {sorted(VALID_LOOP_BAR_LENGTHS)}')
    if not bar_starts:
        raise HTTPException(status_code=400, detail='Beat/bar markers unavailable for loop export')
    if start_bar > len(bar_starts):
        raise HTTPException(status_code=400, detail=f'start_bar exceeds detected bar count ({len(bar_starts)})')

    start_idx = start_bar - 1
    end_idx = start_idx + bars
    start_seconds = float(bar_starts[start_idx])
    if end_idx < len(bar_starts):
        end_seconds = float(bar_starts[end_idx])
    else:
        beat_times = _task_beat_times(task)
        beat_intervals = [
            beat_times[idx + 1] - beat_times[idx]
            for idx in range(len(beat_times) - 1)
            if beat_times[idx + 1] > beat_times[idx]
        ]
        mean_beat_interval = (sum(beat_intervals) / len(beat_intervals)) if beat_intervals else 0.5
        end_seconds = start_seconds + float(bars * beats_per_bar) * mean_beat_interval
    if end_seconds <= start_seconds:
        raise HTTPException(status_code=400, detail='Invalid loop range generated from beatgrid')
    return start_seconds, end_seconds, bar_starts


def _load_mt3_notes(task: dict[str, Any], track_name: str) -> list[dict[str, Any]]:
    track = _mt3_track(task, track_name)
    if not isinstance(track, dict):
        return []
    notes = track.get('notes')
    if isinstance(notes, list):
        return [note for note in notes if isinstance(note, dict)]
    notes_path = track.get('notes_path')
    if not isinstance(notes_path, str) or not notes_path:
        return []
    resolved = _resolve_data_path(notes_path)
    if resolved is None:
        return []
    try:
        payload = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [note for note in payload if isinstance(note, dict)]


def _midi_var_len(value: int) -> bytes:
    value = max(0, int(value))
    output = [value & 0x7F]
    while value > 0x7F:
        value >>= 7
        output.append((value & 0x7F) | 0x80)
    return bytes(reversed(output))


def _write_notes_to_midi(notes: list[dict[str, Any]], bpm: float, output_path: Path) -> None:
    ticks_per_beat = MIDI_TICKS_PER_BEAT
    bpm_value = bpm if bpm > 0 else 120.0
    seconds_per_tick = 60.0 / (bpm_value * ticks_per_beat)
    events: list[tuple[int, int, int, int]] = []
    for note in notes:
        start = note.get('start')
        end = note.get('end')
        pitch = note.get('pitch')
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not isinstance(pitch, (int, float)):
            continue
        if float(end) <= float(start):
            continue
        pitch_value = max(0, min(127, int(round(float(pitch)))))
        velocity_raw = note.get('velocity')
        velocity = max(1, min(127, int(round(float(velocity_raw))))) if isinstance(velocity_raw, (int, float)) else 90
        on_tick = max(0, int(round(float(start) / seconds_per_tick)))
        off_tick = max(on_tick + 1, int(round(float(end) / seconds_per_tick)))
        events.append((on_tick, 1, pitch_value, velocity))
        events.append((off_tick, 0, pitch_value, 0))

    events.sort(key=lambda item: (item[0], item[1]))
    tempo = int(round(60_000_000 / bpm_value))
    track_data = bytearray()
    track_data.extend(b'\x00\xFF\x51\x03' + tempo.to_bytes(3, 'big', signed=False))
    track_data.extend(b'\x00\xFF\x58\x04\x04\x02\x18\x08')

    last_tick = 0
    for tick, event_type, pitch, velocity in events:
        delta = tick - last_tick
        last_tick = tick
        track_data.extend(_midi_var_len(delta))
        if event_type == 1:
            track_data.extend(bytes([0x90, pitch, velocity]))
        else:
            track_data.extend(bytes([0x80, pitch, 0]))

    track_data.extend(b'\x00\xFF\x2F\x00')
    header = b'MThd' + struct.pack('>IHHH', 6, 0, 1, ticks_per_beat)
    track_chunk = b'MTrk' + struct.pack('>I', len(track_data)) + bytes(track_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + track_chunk)


def _clip_notes_to_window(notes: list[dict[str, Any]], start_seconds: float, end_seconds: float) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for note in notes:
        start = note.get('start')
        end = note.get('end')
        pitch = note.get('pitch')
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not isinstance(pitch, (int, float)):
            continue
        note_start = max(float(start), start_seconds)
        note_end = min(float(end), end_seconds)
        if note_end <= note_start:
            continue
        payload = {
            'start': note_start - start_seconds,
            'end': note_end - start_seconds,
            'pitch': int(round(float(pitch))),
            'velocity': note.get('velocity', 90),
        }
        clipped.append(payload)
    return clipped


def _chord_symbol_to_pitches(symbol: str) -> list[int]:
    mapping = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    parsed = re.match(r'^([A-G](?:#|b)?)(.*)$', symbol.strip())
    if not parsed:
        return []
    root_token = parsed.group(1)
    suffix = parsed.group(2).lower()
    is_minor = suffix.startswith('m') and not suffix.startswith('maj')
    root = mapping.get(root_token)
    if root is None:
        return []
    intervals = [0, 3, 7] if is_minor else [0, 4, 7]
    base = CHORD_BASE_MIDI_NOTE
    notes: list[int] = []
    for interval in intervals:
        candidate = base + root + interval
        if notes and candidate <= notes[-1]:
            candidate += 12
        notes.append(candidate)
    return notes


def _chords_to_notes(task: dict[str, Any], start_seconds: float, end_seconds: float) -> list[dict[str, Any]]:
    analysis_raw = task.get('analysis')
    analysis: dict[str, Any] = analysis_raw if isinstance(analysis_raw, dict) else {}
    full_mix_raw = analysis.get('full_mix')
    full_mix: dict[str, Any] = full_mix_raw if isinstance(full_mix_raw, dict) else {}
    chords_raw = full_mix.get('chords')
    chords = chords_raw if isinstance(chords_raw, dict) else task.get('chords')
    segments = chords.get('segments') if isinstance(chords, dict) else None
    if not isinstance(segments, list):
        return []
    notes: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        seg_start = segment.get('start_seconds')
        seg_end = segment.get('end_seconds')
        symbol = segment.get('symbol')
        if not isinstance(seg_start, (int, float)) or not isinstance(seg_end, (int, float)) or not isinstance(symbol, str):
            continue
        clipped_start = max(float(seg_start), start_seconds)
        clipped_end = min(float(seg_end), end_seconds)
        if clipped_end <= clipped_start:
            continue
        for pitch in _chord_symbol_to_pitches(symbol.strip()):
            notes.append({
                'pitch': pitch,
                'start': clipped_start - start_seconds,
                'end': clipped_end - start_seconds,
                'velocity': 72,
            })
    return notes


def _render_audio_loop(input_path: Path, output_path: Path, start_seconds: float, end_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg',
        '-y',
        '-i',
        str(input_path),
        '-ss',
        f'{start_seconds:.6f}',
        '-to',
        f'{end_seconds:.6f}',
        '-ar',
        '44100',
        '-ac',
        '2',
        '-c:a',
        'pcm_s16le',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f'ffmpeg failed while rendering loop: {result.stderr}')


# ---------------------------------------------------------------------------
# Cue point helpers
# ---------------------------------------------------------------------------

def _task_cue_points(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the cue_points list from a task dict."""
    analysis_raw = task.get('analysis')
    analysis: dict[str, Any] = analysis_raw if isinstance(analysis_raw, dict) else {}
    full_mix_raw = analysis.get('full_mix')
    full_mix: dict[str, Any] = full_mix_raw if isinstance(full_mix_raw, dict) else {}
    cue_points_raw = full_mix.get('cue_points')
    if not isinstance(cue_points_raw, list):
        return []
    return [cp for cp in cue_points_raw if isinstance(cp, dict)]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` hex string to an (R, G, B) integer tuple."""
    color = color.lstrip('#')
    if len(color) != 6:
        return (0, 0, 0)
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return (0, 0, 0)
    return (r, g, b)


def _cue_points_rekordbox_xml(
    cue_points: list[dict[str, Any]],
    *,
    title: str = '',
    artist: str = '',
    bpm: float | None = None,
    key: str | None = None,
    duration_seconds: float | None = None,
    file_location: str = '',
) -> str:
    """Render *cue_points* as a Rekordbox-compatible XML library string.

    The output is a standard Pioneer Rekordbox 6 XML library file containing a
    single TRACK entry with POSITION_MARK elements for each hot cue.
    """
    root = ET.Element('DJ_PLAYLISTS', Version='1.0.0')
    ET.SubElement(root, 'PRODUCT', Name='rekordbox', Version='6.6.6', Company='AlphaTheta')

    bpm_str = f'{bpm:.2f}' if bpm is not None else '0.00'
    total_time = str(int(duration_seconds)) if duration_seconds is not None else '0'
    collection = ET.SubElement(root, 'COLLECTION', Entries='1')
    track_attrs: dict[str, str] = {
        'TrackID': '1',
        'Name': title,
        'Artist': artist,
        'Album': '',
        'Genre': '',
        'Kind': 'WAV File',
        'Size': '0',
        'TotalTime': total_time,
        'DiscNumber': '0',
        'TrackNumber': '0',
        'Year': '0',
        'AverageBpm': bpm_str,
        'DateAdded': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'BitRate': '0',
        'SampleRate': '44100',
        'Comments': '',
        'PlayCount': '0',
        'Rating': '0',
        'Location': file_location,
        'Remixer': '',
        'Tonality': key or '',
        'Label': '',
        'Mix': '',
    }
    track = ET.SubElement(collection, 'TRACK', track_attrs)

    for cue in cue_points:
        hot_cue = cue.get('hot_cue')
        if not isinstance(hot_cue, int):
            continue
        time_seconds = cue.get('time_seconds', 0.0)
        color_hex = cue.get('color', '#28E614')
        r, g, b = _hex_to_rgb(str(color_hex))
        ET.SubElement(
            track,
            'POSITION_MARK',
            Name=str(cue.get('name', '')),
            Type='0',
            Start=f'{float(time_seconds):.3f}',
            Num=str(hot_cue),
            Red=str(r),
            Green=str(g),
            Blue=str(b),
        )

    playlists = ET.SubElement(root, 'PLAYLISTS')
    ET.SubElement(playlists, 'NODE', Type='0', Name='ROOT', Count='0')

    ET.indent(root, space='  ')
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode')


def _cue_points_traktor_nml(
    cue_points: list[dict[str, Any]],
    *,
    title: str = '',
    artist: str = '',
    bpm: float | None = None,
    file_location: str = '',
) -> str:
    """Render *cue_points* as a Traktor-compatible NML string.

    Traktor stores cue start positions in **milliseconds**.  Each hot cue
    becomes a CUE_V2 element with ``TYPE="0"`` (standard cue).
    """
    root = ET.Element('NML', VERSION='19')
    ET.SubElement(root, 'HEAD', COMPANY='www.native-instruments.com', PROGRAM='Traktor')

    collection = ET.SubElement(root, 'COLLECTION', ENTRIES='1')
    entry = ET.SubElement(
        collection,
        'ENTRY',
        TITLE=title,
        ARTIST=artist,
    )

    if file_location:
        from pathlib import PurePosixPath
        # Traktor LOCATION splits the path into DIR (parent with Traktor's /: prefix)
        # and FILE (basename).  Strip any file:// scheme prefix first.
        p = PurePosixPath(file_location.lstrip('file:///').lstrip('/'))
        parent_parts = str(p.parent).strip('/')
        dir_str = f'/:{("/" + parent_parts + "/") if parent_parts else "/"}'
        ET.SubElement(
            entry,
            'LOCATION',
            DIR=dir_str,
            FILE=p.name,
            VOLUME='',
            VOLUMEID='',
        )

    if bpm is not None:
        ET.SubElement(entry, 'TEMPO', BPM=f'{bpm:.6f}', BPM_QUALITY='100.000000')

    for idx, cue in enumerate(cue_points):
        hot_cue = cue.get('hot_cue')
        if not isinstance(hot_cue, int):
            continue
        time_seconds = cue.get('time_seconds', 0.0)
        # Traktor uses milliseconds
        start_ms = float(time_seconds) * 1000.0
        ET.SubElement(
            entry,
            'CUE_V2',
            NAME=str(cue.get('name', '')),
            DISPL_ORDER=str(idx),
            TYPE='0',
            START=f'{start_ms:.6f}',
            LEN='0.000000',
            REPEATS='-1',
            HOTCUE=str(hot_cue),
        )

    ET.indent(root, space='  ')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(root, encoding='unicode')


def _cue_points_mixxx_xml(
    cue_points: list[dict[str, Any]],
    *,
    title: str = '',
    artist: str = '',
    file_location: str = '',
    sample_rate: int = 44100,
) -> str:
    """Render *cue_points* as a Mixxx library XML string.

    Mixxx stores cue positions in **samples** (file sample rate × seconds).
    The *sample_rate* parameter defaults to 44100 Hz; pass the actual track
    sample rate for bit-perfect accuracy.
    """
    root = ET.Element('Mixxx-Library')
    track = ET.SubElement(
        root,
        'Track',
        id='1',
        location=file_location,
        title=title,
        artist=artist,
    )
    cues_el = ET.SubElement(track, 'Cues')

    for cue in cue_points:
        hot_cue = cue.get('hot_cue')
        if not isinstance(hot_cue, int):
            continue
        time_seconds = cue.get('time_seconds', 0.0)
        position_samples = int(round(float(time_seconds) * sample_rate))
        color_hex = cue.get('color', '#FFFF00')
        # Mixxx encodes colors as 0xAARRGGBB integers (FF = fully opaque).
        r, g, b = _hex_to_rgb(str(color_hex))
        color_int = (0xFF << 24) | (r << 16) | (g << 8) | b
        ET.SubElement(
            cues_el,
            'Cue',
            id=str(hot_cue + 1),
            type='1',
            position=str(position_samples),
            length='0',
            hotcue=str(hot_cue),
            label=str(cue.get('name', '')),
            color=f'0x{color_int:08X}',
        )

    ET.indent(root, space='  ')
    return "<?xml version='1.0' encoding='UTF-8'?>\n" + ET.tostring(root, encoding='unicode')


def _prepare_loop_exports(task: dict[str, Any], task_id: str, start_bar: int, bars: int) -> dict[str, Any]:
    start_seconds, end_seconds, bar_starts = _loop_time_range(task, start_bar=start_bar, bars=bars)
    safe_task_id = _validated_task_id(task_id)
    bar_end = start_bar + bars - 1
    bpm_raw = task.get('bpm')
    bpm = float(bpm_raw) if isinstance(bpm_raw, (int, float)) else 120.0
    track_slug = _loop_track_slug(task)
    key_slug = _loop_key_slug(task)
    loop_dir = RESULTS_DIR / safe_task_id / 'loops' / f'bars_{start_bar:03d}-{bar_end:03d}'
    loop_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []

    artifact_sources = _task_artifacts(task)
    audio_sources: list[tuple[str, str, Path]] = []
    normalized = artifact_sources.get('normalized_wav')
    if normalized is not None:
        audio_sources.append(('fullmix_wav', 'Full Mix WAV', normalized))
    for stem_name in ('drums', 'bass', 'vocals', 'other'):
        stem_path = artifact_sources.get(f'stem_{stem_name}_wav')
        if stem_path is not None:
            audio_sources.append((f'{stem_name}_wav', f'{stem_name.capitalize()} Stem WAV', stem_path))

    for clip_id, label, source_path in audio_sources:
        ext = 'wav'
        file_name = f'{track_slug}_bars_{start_bar:03d}-{bar_end:03d}_{clip_id.replace("_wav", "")}_{int(round(bpm))}bpm_{key_slug}.{ext}'
        output_path = loop_dir / file_name
        if not output_path.exists():
            _render_audio_loop(source_path, output_path, start_seconds, end_seconds)
        generated.append({
            'clip_id': clip_id,
            'label': label,
            'format': 'wav',
            'path': output_path,
            'filename': file_name,
            'media_type': 'audio/wav',
        })

    full_mix_notes = _load_mt3_notes(task, 'full_mix')
    bass_notes = _load_mt3_notes(task, 'bass') or [
        note
        for note in full_mix_notes
        if isinstance(note.get('pitch'), (int, float)) and float(note['pitch']) < BASS_MELODY_SPLIT_MIDI
    ]
    melody_notes = [
        note
        for note in full_mix_notes
        if isinstance(note.get('pitch'), (int, float)) and float(note['pitch']) >= BASS_MELODY_SPLIT_MIDI
    ]
    drum_notes = _load_mt3_notes(task, 'drums')
    chord_notes = _chords_to_notes(task, start_seconds, end_seconds)
    midi_sources: list[tuple[str, str, list[dict[str, Any]]]] = [
        ('melody_midi', 'Melody MIDI', _clip_notes_to_window(melody_notes, start_seconds, end_seconds)),
        ('bass_midi', 'Bass MIDI', _clip_notes_to_window(bass_notes, start_seconds, end_seconds)),
        ('chord_midi', 'Chord MIDI', chord_notes),
        ('drum_midi', 'Drum MIDI', _clip_notes_to_window(drum_notes, start_seconds, end_seconds)),
    ]

    for clip_id, label, notes in midi_sources:
        if not notes:
            continue
        file_name = f'{track_slug}_bars_{start_bar:03d}-{bar_end:03d}_{clip_id.replace("_midi", "")}_{int(round(bpm))}bpm_{key_slug}.mid'
        output_path = loop_dir / file_name
        if not output_path.exists():
            _write_notes_to_midi(notes, bpm, output_path)
        generated.append({
            'clip_id': clip_id,
            'label': label,
            'format': 'mid',
            'path': output_path,
            'filename': file_name,
            'media_type': 'audio/midi',
        })

    zip_name = f'{track_slug}_bars_{start_bar:03d}-{bar_end:03d}_{int(round(bpm))}bpm_{key_slug}.zip'
    zip_path = loop_dir / zip_name
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for clip in generated:
            zip_file.write(clip['path'], arcname=clip['filename'])

    return {
        'start_bar': start_bar,
        'bars': bars,
        'bar_count': len(bar_starts),
        'start_seconds': round(start_seconds, 6),
        'end_seconds': round(end_seconds, 6),
        'clips': generated,
        'zip_path': zip_path,
    }


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


@app.get('/tasks/fingerprints')
def list_task_fingerprints():
    """Return Audio DNA fingerprints for all completed tasks.

    The response maps each ``task_id`` to its fingerprint dict.  Fingerprints
    are loaded from the on-disk ``fingerprint.json`` artifact written by the
    worker when a task completes.  Tasks that have no fingerprint yet (e.g.
    still processing, or processed before this feature was added) are omitted.

    Callers can use the returned fingerprints for:

    * **Duplicate detection** – find tasks sharing the same ``fingerprint_hash``.
    * **Collection organisation** – group by ``key`` or ``bpm`` range.
    * **Similarity search** – compare ``energy_profile``/``spectral_profile``
      vectors using cosine similarity or nearest-neighbour algorithms.
    """
    _ensure_dirs()
    fingerprints: dict[str, Any] = {}
    for task_file in TASKS_DIR.glob('*.json'):
        try:
            task = json.loads(task_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if task.get('status') != 'done':
            continue
        task_id = task.get('task_id')
        if not isinstance(task_id, str):
            continue
        structured_results = task.get('results')
        if not isinstance(structured_results, dict):
            continue
        fp_path = structured_results.get('fingerprint_json')
        if not isinstance(fp_path, str) or not fp_path:
            continue
        resolved = _resolve_data_path(fp_path)
        if resolved is None:
            continue
        try:
            fingerprint = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(fingerprint, dict) and fingerprint:
            fingerprints[task_id] = fingerprint
    return {'fingerprints': fingerprints}


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

    midi_stems_data = task.get('midi_stems')
    if isinstance(midi_stems_data, dict):
        midi_stems = midi_stems_data.get('stems')
        if isinstance(midi_stems, dict):
            for role, stem_info in midi_stems.items():
                if not isinstance(role, str) or not isinstance(stem_info, dict):
                    continue
                midi_path = stem_info.get('midi_path')
                if not isinstance(midi_path, str) or not midi_path:
                    continue
                resolved = _resolve_data_path(midi_path)
                if resolved is not None:
                    artifacts[f'midi_stem_{role}'] = resolved

    structured_results = task.get('results')
    if isinstance(structured_results, dict):
        structured_files = {
            'results_task_json': structured_results.get('task_json'),
            'results_analysis_json': structured_results.get('analysis_json'),
            'beatgrid_json': structured_results.get('beatgrid_json'),
            'structure_json': structured_results.get('structure_json'),
            'fingerprint_json': structured_results.get('fingerprint_json'),
            'waveform_beats_png': structured_results.get('waveform_beats_png'),
            'tempo_curve_png': structured_results.get('tempo_curve_png'),
            'beatgraph_png': structured_results.get('beatgraph_png'),
            'results_mt3_json': structured_results.get('mt3_json'),
            'results_midi_stems_json': structured_results.get('midi_stems_json'),
            'lyrics_json': structured_results.get('lyrics_json'),
            'credits_json': structured_results.get('credits_json'),
            'song_metadata_json': structured_results.get('song_metadata_json'),
            'musical_profile_json': structured_results.get('musical_profile_json'),
            'ace_step_prompt_json': structured_results.get('ace_step_prompt_json'),
            'results_artifacts_json': structured_results.get('artifacts_json'),
        }
        for artifact_name, artifact_path in structured_files.items():
            if not isinstance(artifact_path, str) or not artifact_path:
                continue
            resolved = _resolve_data_path(artifact_path)
            if resolved is not None:
                artifacts[artifact_name] = resolved

    return artifacts


@app.get('/tasks/{task_id}/report')
def get_task_report(task_id: str, format: str = 'json', request: Request = None):  # type: ignore[assignment]
    """Return a complete song-breakdown report for a completed task.

    Supported formats (via ``?format=`` query parameter or ``Accept`` header):

    * ``json``  – structured JSON (default)
    * ``html``  – self-contained HTML page with inline SVG charts
    * ``pdf``   – multi-page PDF (requires matplotlib)

    The ``Accept`` header is checked when no ``format`` parameter is supplied:
    ``text/html`` → HTML, ``application/pdf`` → PDF, everything else → JSON.
    """
    from api.report import build_report_json, build_report_html, build_report_pdf

    task = _load_task(task_id)

    # Resolve format from query param or Accept header.
    fmt = format.lower().strip()
    if fmt == 'json' and request is not None:
        accept = request.headers.get('accept', '')
        if _get_media_type_quality(accept, 'text/html') > _get_media_type_quality(accept, 'application/json'):
            fmt = 'html'
        elif _get_media_type_quality(accept, 'application/pdf') > _get_media_type_quality(accept, 'application/json'):
            fmt = 'pdf'

    if fmt == 'html':
        html_content = build_report_html(task)
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)

    if fmt == 'pdf':
        try:
            pdf_bytes = build_report_pdf(task)
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f'PDF generation requires matplotlib: {exc}',
            )
        from fastapi.responses import Response
        safe_task_id = _validated_task_id(task_id)
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="shank-report-{safe_task_id}.pdf"'},
        )

    # Default: JSON
    return build_report_json(task)


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
    """Download a MIDI transcription artifact for full mix or a specific stem."""
    task = _load_task(task_id)
    track = _mt3_track(task, track_name)
    midi_path = track.get('midi_path') if isinstance(track, dict) else None
    if not isinstance(midi_path, str) or not midi_path:
        raise HTTPException(status_code=404, detail='MIDI not found')
    resolved = _resolve_data_path(midi_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail='MIDI not found')
    return FileResponse(path=resolved, media_type='audio/midi', filename=resolved.name)


@app.get('/tasks/{task_id}/mt3/notes/{track_name}')
def get_mt3_notes(task_id: str, track_name: str):
    """Return MIDI note metadata JSON for full mix or a specific stem."""
    task = _load_task(task_id)
    track = _mt3_track(task, track_name)
    notes_path = track.get('notes_path') if isinstance(track, dict) else None
    if not isinstance(notes_path, str) or not notes_path:
        raise HTTPException(status_code=404, detail='MIDI note metadata not found')
    resolved = _resolve_data_path(notes_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail='MIDI note metadata not found')
    try:
        return json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail='MIDI note metadata is unreadable')


# ---------------------------------------------------------------------------
# MIDI stem extraction endpoints
# ---------------------------------------------------------------------------

def _midi_stems_from_task(task: dict) -> dict:
    """Return the ``midi_stems.stems`` dict from a task, or an empty dict."""
    midi_stems_data = task.get('midi_stems')
    if not isinstance(midi_stems_data, dict):
        return {}
    stems = midi_stems_data.get('stems')
    return stems if isinstance(stems, dict) else {}


@app.get('/tasks/{task_id}/midi-stems')
def list_midi_stems(task_id: str):
    """List available MIDI stem roles for a task.

    Returns ``{"roles": [...], "status": "...", "backend": "..."}`` where
    ``roles`` is a list of available role names (e.g. ``drums``, ``bass``,
    ``melody``, ``chords``).
    """
    task = _load_task(task_id)
    midi_stems_data = task.get('midi_stems')
    if not isinstance(midi_stems_data, dict):
        return {'roles': [], 'status': 'unavailable', 'backend': None}
    stems = _midi_stems_from_task(task)
    available_roles = [
        role for role, info in stems.items()
        if isinstance(info, dict) and isinstance(info.get('midi_path'), str)
        and _resolve_data_path(info['midi_path']) is not None
    ]
    return {
        'roles': sorted(available_roles),
        'status': midi_stems_data.get('status', 'unknown'),
        'backend': midi_stems_data.get('backend'),
        'warnings': midi_stems_data.get('warnings') or [],
        'errors': midi_stems_data.get('errors') or [],
    }


@app.get('/tasks/{task_id}/midi-stems/{role}')
def download_midi_stem(task_id: str, role: str):
    """Download the MIDI file for a specific stem role (drums, bass, melody, chords)."""
    task = _load_task(task_id)
    stems = _midi_stems_from_task(task)
    stem_info = stems.get(role)
    if not isinstance(stem_info, dict):
        raise HTTPException(status_code=404, detail=f'MIDI stem {role!r} not found')
    midi_path = stem_info.get('midi_path')
    if not isinstance(midi_path, str) or not midi_path:
        raise HTTPException(status_code=404, detail=f'MIDI stem {role!r} has no MIDI file')
    resolved = _resolve_data_path(midi_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f'MIDI stem {role!r} file not found on disk')
    return FileResponse(path=resolved, media_type='audio/midi', filename=f'{role}.mid')


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


@app.get('/tasks/{task_id}/harmonic')
def get_task_harmonic(task_id: str):
    """Return the harmonic analysis for a completed task.

    The response contains:

    * ``key`` – the globally detected key string (e.g. ``'C major'``).
    * ``key_changes`` – list of key-change events, each with
      ``time_seconds``, ``timestamp`` (``MM:SS``), ``key``, and
      ``confidence``.
    * ``segments`` – chord segments enriched with ``roman_numeral``
      (e.g. ``'I'``, ``'vi'``, ``'bVII'``) and ``is_borrowed`` (bool).
    * ``borrowed_chords`` – filtered list of segments where ``is_borrowed``
      is ``true``.
    """
    task = _load_task(task_id)
    harmonic = task.get('harmonic')
    if not isinstance(harmonic, dict):
        raise HTTPException(status_code=404, detail='Harmonic analysis not available for this task')
    return harmonic


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


@app.get('/tasks/{task_id}/fingerprint')
def get_task_fingerprint(task_id: str):
    """Return the Audio DNA fingerprint for a completed task.

    The fingerprint encodes key musical characteristics for duplicate detection,
    similarity search, collection organisation, and recommendation.

    Fields:

    * ``version``          – fingerprint schema version.
    * ``bpm``              – raw BPM float.
    * ``bpm_normalized``   – BPM normalised to [0, 1] on a 0–200 BPM scale.
    * ``key``              – musical key, e.g. ``'A minor'``.
    * ``key_index``        – integer 0–23 encoding tonic + mode.
    * ``chord_profile``    – duration-weighted chord frequency map.
    * ``energy_profile``   – 32-bin normalised energy curve.
    * ``spectral_profile`` – 8-bin normalised mel-spectral summary.
    * ``duration_seconds`` – track length in seconds.
    * ``fingerprint_hash`` – SHA-256 hex digest for quick duplicate lookup.
    """
    task = _load_task(task_id)
    structured_results = task.get('results')
    if isinstance(structured_results, dict):
        fp_path = structured_results.get('fingerprint_json')
        if isinstance(fp_path, str) and fp_path:
            resolved = _resolve_data_path(fp_path)
            if resolved is not None:
                try:
                    fingerprint = json.loads(resolved.read_text())
                    if isinstance(fingerprint, dict) and fingerprint:
                        return fingerprint
                except (OSError, json.JSONDecodeError):
                    pass
    raise HTTPException(status_code=404, detail='Fingerprint not available for this task')


# ---------------------------------------------------------------------------
# Fingerprint & similar-song finder
# ---------------------------------------------------------------------------

# These threshold constants mirror _BPM_SIMILARITY_THRESHOLD,
# _CHORD_SIMILARITY_THRESHOLD, and _ENERGY_SIMILARITY_THRESHOLD in
# worker/analyze.py.  The API and worker run in separate containers so the
# values are intentionally kept in sync manually rather than via a shared
# import.
_FP_BPM_THRESHOLD = 0.95
_FP_CHORD_THRESHOLD = 0.5
_FP_ENERGY_THRESHOLD = 0.8


def _fp_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length float lists.

    This is a dependency-free equivalent of the numpy-based implementation
    used by ``compare_fingerprints`` in ``worker/analyze.py``; the API
    container does not depend on numpy.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = sum(x * x for x in a[:n]) ** 0.5
    norm_b = sum(x * x for x in b[:n]) ** 0.5
    return max(0.0, min(1.0, dot / (norm_a * norm_b))) if (norm_a > 0 and norm_b > 0) else 0.0


def _compare_fingerprints(fp_a: dict[str, Any], fp_b: dict[str, Any]) -> dict[str, Any]:
    """Compare two fingerprint dicts; return similarity score (0–100) and reasons.

    Mirrors ``compare_fingerprints`` in ``worker/analyze.py`` using pure Python
    instead of numpy (the API container has no numpy dependency).
    """
    reasons: list[str] = []
    details: dict[str, Any] = {}
    score = 0.0
    total_weight = 0.0

    # BPM similarity (weight 0.25)
    bpm_a = float(fp_a.get('bpm') or 0.0)
    bpm_b = float(fp_b.get('bpm') or 0.0)
    if bpm_a > 0 and bpm_b > 0:
        bpm_ratio = min(bpm_a, bpm_b) / max(bpm_a, bpm_b)
        if bpm_ratio >= _FP_BPM_THRESHOLD:
            reasons.append('Same BPM range')
        details['bpm_similarity'] = round(bpm_ratio, 3)
        score += bpm_ratio * 0.25
        total_weight += 0.25

    # Key similarity (weight 0.25)
    key_a = str(fp_a.get('key') or '').strip()
    key_b = str(fp_b.get('key') or '').strip()
    if key_a and key_b:
        if key_a == key_b:
            key_score = 1.0
            reasons.append('Same key')
        else:
            tonic_a = key_a.split()[0] if key_a else ''
            tonic_b = key_b.split()[0] if key_b else ''
            key_score = 0.5 if (tonic_a and tonic_a == tonic_b) else 0.0
            if key_score > 0:
                reasons.append('Same tonic, different mode')
        details['key_match'] = key_score > 0.0
        score += key_score * 0.25
        total_weight += 0.25

    # Chord progression similarity (weight 0.25) – Jaccard index
    prog_a = {str(c) for c in (fp_a.get('chord_progression') or []) if str(c).strip()}
    prog_b = {str(c) for c in (fp_b.get('chord_progression') or []) if str(c).strip()}
    if prog_a and prog_b:
        union = len(prog_a | prog_b)
        chord_score = len(prog_a & prog_b) / union if union > 0 else 0.0
        if chord_score >= _FP_CHORD_THRESHOLD:
            reasons.append('Similar chord progression')
        details['chord_similarity'] = round(chord_score, 3)
        score += chord_score * 0.25
        total_weight += 0.25

    # Energy curve similarity (weight 0.25) – cosine
    energy_a = [float(v) for v in (fp_a.get('energy_profile') or []) if isinstance(v, (int, float))]
    energy_b = [float(v) for v in (fp_b.get('energy_profile') or []) if isinstance(v, (int, float))]
    if energy_a and energy_b:
        energy_score = _fp_cosine_similarity(energy_a, energy_b)
        if energy_score >= _FP_ENERGY_THRESHOLD:
            reasons.append('Similar energy curve')
        details['energy_similarity'] = round(energy_score, 3)
        score += energy_score * 0.25
        total_weight += 0.25

    similarity = int(round((score / total_weight) * 100)) if total_weight > 0 else 0
    return {'similarity': similarity, 'reasons': reasons, 'details': details}


def _load_task_fingerprint(task: dict[str, Any]) -> dict[str, Any] | None:
    """Load the pre-computed fingerprint for a task, or return None."""
    results = task.get('results')
    if not isinstance(results, dict):
        return None
    fp_path = results.get('fingerprint_json')
    if not isinstance(fp_path, str) or not fp_path:
        return None
    resolved = _resolve_data_path(fp_path)
    if resolved is None:
        return None
    try:
        data = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


@app.get('/tasks/{task_id}/similar')
def get_similar_tasks(task_id: str, limit: int = 10):
    """Find completed tasks with audio fingerprints similar to *task_id*.

    Similarity is computed across four weighted dimensions:

    * **BPM** – ratio of the lower to higher BPM (weight 25%).
    * **Key** – exact key match scores 100 %; same tonic in different mode
      scores 50 % (weight 25%).
    * **Chord progression** – Jaccard index over the de-duplicated chord sets
      (weight 25%).
    * **Energy curve** – cosine similarity of the coarsened energy profiles
      (weight 25%).

    Returns a list of matches sorted by descending similarity, each with
    ``task_id``, ``similarity`` (0–100), ``reasons``, and ``details``.
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail='limit must be between 1 and 100')
    safe_task_id = _validated_task_id(task_id)
    target_task = _load_task(task_id)
    target_fp = _load_task_fingerprint(target_task)
    if target_fp is None:
        raise HTTPException(status_code=404, detail='Fingerprint not available for this task')

    _ensure_dirs()
    matches: list[dict[str, Any]] = []
    for task_file in TASKS_DIR.glob('*.json'):
        try:
            candidate = json.loads(task_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if candidate.get('status') != 'done':
            continue
        candidate_id = str(candidate.get('task_id') or task_file.stem)
        if candidate_id == safe_task_id:
            continue
        candidate_fp = _load_task_fingerprint(candidate)
        if candidate_fp is None:
            continue
        comparison = _compare_fingerprints(target_fp, candidate_fp)
        matches.append({
            'task_id': candidate_id,
            'similarity': comparison['similarity'],
            'reasons': comparison['reasons'],
            'details': comparison['details'],
        })

    matches.sort(key=lambda m: m['similarity'], reverse=True)
    return {
        'task_id': safe_task_id,
        'matches': matches[:limit],
    }


# ---------------------------------------------------------------------------
# Cue points
# ---------------------------------------------------------------------------


@app.get('/tasks/{task_id}/cue-points')
def get_task_cue_points(task_id: str):
    """Return the DJ cue points for a completed task.

    Each entry contains:

    * ``name`` – display label (e.g. ``'Intro'``, ``'Verse'``, ``'Chorus'``).
    * ``time_seconds`` – cue position in seconds.
    * ``hot_cue`` – zero-based hot cue index (0 = A, 1 = B, …, 7 = H).
    * ``color`` – recommended hex colour string (e.g. ``'#28E614'``).
    """
    task = _load_task(task_id)
    cue_points = _task_cue_points(task)
    if not cue_points:
        raise HTTPException(status_code=404, detail='Cue point data not available for this task')
    return {'cue_points': cue_points}


@app.get('/tasks/{task_id}/cue-points/export/rekordbox')
def export_cue_points_rekordbox(task_id: str):
    """Export cue points as a Rekordbox-compatible XML library file.

    The response is a ``application/xml`` download named
    ``<task_id>_cue_points_rekordbox.xml``.  Import this file into Rekordbox
    via *File → Import → rekordbox xml*.
    """
    task = _load_task(task_id)
    cue_points = _task_cue_points(task)
    if not cue_points:
        raise HTTPException(status_code=404, detail='Cue point data not available for this task')

    analysis_raw = task.get('analysis')
    full_mix: dict[str, Any] = {}
    if isinstance(analysis_raw, dict):
        fm = analysis_raw.get('full_mix')
        if isinstance(fm, dict):
            full_mix = fm

    xml_content = _cue_points_rekordbox_xml(
        cue_points,
        title=str(task.get('title') or task.get('source') or ''),
        artist=str(task.get('artist') or ''),
        bpm=full_mix.get('bpm'),
        key=full_mix.get('key'),
        duration_seconds=full_mix.get('duration_seconds'),
        file_location=str(task.get('normalized_path') or ''),
    )
    filename = f'{task_id}_cue_points_rekordbox.xml'
    return Response(
        content=xml_content,
        media_type='application/xml',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get('/tasks/{task_id}/cue-points/export/traktor')
def export_cue_points_traktor(task_id: str):
    """Export cue points as a Traktor-compatible NML file.

    The response is a ``application/xml`` download named
    ``<task_id>_cue_points_traktor.nml``.  Import into Traktor Pro via
    *Preferences → File Management → Import* or by dropping the NML onto the
    Traktor collection.
    """
    task = _load_task(task_id)
    cue_points = _task_cue_points(task)
    if not cue_points:
        raise HTTPException(status_code=404, detail='Cue point data not available for this task')

    analysis_raw = task.get('analysis')
    full_mix: dict[str, Any] = {}
    if isinstance(analysis_raw, dict):
        fm = analysis_raw.get('full_mix')
        if isinstance(fm, dict):
            full_mix = fm

    nml_content = _cue_points_traktor_nml(
        cue_points,
        title=str(task.get('title') or task.get('source') or ''),
        artist=str(task.get('artist') or ''),
        bpm=full_mix.get('bpm'),
        file_location=str(task.get('normalized_path') or ''),
    )
    filename = f'{task_id}_cue_points_traktor.nml'
    return Response(
        content=nml_content,
        media_type='application/xml',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get('/tasks/{task_id}/cue-points/export/mixxx')
def export_cue_points_mixxx(task_id: str):
    """Export cue points as a Mixxx library XML file.

    The response is a ``application/xml`` download named
    ``<task_id>_cue_points_mixxx.xml``.  Import into Mixxx via
    *Library → Import Library Backup* or by copying into the Mixxx library
    directory.
    """
    task = _load_task(task_id)
    cue_points = _task_cue_points(task)
    if not cue_points:
        raise HTTPException(status_code=404, detail='Cue point data not available for this task')

    xml_content = _cue_points_mixxx_xml(
        cue_points,
        title=str(task.get('title') or task.get('source') or ''),
        artist=str(task.get('artist') or ''),
        file_location=str(task.get('normalized_path') or ''),
    )
    filename = f'{task_id}_cue_points_mixxx.xml'
    return Response(
        content=xml_content,
        media_type='application/xml',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get('/tasks/{task_id}/loops')
def list_task_loop_exports(task_id: str, start_bar: int = 1, bars: int = 4):
    safe_task_id = _validated_task_id(task_id)
    task = _load_task(task_id)
    exports = _prepare_loop_exports(task, task_id, start_bar=start_bar, bars=bars)
    clips: list[dict[str, Any]] = []
    for clip in exports['clips']:
        clips.append({
            'clip_id': clip['clip_id'],
            'label': clip['label'],
            'format': clip['format'],
            'filename': clip['filename'],
            'download_url': (
                f'/tasks/{safe_task_id}/loops/{clip["clip_id"]}'
                f'?start_bar={start_bar}&bars={bars}'
            ),
            'media_type': clip['media_type'],
        })
    return {
        'start_bar': exports['start_bar'],
        'bars': exports['bars'],
        'bar_count': exports['bar_count'],
        'start_seconds': exports['start_seconds'],
        'end_seconds': exports['end_seconds'],
        'clips': clips,
        'zip_url': f'/tasks/{safe_task_id}/loops/zip?start_bar={start_bar}&bars={bars}',
    }


@app.get('/tasks/{task_id}/loops/zip')
def download_task_loop_zip(task_id: str, start_bar: int = 1, bars: int = 4):
    task = _load_task(task_id)
    exports = _prepare_loop_exports(task, task_id, start_bar=start_bar, bars=bars)
    zip_path = exports['zip_path']
    resolved_zip = _resolve_data_path(str(zip_path))
    if resolved_zip is None:
        raise HTTPException(status_code=404, detail='Loop ZIP not available')
    return FileResponse(path=resolved_zip, filename=resolved_zip.name, media_type='application/zip')


@app.get('/tasks/{task_id}/loops/{clip_id}')
def download_task_loop_clip(task_id: str, clip_id: str, start_bar: int = 1, bars: int = 4):
    task = _load_task(task_id)
    exports = _prepare_loop_exports(task, task_id, start_bar=start_bar, bars=bars)
    clip = next((entry for entry in exports['clips'] if entry['clip_id'] == clip_id), None)
    if clip is None:
        raise HTTPException(status_code=404, detail='Loop clip not available')
    resolved_clip = _resolve_data_path(str(clip['path']))
    if resolved_clip is None:
        raise HTTPException(status_code=404, detail='Loop clip not available')
    return FileResponse(path=resolved_clip, filename=clip['filename'], media_type=clip['media_type'])


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
    """Return MIDI transcription availability and current backend configuration."""
    backend = os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip() or 'basic_pitch'
    mt3_enabled = os.getenv('MT3_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
    service_url = os.getenv('MT3_SERVICE_URL', '').strip().rstrip('/')
    # MT3 requires a service URL; other backends (e.g. basic_pitch) run in-process
    service_required = backend == 'mt3'
    return {
        'backend': backend,
        'mt3_enabled': mt3_enabled,
        'service_configured': bool(service_url),
        'service_url': service_url or None,
        'available': mt3_enabled and (not service_required or bool(service_url)),
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
    elif backend == 'mt3' and not service_url:
        # MT3 requires a remote service; other backends (e.g. basic_pitch) run in-process
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
