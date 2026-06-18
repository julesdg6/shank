"""SHANK worker loop – polls for pending tasks and processes them."""
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analyze import analyze_audio, build_fingerprint
from downloader import download_youtube, extract_youtube_metadata
from metadata import collect_song_metadata
from mt3_client import transcribe_with_service
from stems import (
    _is_audio_separator_available,
    _is_demucs_available,
    _prepare_ace_step_stems_for_mt3,
    separate_stems_with_ace_step,
    separate_stems_with_audio_separator,
    separate_stems_with_demucs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mt3_config import DEFAULT_MT3_MODEL, DEFAULT_MT3_SERVICE_URL, DEFAULT_MT3_TIMEOUT, get_mt3_output_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
UPLOADS_DIR = DATA_DIR / 'uploads'
TASKS_DIR = DATA_DIR / 'tasks'
NORMALIZED_DIR = DATA_DIR / 'normalized'
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '10'))
ACE_STEP_API_URL = os.getenv('ACE_STEP_API_URL', '').strip().rstrip('/')
ACE_STEP_STEMS = tuple(
    stem.strip()
    for stem in os.getenv('ACE_STEP_STEMS', 'vocals,drums,bass,other').split(',')
    if stem.strip()
)
# Stem backend selection: 'auto' (default), 'audio_separator', 'acestep', 'demucs', or 'none'
STEM_BACKEND = os.getenv('STEM_BACKEND', 'auto').strip().lower()
MT3_ENABLED = os.getenv('MT3_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
MT3_SERVICE_URL = os.getenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL).strip().rstrip('/')
MT3_MODEL = os.getenv('MT3_MODEL', DEFAULT_MT3_MODEL).strip() or DEFAULT_MT3_MODEL
MT3_TIMEOUT = int(os.getenv('MT3_TIMEOUT', str(DEFAULT_MT3_TIMEOUT)))
MT3_TRANSCRIBE_STEMS = os.getenv('MT3_TRANSCRIBE_STEMS', 'true').strip().lower() in ('1', 'true', 'yes', 'on')
MT3_FAIL_TASK_ON_ERROR = os.getenv('MT3_FAIL_TASK_ON_ERROR', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
TRANSCRIPTION_BACKEND = os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip() or 'basic_pitch'

# Standard WAV output format
WAV_SAMPLE_RATE = '44100'
WAV_CHANNELS = '2'
WAV_CODEC = 'pcm_s16le'
MT3_OUTPUTS_DIR = get_mt3_output_path(DATA_DIR)
STEMS_CACHE_DIR = DATA_DIR / 'stems'
MAX_TASK_LOGS = 50
_MIN_BEAT_INTERVAL_SECONDS = 0.1
RESULTS_DIR = DATA_DIR / 'results'
WORKER_HEARTBEAT_FILE = DATA_DIR / '.worker_heartbeat'


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    STEMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if MT3_ENABLED:
        MT3_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_heartbeat() -> None:
    """Write the current UTC timestamp to the worker heartbeat file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        WORKER_HEARTBEAT_FILE.write_text(datetime.now(timezone.utc).isoformat())
    except OSError as exc:
        log.warning('Failed to write worker heartbeat: %s', exc)


def _read_task(task_file: Path) -> dict | None:
    """Return the parsed task dict, or *None* if the file cannot be read/parsed."""
    try:
        return json.loads(task_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_task(task_file: Path, task: dict) -> None:
    task_file.write_text(json.dumps(task, indent=2))


def _update_task(task_file: Path, updates: dict) -> None:
    task = _read_task(task_file) or {}
    task.update(updates)
    _write_task(task_file, task)


def _record_task_progress(task_file: Path, progress_percent: int, message: str | None = None) -> None:
    task = _read_task(task_file) or {}
    clamped_progress = max(0, min(100, int(progress_percent)))
    task['progress_percent'] = clamped_progress
    if message:
        logs = task.get('logs')
        if not isinstance(logs, list):
            logs = []
        last_message = logs[-1].get('message') if logs and isinstance(logs[-1], dict) else None
        if not logs or last_message != message:
            logs.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': message,
            })
        task['logs'] = logs[-MAX_TASK_LOGS:]
    _write_task(task_file, task)


def _infer_stem_mode(model_name: str | None, stem_tracks: dict[str, str] | None = None) -> str:
    if isinstance(stem_tracks, dict) and {'guitar', 'piano'}.intersection(stem_tracks):
        return '6_stem'
    if isinstance(model_name, str) and '6s' in model_name.lower():
        return '6_stem'
    return '4_stem'


def _task_stem_backend(task: dict[str, Any]) -> str:
    raw_value = task.get('stem_backend')
    if isinstance(raw_value, str) and raw_value.strip():
        normalized = raw_value.strip().lower().replace('-', '_')
    else:
        normalized = STEM_BACKEND
    return 'disabled' if normalized in ('none', 'disabled') else normalized


def _task_stem_model(task: dict[str, Any]) -> str:
    raw_value = task.get('stem_model')
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return os.getenv('AUDIO_SEPARATOR_MODEL', 'htdemucs_ft.yaml').strip() or 'htdemucs_ft.yaml'


def _task_stem_device(task: dict[str, Any], backend: str) -> str:
    raw_value = task.get('stem_device')
    if isinstance(raw_value, str) and raw_value.strip() and raw_value.strip().lower() != 'auto':
        return raw_value.strip().lower()
    if backend == 'demucs':
        return os.getenv('DEMUCS_DEVICE', 'cpu').strip().lower() or 'cpu'
    return os.getenv('AUDIO_SEPARATOR_DEVICE', 'cpu').strip().lower() or 'cpu'


def _task_stem_mode(task: dict[str, Any], stem_model: str, stem_tracks: dict[str, str] | None = None) -> str:
    raw_value = task.get('stem_mode')
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip().lower().replace('-', '_')
    return _infer_stem_mode(stem_model, stem_tracks)


def _resolved_analysis_config(
    *,
    task: dict[str, Any],
    effective_backend: str,
    stem_model: str,
    stem_device: str,
    stem_tracks: dict[str, str] | None,
    midi_enabled: bool,
    midi_reason: str | None = None,
    stem_reason: str | None = None,
) -> dict[str, Any]:
    backend_value = 'disabled' if effective_backend in ('none', 'disabled') else effective_backend
    config = {
        'midi': {
            'enabled': bool(midi_enabled),
            'backend': TRANSCRIPTION_BACKEND,
        },
        'stems': {
            'enabled': backend_value != 'disabled',
            'backend': backend_value,
            'model': stem_model if backend_value == 'audio_separator' else None,
            'device': stem_device,
            'mode': _task_stem_mode(task, stem_model, stem_tracks),
        },
        'reprocess': (
            task.get('analysis_config', {}).get('reprocess')
            if isinstance(task.get('analysis_config'), dict)
            and isinstance(task['analysis_config'].get('reprocess'), dict)
            else {
                'replace_existing': True,
                'archive_previous': False,
                'use_current_settings': True,
            }
        ),
    }
    warnings = []
    existing_warnings = task.get('analysis_config', {}).get('warnings') if isinstance(task.get('analysis_config'), dict) else None
    if isinstance(existing_warnings, list):
        warnings.extend(str(item) for item in existing_warnings if isinstance(item, str))
    if stem_reason:
        warnings.append(stem_reason)
    if midi_reason:
        warnings.append(midi_reason)
    if warnings:
        config['warnings'] = warnings
    return config


def _task_artifact_paths(
    normalized_path: str,
    stem_tracks: dict[str, str] | None,
    mt3_result: dict[str, Any] | None,
    result_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {'normalized_wav': normalized_path}

    if stem_tracks:
        artifacts['stems_wav'] = {stem_name: stem_path for stem_name, stem_path in stem_tracks.items()}

    if mt3_result is not None:
        full_mix = mt3_result.get('full_mix')
        if isinstance(full_mix, dict):
            full_mix_artifacts = {
                key: full_mix[key]
                for key in ('midi_path', 'notes_path')
                if isinstance(full_mix.get(key), str) and full_mix.get(key)
            }
            if full_mix_artifacts:
                artifacts['mt3_full_mix'] = full_mix_artifacts

        mt3_stems = mt3_result.get('stems')
        if isinstance(mt3_stems, dict):
            stem_artifacts: dict[str, dict[str, str]] = {}
            for stem_name, stem_data in mt3_stems.items():
                if not isinstance(stem_name, str) or not isinstance(stem_data, dict):
                    continue
                values = {
                    key: stem_data[key]
                    for key in ('midi_path', 'notes_path')
                    if isinstance(stem_data.get(key), str) and stem_data.get(key)
                }
                if values:
                    stem_artifacts[stem_name] = values
            if stem_artifacts:
                artifacts['mt3_stems'] = stem_artifacts

    if isinstance(result_artifacts, dict):
        for artifact_name, key in (
            ('beatgrid_json', 'beatgrid_json'),
            ('structure_json', 'structure_json'),
            ('fingerprint_json', 'fingerprint_json'),
            ('waveform_beats_png', 'waveform_beats_png'),
            ('tempo_curve_png', 'tempo_curve_png'),
            ('beatgraph_png', 'beatgraph_png'),
            ('lyrics_json', 'lyrics_json'),
            ('credits_json', 'credits_json'),
            ('song_metadata_json', 'song_metadata_json'),
        ):
            value = result_artifacts.get(key)
            if isinstance(value, str) and value:
                artifacts[artifact_name] = value

    return artifacts


def _structured_result_paths(task_id: str) -> dict[str, str]:
    result_dir = RESULTS_DIR / task_id
    return {
        'dir': str(result_dir),
        'task_json': str(result_dir / 'task.json'),
        'analysis_json': str(result_dir / 'analysis.json'),
        'beatgrid_json': str(result_dir / 'beatgrid.json'),
        'structure_json': str(result_dir / 'structure.json'),
        'fingerprint_json': str(result_dir / 'fingerprint.json'),
        'waveform_beats_png': str(result_dir / 'waveform_beats.png'),
        'tempo_curve_png': str(result_dir / 'tempo_curve.png'),
        'beatgraph_png': str(result_dir / 'beatgraph.png'),
        'mt3_json': str(result_dir / 'mt3.json'),
        'lyrics_json': str(result_dir / 'lyrics.json'),
        'credits_json': str(result_dir / 'credits.json'),
        'song_metadata_json': str(result_dir / 'song_metadata.json'),
        'artifacts_json': str(result_dir / 'artifacts.json'),
    }


def _write_beat_outputs(result_artifacts: dict[str, str], analysis_payload: dict[str, Any]) -> None:
    full_mix = analysis_payload.get('full_mix')
    if not isinstance(full_mix, dict):
        return

    beatgrid = full_mix.get('beatgrid')
    if not isinstance(beatgrid, dict):
        beats = full_mix.get('beats')
        bpm = full_mix.get('bpm')
        if isinstance(beats, list) and isinstance(bpm, (int, float)):
            first_beat = beats[0] if beats and isinstance(beats[0], (int, float)) else 0.0
            beatgrid = {
                'bpm': float(bpm),
                'first_beat_seconds': float(first_beat),
                'beats': [
                    {'index': idx + 1, 'time': float(ts)}
                    for idx, ts in enumerate(beats)
                    if isinstance(ts, (int, float))
                ],
            }
        else:
            beatgrid = {'bpm': 0.0, 'first_beat_seconds': 0.0, 'beats': []}

    Path(result_artifacts['beatgrid_json']).write_text(json.dumps(beatgrid, indent=2))

    beat_rows = beatgrid.get('beats')
    waveform = full_mix.get('waveform')
    duration = full_mix.get('duration_seconds')
    if not isinstance(beat_rows, list) or not beat_rows:
        return
    if not isinstance(waveform, list) or not waveform:
        return
    if not isinstance(duration, (int, float)) or duration <= 0:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning('Could not generate beat graphs; matplotlib unavailable: %s', exc)
        return

    beat_times: list[float] = []
    for entry in beat_rows:
        if not isinstance(entry, dict):
            continue
        time_raw = entry.get('time')
        if isinstance(time_raw, (int, float)):
            beat_times.append(float(time_raw))
    if not beat_times:
        return

    wave_values = [float(value) for value in waveform if isinstance(value, (int, float))]
    if not wave_values:
        return
    waveform_times = np.linspace(0.0, float(duration), len(wave_values))
    plt.figure(figsize=(12, 3))
    plt.plot(waveform_times, wave_values, color='#4ea1ff', linewidth=1)
    for beat_time in beat_times:
        plt.axvline(beat_time, color='#ff4d4f', linewidth=0.7, alpha=0.4)
    plt.title('Waveform with Beat Markers')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(result_artifacts['waveform_beats_png'], dpi=120)
    plt.close()

    local_bpms: list[float] = []
    for entry in beat_rows:
        if not isinstance(entry, dict):
            continue
        local_bpm_raw = entry.get('local_bpm')
        if isinstance(local_bpm_raw, (int, float)):
            local_bpms.append(float(local_bpm_raw))
    tempo_times: list[float] = []
    tempo_values: list[float] = []
    if local_bpms and len(local_bpms) == len(beat_times):
        tempo_times = beat_times
        tempo_values = local_bpms
    elif len(beat_times) > 1:
        intervals = np.diff(np.array(beat_times))
        tempo_values = []
        for interval in intervals:
            interval_value = float(interval)
            if interval_value > _MIN_BEAT_INTERVAL_SECONDS and interval_value > 0:
                tempo_values.append(round(float(60.0 / interval_value), 2))
        tempo_times = beat_times[1:1 + len(tempo_values)]
    if tempo_values and tempo_times:
        plt.figure(figsize=(12, 3))
        plt.plot(tempo_times, tempo_values, color='#22c55e', linewidth=1.2)
        plt.title('Tempo Curve')
        plt.xlabel('Time (s)')
        plt.ylabel('BPM')
        plt.tight_layout()
        plt.savefig(result_artifacts['tempo_curve_png'], dpi=120)
        plt.close()

    if len(beat_times) > 1:
        intervals = np.diff(np.array(beat_times))
        interval_values = [float(interval) for interval in intervals if interval > _MIN_BEAT_INTERVAL_SECONDS]
        if interval_values:
            plt.figure(figsize=(12, 3))
            plt.plot(range(1, 1 + len(interval_values)), interval_values, color='#f97316', linewidth=1.2)
            plt.title('Beat Interval Graph')
            plt.xlabel('Beat Index')
            plt.ylabel('Interval (s)')
            plt.tight_layout()
            plt.savefig(result_artifacts['beatgraph_png'], dpi=120)
            plt.close()


def _write_structure_output(result_artifacts: dict[str, str], analysis_payload: dict[str, Any]) -> None:
    full_mix = analysis_payload.get('full_mix')
    structure: list[dict[str, Any]] = []
    if isinstance(full_mix, dict):
        structure_raw = full_mix.get('structure')
        if isinstance(structure_raw, list):
            structure = [entry for entry in structure_raw if isinstance(entry, dict)]
    Path(result_artifacts['structure_json']).write_text(json.dumps(structure, indent=2))


def _write_fingerprint_output(result_artifacts: dict[str, str], analysis_payload: dict[str, Any]) -> None:
    full_mix = analysis_payload.get('full_mix')
    fingerprint: dict[str, Any] = {}
    if isinstance(full_mix, dict):
        fingerprint = build_fingerprint(full_mix)
    Path(result_artifacts['fingerprint_json']).write_text(json.dumps(fingerprint, indent=2))


def _write_structured_results(
    result_artifacts: dict[str, str],
    task_payload: dict[str, Any],
    normalized_path: str,
    analysis_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    mt3_result: dict[str, Any] | None,
    stem_tracks: dict[str, str] | None,
) -> None:
    result_dir = Path(result_artifacts['dir'])
    result_dir.mkdir(parents=True, exist_ok=True)

    task_path = Path(result_artifacts['task_json'])
    analysis_path = Path(result_artifacts['analysis_json'])
    mt3_path = Path(result_artifacts['mt3_json'])
    lyrics_path = Path(result_artifacts['lyrics_json'])
    credits_path = Path(result_artifacts['credits_json'])
    song_metadata_path = Path(result_artifacts['song_metadata_json'])
    artifacts_path = Path(result_artifacts['artifacts_json'])

    task_path.write_text(json.dumps(task_payload, indent=2))
    analysis_path.write_text(json.dumps(analysis_payload, indent=2))
    _write_beat_outputs(result_artifacts, analysis_payload)
    _write_structure_output(result_artifacts, analysis_payload)
    _write_fingerprint_output(result_artifacts, analysis_payload)
    mt3_payload = mt3_result if isinstance(mt3_result, dict) else {}
    lyrics_payload = metadata_payload.get('lyrics') if isinstance(metadata_payload.get('lyrics'), dict) else {}
    credits_payload = metadata_payload.get('credits') if isinstance(metadata_payload.get('credits'), dict) else {}
    mt3_path.write_text(json.dumps(mt3_payload, indent=2))
    lyrics_path.write_text(json.dumps(lyrics_payload, indent=2))
    credits_path.write_text(json.dumps(credits_payload, indent=2))
    song_metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    artifacts_path.write_text(
        json.dumps(_task_artifact_paths(normalized_path, stem_tracks, mt3_payload, result_artifacts), indent=2),
    )


def normalize_audio(input_path: str, output_path: str) -> None:
    """Normalize an audio file to a standard WAV format using ffmpeg.

    Output: 44100 Hz, stereo, 16-bit PCM WAV.
    Raises RuntimeError if ffmpeg exits with a non-zero status.
    """
    cmd = [
        'ffmpeg',
        '-y',              # overwrite output file without prompting
        '-i', input_path,
        '-ar', WAV_SAMPLE_RATE,
        '-ac', WAV_CHANNELS,
        '-c:a', WAV_CODEC,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed (exit {result.returncode}): {result.stderr}')



def transcribe_with_mt3(normalized_path: str, task_id: str, source_name: str = 'full_mix') -> dict[str, Any]:
    """Transcribe a single audio source with MT3 and return the transcription result."""
    output_dir = MT3_OUTPUTS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return transcribe_with_service(
        service_url=MT3_SERVICE_URL,
        audio_path=normalized_path,
        output_dir=output_dir,
        task_id=task_id,
        model=MT3_MODEL,
        source=source_name,
        timeout=MT3_TIMEOUT,
    )


def run_mt3_transcription(
    task_id: str,
    normalized_path: str,
    stems: dict[str, str] | None = None,
    *,
    enabled: bool | None = None,
) -> dict:
    """Run MT3 transcription (full mix first, then optional stems)."""
    mt3_enabled = MT3_ENABLED if enabled is None else bool(enabled)
    result: dict[str, Any] = {
        'enabled': mt3_enabled,
        'backend': TRANSCRIPTION_BACKEND,
        'status': 'disabled',
        'model': MT3_MODEL,
        'output_paths': [],
        'warnings': [],
        'errors': [],
        'full_mix': None,
        'stems': {},
    }

    if not mt3_enabled:
        return result
    if not MT3_SERVICE_URL:
        result['status'] = 'failed'
        result['errors'].append('MT3 is enabled but MT3_SERVICE_URL is not configured')
        result['error'] = result['errors'][0]
        return result

    def transcribe_one(source: str, audio_path: str) -> dict[str, Any]:
        transcription = transcribe_with_mt3(audio_path, task_id, source_name=source)
        backend_name = transcription.get('backend')
        if isinstance(backend_name, str) and backend_name:
            result['backend'] = backend_name
        if isinstance(transcription.get('midi_path'), str):
            result['output_paths'].append(transcription['midi_path'])
        if isinstance(transcription.get('warnings'), list):
            result['warnings'].extend(str(w) for w in transcription['warnings'])
        return transcription

    try:
        result['full_mix'] = transcribe_one('full_mix', normalized_path)
    except Exception as exc:
        log.exception('Task %s MT3 full-mix transcription failed: %s', task_id, exc)
        result['errors'].append(f'full_mix: {exc}')

    if stems:
        if MT3_TRANSCRIBE_STEMS:
            for configured_stem in ACE_STEP_STEMS:
                stem_path = stems.get(configured_stem)
                if not stem_path:
                    continue
                try:
                    result['stems'][configured_stem] = transcribe_one(configured_stem, stem_path)
                except Exception as exc:
                    log.exception('Task %s MT3 stem transcription failed for %s: %s', task_id, configured_stem, exc)
                    result['errors'].append(f'{configured_stem}: {exc}')
        else:
            result['warnings'].append('Stem transcription skipped because MT3_TRANSCRIBE_STEMS is disabled')

    if result['errors'] and result['output_paths']:
        result['status'] = 'partial'
    elif result['errors']:
        result['status'] = 'failed'
    else:
        result['status'] = 'completed'

    if result['errors']:
        result['error'] = '; '.join(str(e) for e in result['errors'])

    return result


def _mt3_disabled_result(reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        'enabled': False,
        'backend': TRANSCRIPTION_BACKEND,
        'status': 'disabled',
        'model': MT3_MODEL,
        'output_paths': [],
        'warnings': [],
        'errors': [],
        'full_mix': None,
        'stems': {},
    }
    if reason:
        result['warnings'].append(reason)
    return result


def _task_requests_mt3(task: dict[str, Any]) -> bool:
    explicit_toggle = task.get('enable_mt3')
    if isinstance(explicit_toggle, bool):
        return explicit_toggle
    return True


def _transcription_payload_from_mt3(mt3_result: dict[str, Any]) -> dict[str, Any]:
    full_mix = mt3_result.get('full_mix')
    notes: list[Any] = []
    midi_file = None
    if isinstance(full_mix, dict):
        midi_candidate = full_mix.get('midi_path')
        if isinstance(midi_candidate, str) and midi_candidate:
            midi_file = midi_candidate
        notes_candidate = full_mix.get('notes')
        if isinstance(notes_candidate, list):
            notes = notes_candidate

    return {
        'enabled': bool(mt3_result.get('enabled')),
        'backend': mt3_result.get('backend') or TRANSCRIPTION_BACKEND,
        'status': mt3_result.get('status'),
        'midi_file': midi_file,
        'notes': notes,
    }


def process_pending_tasks(tasks_dir: Path = TASKS_DIR) -> int:
    """Scan *tasks_dir* for pending tasks and process them.

    - ``url`` tasks: download audio via yt-dlp, normalize to WAV, then analyze BPM/key.
    - ``upload`` tasks with ``file_path``: normalize to WAV, then analyze BPM/key.
    - ``upload`` tasks without ``file_path``: skip (file not yet saved).

    Returns the number of tasks that were picked up.
    """
    if not tasks_dir.exists():
        return 0

    _ensure_dirs()
    picked_up = 0

    for task_file in sorted(tasks_dir.glob('*.json')):
        task = _read_task(task_file)
        if task is None:
            continue
        if task.get('status') != 'pending':
            continue

        task_type = task.get('type')
        raw_task_id = task.get('task_id')

        if not raw_task_id:
            log.warning('Task file %s is missing task_id, skipping', task_file.name)
            continue

        try:
            task_id = str(uuid.UUID(raw_task_id))
        except ValueError:
            log.warning('Task file %s has invalid task_id %r, skipping', task_file.name, raw_task_id)
            continue

        if task_type == 'url':
            url = task.get('source')
            if not url:
                log.warning('Task file %s is missing required fields, skipping', task_file.name)
                continue

            log.info('Picked up URL task %s url=%s', task_id, url)
            picked_up += 1

            _update_task(task_file, {
                'status': 'processing',
                'started_at': datetime.now(timezone.utc).isoformat(),
            })
            _record_task_progress(task_file, 5, 'Task started')
            _record_task_progress(task_file, 12, 'Downloading source audio')

            try:
                downloaded_path = download_youtube(url, UPLOADS_DIR, task_id)
                task_updates: dict[str, Any] = {'file_path': str(downloaded_path)}
                # Capture YouTube metadata and store it alongside the task.
                try:
                    yt_meta = extract_youtube_metadata(url)
                    task_updates['youtube'] = yt_meta
                    task_updates['source_type'] = 'youtube'
                except Exception as meta_exc:
                    log.warning('Task %s: could not fetch YouTube metadata: %s', task_id, meta_exc)
                _update_task(task_file, task_updates)
                _record_task_progress(task_file, 25, 'Download complete')
                log.info('Task %s downloaded → %s', task_id, downloaded_path)
            except Exception as exc:
                log.exception('Task %s download failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                _record_task_progress(task_file, 100, f'Download failed: {str(exc)}')
                continue

            # Re-read so we have the latest file_path written above.
            task = _read_task(task_file) or task

        elif task_type == 'upload':
            if not task.get('file_path'):
                # File not yet present; nothing to process.
                continue

            log.info('Picked up upload task %s', task_id)
            picked_up += 1

            _update_task(task_file, {
                'status': 'processing',
                'started_at': datetime.now(timezone.utc).isoformat(),
            })
            _record_task_progress(task_file, 5, 'Task started')
            task = _read_task(task_file) or task

        else:
            log.warning('Task file %s has unknown type %r, skipping', task_file.name, task_type)
            continue

        # Normalize to WAV, then run BPM/key analysis.
        input_path = task.get('file_path', '')
        normalized_path = str(NORMALIZED_DIR / f'{task_id}.wav')

        try:
            _record_task_progress(task_file, 40, 'Normalizing audio')
            normalize_audio(input_path, normalized_path)
            _record_task_progress(task_file, 55, 'Audio normalized')
            log.info('Task %s normalized → %s', task_id, normalized_path)
        except Exception as exc:
            log.exception('Task %s normalization failed: %s', task_id, exc)
            _update_task(task_file, {
                'status': 'failed',
                'error': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            _record_task_progress(task_file, 100, f'Normalization failed: {str(exc)}')
            continue

        stem_tracks: dict[str, str] | None = None
        requested_backend = _task_stem_backend(task)
        stem_model = _task_stem_model(task)
        stem_device = _task_stem_device(task, requested_backend)
        effective_backend = requested_backend
        stem_reason: str | None = None

        if effective_backend not in ('none', 'disabled'):
            _record_task_progress(task_file, 65, 'Separating stems')

        if effective_backend == 'acestep':
            # Strict Ace-Step mode: fail the task if Ace-Step is unavailable or fails.
            if not ACE_STEP_API_URL:
                stem_reason = 'Stem separation skipped: Ace-Step is not configured.'
                _update_task(task_file, {
                    'status': 'failed',
                    'error': 'STEM_BACKEND=acestep but ACE_STEP_API_URL is not configured',
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                _record_task_progress(task_file, 100, 'Stem separation failed: Ace-Step URL is not configured')
                continue
            try:
                stem_data = separate_stems_with_ace_step(normalized_path)
                task_updates: dict[str, Any] = {
                    'ace_step_task_id': stem_data['task_id'],
                    'stem_backend': 'acestep',
                }
                if stem_data.get('tracks'):
                    stem_tracks = _prepare_ace_step_stems_for_mt3(task_id, stem_data['tracks'])
                    task_updates['stems'] = stem_tracks
                task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                _update_task(task_file, task_updates)
                _record_task_progress(task_file, 70, 'Stem separation enabled via Ace-Step')
            except Exception as exc:
                log.exception('Task %s Ace-Step stem separation failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                _record_task_progress(task_file, 100, f'Stem separation failed: {str(exc)}')
                continue

        elif effective_backend == 'audio_separator':
            # Strict audio-separator mode: fail the task if unavailable or separation fails.
            try:
                stem_data = separate_stems_with_audio_separator(
                    normalized_path,
                    task_id,
                    model_name=stem_model,
                    device=stem_device,
                )
                task_updates = {
                    'stem_backend': 'audio_separator',
                    'stem_model': stem_model,
                    'stem_device': stem_device,
                }
                if stem_data.get('tracks'):
                    stem_tracks = stem_data['tracks']
                    task_updates['stems'] = stem_tracks
                task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                _update_task(task_file, task_updates)
                _record_task_progress(
                    task_file,
                    70,
                    f'Stem separation enabled via Audio Separator ({stem_model}, {stem_device}, {task_updates["stem_mode"]})',
                )
            except Exception as exc:
                log.exception('Task %s audio-separator stem separation failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                _record_task_progress(task_file, 100, f'Stem separation failed: {str(exc)}')
                continue

        elif effective_backend == 'demucs':
            # Strict Demucs mode: fail the task if Demucs is unavailable or fails.
            try:
                stem_data = separate_stems_with_demucs(normalized_path, task_id, device=stem_device)
                task_updates = {
                    'stem_backend': 'demucs',
                    'stem_device': stem_device,
                }
                if stem_data.get('tracks'):
                    stem_tracks = stem_data['tracks']
                    task_updates['stems'] = stem_tracks
                task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                _update_task(task_file, task_updates)
                _record_task_progress(task_file, 70, f'Stem separation enabled via Demucs ({stem_device})')
            except Exception as exc:
                log.exception('Task %s Demucs stem separation failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                _record_task_progress(task_file, 100, f'Stem separation failed: {str(exc)}')
                continue

        elif effective_backend == 'auto':
            # Auto mode: try Ace-Step first (if configured), then audio-separator, then Demucs.
            # Neither failure is fatal – the task continues without stems.
            ace_step_attempted = False
            if ACE_STEP_API_URL:
                ace_step_attempted = True
                try:
                    stem_data = separate_stems_with_ace_step(normalized_path)
                    task_updates = {
                        'ace_step_task_id': stem_data['task_id'],
                        'stem_backend': 'acestep',
                    }
                    if stem_data.get('tracks'):
                        stem_tracks = _prepare_ace_step_stems_for_mt3(task_id, stem_data['tracks'])
                        task_updates['stems'] = stem_tracks
                    task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                    _update_task(task_file, task_updates)
                except Exception as exc:
                    log.warning(
                        'Task %s Ace-Step stem separation failed, trying audio-separator fallback: %s',
                        task_id, exc,
                    )

            audio_separator_attempted = False
            if stem_tracks is None and _is_audio_separator_available():
                audio_separator_attempted = True
                try:
                    stem_data = separate_stems_with_audio_separator(
                        normalized_path,
                        task_id,
                        model_name=stem_model,
                        device=stem_device,
                    )
                    task_updates = {
                        'stem_backend': 'audio_separator',
                        'stem_model': stem_model,
                        'stem_device': stem_device,
                    }
                    if stem_data.get('tracks'):
                        stem_tracks = stem_data['tracks']
                        task_updates['stems'] = stem_tracks
                    task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                    _update_task(task_file, task_updates)
                    if ace_step_attempted:
                        log.info(
                            'Task %s: audio-separator fallback succeeded after Ace-Step failure',
                            task_id,
                        )
                except Exception as exc:
                    log.warning(
                        'Task %s audio-separator fallback failed, trying Demucs: %s',
                        task_id, exc,
                    )

            if stem_tracks is None and _is_demucs_available():
                try:
                    stem_data = separate_stems_with_demucs(normalized_path, task_id, device=stem_device)
                    task_updates = {
                        'stem_backend': 'demucs',
                        'stem_device': stem_device,
                    }
                    if stem_data.get('tracks'):
                        stem_tracks = stem_data['tracks']
                        task_updates['stems'] = stem_tracks
                    task_updates['stem_mode'] = _task_stem_mode(task, stem_model, stem_tracks)
                    _update_task(task_file, task_updates)
                    prior = []
                    if ace_step_attempted:
                        prior.append('Ace-Step')
                    if audio_separator_attempted:
                        prior.append('audio-separator')
                    if prior:
                        log.info(
                            'Task %s: Demucs fallback succeeded after %s failure',
                            task_id, ' and '.join(prior),
                        )
                    else:
                        log.info('Task %s: Demucs separation succeeded', task_id)
                except Exception as exc:
                    log.warning(
                        'Task %s Demucs fallback also failed, continuing without stems: %s',
                        task_id, exc,
                    )

            if stem_tracks is None and (ace_step_attempted or audio_separator_attempted):
                log.warning(
                    'Task %s: no stem backends available or all attempted backends failed;'
                    ' continuing without stems',
                    task_id,
                )
                stem_reason = 'Stem separation skipped: no available backend succeeded.'

            if stem_tracks:
                current_task = _read_task(task_file) or {}
                effective_backend = str(current_task.get('stem_backend') or effective_backend)
                _record_task_progress(task_file, 70, f'Stem separation enabled via {effective_backend}')
            else:
                effective_backend = 'disabled'

        else:
            effective_backend = 'disabled'
            stem_reason = 'Stem separation skipped: disabled by analysis settings.'
            _record_task_progress(task_file, 65, stem_reason)

        task_mt3_override = task.get('enable_mt3')
        mt3_enabled_for_task = task_mt3_override if isinstance(task_mt3_override, bool) else MT3_ENABLED
        midi_reason: str | None = None
        if mt3_enabled_for_task:
            _record_task_progress(task_file, 78, 'Transcribing MIDI')
        else:
            midi_reason = 'MIDI transcription skipped: disabled by analysis settings.'
            _record_task_progress(task_file, 78, midi_reason)
        _update_task(task_file, {
            'analysis_config': _resolved_analysis_config(
                task=task,
                effective_backend=effective_backend,
                stem_model=stem_model,
                stem_device=stem_device,
                stem_tracks=stem_tracks,
                midi_enabled=mt3_enabled_for_task,
                midi_reason=midi_reason,
                stem_reason=stem_reason,
            ),
        })
        mt3_result = run_mt3_transcription(
            task_id,
            normalized_path,
            stems=stem_tracks,
            enabled=mt3_enabled_for_task,
        )
        _update_task(task_file, {
            'mt3': mt3_result,
            'transcription': _transcription_payload_from_mt3(mt3_result),
            'analysis_config': _resolved_analysis_config(
                task=_read_task(task_file) or task,
                effective_backend=effective_backend,
                stem_model=stem_model,
                stem_device=stem_device,
                stem_tracks=stem_tracks,
                midi_enabled=bool(mt3_result.get('enabled')),
                midi_reason=midi_reason,
                stem_reason=stem_reason,
            ),
        })
        if MT3_FAIL_TASK_ON_ERROR and mt3_result.get('status') in ('failed', 'partial'):
            error_msg = '; '.join(mt3_result.get('errors') or []) or 'MT3 transcription failed'
            _update_task(task_file, {
                'status': 'failed',
                'error': error_msg,
                'normalized_path': normalized_path,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            _record_task_progress(task_file, 100, f'MIDI transcription failed: {error_msg}')
            continue

        try:
            _record_task_progress(task_file, 90, 'Running audio analysis')
            full_mix_analysis = analyze_audio(normalized_path)
            stem_analysis: dict[str, Any] = {}
            analysis_warnings: list[str] = []
            if stem_tracks:
                for stem_name, stem_path in stem_tracks.items():
                    try:
                        stem_analysis[stem_name] = analyze_audio(stem_path)
                    except Exception as exc:
                        log.exception('Task %s stem analysis failed for %s', task_id, stem_name)
                        analysis_warnings.append(f'Failed to analyze stem {stem_name}: {str(exc)}')

            analysis_payload: dict[str, Any] = {
                'full_mix': full_mix_analysis,
                'stems': stem_analysis,
            }
            if analysis_warnings:
                analysis_payload['warnings'] = analysis_warnings

            current_task = _read_task(task_file) or {}
            metadata_payload = collect_song_metadata(current_task, input_path)
            completion_updates: dict[str, Any] = {
                'status': 'done',
                'normalized_path': normalized_path,
                'analysis': analysis_payload,
                'lyrics': metadata_payload.get('lyrics'),
                'credits': metadata_payload.get('credits'),
                'song_metadata': metadata_payload,
                'bpm': full_mix_analysis['bpm'],
                'key': full_mix_analysis['key'],
                **({'duration_seconds': full_mix_analysis.get('duration_seconds')} if full_mix_analysis.get('duration_seconds') is not None else {}),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            }
            result_artifacts = _structured_result_paths(task_id)
            completion_updates['results'] = result_artifacts
            completed_task_payload = {**current_task, **completion_updates}
            _write_structured_results(
                result_artifacts=result_artifacts,
                task_payload=completed_task_payload,
                normalized_path=normalized_path,
                analysis_payload=analysis_payload,
                metadata_payload=metadata_payload,
                mt3_result=mt3_result,
                stem_tracks=stem_tracks,
            )
            _update_task(task_file, completion_updates)
            _record_task_progress(task_file, 100, 'Task completed')
            log.info('Task %s done: bpm=%s key=%s', task_id, full_mix_analysis['bpm'], full_mix_analysis['key'])
        except Exception as exc:
            log.exception('Task %s analysis failed: %s', task_id, exc)
            _update_task(task_file, {
                'status': 'failed',
                'error': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            _record_task_progress(task_file, 100, f'Analysis failed: {str(exc)}')

    return picked_up


def run_worker(
    tasks_dir: Path = TASKS_DIR,
    poll_interval: float = POLL_INTERVAL,
    *,
    sleep_fn=time.sleep,
    max_cycles: int | None = None,
) -> None:
    """Continuously poll *tasks_dir* for work and process pending tasks."""
    log.info('Worker starting (poll interval=%ss, DATA_DIR=%s)', poll_interval, DATA_DIR)

    cycles_run = 0
    while max_cycles is None or cycles_run < max_cycles:
        _write_heartbeat()
        count = process_pending_tasks(tasks_dir)
        if count:
            log.info('Processed %d task(s) this cycle', count)

        cycles_run += 1
        if max_cycles is None or cycles_run < max_cycles:
            sleep_fn(poll_interval)


if __name__ == '__main__':
    run_worker()
