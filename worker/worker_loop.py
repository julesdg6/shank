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

from analyze import analyze_audio
from downloader import download_youtube
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


def _task_artifact_paths(
    normalized_path: str,
    stem_tracks: dict[str, str] | None,
    mt3_result: dict[str, Any] | None,
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

    return artifacts


def _structured_result_paths(task_id: str) -> dict[str, str]:
    result_dir = RESULTS_DIR / task_id
    return {
        'dir': str(result_dir),
        'task_json': str(result_dir / 'task.json'),
        'analysis_json': str(result_dir / 'analysis.json'),
        'mt3_json': str(result_dir / 'mt3.json'),
        'artifacts_json': str(result_dir / 'artifacts.json'),
    }


def _write_structured_results(
    result_artifacts: dict[str, str],
    task_payload: dict[str, Any],
    normalized_path: str,
    analysis_payload: dict[str, Any],
    mt3_result: dict[str, Any] | None,
    stem_tracks: dict[str, str] | None,
) -> None:
    result_dir = Path(result_artifacts['dir'])
    result_dir.mkdir(parents=True, exist_ok=True)

    task_path = Path(result_artifacts['task_json'])
    analysis_path = Path(result_artifacts['analysis_json'])
    mt3_path = Path(result_artifacts['mt3_json'])
    artifacts_path = Path(result_artifacts['artifacts_json'])

    task_path.write_text(json.dumps(task_payload, indent=2))
    analysis_path.write_text(json.dumps(analysis_payload, indent=2))
    mt3_payload = mt3_result if isinstance(mt3_result, dict) else {}
    mt3_path.write_text(json.dumps(mt3_payload, indent=2))
    artifacts_path.write_text(json.dumps(_task_artifact_paths(normalized_path, stem_tracks, mt3_payload), indent=2))


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


def run_mt3_transcription(task_id: str, normalized_path: str, stems: dict[str, str] | None = None) -> dict:
    """Run MT3 transcription (full mix first, then optional stems)."""
    result: dict[str, Any] = {
        'enabled': MT3_ENABLED,
        'backend': TRANSCRIPTION_BACKEND,
        'status': 'disabled',
        'model': MT3_MODEL,
        'output_paths': [],
        'warnings': [],
        'errors': [],
        'full_mix': None,
        'stems': {},
    }

    if not MT3_ENABLED:
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
                _update_task(task_file, {'file_path': str(downloaded_path)})
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
        effective_backend = STEM_BACKEND

        if effective_backend != 'none':
            _record_task_progress(task_file, 65, 'Separating stems')

        if effective_backend == 'acestep':
            # Strict Ace-Step mode: fail the task if Ace-Step is unavailable or fails.
            if not ACE_STEP_API_URL:
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
                _update_task(task_file, task_updates)
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
                stem_data = separate_stems_with_audio_separator(normalized_path, task_id)
                task_updates = {'stem_backend': 'audio_separator'}
                if stem_data.get('tracks'):
                    stem_tracks = stem_data['tracks']
                    task_updates['stems'] = stem_tracks
                _update_task(task_file, task_updates)
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
                stem_data = separate_stems_with_demucs(normalized_path, task_id)
                task_updates = {'stem_backend': 'demucs'}
                if stem_data.get('tracks'):
                    stem_tracks = stem_data['tracks']
                    task_updates['stems'] = stem_tracks
                _update_task(task_file, task_updates)
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
                    stem_data = separate_stems_with_audio_separator(normalized_path, task_id)
                    task_updates = {'stem_backend': 'audio_separator'}
                    if stem_data.get('tracks'):
                        stem_tracks = stem_data['tracks']
                        task_updates['stems'] = stem_tracks
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
                    stem_data = separate_stems_with_demucs(normalized_path, task_id)
                    task_updates = {'stem_backend': 'demucs'}
                    if stem_data.get('tracks'):
                        stem_tracks = stem_data['tracks']
                        task_updates['stems'] = stem_tracks
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

        # effective_backend == 'none' (or any unrecognized value): skip stem separation.

        if MT3_ENABLED:
            _record_task_progress(task_file, 78, 'Transcribing MIDI')
        mt3_result = run_mt3_transcription(task_id, normalized_path, stems=stem_tracks)
        _update_task(task_file, {
            'mt3': mt3_result,
            'transcription': _transcription_payload_from_mt3(mt3_result),
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
            completion_updates: dict[str, Any] = {
                'status': 'done',
                'normalized_path': normalized_path,
                'analysis': analysis_payload,
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
