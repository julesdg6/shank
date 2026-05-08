"""SHANK worker loop – polls for pending tasks and processes them."""
import json
import logging
import os
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from analyze import analyze_audio
from downloader import download_youtube
from mt3_client import transcribe_with_service

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
ACE_STEP_API_KEY = os.getenv('ACE_STEP_API_KEY', '').strip()
ACE_STEP_STEMS = tuple(
    stem.strip()
    for stem in os.getenv('ACE_STEP_STEMS', 'vocals,drums,bass,other').split(',')
    if stem.strip()
)
ACE_STEP_POLL_INTERVAL = float(os.getenv('ACE_STEP_POLL_INTERVAL', '2'))
ACE_STEP_TIMEOUT = int(os.getenv('ACE_STEP_TIMEOUT', '300'))
MT3_ENABLED = os.getenv('MT3_ENABLED', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
MT3_SERVICE_URL = os.getenv('MT3_SERVICE_URL', '').strip().rstrip('/')
MT3_MODEL = os.getenv('MT3_MODEL', 'multi_instrument').strip() or 'multi_instrument'
MT3_TIMEOUT = int(os.getenv('MT3_TIMEOUT', '900'))
MT3_TRANSCRIBE_STEMS = os.getenv('MT3_TRANSCRIBE_STEMS', 'true').strip().lower() in ('1', 'true', 'yes', 'on')
MT3_FAIL_TASK_ON_ERROR = os.getenv('MT3_FAIL_TASK_ON_ERROR', 'false').strip().lower() in ('1', 'true', 'yes', 'on')

# Standard WAV output format
WAV_SAMPLE_RATE = '44100'
WAV_CHANNELS = '2'
WAV_CODEC = 'pcm_s16le'
MT3_OUTPUTS_DIR = DATA_DIR / 'mt3'
STEMS_CACHE_DIR = DATA_DIR / 'stems'


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    STEMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MT3_ENABLED:
        MT3_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _ace_step_post(path: str, payload: dict) -> dict:
    """POST JSON payload to the configured Ace-step endpoint and return parsed JSON."""
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{ACE_STEP_API_URL}{path}',
        data=body,
        headers={
            'Content-Type': 'application/json',
            **({'Authorization': f'Bearer {ACE_STEP_API_KEY}'} if ACE_STEP_API_KEY else {}),
        },
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def _ace_step_response_data(response_payload: dict[str, Any]) -> Any:
    """Unwrap Ace-step responses that use a top-level `data` envelope."""
    if isinstance(response_payload, dict) and 'data' in response_payload:
        return response_payload.get('data')
    return response_payload


def _extract_track_files(data: Any) -> dict[str, str]:
    """Collect ``track_name``/``file`` pairs from nested Ace-step result payloads."""
    tracks: dict[str, str] = {}

    def collect(node):
        if isinstance(node, dict):
            track_name = node.get('track_name') or node.get('track') or node.get('name')
            file_url = node.get('file')
            if isinstance(track_name, str) and isinstance(file_url, str):
                tracks[track_name] = file_url
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(data)
    return tracks


def _resolve_ace_step_stem_file(task_id: str, stem_name: str, stem_ref: str) -> str:
    """Return a local file path for an Ace-step stem reference (path or URL)."""
    parsed = urlparse(stem_ref)
    scheme = parsed.scheme.lower()
    if not scheme:
        local_candidate = Path(stem_ref)
        if local_candidate.exists():
            return str(local_candidate)
        if stem_ref.startswith('/'):
            stem_ref = urljoin(f'{ACE_STEP_API_URL}/', stem_ref.lstrip('/'))
            parsed = urlparse(stem_ref)
            scheme = parsed.scheme.lower()
    if scheme == 'file':
        local_candidate = Path(parsed.path)
        if local_candidate.exists():
            return str(local_candidate)
        raise RuntimeError(f'Ace-step stem for {stem_name} is not accessible: {stem_ref}')

    if scheme not in ('http', 'https'):
        raise RuntimeError(f'Ace-step stem for {stem_name} is not accessible: {stem_ref}')

    ext = Path(parsed.path).suffix or '.wav'
    cache_path = STEMS_CACHE_DIR / task_id / f'{stem_name}{ext}'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        stem_ref,
        headers=({'Authorization': f'Bearer {ACE_STEP_API_KEY}'} if ACE_STEP_API_KEY else {}),
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        cache_path.write_bytes(response.read())
    return str(cache_path)


def _prepare_ace_step_stems_for_mt3(task_id: str, tracks: dict[str, str] | None) -> dict[str, str]:
    """Return local file paths for configured Ace-step stems."""
    if not tracks:
        return {}

    normalized = {
        stem_name.strip().lower(): stem_ref
        for stem_name, stem_ref in tracks.items()
        if isinstance(stem_name, str) and isinstance(stem_ref, str)
    }
    prepared: dict[str, str] = {}
    for configured_stem in ACE_STEP_STEMS:
        stem_ref = normalized.get(configured_stem.lower())
        if not stem_ref:
            continue
        prepared[configured_stem] = _resolve_ace_step_stem_file(task_id, configured_stem, stem_ref)
    return prepared


def separate_stems_with_ace_step(src_audio_path: str) -> dict:
    """Run Ace-step extract flow and return ``{'task_id': str, 'tracks': dict[str, str]}``."""
    release_payload = {
        'task_type': 'extract',
        'src_audio_path': src_audio_path,
        'track_classes': list(ACE_STEP_STEMS),
        'audio_format': 'wav',
    }
    release_data = _ace_step_response_data(_ace_step_post('/release_task', release_payload))
    if not isinstance(release_data, dict) or not release_data.get('task_id'):
        raise RuntimeError('Ace-step did not return a task_id for stem separation')
    ace_task_id = release_data['task_id']
    deadline = time.time() + ACE_STEP_TIMEOUT

    while time.time() < deadline:
        query_data = _ace_step_response_data(_ace_step_post('/query_result', {'task_id_list': [ace_task_id]}))
        if isinstance(query_data, list) and query_data:
            task_data = query_data[0]
            status = task_data.get('status')
            status_str = str(status).lower()
            if status_str in ('1', 'succeeded', 'done'):
                result = task_data.get('result')
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        pass
                return {
                    'task_id': ace_task_id,
                    'tracks': _extract_track_files(result),
                }
            if status_str in ('2', 'failed', 'error'):
                raise RuntimeError(task_data.get('error') or 'Ace-step stem separation failed')
        time.sleep(ACE_STEP_POLL_INTERVAL)

    raise RuntimeError('Ace-step stem separation timed out')


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

            try:
                downloaded_path = download_youtube(url, UPLOADS_DIR, task_id)
                _update_task(task_file, {'file_path': str(downloaded_path)})
                log.info('Task %s downloaded → %s', task_id, downloaded_path)
            except Exception as exc:
                log.exception('Task %s download failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
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
            task = _read_task(task_file) or task

        else:
            log.warning('Task file %s has unknown type %r, skipping', task_file.name, task_type)
            continue

        # Normalize to WAV, then run BPM/key analysis.
        input_path = task.get('file_path', '')
        normalized_path = str(NORMALIZED_DIR / f'{task_id}.wav')

        try:
            normalize_audio(input_path, normalized_path)
            log.info('Task %s normalized → %s', task_id, normalized_path)
        except Exception as exc:
            log.exception('Task %s normalization failed: %s', task_id, exc)
            _update_task(task_file, {
                'status': 'failed',
                'error': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            continue

        stem_tracks: dict[str, str] | None = None
        if ACE_STEP_API_URL:
            try:
                stem_data = separate_stems_with_ace_step(normalized_path)
                task_updates = {'ace_step_task_id': stem_data['task_id']}
                if stem_data.get('tracks'):
                    stem_tracks = _prepare_ace_step_stems_for_mt3(task_id, stem_data['tracks'])
                    task_updates['stems'] = stem_tracks
                _update_task(task_file, task_updates)
            except Exception as exc:
                log.exception('Task %s stem separation failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                continue

        mt3_result = run_mt3_transcription(task_id, normalized_path, stems=stem_tracks)
        _update_task(task_file, {'mt3': mt3_result})
        if MT3_FAIL_TASK_ON_ERROR and mt3_result.get('status') in ('failed', 'partial'):
            error_msg = '; '.join(mt3_result.get('errors') or []) or 'MT3 transcription failed'
            _update_task(task_file, {
                'status': 'failed',
                'error': error_msg,
                'normalized_path': normalized_path,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            continue

        try:
            results = analyze_audio(normalized_path)
            _update_task(task_file, {
                'status': 'done',
                'normalized_path': normalized_path,
                'bpm': results['bpm'],
                'key': results['key'],
                **({'duration_seconds': results.get('duration_seconds')} if results.get('duration_seconds') is not None else {}),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            log.info('Task %s done: bpm=%s key=%s', task_id, results['bpm'], results['key'])
        except Exception as exc:
            log.exception('Task %s analysis failed: %s', task_id, exc)
            _update_task(task_file, {
                'status': 'failed',
                'error': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })

    return picked_up


if __name__ == '__main__':
    log.info('Worker starting (poll interval=%ds, DATA_DIR=%s)', POLL_INTERVAL, DATA_DIR)
    while True:
        count = process_pending_tasks()
        if count:
            log.info('Processed %d task(s) this cycle', count)
        time.sleep(POLL_INTERVAL)
