"""SHANK worker loop – polls for pending tasks and processes them."""
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from downloader import download_youtube

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

# Standard WAV output format
WAV_SAMPLE_RATE = '44100'
WAV_CHANNELS = '2'
WAV_CODEC = 'pcm_s16le'


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)


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


def process_pending_tasks(tasks_dir: Path = TASKS_DIR) -> int:
    """Scan *tasks_dir* for pending tasks and process them.

    - URL tasks: download audio via yt-dlp, then normalize to WAV.
    - Upload tasks: normalize the already-saved file to WAV.

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
            log.warning('Task file %s is missing required fields, skipping', task_file.name)
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
                # File not yet present; nothing to normalize.
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

        # Normalize to WAV (applies to both URL and upload tasks).
        input_path = task.get('file_path', '')
        normalized_path = str(NORMALIZED_DIR / f'{task_id}.wav')

        try:
            normalize_audio(input_path, normalized_path)
            _update_task(task_file, {
                'status': 'completed',
                'normalized_path': normalized_path,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            log.info('Task %s normalized → %s', task_id, normalized_path)
        except Exception as exc:
            log.exception('Task %s normalization failed: %s', task_id, exc)
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
