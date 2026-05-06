"""Worker loop: polls for pending tasks and runs librosa-based audio analysis."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from analyze import analyze_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
TASKS_DIR = DATA_DIR / 'tasks'
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))  # seconds between polls


def _load_task(task_file: Path) -> dict:
    return json.loads(task_file.read_text())


def _save_task(task: dict) -> None:
    task_file = TASKS_DIR / f"{task['task_id']}.json"
    task_file.write_text(json.dumps(task, indent=2))


def _process_task(task: dict) -> None:
    """Run analysis on a single pending task and persist the result."""
    task['status'] = 'processing'
    task['started_at'] = datetime.now(timezone.utc).isoformat()
    _save_task(task)

    file_path = task.get('file_path')
    if not file_path or not Path(file_path).exists():
        task['status'] = 'error'
        task['error'] = f"Audio file not found: {file_path}"
        task['finished_at'] = datetime.now(timezone.utc).isoformat()
        _save_task(task)
        log.error('Task %s: audio file not found: %s', task['task_id'], file_path)
        return

    try:
        results = analyze_audio(file_path)
        task['status'] = 'done'
        task['bpm'] = results['bpm']
        task['key'] = results['key']
        task['finished_at'] = datetime.now(timezone.utc).isoformat()
        _save_task(task)
        log.info('Task %s done: bpm=%s key=%s', task['task_id'], results['bpm'], results['key'])
    except Exception as exc:  # noqa: BLE001
        task['status'] = 'error'
        task['error'] = str(exc)
        task['finished_at'] = datetime.now(timezone.utc).isoformat()
        _save_task(task)
        log.exception('Task %s failed: %s', task['task_id'], exc)


def _poll_once() -> None:
    """Scan TASKS_DIR for pending tasks that have an audio file and process them."""
    if not TASKS_DIR.exists():
        return

    for task_file in sorted(TASKS_DIR.glob('*.json')):
        try:
            task = _load_task(task_file)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning('Skipping unreadable task file %s: %s', task_file, exc)
            continue

        # Only process tasks that are pending *and* already have a local audio file.
        # URL-sourced tasks without a file_path require a download step (yt-dlp)
        # that is not yet implemented and will be handled in a later phase.
        if task.get('status') == 'pending' and task.get('file_path'):
            log.info('Processing task %s', task['task_id'])
            _process_task(task)


if __name__ == '__main__':
    log.info('Worker started, polling %s every %ss', TASKS_DIR, POLL_INTERVAL)
    while True:
        _poll_once()
        time.sleep(POLL_INTERVAL)