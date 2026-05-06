"""SHANK worker loop – polls for pending tasks and processes them."""
import json
import logging
import os
import time
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
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '10'))


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


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


def process_pending_tasks() -> None:
    """Scan TASKS_DIR for pending URL tasks and download their audio."""
    _ensure_dirs()
    for task_file in sorted(TASKS_DIR.glob('*.json')):
        task = _read_task(task_file)
        if task is None:
            continue
        if task.get('status') != 'pending' or task.get('type') != 'url':
            continue

        task_id = task['task_id']
        url = task['source']
        log.info('Picked up task %s url=%s', task_id, url)

        _update_task(task_file, {
            'status': 'processing',
            'started_at': datetime.now(timezone.utc).isoformat(),
        })

        try:
            output_path = download_youtube(url, UPLOADS_DIR, task_id)
            _update_task(task_file, {
                'status': 'done',
                'file_path': str(output_path),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })
            log.info('Task %s done → %s', task_id, output_path)
        except Exception as exc:
            log.error('Task %s failed: %s', task_id, exc)
            _update_task(task_file, {
                'status': 'failed',
                'error': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            })


if __name__ == '__main__':
    log.info('Worker starting (poll interval=%ds, DATA_DIR=%s)', POLL_INTERVAL, DATA_DIR)
    while True:
        process_pending_tasks()
        time.sleep(POLL_INTERVAL)
