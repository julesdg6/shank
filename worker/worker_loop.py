"""SHANK worker loop – polls for pending tasks and processes them."""
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from analyze import analyze_audio
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


def process_pending_tasks(tasks_dir: Path = TASKS_DIR) -> int:
    """Scan *tasks_dir* for pending tasks and process them.

    - ``upload`` tasks: run librosa analysis on the existing ``file_path``.
    - ``url`` tasks: download audio via yt-dlp, then run librosa analysis.

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

        if task_type == 'upload':
            file_path = task.get('file_path')
            if not file_path:
                log.warning('Task %s has no file_path, skipping', task_id)
                continue

            picked_up += 1
            log.info('Picked up upload task %s', task_id)
            _update_task(task_file, {
                'status': 'processing',
                'started_at': datetime.now(timezone.utc).isoformat(),
            })

            try:
                results = analyze_audio(file_path)
                _update_task(task_file, {
                    'status': 'done',
                    'bpm': results['bpm'],
                    'key': results['key'],
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                log.info('Task %s done: bpm=%s key=%s', task_id, results['bpm'], results['key'])
            except Exception as exc:  # noqa: BLE001
                log.exception('Task %s failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })

        elif task_type == 'url':
            url = task.get('source')
            if not url:
                log.warning('Task file %s is missing required fields, skipping', task_file.name)
                continue

            picked_up += 1
            log.info('Picked up task %s url=%s', task_id, url)
            _update_task(task_file, {
                'status': 'processing',
                'started_at': datetime.now(timezone.utc).isoformat(),
            })

            try:
                output_path = download_youtube(url, UPLOADS_DIR, task_id)
                results = analyze_audio(str(output_path))
                _update_task(task_file, {
                    'status': 'done',
                    'file_path': str(output_path),
                    'bpm': results['bpm'],
                    'key': results['key'],
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })
                log.info('Task %s done → %s bpm=%s key=%s', task_id, output_path, results['bpm'], results['key'])
            except Exception as exc:
                log.exception('Task %s failed: %s', task_id, exc)
                _update_task(task_file, {
                    'status': 'failed',
                    'error': str(exc),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                })

        else:
            log.warning('Task %s has unknown type %r, skipping', task_id, task_type)

    return picked_up


if __name__ == '__main__':
    log.info('Worker starting (poll interval=%ds, DATA_DIR=%s)', POLL_INTERVAL, DATA_DIR)
    while True:
        count = process_pending_tasks()
        if count:
            log.info('Processed %d task(s) this cycle', count)
        time.sleep(POLL_INTERVAL)

