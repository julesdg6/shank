"""Worker loop – polls the tasks directory for pending JSON task files."""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
TASKS_DIR = DATA_DIR / 'tasks'
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))  # seconds


def _load_task(task_file: Path) -> dict | None:
    """Read and parse a task JSON file.  Returns None if the file is unreadable."""
    try:
        return json.loads(task_file.read_text())
    except Exception as exc:
        log.warning('Could not read task file %s: %s', task_file, exc)
        return None


def _save_task(task_file: Path, task: dict) -> None:
    """Write the task dict back to its JSON file."""
    task_file.write_text(json.dumps(task, indent=2))


def _process_task(task: dict) -> None:
    """Execute the work for a single task.

    Raises an exception on failure so the caller can mark the task as failed.
    """
    task_type = task.get('type')
    task_id = task.get('task_id')

    if task_type == 'upload':
        log.info('Processing upload task %s (source: %s)', task_id, task.get('source'))
        # TODO: run librosa analysis on task['file_path']

    elif task_type == 'url':
        log.info('Processing URL task %s (source: %s)', task_id, task.get('source'))
        # TODO: download with yt-dlp, then run librosa analysis

    else:
        raise ValueError(f'Unknown task type: {task_type!r}')


def process_pending_tasks(tasks_dir: Path) -> int:
    """Scan *tasks_dir* and process every pending task.

    Returns the number of tasks that were picked up.
    """
    if not tasks_dir.exists():
        return 0

    picked_up = 0
    for task_file in sorted(tasks_dir.glob('*.json')):
        task = _load_task(task_file)
        if task is None or task.get('status') != 'pending':
            continue

        picked_up += 1
        task_id = task.get('task_id', task_file.stem)
        log.info('Picked up task %s', task_id)

        # Mark as processing immediately so another worker won't pick it up.
        task['status'] = 'processing'
        task['started_at'] = datetime.now(timezone.utc).isoformat()
        _save_task(task_file, task)

        try:
            _process_task(task)
            task['status'] = 'done'
            task['finished_at'] = datetime.now(timezone.utc).isoformat()
            log.info('Task %s completed successfully', task_id)
        except Exception as exc:
            task['status'] = 'failed'
            task['error'] = str(exc)
            task['finished_at'] = datetime.now(timezone.utc).isoformat()
            log.error('Task %s failed: %s', task_id, exc)

        _save_task(task_file, task)

    return picked_up


def main() -> None:
    log.info('Worker starting – polling %s every %ds', TASKS_DIR, POLL_INTERVAL)
    while True:
        try:
            count = process_pending_tasks(TASKS_DIR)
            if count:
                log.info('Processed %d task(s) this cycle', count)
        except Exception as exc:
            log.error('Unhandled error in poll cycle: %s', exc)
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
