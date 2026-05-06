"""Tests for the worker loop task-polling logic."""
import json
import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def tasks_dir(tmp_path):
    """Return a fresh temporary tasks directory."""
    d = tmp_path / 'tasks'
    d.mkdir()
    return d


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    """Import worker_loop with DATA_DIR pointing at tmp_path."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    # Add the worker directory to sys.path so the bare module can be imported.
    worker_dir = str(Path(__file__).resolve().parents[1])
    if worker_dir not in sys.path:
        sys.path.insert(0, worker_dir)
    import worker_loop as wl
    importlib.reload(wl)
    return wl


def _write_task(tasks_dir: Path, task: dict) -> Path:
    task_file = tasks_dir / f"{task['task_id']}.json"
    task_file.write_text(json.dumps(task, indent=2))
    return task_file


# ---------------------------------------------------------------------------
# process_pending_tasks
# ---------------------------------------------------------------------------

def test_no_tasks_returns_zero(worker, tasks_dir):
    assert worker.process_pending_tasks(tasks_dir) == 0


def test_missing_dir_returns_zero(worker, tmp_path):
    missing = tmp_path / 'nonexistent'
    assert worker.process_pending_tasks(missing) == 0


def test_pending_upload_task_is_processed(worker, tasks_dir):
    task = {
        'task_id': 'aaaaaaaa-0000-0000-0000-000000000001',
        'type': 'upload',
        'source': 'song.mp3',
        'file_path': '/tmp/song.mp3',
        'status': 'pending',
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    task_file = _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)

    assert count == 1
    result = json.loads(task_file.read_text())
    assert result['status'] == 'done'
    assert 'started_at' in result
    assert 'finished_at' in result


def test_pending_url_task_is_processed(worker, tasks_dir):
    task = {
        'task_id': 'aaaaaaaa-0000-0000-0000-000000000002',
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'pending',
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    task_file = _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)

    assert count == 1
    result = json.loads(task_file.read_text())
    assert result['status'] == 'done'


def test_already_processing_task_is_skipped(worker, tasks_dir):
    task = {
        'task_id': 'aaaaaaaa-0000-0000-0000-000000000003',
        'type': 'upload',
        'source': 'song.mp3',
        'status': 'processing',
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)
    assert count == 0


def test_done_task_is_skipped(worker, tasks_dir):
    task = {
        'task_id': 'aaaaaaaa-0000-0000-0000-000000000004',
        'type': 'upload',
        'source': 'song.mp3',
        'status': 'done',
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)
    assert count == 0


def test_unknown_task_type_marked_failed(worker, tasks_dir):
    task = {
        'task_id': 'aaaaaaaa-0000-0000-0000-000000000005',
        'type': 'unknown_type',
        'source': 'whatever',
        'status': 'pending',
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    task_file = _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)

    assert count == 1
    result = json.loads(task_file.read_text())
    assert result['status'] == 'failed'
    assert 'error' in result


def test_multiple_pending_tasks_all_processed(worker, tasks_dir):
    for i in range(3):
        task = {
            'task_id': f'aaaaaaaa-0000-0000-0000-00000000000{i + 6}',
            'type': 'url',
            'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'status': 'pending',
            'created_at': '2026-01-01T00:00:00+00:00',
        }
        _write_task(tasks_dir, task)

    count = worker.process_pending_tasks(tasks_dir)
    assert count == 3

    for task_file in tasks_dir.glob('*.json'):
        assert json.loads(task_file.read_text())['status'] == 'done'
