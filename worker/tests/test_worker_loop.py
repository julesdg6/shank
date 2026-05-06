"""Tests for worker/worker_loop.py – polling loop and task handling."""
import importlib
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the worker package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

import worker_loop  # noqa: E402


YOUTUBE_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Point worker_loop at a temporary DATA_DIR and reload the module."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)
    return tmp_path


def _make_task(data_dir: Path, *, task_type: str = 'url', status: str = 'pending') -> tuple[str, Path]:
    task_id = str(uuid.uuid4())
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task = {
        'task_id': task_id,
        'type': task_type,
        'source': YOUTUBE_URL,
        'status': status,
    }
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))
    return task_id, task_file


# ---------------------------------------------------------------------------
# process_pending_tasks – happy path
# ---------------------------------------------------------------------------

def test_pending_url_task_is_downloaded(data_dir):
    """A pending url task should be downloaded and marked done."""
    task_id, task_file = _make_task(data_dir)
    uploads_dir = data_dir / 'uploads'
    fake_output = uploads_dir / f'{task_id}.mp3'

    with patch('worker_loop.download_youtube', return_value=fake_output) as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_called_once_with(YOUTUBE_URL, worker_loop.UPLOADS_DIR, task_id)
    task = json.loads(task_file.read_text())
    assert task['status'] == 'done'
    assert task['file_path'] == str(fake_output)
    assert 'completed_at' in task


def test_done_task_is_skipped(data_dir):
    """Tasks that are already done must not be re-processed."""
    _make_task(data_dir, status='done')

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


def test_upload_type_task_is_skipped(data_dir):
    """Tasks of type 'upload' are not URL tasks and must be ignored."""
    _make_task(data_dir, task_type='upload', status='pending')

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


def test_processing_task_is_skipped(data_dir):
    """Tasks already marked 'processing' should not be picked up again."""
    _make_task(data_dir, status='processing')

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


def test_task_missing_task_id_field_is_skipped(data_dir):
    """A task file missing 'task_id' should be skipped without crashing."""
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / 'no-id.json'
    task_file.write_text(json.dumps({'type': 'url', 'source': YOUTUBE_URL, 'status': 'pending'}))

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


def test_task_missing_source_field_is_skipped(data_dir):
    """A task file missing 'source' should be skipped without crashing."""
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(uuid.uuid4())
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps({'task_id': task_id, 'type': 'url', 'status': 'pending'}))

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


def test_task_invalid_uuid_task_id_is_skipped(data_dir):
    """A task file with a non-UUID task_id should be skipped without crashing."""
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / 'bad-uuid.json'
    task_file.write_text(json.dumps({
        'task_id': '../../etc/passwd',
        'type': 'url',
        'source': YOUTUBE_URL,
        'status': 'pending',
    }))

    with patch('worker_loop.download_youtube') as mock_dl:
        worker_loop.process_pending_tasks()

    mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# process_pending_tasks – error handling
# ---------------------------------------------------------------------------

def test_failed_download_marks_task_failed(data_dir):
    """If download_youtube raises, the task status must be set to 'failed'."""
    task_id, task_file = _make_task(data_dir)

    with patch('worker_loop.download_youtube', side_effect=RuntimeError('boom')):
        worker_loop.process_pending_tasks()

    task = json.loads(task_file.read_text())
    assert task['status'] == 'failed'
    assert 'boom' in task['error']
    assert 'completed_at' in task


def test_task_status_set_to_processing_before_download(data_dir):
    """The task file must be written with status='processing' before the download starts."""
    task_id, task_file = _make_task(data_dir)
    observed_statuses: list[str] = []

    def fake_download(url, output_dir, tid):
        task = json.loads(task_file.read_text())
        observed_statuses.append(task['status'])
        return output_dir / f'{tid}.mp3'

    with patch('worker_loop.download_youtube', side_effect=fake_download):
        worker_loop.process_pending_tasks()

    assert observed_statuses == ['processing']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_read_task_returns_none_for_corrupt_file(data_dir, tmp_path):
    """_read_task must return None for files with invalid JSON."""
    bad_file = tmp_path / 'bad.json'
    bad_file.write_text('not valid json {{{')
    assert worker_loop._read_task(bad_file) is None


def test_multiple_pending_tasks_all_processed(data_dir):
    """All pending URL tasks in the directory should be processed."""
    ids = [_make_task(data_dir)[0] for _ in range(3)]
    uploads_dir = data_dir / 'uploads'

    def fake_download(url, output_dir, task_id):
        return output_dir / f'{task_id}.mp3'

    with patch('worker_loop.download_youtube', side_effect=fake_download) as mock_dl:
        worker_loop.process_pending_tasks()

    assert mock_dl.call_count == 3
