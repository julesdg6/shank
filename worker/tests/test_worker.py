"""Tests for the SHANK worker ffmpeg normalization pipeline."""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(tmp_path: Path, status: str = 'pending', include_file: bool = True) -> tuple[dict, Path]:
    """Create a minimal task dict and write it as a JSON file."""
    import uuid
    task_id = str(uuid.uuid4())
    upload_file = tmp_path / 'uploads' / f'{task_id}.mp3'
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_bytes(b'\xff\xfb' + b'\x00' * 100)  # MP3 frame-sync header bytes

    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': 'song.mp3',
        'file_path': str(upload_file) if include_file else None,
        'status': status,
    }
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))
    return task, task_file


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------

def test_normalize_audio_calls_ffmpeg_with_correct_args(tmp_path, monkeypatch):
    """normalize_audio must invoke ffmpeg with the right WAV conversion flags."""
    import worker.worker_loop as wl

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    input_p = str(tmp_path / 'in.mp3')
    output_p = str(tmp_path / 'out.wav')
    wl.normalize_audio(input_p, output_p)

    cmd = captured['cmd']
    assert cmd[0] == 'ffmpeg'
    assert '-y' in cmd
    assert input_p in cmd
    assert output_p in cmd
    assert '-ar' in cmd
    assert wl.WAV_SAMPLE_RATE in cmd
    assert '-ac' in cmd
    assert wl.WAV_CHANNELS in cmd
    assert '-c:a' in cmd
    assert wl.WAV_CODEC in cmd


def test_normalize_audio_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    """normalize_audio must raise RuntimeError when ffmpeg returns non-zero."""
    import worker.worker_loop as wl

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = 'some ffmpeg error'
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='ffmpeg failed'):
        wl.normalize_audio('in.mp3', 'out.wav')


# ---------------------------------------------------------------------------
# process_task
# ---------------------------------------------------------------------------

def test_process_task_normalizes_pending_upload(tmp_path, monkeypatch):
    """process_task must normalize a pending upload task and mark it completed."""
    import importlib
    import worker.worker_loop as wl

    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(wl)

    task, task_file = _make_task(tmp_path)

    with patch.object(wl, 'normalize_audio') as mock_norm:
        wl.process_task(task_file)

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'completed'
    assert 'normalized_path' in updated
    assert updated['normalized_path'].endswith('.wav')
    mock_norm.assert_called_once()


def test_process_task_marks_failed_on_ffmpeg_error(tmp_path, monkeypatch):
    """process_task must mark the task as failed when ffmpeg raises."""
    import importlib
    import worker.worker_loop as wl

    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(wl)

    task, task_file = _make_task(tmp_path)

    with patch.object(wl, 'normalize_audio', side_effect=RuntimeError('ffmpeg failed: oops')):
        wl.process_task(task_file)

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'error' in updated


def test_process_task_skips_non_pending(tmp_path, monkeypatch):
    """process_task must not touch tasks that are not in 'pending' status."""
    import importlib
    import worker.worker_loop as wl

    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(wl)

    task, task_file = _make_task(tmp_path, status='completed')

    with patch.object(wl, 'normalize_audio') as mock_norm:
        wl.process_task(task_file)

    mock_norm.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'completed'


def test_process_task_skips_url_task_without_file_path(tmp_path, monkeypatch):
    """URL tasks with no file_path should be skipped (yt-dlp not yet run)."""
    import importlib
    import worker.worker_loop as wl

    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(wl)

    task, task_file = _make_task(tmp_path, include_file=False)

    with patch.object(wl, 'normalize_audio') as mock_norm:
        wl.process_task(task_file)

    mock_norm.assert_not_called()
    # Status should remain unchanged
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'pending'


def test_process_task_sets_processing_before_normalization(tmp_path, monkeypatch):
    """Task status must be written as 'processing' before ffmpeg is invoked."""
    import importlib
    import worker.worker_loop as wl

    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(wl)

    task, task_file = _make_task(tmp_path)
    observed_statuses = []

    def capture_and_succeed(input_path, output_path):
        observed_statuses.append(json.loads(task_file.read_text())['status'])

    with patch.object(wl, 'normalize_audio', side_effect=capture_and_succeed):
        wl.process_task(task_file)

    assert 'processing' in observed_statuses
