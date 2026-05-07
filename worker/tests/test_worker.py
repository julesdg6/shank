"""Tests for the SHANK worker ffmpeg normalization pipeline."""
import importlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the worker package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

import worker_loop  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload_task(data_dir: Path, status: str = 'pending', include_file: bool = True) -> tuple[dict, Path]:
    """Create a minimal upload task dict and write it as a JSON file."""
    task_id = str(uuid.uuid4())
    upload_file = data_dir / 'uploads' / f'{task_id}.mp3'
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_bytes(b'\xff\xfb' + b'\x00' * 100)  # MP3 frame-sync header bytes

    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': 'song.mp3',
        'file_path': str(upload_file) if include_file else None,
        'status': status,
    }
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))
    return task, task_file


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Point worker_loop at a temporary DATA_DIR and reload the module."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)
    return tmp_path


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------

def test_normalize_audio_calls_ffmpeg_with_correct_args(tmp_path, monkeypatch):
    """normalize_audio must invoke ffmpeg with the right WAV conversion flags."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    input_p = str(tmp_path / 'in.mp3')
    output_p = str(tmp_path / 'out.wav')
    worker_loop.normalize_audio(input_p, output_p)

    cmd = captured['cmd']
    assert cmd[0] == 'ffmpeg'
    assert '-y' in cmd
    assert input_p in cmd
    assert output_p in cmd
    assert '-ar' in cmd
    assert worker_loop.WAV_SAMPLE_RATE in cmd
    assert '-ac' in cmd
    assert worker_loop.WAV_CHANNELS in cmd
    assert '-c:a' in cmd
    assert worker_loop.WAV_CODEC in cmd


def test_normalize_audio_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    """normalize_audio must raise RuntimeError when ffmpeg returns non-zero."""
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = 'some ffmpeg error'
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='ffmpeg failed'):
        worker_loop.normalize_audio('in.mp3', 'out.wav')


# ---------------------------------------------------------------------------
# process_pending_tasks – upload tasks
# ---------------------------------------------------------------------------

def test_pending_upload_task_is_normalized(data_dir):
    """process_pending_tasks must normalize and analyze a pending upload task and mark it done."""
    task, task_file = _make_upload_task(data_dir)
    fake_analysis = {'bpm': 128.0, 'key': 'A minor'}

    with patch('worker_loop.normalize_audio') as mock_norm, \
         patch('worker_loop.analyze_audio', return_value=fake_analysis):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert 'normalized_path' in updated
    assert updated['normalized_path'].endswith('.wav')
    assert updated['bpm'] == 128.0
    assert updated['key'] == 'A minor'
    mock_norm.assert_called_once()


def test_upload_task_normalization_failure_marks_failed(data_dir):
    """If ffmpeg raises during upload normalization, the task must be marked failed."""
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio', side_effect=RuntimeError('ffmpeg failed: oops')):
        worker_loop.process_pending_tasks()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'error' in updated


def test_upload_task_skips_non_pending(data_dir):
    """Upload tasks not in 'pending' status must not be re-processed."""
    task, task_file = _make_upload_task(data_dir, status='completed')

    with patch('worker_loop.normalize_audio') as mock_norm:
        count = worker_loop.process_pending_tasks()

    mock_norm.assert_not_called()
    assert count == 0
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'completed'


def test_upload_task_without_file_path_is_skipped(data_dir):
    """Upload tasks with no file_path should be skipped (file not yet saved)."""
    task, task_file = _make_upload_task(data_dir, include_file=False)

    with patch('worker_loop.normalize_audio') as mock_norm:
        count = worker_loop.process_pending_tasks()

    mock_norm.assert_not_called()
    assert count == 0
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'pending'


def test_upload_task_sets_processing_before_normalization(data_dir):
    """Task status must be written as 'processing' before ffmpeg is invoked."""
    task, task_file = _make_upload_task(data_dir)
    observed_statuses = []

    def capture_and_succeed(input_path, output_path):
        observed_statuses.append(json.loads(task_file.read_text())['status'])

    with patch('worker_loop.normalize_audio', side_effect=capture_and_succeed), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        worker_loop.process_pending_tasks()

    assert 'processing' in observed_statuses


def test_pending_upload_task_records_ace_step_stems_when_enabled(data_dir, monkeypatch):
    """When configured, Ace-step stem extraction output should be saved to the task."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', return_value={
             'task_id': 'ace-task-1',
             'tracks': {
                 'vocals': '/v1/audio?path=/tmp/vocals.wav',
                 'drums': '/v1/audio?path=/tmp/drums.wav',
                 'bass': '/v1/audio?path=/tmp/bass.wav',
                 'other': '/v1/audio?path=/tmp/other.wav',
             },
         }), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 128.0, 'key': 'A minor'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['ace_step_task_id'] == 'ace-task-1'
    assert updated['stems']['vocals'].endswith('vocals.wav')
    assert updated['stems']['drums'].endswith('drums.wav')
    assert updated['stems']['bass'].endswith('bass.wav')
    assert updated['stems']['other'].endswith('other.wav')


def test_pending_upload_task_marks_failed_when_ace_step_fails(data_dir, monkeypatch):
    """If Ace-step stem extraction fails, the task should be marked as failed."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step unavailable')), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        count = worker_loop.process_pending_tasks()

    assert count == 1
    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'Ace-step unavailable' in updated['error']
