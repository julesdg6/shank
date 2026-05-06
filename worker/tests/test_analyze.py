"""Tests for worker/analyze.py – librosa-based BPM and key detection."""

import importlib
import json
import wave

import numpy as np
import pytest

# conftest.py adds the worker directory to sys.path, so we can import directly.
from analyze import analyze_audio
import worker_loop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_VALID_KEYS = {f'{p} {m}' for p in _PITCH_CLASSES for m in ('major', 'minor')}

SAMPLE_RATE = 22050


def _write_sine_wav(path, frequency=440.0, duration=5.0, sr=SAMPLE_RATE):
    """Write a single-frequency sine wave to a PCM WAV file at *path*."""
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def _write_rhythmic_wav(path, bpm=120.0, frequency=440.0, duration=8.0, sr=SAMPLE_RATE):
    """Write a rhythmic click-track WAV with impulse bursts at *bpm* beats per minute.

    The periodic transients give librosa enough rhythmic information to estimate
    a non-zero BPM.
    """
    n_samples = int(duration * sr)
    signal = np.zeros(n_samples, dtype=np.float32)
    beat_period = sr * 60.0 / bpm
    beat_idx = 0
    while True:
        pos = int(round(beat_idx * beat_period))
        if pos >= n_samples:
            break
        burst_len = min(int(0.02 * sr), n_samples - pos)
        t_burst = np.arange(burst_len) / sr
        signal[pos:pos + burst_len] += (
            np.sin(2 * np.pi * frequency * t_burst) * np.linspace(1, 0, burst_len)
        )
        beat_idx += 1
    samples = (signal * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# Tests for analyze_audio
# ---------------------------------------------------------------------------

def test_analyze_audio_returns_bpm_and_key(tmp_path):
    """analyze_audio must return a dict with 'bpm' and 'key' keys."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))

    assert isinstance(result, dict)
    assert 'bpm' in result
    assert 'key' in result


def test_analyze_audio_bpm_is_positive(tmp_path):
    """BPM must be a positive float for audio with a detectable beat."""
    wav = _write_rhythmic_wav(tmp_path / 'rhythmic.wav')
    result = analyze_audio(str(wav))

    assert isinstance(result['bpm'], float)
    assert result['bpm'] > 0


def test_analyze_audio_key_is_valid(tmp_path):
    """Key must be one of the 24 recognised major/minor keys."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))

    assert result['key'] in _VALID_KEYS, f"Unexpected key: {result['key']!r}"


def test_analyze_audio_missing_file_raises(tmp_path):
    """analyze_audio must raise an exception for a non-existent file."""
    with pytest.raises(Exception):
        analyze_audio(str(tmp_path / 'nonexistent.wav'))


# ---------------------------------------------------------------------------
# Tests for the worker loop helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def reloaded_worker_loop(monkeypatch, tmp_path):
    """Reload worker_loop with DATA_DIR pointed at tmp_path."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)
    return worker_loop


def test_poll_once_processes_pending_task(tmp_path, reloaded_worker_loop):
    """_poll_once must pick up a pending task with a file_path and process it."""
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    wav = _write_sine_wav(tmp_path / 'audio.wav')
    task_id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': 'audio.wav',
        'file_path': str(wav),
        'status': 'pending',
        'created_at': '2025-01-01T00:00:00+00:00',
    }
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))

    reloaded_worker_loop._poll_once()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert isinstance(updated['bpm'], float)
    assert updated['key'] in _VALID_KEYS


def test_poll_once_error_on_missing_audio(tmp_path, reloaded_worker_loop):
    """_poll_once must set status='error' when the audio file does not exist."""
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_id = 'ffffffff-0000-1111-2222-333333333333'
    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': 'ghost.wav',
        'file_path': str(tmp_path / 'ghost.wav'),  # does not exist
        'status': 'pending',
        'created_at': '2025-01-01T00:00:00+00:00',
    }
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))

    reloaded_worker_loop._poll_once()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'error'
    assert 'error' in updated


def test_poll_once_skips_url_task_without_file(tmp_path, reloaded_worker_loop):
    """URL tasks without a file_path must be left as 'pending' (not yet supported)."""
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_id = '11111111-2222-3333-4444-555555555555'
    task = {
        'task_id': task_id,
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'pending',
        'created_at': '2025-01-01T00:00:00+00:00',
    }
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))

    reloaded_worker_loop._poll_once()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'pending'
