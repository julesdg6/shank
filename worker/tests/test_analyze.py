"""Tests for worker/analyze.py – librosa-based BPM and key detection."""

import importlib
import json
import wave

import librosa
import numpy as np
import pytest

# conftest.py adds the worker directory to sys.path, so we can import directly.
from analyze import _detect_chords, analyze_audio
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


def _write_chord_wav(file_path, frequencies=(261.63, 329.63, 392.0), duration=4.0, sr=SAMPLE_RATE):
    """Write a simple chord (sum of sines) to a PCM WAV file."""
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float32)
    for frequency in frequencies:
        signal += np.sin(2 * np.pi * frequency * t)
    signal /= max(len(frequencies), 1)
    samples = (signal * 32767).astype(np.int16)
    with wave.open(str(file_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return file_path


# ---------------------------------------------------------------------------
# Tests for analyze_audio
# ---------------------------------------------------------------------------

def test_analyze_audio_returns_bpm_and_key(tmp_path):
    """analyze_audio must return base stats plus summary visualization artefacts."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))

    assert isinstance(result, dict)
    assert 'bpm' in result
    assert 'bpm_confidence' in result
    assert 'key' in result
    assert 'key_confidence' in result
    assert 'lufs' in result
    assert 'duration_seconds' in result
    assert 'beats' in result
    assert 'downbeats' in result
    assert 'sections' in result
    assert 'cue_points' in result
    assert 'chords' in result
    assert 'tempo_changes' in result
    assert 'waveform' in result
    assert 'frequency_histogram' in result
    assert 'spectrogram_summary' in result
    assert 'loudness_curve' in result
    assert 'energy_over_time' in result
    assert isinstance(result['chords'], dict)
    assert isinstance(result['beats'], list)
    assert isinstance(result['downbeats'], list)
    assert isinstance(result['sections'], list)
    assert isinstance(result['cue_points'], list)
    assert isinstance(result['tempo_changes'], list)
    assert 'segments' in result['chords']
    assert 'progression' in result['chords']


def test_analyze_audio_bpm_is_positive(tmp_path):
    """BPM must be a positive float for audio with a detectable beat."""
    wav = _write_rhythmic_wav(tmp_path / 'rhythmic.wav')
    result = analyze_audio(str(wav))

    assert isinstance(result['bpm'], float)
    assert result['bpm'] > 0
    assert 0.0 <= result['bpm_confidence'] <= 1.0
    assert result['beats']
    assert all(isinstance(ts, float) for ts in result['beats'])
    assert all(isinstance(ts, float) for ts in result['downbeats'])


def test_analyze_audio_key_is_valid(tmp_path):
    """Key must be one of the 24 recognised major/minor keys."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))

    assert result['key'] in _VALID_KEYS, f"Unexpected key: {result['key']!r}"
    assert 0.0 <= result['key_confidence'] <= 1.0


def test_analyze_audio_duration_is_positive(tmp_path):
    """Duration should be reported as a positive float."""
    wav = _write_sine_wav(tmp_path / 'test.wav', duration=3.5)
    result = analyze_audio(str(wav))
    assert isinstance(result['duration_seconds'], float)
    assert result['duration_seconds'] > 0


def test_analyze_audio_missing_file_raises(tmp_path):
    """analyze_audio must raise an exception for a non-existent file."""
    with pytest.raises(Exception):
        analyze_audio(str(tmp_path / 'nonexistent.wav'))


def test_detect_chords_returns_structured_data(tmp_path):
    """Chord detection should return segment + progression metadata."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    y, sr = librosa.load(str(wav), mono=True)
    result = _detect_chords(y, sr)

    assert isinstance(result, dict)
    assert 'segments' in result
    assert 'progression' in result
    assert isinstance(result['segments'], list)
    assert isinstance(result['progression'], list)

    if result['segments']:
        first = result['segments'][0]
        assert all(key in first for key in ['symbol', 'root', 'quality', 'start_seconds', 'end_seconds'])
        assert first['quality'] in {'major', 'minor'}
        assert first['end_seconds'] >= first['start_seconds']


def test_detect_chords_identifies_c_major_triad(tmp_path):
    """A synthetic C-major triad should produce a C major segment."""
    wav = _write_chord_wav(tmp_path / 'c_major.wav', frequencies=(261.63, 329.63, 392.0))
    y, sr = librosa.load(str(wav), mono=True)
    result = _detect_chords(y, sr)

    assert result['segments'], 'Expected at least one detected chord segment'
    first = result['segments'][0]
    assert first['root'] == 'C'
    assert first['quality'] == 'major'
    assert result['progression'][0] == 'C'


def test_detect_chords_identifies_a_minor_triad(tmp_path):
    """A synthetic A-minor triad should produce an A minor segment."""
    wav = _write_chord_wav(tmp_path / 'a_minor.wav', frequencies=(220.0, 261.63, 329.63))
    y, sr = librosa.load(str(wav), mono=True)
    result = _detect_chords(y, sr)

    assert result['segments'], 'Expected at least one detected chord segment'
    first = result['segments'][0]
    assert first['root'] == 'A'
    assert first['quality'] == 'minor'
    assert result['progression'][0] == 'Am'


def test_detect_chords_returns_empty_for_silence(tmp_path):
    """Silent audio should not produce chord segments."""
    wav = _write_chord_wav(tmp_path / 'silence.wav', frequencies=())
    y, sr = librosa.load(str(wav), mono=True)
    result = _detect_chords(y, sr)
    assert result == {'segments': [], 'progression': []}


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
    """process_pending_tasks must pick up a pending upload task with a file_path and analyze it."""
    from unittest.mock import patch  # noqa: PLC0415
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

    normalized_dir = tmp_path / 'normalized'
    normalized_dir.mkdir(parents=True, exist_ok=True)

    def fake_normalize(input_path, output_path):
        # Copy the source WAV so analyze_audio has a real file to read
        import shutil  # noqa: PLC0415
        shutil.copy(input_path, output_path)

    with patch('worker_loop.normalize_audio', side_effect=fake_normalize):
        reloaded_worker_loop.process_pending_tasks(tasks_dir)

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert isinstance(updated['bpm'], float)
    assert updated['key'] in _VALID_KEYS
    assert updated['duration_seconds'] > 0
    assert 'normalized_path' in updated


def test_poll_once_error_on_missing_audio(tmp_path, reloaded_worker_loop):
    """process_pending_tasks must set status='failed' when normalization fails for a missing file."""
    from unittest.mock import patch  # noqa: PLC0415
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

    with patch('worker_loop.normalize_audio', side_effect=RuntimeError('No such file')):
        reloaded_worker_loop.process_pending_tasks(tasks_dir)

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'error' in updated


def test_poll_once_skips_url_task_without_file(tmp_path, reloaded_worker_loop):
    """URL tasks should be downloaded, normalized, and analyzed."""
    from unittest.mock import patch  # noqa: PLC0415
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

    # Mock download, normalize, and analysis so no real network/file access occurs
    with patch('worker_loop.download_youtube', return_value=tmp_path / 'audio.mp3'), \
         patch('worker_loop.normalize_audio'), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        reloaded_worker_loop.process_pending_tasks(tasks_dir)

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
