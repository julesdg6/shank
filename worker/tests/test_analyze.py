"""Tests for worker/analyze.py – librosa-based BPM and key detection."""

import builtins
import importlib
import json
import sys
import wave

import librosa
import numpy as np
import pytest

# conftest.py adds the worker directory to sys.path, so we can import directly.
from analyze import _derive_song_structure, _detect_chords, analyze_audio, build_fingerprint, compare_fingerprints
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
    assert 'structure' in result
    assert 'cue_points' in result
    assert 'chords' in result
    assert 'bpm_source' in result
    assert 'beat_detection' in result
    assert 'beatgrid' in result
    assert 'tempo_changes' in result
    assert 'waveform' in result
    assert 'frequency_histogram' in result
    assert 'spectrogram_summary' in result
    assert 'loudness_curve' in result
    assert 'energy_over_time' in result
    assert isinstance(result['chords'], dict)
    assert isinstance(result['beat_detection'], dict)
    assert isinstance(result['beatgrid'], dict)
    assert isinstance(result['beats'], list)
    assert isinstance(result['downbeats'], list)
    assert isinstance(result['sections'], list)
    assert isinstance(result['structure'], list)
    assert isinstance(result['cue_points'], list)
    assert isinstance(result['tempo_changes'], list)
    assert 'segments' in result['chords']
    assert 'progression' in result['chords']
    assert 'beats' in result['beatgrid']


def test_derive_song_structure_uses_expected_labels():
    sections = [
        {'start_seconds': 0.0, 'end_seconds': 16.0, 'label': 'section_1'},
        {'start_seconds': 16.0, 'end_seconds': 48.0, 'label': 'section_2'},
        {'start_seconds': 48.0, 'end_seconds': 80.0, 'label': 'section_3'},
        {'start_seconds': 80.0, 'end_seconds': 112.0, 'label': 'section_4'},
        {'start_seconds': 112.0, 'end_seconds': 144.0, 'label': 'section_5'},
        {'start_seconds': 144.0, 'end_seconds': 176.0, 'label': 'section_6'},
        {'start_seconds': 176.0, 'end_seconds': 208.0, 'label': 'section_7'},
        {'start_seconds': 208.0, 'end_seconds': 240.0, 'label': 'section_8'},
    ]

    structure = _derive_song_structure(sections, duration_seconds=240.0)

    assert [entry['label'] for entry in structure] == [
        'Intro',
        'Verse',
        'Chorus',
        'Verse',
        'Chorus',
        'Bridge',
        'Breakdown',
        'Outro',
    ]
    assert structure[0]['timestamp'] == '00:00'
    assert structure[1]['timestamp'] == '00:16'


def test_analyze_audio_uses_mixxx_backend_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv('BEAT_DETECTION_ENGINE', 'mixxx')
    wav = _write_rhythmic_wav(tmp_path / 'mixxx.wav', duration=4.0)

    def fake_mixxx(_):
        return {
            'bpm': 128.02,
            'mode': 'constant_tempo',
            'first_beat_seconds': 0.423,
            'confidence': None,
            'beats': [
                {'time': 0.423},
                {'time': 0.892},
                {'time': 1.361},
            ],
        }

    monkeypatch.setattr('analyze._mixxx_beats', fake_mixxx)
    result = analyze_audio(str(wav))
    assert result['bpm_source'] == 'mixxx'
    assert result['beat_detection']['engine'] == 'mixxx'
    assert result['beat_detection']['mode'] == 'constant_tempo'
    assert result['beat_detection']['first_beat_seconds'] == 0.423
    assert result['beat_detection']['beat_count'] == 3
    assert result['beat_detection']['confidence'] is None
    assert result['beatgrid']['bpm'] == 128.02
    assert result['beatgrid']['first_beat_seconds'] == 0.423
    assert result['beatgrid']['beats'][0] == {'index': 1, 'time': 0.423}


def test_analyze_audio_mixxx_failure_falls_back_to_default_detector(tmp_path, monkeypatch):
    monkeypatch.setenv('BEAT_DETECTION_ENGINE', 'mixxx')
    wav = _write_rhythmic_wav(tmp_path / 'fallback.wav', duration=4.0)
    monkeypatch.setattr('analyze._mixxx_beats', lambda _: None)
    result = analyze_audio(str(wav))
    assert result['bpm_source'] in {'librosa', 'madmom'}
    assert result['beat_detection']['engine'] in {'librosa', 'madmom'}
    assert result['beat_detection']['beat_count'] == len(result['beats'])


def test_analyze_audio_mixxx_variable_tempo_populates_local_bpm(tmp_path, monkeypatch):
    monkeypatch.setenv('BEAT_DETECTION_ENGINE', 'mixxx')
    wav = _write_rhythmic_wav(tmp_path / 'variable.wav', duration=4.0)

    def fake_mixxx(_):
        return {
            'bpm': 128.0,
            'mode': 'variable_tempo',
            'first_beat_seconds': 0.42,
            'confidence': None,
            'beats': [
                {'time': 0.42, 'local_bpm': 127.8},
                {'time': 0.89, 'local_bpm': 128.1},
            ],
        }

    monkeypatch.setattr('analyze._mixxx_beats', fake_mixxx)
    result = analyze_audio(str(wav))
    assert result['beat_detection']['mode'] == 'variable_tempo'
    assert result['beatgrid']['mode'] == 'variable_tempo'
    assert result['beatgrid']['beats'][0] == {'index': 1, 'time': 0.42, 'local_bpm': 127.8}


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
        assert all(key in first for key in ['symbol', 'root', 'quality', 'confidence', 'start_seconds', 'end_seconds'])
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


def test_detect_chords_segments_include_confidence(tmp_path):
    """Each chord segment should include a numeric confidence field between 0 and 1."""
    wav = _write_chord_wav(tmp_path / 'c_major.wav', frequencies=(261.63, 329.63, 392.0))
    y, sr = librosa.load(str(wav), mono=True)
    result = _detect_chords(y, sr)

    assert result['segments'], 'Expected at least one chord segment'
    for seg in result['segments']:
        assert 'confidence' in seg, f'Segment missing confidence key: {seg!r}'
        assert isinstance(seg['confidence'], float)
        assert 0.0 <= seg['confidence'] <= 1.0


def test_analyze_audio_chord_backend_disabled(tmp_path, monkeypatch):
    """When CHORD_BACKEND=disabled the chords result must be empty."""
    monkeypatch.setenv('CHORD_BACKEND', 'disabled')
    wav = _write_chord_wav(tmp_path / 'c_major.wav', frequencies=(261.63, 329.63, 392.0))
    result = analyze_audio(str(wav))
    assert result['chords'] == {'segments': [], 'progression': []}


def test_analyze_audio_chord_backend_auto_returns_chords(tmp_path, monkeypatch):
    """When CHORD_BACKEND=auto (default) chord detection runs and returns results."""
    monkeypatch.setenv('CHORD_BACKEND', 'auto')
    wav = _write_chord_wav(tmp_path / 'c_major.wav', frequencies=(261.63, 329.63, 392.0))
    result = analyze_audio(str(wav))
    chords = result['chords']
    assert isinstance(chords, dict)
    assert 'segments' in chords
    assert 'progression' in chords
    assert chords['segments'], 'Expected at least one chord segment for non-silent audio'


def test_analyze_audio_chord_backend_madmom_falls_back_to_librosa(tmp_path, monkeypatch):
    """When CHORD_BACKEND=madmom but madmom is unavailable, librosa fallback is used."""
    monkeypatch.setenv('CHORD_BACKEND', 'madmom')
    wav = _write_chord_wav(tmp_path / 'c_major.wav', frequencies=(261.63, 329.63, 392.0))

    # Ensure madmom import fails so the fallback path is exercised.
    madmom_backup = sys.modules.pop('madmom', None)
    madmom_audio_backup = sys.modules.pop('madmom.audio', None)
    madmom_audio_chroma_backup = sys.modules.pop('madmom.audio.chroma', None)
    madmom_features_backup = sys.modules.pop('madmom.features', None)
    madmom_features_chords_backup = sys.modules.pop('madmom.features.chords', None)
    try:
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name.startswith('madmom'):
                raise ImportError(f'Mocked: {name} not available')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', patched_import)

        result = analyze_audio(str(wav))
        chords = result['chords']
        assert isinstance(chords, dict)
        assert 'segments' in chords
        assert 'progression' in chords
        # The fallback librosa implementation should still detect a chord.
        assert chords['segments'], 'Fallback librosa chord detection should produce results'
    finally:
        # Restore any madmom modules that were popped.
        for mod_name, mod in [
            ('madmom', madmom_backup),
            ('madmom.audio', madmom_audio_backup),
            ('madmom.audio.chroma', madmom_audio_chroma_backup),
            ('madmom.features', madmom_features_backup),
            ('madmom.features.chords', madmom_features_chords_backup),
        ]:
            if mod is not None:
                sys.modules[mod_name] = mod


# ---------------------------------------------------------------------------
# Tests for build_fingerprint and compare_fingerprints
# ---------------------------------------------------------------------------


def test_build_fingerprint_returns_expected_keys(tmp_path):
    """build_fingerprint must return the required keys from an analysis dict."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    analysis = analyze_audio(str(wav))
    fp = build_fingerprint(analysis)

    assert isinstance(fp, dict)
    assert 'bpm' in fp
    assert 'key' in fp
    assert 'chord_progression' in fp
    assert 'energy_profile' in fp
    assert 'spectral_centroid' in fp


def test_build_fingerprint_bpm_matches_analysis(tmp_path):
    """Fingerprint BPM must equal the bpm from analyze_audio."""
    wav = _write_rhythmic_wav(tmp_path / 'test.wav')
    analysis = analyze_audio(str(wav))
    fp = build_fingerprint(analysis)

    assert fp['bpm'] == analysis['bpm']


def test_build_fingerprint_energy_profile_length():
    """Energy profile must always have the expected number of bins."""
    from analyze import _FINGERPRINT_ENERGY_BINS  # noqa: PLC0415
    analysis = {
        'bpm': 120.0,
        'key': 'C major',
        'energy_over_time': [float(i) / 128 for i in range(128)],
        'chords': {'progression': ['C', 'Am', 'F', 'G']},
        'frequency_histogram': [1.0] * 64,
    }
    fp = build_fingerprint(analysis)
    assert len(fp['energy_profile']) == _FINGERPRINT_ENERGY_BINS


def test_build_fingerprint_chord_progression_is_deduped():
    """Repeated chord tokens should appear only once in the fingerprint progression."""
    analysis = {
        'bpm': 120.0,
        'key': 'C major',
        'energy_over_time': [],
        'chords': {'progression': ['C', 'Am', 'C', 'F', 'Am']},
        'frequency_histogram': [],
    }
    fp = build_fingerprint(analysis)
    assert fp['chord_progression'] == ['C', 'Am', 'F']


def test_build_fingerprint_handles_missing_fields():
    """build_fingerprint must not raise when optional analysis fields are absent."""
    fp = build_fingerprint({'bpm': 128.0})
    assert fp['bpm'] == 128.0
    assert fp['key'] == ''
    assert fp['chord_progression'] == []


def test_compare_fingerprints_identical():
    """Comparing a fingerprint with itself should yield 100% similarity."""
    fp = {
        'bpm': 128.0,
        'key': 'A minor',
        'chord_progression': ['Am', 'F', 'C', 'G'],
        'energy_profile': [0.1] * 16,
        'spectral_centroid': 5.0,
    }
    result = compare_fingerprints(fp, fp)
    assert result['similarity'] == 100
    assert 'Same BPM range' in result['reasons']
    assert 'Same key' in result['reasons']


def test_compare_fingerprints_different_bpm():
    """Large BPM difference should lower the similarity score."""
    fp_a = {'bpm': 128.0, 'key': 'C major', 'chord_progression': [], 'energy_profile': []}
    fp_b = {'bpm': 80.0, 'key': 'C major', 'chord_progression': [], 'energy_profile': []}
    result = compare_fingerprints(fp_a, fp_b)
    assert result['similarity'] < 100
    assert 'Same BPM range' not in result['reasons']


def test_compare_fingerprints_same_key():
    """Exact key match must add the 'Same key' reason."""
    fp_a = {'bpm': 120.0, 'key': 'G major', 'chord_progression': [], 'energy_profile': []}
    fp_b = {'bpm': 120.0, 'key': 'G major', 'chord_progression': [], 'energy_profile': []}
    result = compare_fingerprints(fp_a, fp_b)
    assert 'Same key' in result['reasons']
    assert result['details']['key_match'] is True


def test_compare_fingerprints_same_tonic_different_mode():
    """Same root in major vs minor mode should produce a partial key match."""
    fp_a = {'bpm': 120.0, 'key': 'A major', 'chord_progression': [], 'energy_profile': []}
    fp_b = {'bpm': 120.0, 'key': 'A minor', 'chord_progression': [], 'energy_profile': []}
    result = compare_fingerprints(fp_a, fp_b)
    assert 'Same tonic, different mode' in result['reasons']


def test_compare_fingerprints_similar_chords():
    """Sufficient chord overlap should add a 'Similar chord progression' reason."""
    fp_a = {'bpm': 120.0, 'key': 'C major', 'chord_progression': ['C', 'G', 'Am', 'F'], 'energy_profile': []}
    fp_b = {'bpm': 120.0, 'key': 'C major', 'chord_progression': ['C', 'G', 'Am', 'Em'], 'energy_profile': []}
    result = compare_fingerprints(fp_a, fp_b)
    assert result['details']['chord_similarity'] > 0.0


def test_compare_fingerprints_energy_similarity():
    """Cosine similarity of matching energy profiles should appear in details."""
    profile = [0.1, 0.2, 0.15, 0.3] * 4
    fp_a = {'bpm': 120.0, 'key': '', 'chord_progression': [], 'energy_profile': profile}
    fp_b = {'bpm': 120.0, 'key': '', 'chord_progression': [], 'energy_profile': profile}
    result = compare_fingerprints(fp_a, fp_b)
    assert result['details']['energy_similarity'] == 1.0
    assert 'Similar energy curve' in result['reasons']


def test_compare_fingerprints_returns_integer_similarity():
    """Similarity value must be an integer between 0 and 100."""
    fp = {'bpm': 130.0, 'key': 'D minor', 'chord_progression': ['Dm'], 'energy_profile': [0.05] * 16}
    result = compare_fingerprints(fp, fp)
    assert isinstance(result['similarity'], int)
    assert 0 <= result['similarity'] <= 100


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
