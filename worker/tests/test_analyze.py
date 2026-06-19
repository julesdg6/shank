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
from analyze import _derive_song_structure, _detect_chords, analyze_audio, build_fingerprint, _derive_cue_points
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
# Tests for the worker loop helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def reloaded_worker_loop(monkeypatch, tmp_path):
    """Reload worker_loop with DATA_DIR pointed at tmp_path."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)
    return worker_loop


# ---------------------------------------------------------------------------
# _derive_cue_points
# ---------------------------------------------------------------------------

def test_derive_cue_points_uses_structure_labels():
    """Cue points must be derived from the song structure in label order."""
    structure = [
        {'label': 'Intro', 'start_seconds': 0.0, 'end_seconds': 16.0, 'timestamp': '00:00'},
        {'label': 'Verse', 'start_seconds': 16.0, 'end_seconds': 48.0, 'timestamp': '00:16'},
        {'label': 'Chorus', 'start_seconds': 48.0, 'end_seconds': 80.0, 'timestamp': '00:48'},
        {'label': 'Breakdown', 'start_seconds': 80.0, 'end_seconds': 112.0, 'timestamp': '01:20'},
        {'label': 'Outro', 'start_seconds': 224.0, 'end_seconds': 240.0, 'timestamp': '03:44'},
    ]
    cues = _derive_cue_points([], [], 240.0, structure=structure)
    names = [c['name'] for c in cues]
    assert names == ['Intro', 'Verse', 'Chorus', 'Breakdown', 'Outro']
    assert cues[0]['time_seconds'] == 0.0
    assert cues[1]['time_seconds'] == 16.0
    assert cues[2]['time_seconds'] == 48.0


def test_derive_cue_points_assigns_hot_cue_indices():
    """Each cue point must have a zero-based hot_cue index."""
    structure = [
        {'label': 'Intro', 'start_seconds': 0.0, 'end_seconds': 16.0, 'timestamp': '00:00'},
        {'label': 'Verse', 'start_seconds': 16.0, 'end_seconds': 48.0, 'timestamp': '00:16'},
        {'label': 'Chorus', 'start_seconds': 48.0, 'end_seconds': 80.0, 'timestamp': '00:48'},
    ]
    cues = _derive_cue_points([], [], 240.0, structure=structure)
    assert cues[0]['hot_cue'] == 0
    assert cues[1]['hot_cue'] == 1
    assert cues[2]['hot_cue'] == 2


def test_derive_cue_points_assigns_colors():
    """Each cue point must carry a non-empty hex color string."""
    structure = [
        {'label': 'Intro', 'start_seconds': 0.0, 'end_seconds': 16.0, 'timestamp': '00:00'},
        {'label': 'Verse', 'start_seconds': 16.0, 'end_seconds': 48.0, 'timestamp': '00:16'},
    ]
    cues = _derive_cue_points([], [], 240.0, structure=structure)
    for cue in cues:
        assert isinstance(cue['color'], str)
        assert cue['color'].startswith('#')
        assert len(cue['color']) == 7


def test_derive_cue_points_deduplicated_labels():
    """Repeated section labels should appear only once as a cue point."""
    structure = [
        {'label': 'Intro', 'start_seconds': 0.0, 'end_seconds': 16.0, 'timestamp': '00:00'},
        {'label': 'Verse', 'start_seconds': 16.0, 'end_seconds': 48.0, 'timestamp': '00:16'},
        {'label': 'Chorus', 'start_seconds': 48.0, 'end_seconds': 80.0, 'timestamp': '00:48'},
        {'label': 'Verse', 'start_seconds': 80.0, 'end_seconds': 112.0, 'timestamp': '01:20'},
        {'label': 'Chorus', 'start_seconds': 112.0, 'end_seconds': 144.0, 'timestamp': '01:52'},
        {'label': 'Outro', 'start_seconds': 224.0, 'end_seconds': 240.0, 'timestamp': '03:44'},
    ]
    cues = _derive_cue_points([], [], 240.0, structure=structure)
    names = [c['name'] for c in cues]
    assert names.count('Verse') == 1
    assert names.count('Chorus') == 1


def test_derive_cue_points_fallback_without_structure():
    """Without structure, fall back to beat/downbeat anchors."""
    beats = [0.5, 0.97, 1.44]
    downbeats = [0.5, 2.0]
    cues = _derive_cue_points(downbeats, beats, 120.0)
    names = [c['name'] for c in cues]
    assert 'Intro' in names
    assert 'Outro' in names
    # Verify hot_cue indices are assigned sequentially
    for idx, cue in enumerate(cues):
        assert cue['hot_cue'] == idx


def test_derive_cue_points_outro_added_when_missing_from_structure():
    """An Outro cue must be synthesised from duration when the structure lacks one."""
    structure = [
        {'label': 'Intro', 'start_seconds': 0.0, 'end_seconds': 16.0, 'timestamp': '00:00'},
        {'label': 'Verse', 'start_seconds': 16.0, 'end_seconds': 48.0, 'timestamp': '00:16'},
    ]
    cues = _derive_cue_points([], [], 120.0, structure=structure)
    names = [c['name'] for c in cues]
    assert 'Outro' in names
    outro = next(c for c in cues if c['name'] == 'Outro')
    # Should be duration - OUTRO_OFFSET_SECONDS (8 s)
    assert outro['time_seconds'] == 112.0


def test_analyze_audio_cue_points_have_hot_cue_and_color(tmp_path):
    """analyze_audio output must include cue points with hot_cue and color fields."""
    wav = _write_rhythmic_wav(tmp_path / 'rhythmic.wav', duration=30.0)
    result = analyze_audio(str(wav))
    cue_points = result['cue_points']
    assert isinstance(cue_points, list)
    assert len(cue_points) >= 1
    for cue in cue_points:
        assert 'name' in cue
        assert 'time_seconds' in cue
        assert 'hot_cue' in cue
        assert isinstance(cue['hot_cue'], int)
        assert 'color' in cue
        assert cue['color'].startswith('#')


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


# ---------------------------------------------------------------------------
# Tests for build_fingerprint
# ---------------------------------------------------------------------------

def test_build_fingerprint_returns_expected_fields(tmp_path):
    """build_fingerprint must return all required Audio DNA fields."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))
    fp = build_fingerprint(result)

    assert isinstance(fp, dict)
    assert fp['version'] == '1'
    assert isinstance(fp['bpm'], float)
    assert isinstance(fp['bpm_normalized'], float)
    assert 0.0 <= fp['bpm_normalized'] <= 1.0
    assert isinstance(fp['key'], str)
    assert isinstance(fp['key_index'], int)
    assert 0 <= fp['key_index'] <= 23
    assert isinstance(fp['chord_profile'], dict)
    assert isinstance(fp['energy_profile'], list)
    assert len(fp['energy_profile']) == 32
    assert isinstance(fp['spectral_profile'], list)
    assert len(fp['spectral_profile']) == 8
    assert isinstance(fp['duration_seconds'], float)
    assert isinstance(fp['fingerprint_hash'], str)
    assert len(fp['fingerprint_hash']) == 64  # SHA-256 hex digest


def test_build_fingerprint_bpm_normalized(tmp_path):
    """bpm_normalized must be bpm / 200 clamped to [0, 1]."""
    result = {'bpm': 120.0, 'key': 'C major'}
    fp = build_fingerprint(result)
    assert fp['bpm_normalized'] == round(120.0 / 200.0, 4)

    result_high = {'bpm': 300.0, 'key': 'C major'}
    fp_high = build_fingerprint(result_high)
    assert fp_high['bpm_normalized'] == 1.0


def test_build_fingerprint_key_index_major():
    """Key index for major keys should be 0–11."""
    _PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, pitch in enumerate(_PITCH_CLASSES):
        fp = build_fingerprint({'bpm': 120.0, 'key': f'{pitch} major'})
        assert fp['key_index'] == i, f'Expected {i} for {pitch} major, got {fp["key_index"]}'


def test_build_fingerprint_key_index_minor():
    """Key index for minor keys should be 12–23."""
    _PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, pitch in enumerate(_PITCH_CLASSES):
        fp = build_fingerprint({'bpm': 120.0, 'key': f'{pitch} minor'})
        assert fp['key_index'] == i + 12, f'Expected {i + 12} for {pitch} minor, got {fp["key_index"]}'


def test_build_fingerprint_chord_profile_sums_to_one(tmp_path):
    """chord_profile values must sum to 1.0 when chord segments are present."""
    wav = _write_chord_wav(tmp_path / 'chord.wav')
    result = analyze_audio(str(wav))
    fp = build_fingerprint(result)

    if fp['chord_profile']:
        total = sum(fp['chord_profile'].values())
        assert abs(total - 1.0) < 1e-3, f'chord_profile values sum to {total}, expected ~1.0'


def test_build_fingerprint_profiles_normalized(tmp_path):
    """energy_profile and spectral_profile values must be in [0, 1]."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))
    fp = build_fingerprint(result)

    assert all(0.0 <= v <= 1.0 for v in fp['energy_profile']), 'energy_profile contains out-of-range values'
    assert all(0.0 <= v <= 1.0 for v in fp['spectral_profile']), 'spectral_profile contains out-of-range values'


def test_build_fingerprint_hash_is_deterministic(tmp_path):
    """The same analysis result must always produce the same fingerprint_hash."""
    wav = _write_sine_wav(tmp_path / 'test.wav')
    result = analyze_audio(str(wav))
    fp1 = build_fingerprint(result)
    fp2 = build_fingerprint(result)
    assert fp1['fingerprint_hash'] == fp2['fingerprint_hash']


def test_build_fingerprint_hash_differs_for_different_bpm():
    """Different BPM values must produce different fingerprint hashes."""
    fp_a = build_fingerprint({'bpm': 120.0, 'key': 'C major'})
    fp_b = build_fingerprint({'bpm': 140.0, 'key': 'C major'})
    assert fp_a['fingerprint_hash'] != fp_b['fingerprint_hash']


def test_build_fingerprint_empty_analysis():
    """build_fingerprint must not raise for a minimal/empty analysis dict."""
    fp = build_fingerprint({})
    assert isinstance(fp, dict)
    assert fp['bpm'] == 0.0
    assert fp['key'] == 'C major'
    assert fp['key_index'] == 0
    assert fp['chord_profile'] == {}
    assert len(fp['energy_profile']) == 32
    assert len(fp['spectral_profile']) == 8
