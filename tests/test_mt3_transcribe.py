"""Tests for mt3/transcribe.py standalone transcription wrapper."""

import base64
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the repo root is on sys.path so `mt3` package is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mt3.transcribe as mt3_transcribe  # noqa: E402

# Minimal valid MIDI bytes (single empty track) used in test fixtures.
_EMPTY_MIDI = (
    b'MThd'
    b'\x00\x00\x00\x06'
    b'\x00\x00'
    b'\x00\x01'
    b'\x01\xe0'
    b'MTrk'
    b'\x00\x00\x00\x04'
    b'\x00\xff\x2f\x00'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path) -> Path:
    """Write a minimal stub WAV file so the path exists."""
    path.write_bytes(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00'
                     b'\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00'
                     b'data\x00\x00\x00\x00')
    return path


# ---------------------------------------------------------------------------
# transcribe() – basic behaviour
# ---------------------------------------------------------------------------

def test_transcribe_writes_midi_file(tmp_path):
    """transcribe() must produce a .mid file."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(wav, output_dir=tmp_path / 'out', service_url='')

    midi = Path(result['midi_path'])
    assert midi.exists()
    assert midi.suffix == '.mid'


def test_transcribe_writes_notes_json_by_default(tmp_path):
    """transcribe() must write a .notes.json file when save_notes=True."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(wav, output_dir=tmp_path / 'out', service_url='')

    assert result['notes_path'] is not None
    notes_file = Path(result['notes_path'])
    assert notes_file.exists()
    assert notes_file.suffix == '.json'


def test_transcribe_skips_notes_when_save_notes_false(tmp_path):
    """transcribe() must not write a notes file when save_notes=False."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(
        wav, output_dir=tmp_path / 'out', service_url='', save_notes=False,
    )

    assert result['notes_path'] is None
    # Ensure no stray notes file was created either
    out = tmp_path / 'out'
    assert not list(out.glob('*.notes.json'))


def test_transcribe_writes_meta_json(tmp_path):
    """transcribe() must write a .meta.json file with summary metadata."""
    wav = _make_wav(tmp_path / 'audio.wav')
    out = tmp_path / 'out'
    result = mt3_transcribe.transcribe(wav, output_dir=out, service_url='')

    meta_file = out / f'{result["task_id"]}.meta.json'
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta['wav_path'] == str(wav)
    assert meta['midi_path'] == result['midi_path']
    assert meta['task_id'] == result['task_id']
    assert 'transcribed_at' in meta


def test_transcribe_returns_expected_keys(tmp_path):
    """Result dict must contain all documented keys."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(wav, output_dir=tmp_path / 'out', service_url='')

    for key in ('wav_path', 'midi_path', 'notes_path', 'note_count',
                'pitch_range', 'duration_seconds', 'program_count',
                'model', 'task_id', 'output_dir', 'transcribed_at', 'warnings'):
        assert key in result, f'Missing key: {key}'


def test_transcribe_raises_for_missing_wav(tmp_path):
    """transcribe() must raise FileNotFoundError when WAV does not exist."""
    with pytest.raises(FileNotFoundError, match='WAV file not found'):
        mt3_transcribe.transcribe(tmp_path / 'missing.wav', service_url='')


def test_transcribe_uses_custom_task_id(tmp_path):
    """Custom task_id must be reflected in the output filenames."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(
        wav, output_dir=tmp_path / 'out', service_url='', task_id='my_task_abc',
    )

    assert result['task_id'] == 'my_task_abc'
    assert Path(result['midi_path']).name == 'my_task_abc.mid'


def test_transcribe_uses_custom_model(tmp_path):
    """Custom model name must appear in the returned metadata."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(
        wav, output_dir=tmp_path / 'out', service_url='', model='ismir2021',
    )

    assert result['model'] == 'ismir2021'


def test_transcribe_default_output_dir_uses_data_dir(tmp_path, monkeypatch):
    """Without output_dir, outputs must go to DATA_DIR/mt3/<task_id>/."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(mt3_transcribe)

    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(wav, service_url='', task_id='tid_default')

    expected = tmp_path / 'mt3' / 'tid_default'
    assert Path(result['output_dir']) == expected
    assert Path(result['midi_path']).parent == expected

    # Reload module to restore original DATA_DIR for subsequent tests
    importlib.reload(mt3_transcribe)


# ---------------------------------------------------------------------------
# transcribe() – HTTP service path
# ---------------------------------------------------------------------------

def test_transcribe_calls_http_service(tmp_path):
    """When service_url is set, _call_service must be invoked."""
    wav = _make_wav(tmp_path / 'audio.wav')
    empty_midi_b64 = base64.b64encode(_EMPTY_MIDI).decode()
    fake_payload = {
        'status': 'completed',
        'model': 'svc_model',
        'midi_base64': empty_midi_b64,
        'notes': [{'pitch': 60, 'start': 0.0, 'end': 1.0, 'program': 5}],
        'warnings': [],
    }

    with patch.object(mt3_transcribe, '_call_service', return_value=fake_payload) as mock_call:
        result = mt3_transcribe.transcribe(
            wav,
            output_dir=tmp_path / 'out',
            service_url='http://localhost:8090',
            task_id='svc_task',
        )

    mock_call.assert_called_once_with(
        'http://localhost:8090', wav.resolve(), 'svc_task', mt3_transcribe.MT3_MODEL, mt3_transcribe.MT3_TIMEOUT,
    )
    assert result['model'] == 'svc_model'
    assert result['note_count'] == 1
    assert result['pitch_range'] == {'min': 60, 'max': 60}
    assert result['duration_seconds'] == 1.0
    assert result['program_count'] == 1


def test_transcribe_service_warnings_forwarded(tmp_path):
    """Warnings from the service must appear in the result."""
    wav = _make_wav(tmp_path / 'audio.wav')
    empty_midi_b64 = base64.b64encode(_EMPTY_MIDI).decode()
    fake_payload = {
        'model': 'mt3',
        'midi_base64': empty_midi_b64,
        'notes': [],
        'warnings': ['slow inference detected'],
    }

    with patch.object(mt3_transcribe, '_call_service', return_value=fake_payload):
        result = mt3_transcribe.transcribe(
            wav, output_dir=tmp_path / 'out', service_url='http://localhost:8090',
        )

    assert 'slow inference detected' in result['warnings']


# ---------------------------------------------------------------------------
# transcribe() – inline fallback path
# ---------------------------------------------------------------------------

def test_transcribe_inline_uses_services_mt3(tmp_path, monkeypatch):
    """When service_url is empty, _transcribe_inline is invoked."""
    wav = _make_wav(tmp_path / 'audio.wav')
    fake_notes = [{'pitch': 62, 'start': 0.5, 'end': 1.5}]

    with patch.object(
        mt3_transcribe, '_transcribe_inline', return_value=(_EMPTY_MIDI, fake_notes, []),
    ) as mock_inline:
        result = mt3_transcribe.transcribe(
            wav, output_dir=tmp_path / 'out', service_url='',
        )

    mock_inline.assert_called_once()
    assert result['note_count'] == 1


# ---------------------------------------------------------------------------
# MIDI output validity
# ---------------------------------------------------------------------------

def test_midi_output_has_valid_header(tmp_path):
    """The produced MIDI file must start with the MThd magic bytes."""
    wav = _make_wav(tmp_path / 'audio.wav')
    result = mt3_transcribe.transcribe(wav, output_dir=tmp_path / 'out', service_url='')
    midi_bytes = Path(result['midi_path']).read_bytes()
    assert midi_bytes[:4] == b'MThd'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_json_output(tmp_path, capsys):
    """CLI with --json must print valid JSON to stdout."""
    wav = _make_wav(tmp_path / 'audio.wav')

    mt3_transcribe.main([str(wav), '--output-dir', str(tmp_path / 'out'),
                         '--service-url', '', '--task-id', 'cli_task', '--json'])

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data['task_id'] == 'cli_task'
    assert data['midi_path'].endswith('.mid')


def test_cli_plain_output(tmp_path, capsys):
    """CLI without --json must print human-readable MIDI path."""
    wav = _make_wav(tmp_path / 'audio.wav')

    mt3_transcribe.main([str(wav), '--output-dir', str(tmp_path / 'out'),
                         '--service-url', '', '--task-id', 'cli_plain'])

    captured = capsys.readouterr()
    assert 'MIDI:' in captured.out
    assert 'cli_plain.mid' in captured.out


def test_cli_no_notes_flag(tmp_path):
    """CLI --no-notes must not write a notes JSON file."""
    wav = _make_wav(tmp_path / 'audio.wav')
    out = tmp_path / 'out'

    mt3_transcribe.main([str(wav), '--output-dir', str(out),
                         '--service-url', '', '--no-notes', '--task-id', 'nonotes'])

    assert not list(out.glob('*.notes.json'))
