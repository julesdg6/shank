import importlib
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

from transcription.base import (
    BackendDependencyError,
    EmptyTranscriptionError,
    InvalidAudioError,
    TranscriptionResult,
)


def _load_client(monkeypatch, tmp_path):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('MT3_MODEL', 'multi_instrument')
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'basic_pitch')
    import services.mt3.main as mt3_main
    importlib.reload(mt3_main)
    return TestClient(mt3_main.app), mt3_main


def test_mt3_health(monkeypatch, tmp_path):
    client, _ = _load_client(monkeypatch, tmp_path)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'mt3'}


def test_mt3_transcribe_writes_artifacts_in_data_dir(monkeypatch, tmp_path):
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1, -0.1, 0.0], 16000)

    class FakeBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            return TranscriptionResult(
                backend='basic_pitch',
                midi_bytes=mt3_main._empty_midi_bytes(),
                notes=[{
                    'start': 0.0,
                    'end': 0.25,
                    'pitch': 60,
                    'velocity': 88,
                    'confidence': 0.9,
                }],
            )

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: FakeBackend())

    response = client.post('/transcribe', json={
        'path': 'uploads/sample.wav',
        'task_id': 'task-1',
        'source': 'full_mix',
    })

    payload = response.json()
    midi_path = Path(payload['midi_path'])
    notes_path = Path(payload['notes_path'])

    assert response.status_code == 200
    assert payload['status'] == 'completed'
    assert payload['error'] is None
    assert payload['backend'] == 'basic_pitch'
    assert payload['model'] == 'multi_instrument'
    assert payload['note_count'] == 1
    assert midi_path.exists()
    assert notes_path.exists()
    assert midi_path.suffix == '.mid'
    assert notes_path.suffix == '.json'
    assert payload['notes'][0]['pitch'] == 60


def test_mt3_transcribe_rejects_path_outside_data_dir(monkeypatch, tmp_path):
    client, _ = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': '/etc/passwd'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'


def test_mt3_transcribe_rejects_relative_path_traversal(monkeypatch, tmp_path):
    client, _ = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': '../../../etc/passwd'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'


def test_mt3_transcribe_rejects_nonexistent_file(monkeypatch, tmp_path):
    client, _ = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': 'uploads/nonexistent.wav'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'does not exist' in response.json()['error']


def test_mt3_transcribe_uses_model_from_request(monkeypatch, tmp_path):
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1, -0.1, 0.0], 16000)

    class FakeBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            return TranscriptionResult(
                backend='basic_pitch',
                midi_bytes=mt3_main._empty_midi_bytes(),
                notes=[{'start': 0.0, 'end': 0.2, 'pitch': 64, 'velocity': 90}],
            )

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: FakeBackend())

    response = client.post('/transcribe', json={
        'path': 'uploads/sample.wav',
        'task_id': 'task-2',
        'source': 'vocals',
        'model': 'ismir2021',
    })

    assert response.status_code == 200
    assert response.json()['model'] == 'ismir2021'


def test_mt3_transcribe_accepts_audio_path_field(monkeypatch, tmp_path):
    """The 'audio_path' field should be accepted as an alias for 'path'."""
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample2.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1], 16000)

    class FakeBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            return TranscriptionResult(
                backend='basic_pitch',
                midi_bytes=mt3_main._empty_midi_bytes(),
                notes=[{'start': 0.0, 'end': 0.1, 'pitch': 72, 'velocity': 100}],
            )

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: FakeBackend())

    response = client.post('/transcribe', json={
        'audio_path': 'uploads/sample2.wav',
        'task_id': 'task-3',
    })

    assert response.status_code == 200
    assert response.json()['status'] == 'completed'


def test_mt3_transcribe_returns_failed_when_no_path(monkeypatch, tmp_path):
    client, _ = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'path is required' in response.json()['error']


def test_mt3_transcribe_returns_failed_when_backend_dependency_missing(monkeypatch, tmp_path):
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1], 16000)

    class MissingBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            raise BackendDependencyError('basic_pitch backend is not installed')

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: MissingBackend())
    response = client.post('/transcribe', json={'path': 'uploads/sample.wav'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'dependency is unavailable' in response.json()['error']


def test_mt3_transcribe_returns_failed_for_invalid_audio(monkeypatch, tmp_path):
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'invalid.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_text('not-audio')

    class InvalidAudioBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            raise InvalidAudioError('invalid or unreadable audio input')

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: InvalidAudioBackend())
    response = client.post('/transcribe', json={'path': 'uploads/invalid.wav'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'invalid audio input' in response.json()['error']


def test_mt3_transcribe_returns_failed_for_empty_transcription(monkeypatch, tmp_path):
    client, mt3_main = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'empty.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1], 16000)

    class EmptyBackend:
        def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
            raise EmptyTranscriptionError('transcription produced no note events')

    monkeypatch.setattr(mt3_main, 'get_backend', lambda name: EmptyBackend())
    response = client.post('/transcribe', json={'path': 'uploads/empty.wav'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'no note events' in response.json()['error']
