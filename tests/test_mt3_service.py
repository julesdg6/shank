import importlib
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient


def _load_client(monkeypatch, tmp_path):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('MT3_MODEL', 'multi_instrument')
    import services.mt3.main as mt3_main
    importlib.reload(mt3_main)
    return TestClient(mt3_main.app)


def test_mt3_health(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'mt3'}


def test_mt3_transcribe_writes_artifacts_in_data_dir(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1, -0.1, 0.0], 16000)

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
    assert payload['model'] == 'multi_instrument'
    assert midi_path.exists()
    assert notes_path.exists()
    assert midi_path.suffix == '.mid'
    assert notes_path.suffix == '.json'
    assert notes_path.read_text().strip() == '[]'


def test_mt3_transcribe_rejects_path_outside_data_dir(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': '/etc/passwd'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'


def test_mt3_transcribe_rejects_relative_path_traversal(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': '../../../etc/passwd'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'


def test_mt3_transcribe_rejects_nonexistent_file(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={'path': 'uploads/nonexistent.wav'})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'does not exist' in response.json()['error']


def test_mt3_transcribe_uses_model_from_request(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1, -0.1, 0.0], 16000)

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
    client = _load_client(monkeypatch, tmp_path)
    audio_path = tmp_path / 'uploads' / 'sample2.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), [0.0, 0.1], 16000)

    response = client.post('/transcribe', json={
        'audio_path': 'uploads/sample2.wav',
        'task_id': 'task-3',
    })

    assert response.status_code == 200
    assert response.json()['status'] == 'completed'


def test_mt3_transcribe_returns_failed_when_no_path(monkeypatch, tmp_path):
    client = _load_client(monkeypatch, tmp_path)

    response = client.post('/transcribe', json={})

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert 'path is required' in response.json()['error']
