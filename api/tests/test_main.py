"""Tests for the SHANK FastAPI endpoints."""
import io
import json
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Return a TestClient with DATA_DIR pointed at a temporary directory."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))

    # Re-import main so that DATA_DIR / UPLOADS_DIR / TASKS_DIR are
    # re-evaluated with the patched environment variable.
    import importlib
    import api.main as main_module  # noqa: PLC0415

    importlib.reload(main_module)
    return TestClient(main_module.app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'online'
    assert data['service'] == 'SHANK API'


# ---------------------------------------------------------------------------
# POST /tasks/upload
# ---------------------------------------------------------------------------

def test_upload_mp3(client, tmp_path):
    audio_bytes = b'\xff\xfb\x90\x00' + b'\x00' * 100  # minimal fake MP3 header
    response = client.post(
        '/tasks/upload',
        files={'file': ('song.mp3', io.BytesIO(audio_bytes), 'audio/mpeg')},
    )
    assert response.status_code == 202
    body = response.json()
    assert body['status'] == 'pending'
    assert 'task_id' in body


def test_upload_wav(client):
    response = client.post(
        '/tasks/upload',
        files={'file': ('track.wav', io.BytesIO(b'RIFF' + b'\x00' * 36), 'audio/wav')},
    )
    assert response.status_code == 202


def test_upload_flac(client):
    response = client.post(
        '/tasks/upload',
        files={'file': ('track.flac', io.BytesIO(b'fLaC' + b'\x00' * 36), 'audio/flac')},
    )
    assert response.status_code == 202


def test_submit_melody_upload(client, tmp_path):
    response = client.post(
        '/tasks/melody',
        files={'file': ('melody.wav', io.BytesIO(b'RIFF' + b'\x00' * 36), 'audio/wav')},
    )
    assert response.status_code == 202
    body = response.json()
    assert body['status'] == 'pending'
    task_id = body['task_id']

    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    assert task_file.exists()
    task = json.loads(task_file.read_text())
    assert task['type'] == 'upload'
    assert task['requested_type'] == 'melody'


def test_upload_invalid_extension(client):
    response = client.post(
        '/tasks/upload',
        files={'file': ('video.mp4', io.BytesIO(b'\x00' * 16), 'video/mp4')},
    )
    assert response.status_code == 400


def test_upload_creates_task_file(client, tmp_path):
    audio_bytes = b'\xff\xfb\x90\x00' + b'\x00' * 100
    response = client.post(
        '/tasks/upload',
        files={'file': ('song.mp3', io.BytesIO(audio_bytes), 'audio/mpeg')},
    )
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    assert task_file.exists()
    task = json.loads(task_file.read_text())
    assert task['task_id'] == task_id
    assert task['type'] == 'upload'
    assert task['status'] == 'pending'
    assert task['source'] == 'song.mp3'


# ---------------------------------------------------------------------------
# POST /tasks/url
# ---------------------------------------------------------------------------

def test_submit_youtube_url(client):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'},
    )
    assert response.status_code == 202
    body = response.json()
    assert body['status'] == 'pending'
    assert 'task_id' in body


def test_submit_youtu_be_shortlink(client):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://youtu.be/dQw4w9WgXcQ'},
    )
    assert response.status_code == 202


def test_submit_non_youtube_url_rejected(client):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://example.com/audio.mp3'},
    )
    assert response.status_code == 422


def test_submit_http_youtube_url_rejected(client):
    """Plain HTTP YouTube URLs must be rejected (HTTPS only)."""
    response = client.post(
        '/tasks/url',
        json={'url': 'http://www.youtube.com/watch?v=dQw4w9WgXcQ'},
    )
    assert response.status_code == 422


def test_submit_url_creates_task_file(client, tmp_path):
    url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    response = client.post('/tasks/url', json={'url': url})
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    assert task_file.exists()
    task = json.loads(task_file.read_text())
    assert task['task_id'] == task_id
    assert task['type'] == 'url'
    assert task['source'] == url
    assert task['status'] == 'pending'


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------

def test_get_task(client):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'},
    )
    task_id = response.json()['task_id']

    status_response = client.get(f'/tasks/{task_id}')
    assert status_response.status_code == 200
    data = status_response.json()
    assert data['task_id'] == task_id
    assert data['status'] == 'pending'


def test_get_task_not_found(client):
    """A valid UUID format that has no corresponding task file returns 404."""
    missing_uuid = str(uuid.uuid4())
    response = client.get(f'/tasks/{missing_uuid}')
    assert response.status_code == 404


def test_get_task_invalid_id_returns_404(client):
    """A non-UUID task_id (including path traversal attempts) returns 404."""
    for bad_id in ('nonexistent-task-id', '../../etc/passwd', 'not-a-uuid-at-all'):
        response = client.get(f'/tasks/{bad_id}')
        assert response.status_code == 404, f"Expected 404 for task_id={bad_id!r}"


def test_upload_oversized_file_rejected(client, monkeypatch):
    """Files larger than MAX_UPLOAD_SIZE must be rejected with 413."""
    import api.main as main_module  # noqa: PLC0415

    monkeypatch.setattr(main_module, 'MAX_UPLOAD_SIZE', 10)
    response = client.post(
        '/tasks/upload',
        files={'file': ('big.mp3', io.BytesIO(b'\xff\xfb' + b'\x00' * 20), 'audio/mpeg')},
    )
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# GET /tasks/completed
# ---------------------------------------------------------------------------

def test_list_completed_tasks_returns_only_done_tasks(client, tmp_path):
    done_old = str(uuid.uuid4())
    done_new = str(uuid.uuid4())
    pending = str(uuid.uuid4())

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    (tasks_dir / f'{done_old}.json').write_text(json.dumps({
        'task_id': done_old,
        'status': 'done',
        'source': 'older-song.mp3',
        'bpm': 100.0,
        'key': 'C major',
        'duration_seconds': 120.5,
        'completed_at': '2026-01-01T00:00:00+00:00',
    }))
    (tasks_dir / f'{done_new}.json').write_text(json.dumps({
        'task_id': done_new,
        'status': 'done',
        'source': 'newer-song.mp3',
        'bpm': 110.0,
        'key': 'D minor',
        'duration_seconds': 140.2,
        'completed_at': '2026-01-02T00:00:00+00:00',
    }))
    (tasks_dir / f'{pending}.json').write_text(json.dumps({
        'task_id': pending,
        'status': 'pending',
        'source': 'pending-song.mp3',
    }))

    response = client.get('/tasks/completed')
    assert response.status_code == 200
    tasks = response.json()['tasks']
    assert [task['task_id'] for task in tasks] == [done_new, done_old]
    assert all(task['status'] == 'done' for task in tasks)


def test_list_completed_tasks_skips_invalid_json_files(client, tmp_path):
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / 'bad.json').write_text('{ not-valid-json')

    response = client.get('/tasks/completed')
    assert response.status_code == 200
    assert response.json() == {'tasks': []}


def test_download_mt3_midi_for_full_mix(client, tmp_path):
    task_id = str(uuid.uuid4())
    midi_file = tmp_path / 'mt3' / task_id / 'full_mix.mid'
    midi_file.parent.mkdir(parents=True, exist_ok=True)
    midi_file.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': {'midi_path': str(midi_file)},
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/midi/full_mix')
    assert response.status_code == 200
    assert response.content.startswith(b'MThd')


def test_download_mt3_midi_rejects_path_outside_data_dir(client, tmp_path):
    task_id = str(uuid.uuid4())
    outside = tmp_path.parent / 'outside.mid'
    outside.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': {'midi_path': str(outside)},
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/midi/full_mix')
    assert response.status_code == 404


def test_list_task_artifacts_includes_normalized_and_mt3_outputs(client, tmp_path):
    task_id = str(uuid.uuid4())
    normalized = tmp_path / 'normalized' / f'{task_id}.wav'
    midi_file = tmp_path / 'mt3' / task_id / 'full_mix.mid'
    notes_file = tmp_path / 'mt3' / task_id / 'full_mix_notes.json'
    stem_midi = tmp_path / 'mt3' / task_id / 'vocals.mid'
    stem_wav = tmp_path / 'stems' / task_id / 'vocals.wav'

    normalized.parent.mkdir(parents=True, exist_ok=True)
    midi_file.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_bytes(b'RIFF' + b'\x00' * 36)
    midi_file.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    notes_file.write_text(json.dumps({'notes': []}))
    stem_midi.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    stem_wav.parent.mkdir(parents=True, exist_ok=True)
    stem_wav.write_bytes(b'RIFF' + b'\x00' * 36)

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'normalized_path': str(normalized),
        'stems': {'vocals': str(stem_wav)},
        'mt3': {
            'status': 'completed',
            'full_mix': {
                'midi_path': str(midi_file),
                'notes_path': str(notes_file),
            },
            'stems': {
                'vocals': {'midi_path': str(stem_midi)},
            },
        },
    }))

    response = client.get(f'/tasks/{task_id}/artifacts')
    assert response.status_code == 200
    assert response.json() == {
        'artifacts': ['midi', 'normalized_wav', 'notes_json', 'stem_vocals_midi', 'stem_vocals_wav'],
    }


def test_download_task_artifact_by_name(client, tmp_path):
    task_id = str(uuid.uuid4())
    normalized = tmp_path / 'normalized' / f'{task_id}.wav'
    normalized.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_bytes(b'RIFF' + b'\x00' * 36)

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'normalized_path': str(normalized),
    }))

    response = client.get(f'/tasks/{task_id}/artifacts/normalized_wav')
    assert response.status_code == 200
    assert response.content.startswith(b'RIFF')


def test_artifacts_endpoints_reject_outside_data_dir_paths(client, tmp_path):
    task_id = str(uuid.uuid4())
    outside = tmp_path.parent / 'outside.wav'
    outside.write_bytes(b'RIFF' + b'\x00' * 36)

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'normalized_path': str(outside),
    }))

    list_response = client.get(f'/tasks/{task_id}/artifacts')
    assert list_response.status_code == 200
    assert list_response.json() == {'artifacts': []}

    download_response = client.get(f'/tasks/{task_id}/artifacts/normalized_wav')
    assert download_response.status_code == 404


def test_download_mt3_midi_for_stem(client, tmp_path):
    task_id = str(uuid.uuid4())
    stem_midi = tmp_path / 'mt3' / task_id / 'vocals.mid'
    stem_midi.parent.mkdir(parents=True, exist_ok=True)
    stem_midi.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': None,
            'stems': {
                'vocals': {'midi_path': str(stem_midi)},
            },
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/midi/vocals')
    assert response.status_code == 200
    assert response.content.startswith(b'MThd')


def test_download_mt3_midi_returns_404_when_track_not_found(client, tmp_path):
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': None,
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/midi/full_mix')
    assert response.status_code == 404


def test_get_mt3_notes_for_full_mix(client, tmp_path):
    task_id = str(uuid.uuid4())
    notes_file = tmp_path / 'mt3' / task_id / 'full_mix_notes.json'
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    notes_data = [{'pitch': 60, 'start': 0.0, 'end': 1.0}]
    notes_file.write_text(json.dumps(notes_data))
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': {'notes_path': str(notes_file)},
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/notes/full_mix')
    assert response.status_code == 200
    assert response.json() == notes_data


def test_get_mt3_notes_returns_404_when_outside_data_dir(client, tmp_path):
    task_id = str(uuid.uuid4())
    outside = tmp_path.parent / 'outside_notes.json'
    outside.write_text(json.dumps([{'pitch': 60}]))
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': {'notes_path': str(outside)},
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/notes/full_mix')
    assert response.status_code == 404


def test_get_mt3_notes_returns_404_when_no_notes_path(client, tmp_path):
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'mt3': {
            'status': 'completed',
            'full_mix': {'midi_path': '/some/path.mid'},
            'stems': {},
        },
    }))

    response = client.get(f'/tasks/{task_id}/mt3/notes/full_mix')
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /stem-backend/status
# ---------------------------------------------------------------------------

def test_stem_backend_status_returns_expected_shape(client, monkeypatch):
    """The /stem-backend/status endpoint must return the correct structure."""
    monkeypatch.delenv('ACE_STEP_API_URL', raising=False)
    monkeypatch.delenv('STEM_BACKEND', raising=False)
    import importlib
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    from fastapi.testclient import TestClient
    c = TestClient(main_module.app)

    response = c.get('/stem-backend/status')
    assert response.status_code == 200
    data = response.json()
    assert 'configured_backend' in data
    assert 'active_backend' in data
    assert 'acestep' in data
    assert 'demucs' in data
    assert isinstance(data['acestep']['configured'], bool)
    assert isinstance(data['acestep']['healthy'], bool)
    assert isinstance(data['demucs']['available'], bool)


def test_stem_backend_status_active_none_when_no_backend_configured(client, monkeypatch):
    """active_backend must be 'none' when no backend is reachable."""
    monkeypatch.delenv('ACE_STEP_API_URL', raising=False)
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    import importlib
    import shutil
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    with patch.object(shutil, 'which', return_value=None):
        c = TestClient(main_module.app)
        response = c.get('/stem-backend/status')

    assert response.status_code == 200
    data = response.json()
    assert data['active_backend'] == 'none'
    assert data['acestep']['configured'] is False
    assert data['demucs']['available'] is False
