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
