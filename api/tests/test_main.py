"""Tests for the SHANK FastAPI endpoints."""
import importlib
import io
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import api.main as main_module
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
    response = client.get('/', headers={'accept': 'application/json'})
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'online'
    assert data['service'] == 'SHANK API'


def test_root_serves_dashboard_html_for_browser_requests(client):
    response = client.get('/', headers={'accept': 'text/html,application/xhtml+xml'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert 'SHANK — AI Song Analyzer' in response.text
    assert 'Upload Audio File' in response.text
    assert 'Analysis Settings' in response.text
    assert 'Stem Separation' in response.text
    assert 'Reprocess Behaviour' in response.text
    assert 'Spectrogram Preview' in response.text
    assert 'MIDI Piano Roll' in response.text


def test_root_dashboard_shows_container_stem_download_command(client):
    response = client.get('/', headers={'accept': 'text/html,application/xhtml+xml'})

    assert response.status_code == 200
    assert 'docker compose exec shank python3 scripts/download_stem_models.py' in response.text
    assert 'docker logs shank' in response.text


def test_root_keeps_json_for_non_html_accept_headers(client):
    response = client.get('/', headers={'accept': 'application/xhtml+xml,application/json'})
    assert response.status_code == 200
    assert response.json() == {'status': 'online', 'service': 'SHANK API'}


def test_root_defaults_to_html_when_accept_header_is_missing(client):
    client.headers.pop('accept', None)
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert 'Upload Audio File' in response.text


def test_root_serves_html_for_wildcard_accept_when_json_is_not_preferred(client):
    response = client.get('/', headers={'accept': 'application/json;q=0.5,*/*;q=0.8'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert 'Upload Audio File' in response.text


def test_root_logs_warning_when_dashboard_file_is_missing(client, monkeypatch, caplog, tmp_path):
    missing_ui_dir = tmp_path / 'missing-ui'
    missing_index = missing_ui_dir / 'index.html'
    monkeypatch.setattr(main_module, '_UI_DIR', missing_ui_dir)
    with caplog.at_level('WARNING'):
        response = client.get('/', headers={'accept': 'text/html'})

    assert response.status_code == 200
    assert response.json() == {'status': 'online', 'service': 'SHANK API'}
    assert f'Dashboard HTML requested at / but {missing_index} is missing' in caplog.text


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
    assert task['enable_mt3'] is True


def test_upload_persists_enable_mt3_flag(client, tmp_path):
    response = client.post(
        '/tasks/upload',
        files={'file': ('song.wav', io.BytesIO(b'RIFF' + b'\x00' * 36), 'audio/wav')},
        data={'enable_mt3': 'false'},
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']

    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert task['enable_mt3'] is False


def test_submit_melody_upload_persists_mt3_override(client, tmp_path):
    response = client.post(
        '/tasks/melody',
        files={'file': ('melody.wav', io.BytesIO(b'RIFF' + b'\x00' * 36), 'audio/wav')},
        data={'enable_mt3': 'true'},
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert task['enable_mt3'] is True


def test_upload_persists_mt3_disable_override(client, tmp_path):
    response = client.post(
        '/tasks/upload',
        files={'file': ('song.mp3', io.BytesIO(b'\xff\xfb\x90\x00' + b'\x00' * 100), 'audio/mpeg')},
        data={'enable_mt3': 'false'},
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert task['enable_mt3'] is False


def test_upload_persists_analysis_settings_snapshot(client, tmp_path):
    response = client.post(
        '/tasks/upload',
        files={'file': ('song.wav', io.BytesIO(b'RIFF' + b'\x00' * 36), 'audio/wav')},
        data={
            'enable_mt3': 'true',
            'stem_backend': 'audio_separator',
            'stem_model': 'htdemucs_6s.yaml',
            'stem_device': 'cpu',
            'stem_mode': '6_stem',
        },
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task = json.loads((tmp_path / 'tasks' / f'{task_id}.json').read_text())
    assert task['enable_mt3'] is True
    assert task['stem_backend'] == 'audio_separator'
    assert task['stem_model'] == 'htdemucs_6s.yaml'
    assert task['stem_device'] == 'cpu'
    assert task['stem_mode'] == '6_stem'
    assert task['analysis_config']['midi']['enabled'] is True
    assert task['analysis_config']['stems']['backend'] == 'audio_separator'
    assert task['analysis_config']['stems']['mode'] == '6_stem'


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


def test_submit_url_persists_mt3_override(client, tmp_path):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'enable_mt3': False},
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert task['enable_mt3'] is False


def test_submit_url_persists_mt3_enabled_override(client, tmp_path):
    response = client.post(
        '/tasks/url',
        json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'enable_mt3': True},
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert task['enable_mt3'] is True


def test_submit_url_uses_default_analysis_settings_when_not_provided(client, tmp_path):
    response = client.post('/tasks/url', json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'})
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task_file = tmp_path / 'tasks' / f'{task_id}.json'
    task = json.loads(task_file.read_text())
    assert 'enable_mt3' in task
    assert 'analysis_config' in task


def test_submit_url_persists_shared_analysis_settings(client, tmp_path):
    response = client.post(
        '/tasks/url',
        json={
            'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'enable_mt3': True,
            'stem_backend': 'demucs',
            'stem_device': 'cpu',
            'stem_mode': '4_stem',
        },
    )
    assert response.status_code == 202
    task_id = response.json()['task_id']
    task = json.loads((tmp_path / 'tasks' / f'{task_id}.json').read_text())
    assert task['enable_mt3'] is True
    assert task['stem_backend'] == 'demucs'
    assert task['stem_device'] == 'cpu'
    assert task['analysis_config']['stems']['backend'] == 'demucs'
    assert task['analysis_config']['stems']['mode'] == '4_stem'


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
    results_dir = tmp_path / 'results' / task_id
    results_task_json = results_dir / 'task.json'
    results_analysis_json = results_dir / 'analysis.json'
    results_beatgrid_json = results_dir / 'beatgrid.json'
    results_structure_json = results_dir / 'structure.json'
    waveform_beats_png = results_dir / 'waveform_beats.png'
    tempo_curve_png = results_dir / 'tempo_curve.png'
    beatgraph_png = results_dir / 'beatgraph.png'
    results_mt3_json = results_dir / 'mt3.json'
    results_lyrics_json = results_dir / 'lyrics.json'
    results_credits_json = results_dir / 'credits.json'
    results_song_metadata_json = results_dir / 'song_metadata.json'
    results_artifacts_json = results_dir / 'artifacts.json'

    normalized.parent.mkdir(parents=True, exist_ok=True)
    midi_file.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_bytes(b'RIFF' + b'\x00' * 36)
    midi_file.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    notes_file.write_text(json.dumps({'notes': []}))
    stem_midi.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00')
    stem_wav.parent.mkdir(parents=True, exist_ok=True)
    stem_wav.write_bytes(b'RIFF' + b'\x00' * 36)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_task_json.write_text(json.dumps({'status': 'done'}))
    results_analysis_json.write_text(json.dumps({'full_mix': {'bpm': 120.0}}))
    results_beatgrid_json.write_text(json.dumps({'bpm': 120.0, 'beats': []}))
    results_structure_json.write_text(json.dumps([{'label': 'Intro', 'start_seconds': 0.0}]))
    waveform_beats_png.write_bytes(b'\x89PNG\r\n\x1a\n')
    tempo_curve_png.write_bytes(b'\x89PNG\r\n\x1a\n')
    beatgraph_png.write_bytes(b'\x89PNG\r\n\x1a\n')
    results_mt3_json.write_text(json.dumps({'status': 'completed'}))
    results_lyrics_json.write_text(json.dumps({'plain_lyrics': 'Never gonna give you up'}))
    results_credits_json.write_text(json.dumps({'track_title': 'Never Gonna Give You Up'}))
    results_song_metadata_json.write_text(json.dumps({'enabled': True}))
    results_artifacts_json.write_text(json.dumps({'normalized_wav': str(normalized)}))

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
        'results': {
            'dir': str(results_dir),
            'task_json': str(results_task_json),
            'analysis_json': str(results_analysis_json),
            'beatgrid_json': str(results_beatgrid_json),
            'structure_json': str(results_structure_json),
            'waveform_beats_png': str(waveform_beats_png),
            'tempo_curve_png': str(tempo_curve_png),
            'beatgraph_png': str(beatgraph_png),
            'mt3_json': str(results_mt3_json),
            'lyrics_json': str(results_lyrics_json),
            'credits_json': str(results_credits_json),
            'song_metadata_json': str(results_song_metadata_json),
            'artifacts_json': str(results_artifacts_json),
        },
    }))

    response = client.get(f'/tasks/{task_id}/artifacts')
    assert response.status_code == 200
    assert response.json() == {
        'artifacts': [
            'beatgraph_png',
            'beatgrid_json',
            'credits_json',
            'lyrics_json',
            'midi',
            'normalized_wav',
            'notes_json',
            'results_analysis_json',
            'results_artifacts_json',
            'results_mt3_json',
            'results_task_json',
            'song_metadata_json',
            'stem_vocals_midi',
            'stem_vocals_wav',
            'structure_json',
            'tempo_curve_png',
            'waveform_beats_png',
        ],
    }

    beatgrid_response = client.get(f'/tasks/{task_id}/artifacts/beatgrid_json')
    assert beatgrid_response.status_code == 200
    assert beatgrid_response.json() == {'bpm': 120.0, 'beats': []}


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


def test_loop_exports_endpoint_prepares_audio_midi_and_zip(client, tmp_path, monkeypatch):
    task_id = str(uuid.uuid4())
    normalized = tmp_path / 'normalized' / f'{task_id}.wav'
    bass_stem = tmp_path / 'stems' / task_id / 'bass.wav'
    notes_file = tmp_path / 'mt3' / task_id / 'full_mix.notes.json'
    normalized.parent.mkdir(parents=True, exist_ok=True)
    bass_stem.parent.mkdir(parents=True, exist_ok=True)
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_bytes(b'RIFF' + b'\x00' * 36)
    bass_stem.write_bytes(b'RIFF' + b'\x00' * 36)
    notes_file.write_text(json.dumps([
        {'pitch': 72, 'start': 0.2, 'end': 1.0, 'velocity': 100},
        {'pitch': 48, 'start': 0.3, 'end': 1.2, 'velocity': 100},
    ]))

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'source': 'track.wav',
        'bpm': 128.0,
        'key': 'C minor',
        'normalized_path': str(normalized),
        'stems': {'bass': str(bass_stem)},
        'analysis': {
            'full_mix': {
                'beatgrid': {
                    'beats': [{'index': idx + 1, 'time': float(idx) * 0.5} for idx in range(48)],
                },
                'chords': {
                    'segments': [{'symbol': 'Cm', 'start_seconds': 0.0, 'end_seconds': 8.0}],
                },
            },
        },
        'mt3': {
            'status': 'completed',
            'full_mix': {'notes_path': str(notes_file)},
            'stems': {},
        },
    }))

    def fake_ffmpeg(cmd, capture_output, text):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b'RIFF' + b'\x00' * 36)
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr(main_module.subprocess, 'run', fake_ffmpeg)

    response = client.get(f'/tasks/{task_id}/loops?start_bar=1&bars=4')
    assert response.status_code == 200
    body = response.json()
    clip_ids = {clip['clip_id'] for clip in body['clips']}
    assert 'fullmix_wav' in clip_ids
    assert 'bass_wav' in clip_ids
    assert 'melody_midi' in clip_ids
    assert 'bass_midi' in clip_ids
    assert 'chord_midi' in clip_ids
    assert body['zip_url'].endswith('start_bar=1&bars=4')

    zip_response = client.get(body['zip_url'])
    assert zip_response.status_code == 200
    assert zip_response.headers['content-type'].startswith('application/zip')

    first_clip = body['clips'][0]
    clip_response = client.get(first_clip['download_url'])
    assert clip_response.status_code == 200


def test_loop_exports_rejects_invalid_bar_length(client, tmp_path):
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
        'analysis': {'full_mix': {'beatgrid': {'beats': [{'index': idx + 1, 'time': float(idx)} for idx in range(16)]}}},
    }))

    response = client.get(f'/tasks/{task_id}/loops?start_bar=1&bars=3')
    assert response.status_code == 400


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
    assert 'audio_separator' in data
    assert 'demucs' in data
    assert isinstance(data['acestep']['configured'], bool)
    assert isinstance(data['acestep']['healthy'], bool)
    assert isinstance(data['audio_separator']['available'], bool)
    assert isinstance(data['audio_separator']['model_ready'], bool)
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
    with (
        patch.object(shutil, 'which', return_value=None),
        patch.object(main_module.importlib.util, 'find_spec', return_value=None),
    ):
        c = TestClient(main_module.app)
        response = c.get('/stem-backend/status')

    assert response.status_code == 200
    data = response.json()
    assert data['active_backend'] == 'none'
    assert data['acestep']['configured'] is False
    assert data['demucs']['available'] is False


def test_stem_backend_status_prefers_healthy_ace_step(client, monkeypatch):
    """A configured, reachable Ace-Step service should be reported as active."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('ACE_STEP_API_KEY', 'secret-token')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    import api.main as main_module  # noqa: PLC0415 - import inside test so monkey-patched env can be reloaded
    importlib.reload(main_module)

    mock_urlopen_response = MagicMock()

    with (
        patch.object(main_module.urllib.request, 'urlopen', return_value=mock_urlopen_response) as mock_urlopen,
        patch.object(shutil, 'which', return_value=None),
    ):
        c = TestClient(main_module.app)
        response = c.get('/stem-backend/status')

    assert response.status_code == 200
    data = response.json()
    assert data['active_backend'] == 'acestep'
    assert data['acestep']['configured'] is True
    assert data['acestep']['healthy'] is True
    assert data['acestep']['url'] == 'http://ace-step:8001'
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == 'http://ace-step:8001'
    assert request.get_header('Authorization') == 'Bearer secret-token'


def test_stem_backend_status_audio_separator_active_when_available_and_ready(client, monkeypatch, tmp_path):
    model_dir = tmp_path / 'models' / 'separator'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'htdemucs_ft.yaml').write_text('version: 1\n')
    (model_dir / 'htdemucs_ft.ckpt').write_bytes(b'\0' * (main_module._MODEL_WEIGHT_MIN_BYTES + 1))
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL_DIR', str(model_dir))
    monkeypatch.setenv('STEM_BACKEND', 'audio_separator')
    monkeypatch.delenv('ACE_STEP_API_URL', raising=False)

    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)
    with patch.object(reloaded_main.importlib.util, 'find_spec', return_value=object()):
        c = TestClient(reloaded_main.app)
        response = c.get('/stem-backend/status')

    assert response.status_code == 200
    data = response.json()
    assert data['active_backend'] == 'audio_separator'
    assert data['audio_separator']['available'] is True
    assert data['audio_separator']['model_exists'] is True
    assert data['audio_separator']['model_ready'] is True
    assert data['audio_separator']['config_only'] is False


# ---------------------------------------------------------------------------
# GET /mt3/status
# ---------------------------------------------------------------------------

def test_mt3_status_reports_available(client, monkeypatch):
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://127.0.0.1:8090')
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'mt3')
    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)
    c = TestClient(reloaded_main.app)

    response = c.get('/mt3/status')
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is True
    assert data['state'] == 'available'
    assert data['reason'] == 'ok'
    assert data['enabled'] is True
    assert data['backend'] == 'mt3'
    assert 'available' in data['reason_detail'].lower()


def test_mt3_status_reports_disabled_when_mt3_off(client, monkeypatch):
    monkeypatch.setenv('MT3_ENABLED', 'false')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://127.0.0.1:8090')
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'mt3')
    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)
    c = TestClient(reloaded_main.app)

    response = c.get('/mt3/status')
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is False
    assert data['state'] == 'unavailable'
    assert data['reason'] == 'transcription_disabled'
    assert 'disabled' in data['reason_detail']


def test_mt3_status_reports_backend_disabled_reason(client, monkeypatch):
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://127.0.0.1:8090')
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'disabled')
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    response = c.get('/mt3/status')
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is False
    assert data['state'] == 'unavailable'
    assert data['reason'] == 'backend_disabled'
    assert 'disabled' in data['reason_detail']


def test_mt3_status_reports_service_unconfigured_reason(client, monkeypatch):
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.delenv('MT3_SERVICE_URL', raising=False)
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'mt3')
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    response = c.get('/mt3/status')
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is False
    assert data['state'] == 'unavailable'
    assert data['reason'] == 'service_unconfigured'
    assert 'not configured' in data['reason_detail']


def test_mt3_status_basic_pitch_available_without_service_url(client, monkeypatch):
    """basic_pitch runs in-process and must be available even without MT3_SERVICE_URL."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.delenv('MT3_SERVICE_URL', raising=False)
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'basic_pitch')
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    response = c.get('/mt3/status')
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is True
    assert data['state'] == 'available'
    assert data['backend'] == 'basic_pitch'
    assert data['backend_display'] == 'Basic Pitch'
    assert 'Basic Pitch' in data['reason_detail']


def test_mt3_status_basic_pitch_shows_backend_display(client, monkeypatch):
    """backend_display must be returned and use friendly name for basic_pitch."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.delenv('MT3_SERVICE_URL', raising=False)
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'basic_pitch')
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    response = c.get('/mt3/status')
    data = response.json()
    assert data['backend_display'] == 'Basic Pitch'
    assert 'MT3' not in data['reason_detail']


def test_analysis_settings_returns_defaults_and_availability(client, monkeypatch, tmp_path):
    model_dir = tmp_path / 'models' / 'separator'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'htdemucs_ft.yaml').write_text('version: 1\n')
    (model_dir / 'htdemucs_ft.ckpt').write_bytes(b'\0' * (main_module._MODEL_WEIGHT_MIN_BYTES + 1))
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL_DIR', str(model_dir))
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL', 'htdemucs_ft.yaml')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('TRANSCRIPTION_BACKEND', 'basic_pitch')
    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)

    with (
        patch.object(reloaded_main.importlib.util, 'find_spec', return_value=object()),
        patch.object(shutil, 'which', side_effect=lambda binary: '/usr/bin/nvidia-smi' if binary == 'nvidia-smi' else None),
    ):
        c = TestClient(reloaded_main.app)
        response = c.get('/analysis/settings')

    assert response.status_code == 200
    data = response.json()
    assert data['defaults']['midi']['selection'] == 'auto'
    assert data['defaults']['stems']['backend'] == 'auto'
    assert data['stem_backends']['active_backend'] == 'audio_separator'
    assert any(model['name'] == 'htdemucs_ft.yaml' for model in data['available_models'])
    assert data['devices']['cuda_available'] is True


# ---------------------------------------------------------------------------
# /api/models/*
# ---------------------------------------------------------------------------

def test_models_status_reports_missing_by_default(client, monkeypatch, tmp_path):
    model_dir = tmp_path / 'models' / 'separator'
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL_DIR', str(model_dir))
    import api.main as main_module  # noqa: PLC0415
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    response = c.get('/api/models/status')
    assert response.status_code == 200
    data = response.json()
    assert data['models_ready'] is False
    assert data['status'] in {'not_found', 'idle'}
    assert data['models']['htdemucs_ft.yaml']['exists'] is False


def test_models_status_reports_ready_when_model_exists(client, monkeypatch, tmp_path):
    model_dir = tmp_path / 'models' / 'separator'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'htdemucs_ft.yaml').write_text('version: 1\n')
    (model_dir / 'htdemucs_ft.ckpt').write_bytes(b'\0' * (main_module._MODEL_WEIGHT_MIN_BYTES + 1))
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL_DIR', str(model_dir))
    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)
    c = TestClient(reloaded_main.app)

    response = c.get('/api/models/status')
    assert response.status_code == 200
    data = response.json()
    assert data['models_ready'] is True
    assert data['progress_percent'] == 100
    assert data['models']['htdemucs_ft.yaml']['exists'] is True
    assert data['models']['htdemucs_ft.yaml']['ready'] is True
    assert data['models']['htdemucs_ft.yaml']['config_only'] is False


def test_models_status_reports_config_only_when_yaml_exists_without_weights(client, monkeypatch, tmp_path):
    model_dir = tmp_path / 'models' / 'separator'
    model_dir.mkdir(parents=True, exist_ok=True)
    # Keep this file exactly one byte below the config-only threshold.
    (model_dir / 'htdemucs_ft.yaml').write_text('x' * (main_module._MODEL_CONFIG_MIN_BYTES - 1))
    monkeypatch.setenv('AUDIO_SEPARATOR_MODEL_DIR', str(model_dir))
    import api.main as reloaded_main  # noqa: PLC0415
    importlib.reload(reloaded_main)
    c = TestClient(reloaded_main.app)

    response = c.get('/api/models/status')
    assert response.status_code == 200
    data = response.json()
    assert data['models_ready'] is False
    assert data['models']['htdemucs_ft.yaml']['exists'] is True
    assert data['models']['htdemucs_ft.yaml']['ready'] is False
    assert data['models']['htdemucs_ft.yaml']['config_only'] is True


def test_models_download_endpoint_forwards_six_stems(client, monkeypatch):
    captured = {}

    def fake_start(*, six_stems: bool, model_dir: str | None):
        captured['six_stems'] = six_stems
        captured['model_dir'] = model_dir
        return {'started': True, 'status': 'downloading', 'models_ready': False}

    monkeypatch.setattr(main_module, '_start_model_download', fake_start)

    response = client.post('/api/models/download?six_stems=true')
    assert response.status_code == 200
    assert response.json()['started'] is True
    assert captured == {'six_stems': True, 'model_dir': None}


def test_models_download_endpoint_rejects_custom_model_dir(client):
    response = client.post('/api/models/download?model_dir=models/separator')
    assert response.status_code == 400


def test_models_download_returns_500_when_script_missing(client, monkeypatch):
    original_is_file = main_module.Path.is_file

    def _missing_script(self: main_module.Path) -> bool:
        if 'download_stem_models' in str(self):
            return False
        return original_is_file(self)

    monkeypatch.setattr(main_module.Path, 'is_file', _missing_script)
    response = client.post('/api/models/download')
    assert response.status_code == 500
    assert 'missing' in response.json()['detail'].lower()
    # State must not have been mutated — no download was started.
    assert not main_module._MODEL_DOWNLOAD_STATE.get('is_downloading')


def test_models_cancel_returns_no_active_download(client):
    response = client.post('/api/models/cancel')
    assert response.status_code == 200
    assert response.json()['cancelled'] is False


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/chords
# ---------------------------------------------------------------------------

def test_get_task_chords_returns_chord_data(client, tmp_path):
    """GET /tasks/{task_id}/chords must return the chords dict from the task."""
    task_id = str(uuid.uuid4())
    chords_data = {
        'segments': [
            {
                'symbol': 'Am',
                'root': 'A',
                'quality': 'minor',
                'confidence': 0.72,
                'start_seconds': 0.0,
                'end_seconds': 3.8,
            },
            {
                'symbol': 'F',
                'root': 'F',
                'quality': 'major',
                'confidence': 0.68,
                'start_seconds': 3.8,
                'end_seconds': 7.6,
            },
        ],
        'progression': ['Am', 'F'],
    }
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'chords': chords_data,
    }))

    response = client.get(f'/tasks/{task_id}/chords')
    assert response.status_code == 200
    assert response.json() == chords_data


def test_get_task_chords_returns_404_when_no_chords(client, tmp_path):
    """GET /tasks/{task_id}/chords must return 404 when the task has no chord data."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'bpm': 120.0,
    }))

    response = client.get(f'/tasks/{task_id}/chords')
    assert response.status_code == 404


def test_get_task_chords_returns_404_for_unknown_task(client):
    """GET /tasks/{task_id}/chords must return 404 for a nonexistent task."""
    response = client.get(f'/tasks/{uuid.uuid4()}/chords')
    assert response.status_code == 404


def test_get_task_chords_returns_empty_when_disabled_backend(client, tmp_path):
    """When chords was captured with disabled backend, the endpoint returns empty segments."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'chords': {'segments': [], 'progression': []},
    }))

    response = client.get(f'/tasks/{task_id}/chords')
    assert response.status_code == 200
    assert response.json() == {'segments': [], 'progression': []}


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/beatgrid
# ---------------------------------------------------------------------------

def test_get_task_beatgrid_returns_beatgrid_and_detection_data(client, tmp_path):
    """GET /tasks/{task_id}/beatgrid must return beatgrid and beat_detection dicts."""
    task_id = str(uuid.uuid4())
    beatgrid_data = {
        'bpm': 128.02,
        'first_beat_seconds': 0.423,
        'beats': [
            {'index': 1, 'time': 0.423},
            {'index': 2, 'time': 0.892},
            {'index': 3, 'time': 1.361},
        ],
    }
    beat_detection_data = {
        'engine': 'mixxx',
        'mode': 'constant_tempo',
        'first_beat_seconds': 0.423,
        'beat_count': 3,
        'confidence': 0.95,
    }
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'beatgrid': beatgrid_data,
        'beat_detection': beat_detection_data,
    }))

    response = client.get(f'/tasks/{task_id}/beatgrid')
    assert response.status_code == 200
    body = response.json()
    assert body['beatgrid'] == beatgrid_data
    assert body['beat_detection'] == beat_detection_data


def test_get_task_beatgrid_returns_beatgrid_without_detection_when_absent(client, tmp_path):
    """GET /tasks/{task_id}/beatgrid returns only beatgrid when beat_detection is missing."""
    task_id = str(uuid.uuid4())
    beatgrid_data = {
        'bpm': 120.0,
        'first_beat_seconds': 0.5,
        'beats': [{'index': 1, 'time': 0.5}],
    }
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'beatgrid': beatgrid_data,
    }))

    response = client.get(f'/tasks/{task_id}/beatgrid')
    assert response.status_code == 200
    body = response.json()
    assert body['beatgrid'] == beatgrid_data
    assert 'beat_detection' not in body


def test_get_task_beatgrid_returns_variable_tempo_grid(client, tmp_path):
    """GET /tasks/{task_id}/beatgrid must surface variable-tempo beat grids correctly."""
    task_id = str(uuid.uuid4())
    beatgrid_data = {
        'bpm': 128.0,
        'first_beat_seconds': 0.42,
        'mode': 'variable_tempo',
        'beats': [
            {'index': 1, 'time': 0.42, 'local_bpm': 127.8},
            {'index': 2, 'time': 0.89, 'local_bpm': 128.1},
        ],
    }
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'beatgrid': beatgrid_data,
    }))

    response = client.get(f'/tasks/{task_id}/beatgrid')
    assert response.status_code == 200
    body = response.json()
    assert body['beatgrid']['mode'] == 'variable_tempo'
    assert body['beatgrid']['beats'][0]['local_bpm'] == 127.8


def test_get_task_beatgrid_returns_404_when_no_beatgrid(client, tmp_path):
    """GET /tasks/{task_id}/beatgrid must return 404 when the task has no beatgrid data."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'bpm': 120.0,
    }))

    response = client.get(f'/tasks/{task_id}/beatgrid')
    assert response.status_code == 404


def test_get_task_beatgrid_returns_404_for_unknown_task(client):
    """GET /tasks/{task_id}/beatgrid must return 404 for a nonexistent task."""
    response = client.get(f'/tasks/{uuid.uuid4()}/beatgrid')
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/cue-points  &  export endpoints
# ---------------------------------------------------------------------------

_SAMPLE_CUE_POINTS = [
    {'name': 'Intro', 'time_seconds': 0.5, 'hot_cue': 0, 'color': '#28E614'},
    {'name': 'Verse', 'time_seconds': 16.0, 'hot_cue': 1, 'color': '#F8821A'},
    {'name': 'Chorus', 'time_seconds': 48.0, 'hot_cue': 2, 'color': '#C02626'},
    {'name': 'Breakdown', 'time_seconds': 96.0, 'hot_cue': 3, 'color': '#1F9BFA'},
    {'name': 'Outro', 'time_seconds': 220.0, 'hot_cue': 4, 'color': '#FAC000'},
]


def _make_cue_task(tmp_path, task_id: str, cue_points=None):
    """Write a minimal task JSON file containing cue point data."""
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'title': 'Test Track',
        'artist': 'Test Artist',
        'analysis': {
            'full_mix': {
                'bpm': 128.0,
                'key': 'A minor',
                'duration_seconds': 240.0,
                'cue_points': cue_points if cue_points is not None else _SAMPLE_CUE_POINTS,
            },
        },
    }))


def test_get_task_cue_points_returns_cue_data(client, tmp_path):
    """GET /tasks/{task_id}/cue-points must return the hot cue list."""
    task_id = str(uuid.uuid4())
    _make_cue_task(tmp_path, task_id)

    response = client.get(f'/tasks/{task_id}/cue-points')
    assert response.status_code == 200
    data = response.json()
    assert 'cue_points' in data
    cues = data['cue_points']
    assert isinstance(cues, list)
    assert len(cues) == len(_SAMPLE_CUE_POINTS)
    first = cues[0]
    assert first['name'] == 'Intro'
    assert first['time_seconds'] == 0.5
    assert first['hot_cue'] == 0
    assert first['color'].startswith('#')


def test_get_task_cue_points_returns_404_when_no_cue_points(client, tmp_path):
    """GET /tasks/{task_id}/cue-points must return 404 when no cue data is stored."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'bpm': 128.0,
    }))
    response = client.get(f'/tasks/{task_id}/cue-points')
    assert response.status_code == 404


def test_get_task_cue_points_returns_404_for_unknown_task(client):
    """GET /tasks/{task_id}/cue-points must return 404 for a nonexistent task."""
    response = client.get(f'/tasks/{uuid.uuid4()}/cue-points')
    assert response.status_code == 404


def test_export_cue_points_rekordbox_returns_xml(client, tmp_path):
    """Rekordbox export must return XML with POSITION_MARK elements."""
    task_id = str(uuid.uuid4())
    _make_cue_task(tmp_path, task_id)

    response = client.get(f'/tasks/{task_id}/cue-points/export/rekordbox')
    assert response.status_code == 200
    assert 'xml' in response.headers.get('content-type', '')
    body = response.text
    assert 'DJ_PLAYLISTS' in body
    assert 'POSITION_MARK' in body
    assert 'Intro' in body
    assert 'Verse' in body
    assert 'Chorus' in body
    assert f'attachment; filename="{task_id}_cue_points_rekordbox.xml"' in response.headers.get('content-disposition', '')


def test_export_cue_points_traktor_returns_nml(client, tmp_path):
    """Traktor export must return NML XML with CUE_V2 elements in milliseconds."""
    task_id = str(uuid.uuid4())
    _make_cue_task(tmp_path, task_id)

    response = client.get(f'/tasks/{task_id}/cue-points/export/traktor')
    assert response.status_code == 200
    assert 'xml' in response.headers.get('content-type', '')
    body = response.text
    assert 'NML' in body
    assert 'CUE_V2' in body
    # Intro at 0.5 s → 500 ms
    assert '500.000000' in body
    assert 'Intro' in body
    assert f'attachment; filename="{task_id}_cue_points_traktor.nml"' in response.headers.get('content-disposition', '')


def test_export_cue_points_mixxx_returns_xml(client, tmp_path):
    """Mixxx export must return XML with Cue elements in sample positions."""
    task_id = str(uuid.uuid4())
    _make_cue_task(tmp_path, task_id)

    response = client.get(f'/tasks/{task_id}/cue-points/export/mixxx')
    assert response.status_code == 200
    assert 'xml' in response.headers.get('content-type', '')
    body = response.text
    assert 'Mixxx-Library' in body
    assert '<Cue' in body
    assert 'Intro' in body
    # Intro at 0.5 s × 44100 = 22050 samples
    assert '22050' in body
    assert f'attachment; filename="{task_id}_cue_points_mixxx.xml"' in response.headers.get('content-disposition', '')


def test_export_cue_points_returns_404_when_no_cue_data(client, tmp_path):
    """Export endpoints must return 404 when no cue points are available."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
    }))
    for endpoint in ('rekordbox', 'traktor', 'mixxx'):
        response = client.get(f'/tasks/{task_id}/cue-points/export/{endpoint}')
        assert response.status_code == 404, f'Expected 404 for {endpoint}'


def test_rekordbox_xml_contains_correct_rgb_values(client, tmp_path):
    """Rekordbox POSITION_MARK must carry split RGB components for the cue colour."""
    task_id = str(uuid.uuid4())
    # Single green cue (#28E614 → R=40, G=230, B=20)
    _make_cue_task(tmp_path, task_id, cue_points=[
        {'name': 'Intro', 'time_seconds': 1.0, 'hot_cue': 0, 'color': '#28E614'},
    ])
    response = client.get(f'/tasks/{task_id}/cue-points/export/rekordbox')
    assert response.status_code == 200
    body = response.text
    assert 'Red="40"' in body
    assert 'Green="230"' in body
    assert 'Blue="20"' in body


# ---------------------------------------------------------------------------
# GET /worker/status
# ---------------------------------------------------------------------------

def test_worker_status_offline_when_no_heartbeat(client):
    """Without a heartbeat file the worker must be reported as offline."""
    response = client.get('/worker/status')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'offline'
    assert data['last_heartbeat'] is None


def test_worker_status_online_with_fresh_heartbeat(client, tmp_path):
    """A freshly written heartbeat must cause the worker to be reported online."""
    from datetime import datetime, timezone
    heartbeat_file = tmp_path / '.worker_heartbeat'
    heartbeat_file.write_text(datetime.now(timezone.utc).isoformat())

    response = client.get('/worker/status')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'online'
    assert data['last_heartbeat'] is not None
    assert data['age_seconds'] is not None


def test_worker_status_offline_with_stale_heartbeat(client, tmp_path, monkeypatch):
    """A heartbeat older than the stale threshold must report worker as offline."""
    from datetime import datetime, timezone, timedelta
    heartbeat_file = tmp_path / '.worker_heartbeat'
    old_time = datetime.now(timezone.utc) - timedelta(seconds=3600)
    heartbeat_file.write_text(old_time.isoformat())

    response = client.get('/worker/status')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'offline'


# ---------------------------------------------------------------------------
# GET /doctor
# ---------------------------------------------------------------------------

def test_doctor_status_aggregates_deployment_checks(monkeypatch):
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    monkeypatch.setattr(main_module, 'get_worker_status', lambda: {'status': 'online'})
    monkeypatch.setattr(main_module, 'get_stem_backend_status', lambda: {'configured_backend': 'auto', 'active_backend': 'demucs'})
    monkeypatch.setattr(
        main_module,
        'get_transcription_status',
        lambda: {'backend': 'mt3', 'available': True},
    )
    monkeypatch.setattr(
        main_module,
        '_snapshot_model_download_status',
        lambda: {
            'model_dir': '/models',
            'models_ready': False,
            'models': {
                'htdemucs_ft.yaml': {'exists': True},
                'htdemucs_6s.yaml': {'exists': False},
            },
        },
    )
    monkeypatch.setattr(main_module, '_is_dir_writable', lambda _path: True)
    monkeypatch.setattr(main_module, '_disk_free_gb', lambda _path: 12.34)
    monkeypatch.setattr(shutil, 'which', lambda binary: f'/usr/bin/{binary}')

    response = c.get('/doctor')
    assert response.status_code == 200
    data = response.json()
    assert data['api'] == {'ok': True, 'service': 'SHANK API'}
    assert data['worker']['status'] == 'online'
    assert data['ffmpeg'] == {'available': True, 'path': '/usr/bin/ffmpeg'}
    assert data['yt_dlp'] == {'available': True, 'path': '/usr/bin/yt-dlp'}
    assert data['stem_backend']['active_backend'] == 'demucs'
    assert data['models']['models_ready'] is False
    assert data['models']['found'] == ['htdemucs_ft.yaml']
    assert data['models']['missing'] == ['htdemucs_6s.yaml']
    assert data['transcription'] == {'backend': 'mt3', 'available': True}
    assert data['data_dir']['writable'] is True
    assert data['disk']['free_gb'] == 12.34


def test_doctor_status_reports_missing_binaries(monkeypatch):
    importlib.reload(main_module)
    c = TestClient(main_module.app)

    monkeypatch.setattr(shutil, 'which', lambda _binary: None)

    response = c.get('/doctor')
    assert response.status_code == 200
    data = response.json()
    assert data['ffmpeg']['available'] is False
    assert data['ffmpeg']['path'] is None
    assert data['yt_dlp']['available'] is False
    assert data['yt_dlp']['path'] is None

# POST /tasks/{task_id}/reprocess
# ---------------------------------------------------------------------------

def test_reprocess_url_task_replaces_existing_task_by_default(client, tmp_path):
    """Reprocessing should reset the existing task instead of creating a duplicate."""
    original_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{original_id}.json').write_text(json.dumps({
        'task_id': original_id,
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'done',
        'analysis': {'full_mix': {'bpm': 120.0}},
    }))

    response = client.post(f'/tasks/{original_id}/reprocess', json={'mode': 'all'})
    assert response.status_code == 202
    data = response.json()
    assert data['source_task_id'] == original_id
    assert data['status'] == 'pending'
    assert data['task_id'] == original_id

    updated_task = json.loads((tasks_dir / f'{original_id}.json').read_text())
    assert updated_task['task_id'] == original_id
    assert updated_task['type'] == 'url'
    assert updated_task['source'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    assert updated_task['status'] == 'pending'
    assert updated_task['reprocess_mode'] == 'use_current_replace'
    assert updated_task['reprocess_target'] == 'all'
    assert 'analysis' not in updated_task


def test_reprocess_upload_task_carries_file_path(client, tmp_path):
    """Reprocessing an upload task should carry the existing file_path forward."""
    original_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(tmp_path / 'uploads' / f'{original_id}.mp3')
    (tasks_dir / f'{original_id}.json').write_text(json.dumps({
        'task_id': original_id,
        'type': 'upload',
        'source': 'song.mp3',
        'file_path': file_path,
        'status': 'done',
    }))

    response = client.post(f'/tasks/{original_id}/reprocess', json={'mode': 'audio_analysis'})
    assert response.status_code == 202
    updated_task = json.loads((tasks_dir / f'{original_id}.json').read_text())
    assert updated_task['file_path'] == file_path
    assert updated_task['reprocess_target'] == 'audio_analysis'
    assert updated_task['reprocess_mode'] == 'use_current_replace'


def test_reprocess_not_found_returns_404(client):
    """Reprocessing a non-existent task should return 404."""
    missing = str(uuid.uuid4())
    response = client.post(f'/tasks/{missing}/reprocess', json={})
    assert response.status_code == 404


def test_reprocess_invalid_mode_returns_400(client, tmp_path):
    """An unrecognised reprocess mode should return 400."""
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'done',
    }))

    response = client.post(f'/tasks/{task_id}/reprocess', json={'mode': 'nonexistent_mode'})
    assert response.status_code == 400


def test_reprocess_with_enable_mt3_override(client, tmp_path):
    """Providing enable_mt3 in the reprocess request should override the original."""
    original_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{original_id}.json').write_text(json.dumps({
        'task_id': original_id,
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'done',
        'enable_mt3': False,
    }))

    response = client.post(
        f'/tasks/{original_id}/reprocess',
        json={'mode': 'all', 'enable_mt3': True, 'reprocess_mode': 'use_current_archive'},
    )
    assert response.status_code == 202
    updated_task = json.loads((tasks_dir / f'{original_id}.json').read_text())
    assert updated_task['enable_mt3'] is True
    assert updated_task['analysis_config']['reprocess']['archive_previous'] is True
    assert updated_task['analysis_config']['reprocess']['use_current_settings'] is True
    assert updated_task['archived_analysis']


def test_reprocess_can_reuse_original_settings(client, tmp_path):
    original_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{original_id}.json').write_text(json.dumps({
        'task_id': original_id,
        'type': 'url',
        'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'status': 'done',
        'enable_mt3': False,
        'stem_backend': 'demucs',
        'stem_device': 'cpu',
        'analysis_config': {
            'midi': {'enabled': False, 'backend': 'basic_pitch'},
            'stems': {'enabled': True, 'backend': 'demucs', 'model': None, 'device': 'cpu', 'mode': '4_stem'},
            'reprocess': {'replace_existing': True, 'archive_previous': False, 'use_current_settings': True},
        },
    }))

    response = client.post(
        f'/tasks/{original_id}/reprocess',
        json={
            'mode': 'all',
            'enable_mt3': True,
            'stem_backend': 'audio_separator',
            'reprocess_mode': 'reuse_original_replace',
        },
    )
    assert response.status_code == 202
    updated_task = json.loads((tasks_dir / f'{original_id}.json').read_text())
    assert updated_task['enable_mt3'] is False
    assert updated_task['stem_backend'] == 'demucs'
    assert updated_task['analysis_config']['reprocess']['use_current_settings'] is False


def test_reprocess_all_valid_modes(client, tmp_path):
    """All documented reprocess modes should be accepted."""
    valid_modes = ['all', 'audio_analysis', 'stems', 'midi', 'metadata', 'ai_prompts']
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for mode in valid_modes:
        task_id = str(uuid.uuid4())
        (tasks_dir / f'{task_id}.json').write_text(json.dumps({
            'task_id': task_id,
            'type': 'url',
            'source': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'status': 'done',
        }))
        response = client.post(f'/tasks/{task_id}/reprocess', json={'mode': mode})
        assert response.status_code == 202, f"Expected 202 for mode={mode!r}"


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/fingerprint, GET /tasks/fingerprints, GET /tasks/{task_id}/similar
# ---------------------------------------------------------------------------

def _make_fingerprint_task(tmp_path, task_id, bpm=120.0, key='C major',
                            chord_progression=None, energy_profile=None,
                            status='done'):
    """Helper: write a task file plus a fingerprint.json in results dir."""
    if chord_progression is None:
        chord_progression = ['C', 'Am', 'F', 'G']
    if energy_profile is None:
        energy_profile = [0.1] * 16

    results_dir = tmp_path / 'results' / task_id
    results_dir.mkdir(parents=True, exist_ok=True)
    fp_path = results_dir / 'fingerprint.json'
    fp_path.write_text(json.dumps({
        'bpm': bpm,
        'key': key,
        'chord_progression': chord_progression,
        'energy_profile': energy_profile,
        'spectral_centroid': 5.0,
    }))

    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': status,
        'bpm': bpm,
        'key': key,
        'results': {
            'dir': str(results_dir),
            'fingerprint_json': str(fp_path),
        },
    }))
    return fp_path


def test_get_task_fingerprint_returns_fingerprint(client, tmp_path):
    task_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, task_id, bpm=128.0, key='A minor')
    response = client.get(f'/tasks/{task_id}/fingerprint')
    assert response.status_code == 200
    data = response.json()
    assert data['bpm'] == 128.0
    assert data['key'] == 'A minor'
    assert 'chord_progression' in data
    assert 'energy_profile' in data


def test_get_task_fingerprint_returns_404_when_no_fingerprint(client, tmp_path):
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'bpm': 120.0,
    }))
    response = client.get(f'/tasks/{task_id}/fingerprint')
    assert response.status_code == 404


def test_list_task_fingerprints_returns_completed_only(client, tmp_path):
    done_id = str(uuid.uuid4())
    pending_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, done_id, bpm=130.0, key='D minor')
    _make_fingerprint_task(tmp_path, pending_id, bpm=100.0, key='G major', status='pending')

    response = client.get('/tasks/fingerprints')
    assert response.status_code == 200
    data = response.json()
    assert 'fingerprints' in data
    ids = {fp['task_id'] for fp in data['fingerprints']}
    assert done_id in ids
    assert pending_id not in ids


def test_list_task_fingerprints_includes_task_id(client, tmp_path):
    task_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, task_id)
    response = client.get('/tasks/fingerprints')
    assert response.status_code == 200
    fps = response.json()['fingerprints']
    assert any(fp['task_id'] == task_id for fp in fps)


def test_get_similar_tasks_returns_matches(client, tmp_path):
    target_id = str(uuid.uuid4())
    similar_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, target_id, bpm=128.0, key='A minor',
                           chord_progression=['Am', 'F', 'C', 'G'],
                           energy_profile=[0.1] * 16)
    _make_fingerprint_task(tmp_path, similar_id, bpm=128.0, key='A minor',
                           chord_progression=['Am', 'F', 'C', 'G'],
                           energy_profile=[0.1] * 16)

    response = client.get(f'/tasks/{target_id}/similar')
    assert response.status_code == 200
    data = response.json()
    assert data['task_id'] == target_id
    assert 'matches' in data
    assert len(data['matches']) == 1
    assert data['matches'][0]['task_id'] == similar_id
    assert data['matches'][0]['similarity'] == 100


def test_get_similar_tasks_excludes_self(client, tmp_path):
    task_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, task_id)
    response = client.get(f'/tasks/{task_id}/similar')
    assert response.status_code == 200
    ids = [m['task_id'] for m in response.json()['matches']]
    assert task_id not in ids


def test_get_similar_tasks_returns_404_when_no_fingerprint(client, tmp_path):
    task_id = str(uuid.uuid4())
    tasks_dir = tmp_path / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / f'{task_id}.json').write_text(json.dumps({
        'task_id': task_id,
        'status': 'done',
        'bpm': 120.0,
    }))
    response = client.get(f'/tasks/{task_id}/similar')
    assert response.status_code == 404


def test_get_similar_tasks_sorted_by_similarity(client, tmp_path):
    target_id = str(uuid.uuid4())
    high_id = str(uuid.uuid4())
    low_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, target_id, bpm=128.0, key='A minor',
                           chord_progression=['Am', 'F', 'C', 'G'],
                           energy_profile=[0.1] * 16)
    # High similarity: same key, BPM, chords
    _make_fingerprint_task(tmp_path, high_id, bpm=128.0, key='A minor',
                           chord_progression=['Am', 'F', 'C', 'G'],
                           energy_profile=[0.1] * 16)
    # Low similarity: different BPM, key, chords
    _make_fingerprint_task(tmp_path, low_id, bpm=72.0, key='B major',
                           chord_progression=['B', 'E', 'F#'],
                           energy_profile=[0.9] * 16)

    response = client.get(f'/tasks/{target_id}/similar')
    assert response.status_code == 200
    matches = response.json()['matches']
    assert matches[0]['task_id'] == high_id
    assert matches[0]['similarity'] > matches[1]['similarity']


def test_get_similar_tasks_limit_parameter(client, tmp_path):
    target_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, target_id)
    for _ in range(5):
        other_id = str(uuid.uuid4())
        _make_fingerprint_task(tmp_path, other_id)

    response = client.get(f'/tasks/{target_id}/similar?limit=2')
    assert response.status_code == 200
    assert len(response.json()['matches']) <= 2


def test_get_similar_tasks_invalid_limit(client, tmp_path):
    task_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, task_id)
    response = client.get(f'/tasks/{task_id}/similar?limit=0')
    assert response.status_code == 400


def test_similar_tasks_response_includes_reasons_and_details(client, tmp_path):
    target_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    _make_fingerprint_task(tmp_path, target_id, bpm=128.0, key='C major',
                           chord_progression=['C', 'G'], energy_profile=[0.2] * 16)
    _make_fingerprint_task(tmp_path, other_id, bpm=128.0, key='C major',
                           chord_progression=['C', 'G'], energy_profile=[0.2] * 16)

    response = client.get(f'/tasks/{target_id}/similar')
    assert response.status_code == 200
    match = response.json()['matches'][0]
    assert 'reasons' in match
    assert 'details' in match
    assert isinstance(match['reasons'], list)
    assert isinstance(match['details'], dict)
