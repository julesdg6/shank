import json
from pathlib import Path

import pytest

import api.mcp_server as mcp_server


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode('utf-8')


def test_request_json_posts_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured['url'] = req.full_url
        captured['method'] = req.get_method()
        captured['content_type'] = req.headers.get('Content-type')
        captured['body'] = req.data
        captured['timeout'] = timeout
        return _FakeResponse({'ok': True})

    monkeypatch.setattr(mcp_server.request, 'urlopen', fake_urlopen)

    result = mcp_server._request_json(
        'POST',
        '/tasks/url',
        api_base_url='http://localhost:8088',
        payload={'url': 'https://youtu.be/demo'},
    )

    assert result == {'ok': True}
    assert captured['url'] == 'http://localhost:8088/tasks/url'
    assert captured['method'] == 'POST'
    assert captured['content_type'] == 'application/json'
    assert json.loads(captured['body'].decode('utf-8')) == {'url': 'https://youtu.be/demo'}
    assert captured['timeout'] == mcp_server.DEFAULT_TIMEOUT_SECONDS


def test_submit_audio_file_uploads_multipart(monkeypatch, tmp_path):
    wav_path = tmp_path / 'clip.wav'
    wav_path.write_bytes(b'RIFF' + b'\x00' * 20)
    captured = {}

    def fake_urlopen(req, timeout):
        captured['url'] = req.full_url
        captured['method'] = req.get_method()
        captured['content_type'] = req.headers.get('Content-type')
        captured['body'] = req.data
        captured['timeout'] = timeout
        return _FakeResponse({'task_id': '123', 'status': 'pending'})

    monkeypatch.setattr(mcp_server.request, 'urlopen', fake_urlopen)

    result = mcp_server.submit_audio_file(str(wav_path), api_base_url='http://localhost:8088')

    assert result == {'task_id': '123', 'status': 'pending'}
    assert captured['url'] == 'http://localhost:8088/tasks/upload'
    assert captured['method'] == 'POST'
    assert captured['content_type'].startswith('multipart/form-data; boundary=')
    assert b'filename="clip.wav"' in captured['body']
    assert b'RIFF' in captured['body']
    assert captured['timeout'] == 120


def test_submit_audio_file_melody_uses_melody_endpoint(monkeypatch, tmp_path):
    wav_path = tmp_path / 'clip.wav'
    wav_path.write_bytes(b'RIFF' + b'\x00' * 20)
    captured = {}

    def fake_urlopen(req, timeout):
        captured['url'] = req.full_url
        return _FakeResponse({'task_id': '456', 'status': 'pending'})

    monkeypatch.setattr(mcp_server.request, 'urlopen', fake_urlopen)

    mcp_server.submit_audio_file(
        str(wav_path),
        requested_type='melody',
        api_base_url='http://localhost:8088',
    )

    assert captured['url'] == 'http://localhost:8088/tasks/melody'


def test_submit_audio_file_raises_for_missing_file(tmp_path):
    missing = tmp_path / 'missing.wav'
    with pytest.raises(FileNotFoundError):
        mcp_server.submit_audio_file(str(missing))


def test_request_json_rejects_payload_and_raw_body():
    with pytest.raises(ValueError):
        mcp_server._request_json(
            'POST',
            '/tasks/upload',
            payload={'x': 1},
            raw_body=b'body',
        )
