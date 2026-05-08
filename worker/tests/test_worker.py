"""Tests for the SHANK worker ffmpeg normalization pipeline."""
import importlib
import json
import subprocess
import sys
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the worker package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

import worker_loop  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload_task(data_dir: Path, status: str = 'pending', include_file: bool = True) -> tuple[dict, Path]:
    """Create a minimal upload task dict and write it as a JSON file."""
    task_id = str(uuid.uuid4())
    upload_file = data_dir / 'uploads' / f'{task_id}.mp3'
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_bytes(b'\xff\xfb' + b'\x00' * 100)  # MP3 frame-sync header bytes

    task = {
        'task_id': task_id,
        'type': 'upload',
        'source': 'song.mp3',
        'file_path': str(upload_file) if include_file else None,
        'status': status,
    }
    tasks_dir = data_dir / 'tasks'
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / f'{task_id}.json'
    task_file.write_text(json.dumps(task))
    return task, task_file


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Point worker_loop at a temporary DATA_DIR and reload the module."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)
    return tmp_path


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------

def test_normalize_audio_calls_ffmpeg_with_correct_args(tmp_path, monkeypatch):
    """normalize_audio must invoke ffmpeg with the right WAV conversion flags."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    input_p = str(tmp_path / 'in.mp3')
    output_p = str(tmp_path / 'out.wav')
    worker_loop.normalize_audio(input_p, output_p)

    cmd = captured['cmd']
    assert cmd[0] == 'ffmpeg'
    assert '-y' in cmd
    assert input_p in cmd
    assert output_p in cmd
    assert '-ar' in cmd
    assert worker_loop.WAV_SAMPLE_RATE in cmd
    assert '-ac' in cmd
    assert worker_loop.WAV_CHANNELS in cmd
    assert '-c:a' in cmd
    assert worker_loop.WAV_CODEC in cmd


def test_normalize_audio_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    """normalize_audio must raise RuntimeError when ffmpeg returns non-zero."""
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = 'some ffmpeg error'
        return result

    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='ffmpeg failed'):
        worker_loop.normalize_audio('in.mp3', 'out.wav')


# ---------------------------------------------------------------------------
# process_pending_tasks – upload tasks
# ---------------------------------------------------------------------------

def test_pending_upload_task_is_normalized(data_dir):
    """process_pending_tasks must normalize and analyze a pending upload task and mark it done."""
    task, task_file = _make_upload_task(data_dir)
    fake_analysis = {'bpm': 128.0, 'key': 'A minor', 'duration_seconds': 95.75}

    with patch('worker_loop.normalize_audio') as mock_norm, \
         patch('worker_loop.analyze_audio', return_value=fake_analysis):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert 'normalized_path' in updated
    assert updated['normalized_path'].endswith('.wav')
    assert updated['bpm'] == 128.0
    assert updated['key'] == 'A minor'
    assert updated['duration_seconds'] == 95.75
    mock_norm.assert_called_once()


def test_upload_task_normalization_failure_marks_failed(data_dir):
    """If ffmpeg raises during upload normalization, the task must be marked failed."""
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio', side_effect=RuntimeError('ffmpeg failed: oops')):
        worker_loop.process_pending_tasks()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'error' in updated


def test_upload_task_skips_non_pending(data_dir):
    """Upload tasks not in 'pending' status must not be re-processed."""
    task, task_file = _make_upload_task(data_dir, status='completed')

    with patch('worker_loop.normalize_audio') as mock_norm:
        count = worker_loop.process_pending_tasks()

    mock_norm.assert_not_called()
    assert count == 0
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'completed'


def test_upload_task_without_file_path_is_skipped(data_dir):
    """Upload tasks with no file_path should be skipped (file not yet saved)."""
    task, task_file = _make_upload_task(data_dir, include_file=False)

    with patch('worker_loop.normalize_audio') as mock_norm:
        count = worker_loop.process_pending_tasks()

    mock_norm.assert_not_called()
    assert count == 0
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'pending'


def test_upload_task_sets_processing_before_normalization(data_dir):
    """Task status must be written as 'processing' before ffmpeg is invoked."""
    task, task_file = _make_upload_task(data_dir)
    observed_statuses = []

    def capture_and_succeed(input_path, output_path):
        observed_statuses.append(json.loads(task_file.read_text())['status'])

    with patch('worker_loop.normalize_audio', side_effect=capture_and_succeed), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        worker_loop.process_pending_tasks()

    assert 'processing' in observed_statuses


def test_pending_upload_task_records_ace_step_stems_when_enabled(data_dir, monkeypatch):
    """When configured, Ace-step stem extraction output should be saved to the task."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)
    vocals = data_dir / 'stems' / 'vocals.wav'
    drums = data_dir / 'stems' / 'drums.wav'
    bass = data_dir / 'stems' / 'bass.wav'
    other = data_dir / 'stems' / 'other.wav'
    for stem_file in (vocals, drums, bass, other):
        stem_file.parent.mkdir(parents=True, exist_ok=True)
        stem_file.write_bytes(b'fake-wav')

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', return_value={
             'task_id': 'ace-task-1',
             'tracks': {
                 'vocals': str(vocals),
                 'drums': str(drums),
                 'bass': str(bass),
                 'other': str(other),
             },
         }), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 128.0, 'key': 'A minor'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['ace_step_task_id'] == 'ace-task-1'
    assert updated['stems']['vocals'].endswith('vocals.wav')
    assert updated['stems']['drums'].endswith('drums.wav')
    assert updated['stems']['bass'].endswith('bass.wav')
    assert updated['stems']['other'].endswith('other.wav')


def test_pending_upload_task_marks_failed_when_ace_step_fails(data_dir, monkeypatch):
    """If Ace-step stem extraction fails, the task should be marked as failed."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step unavailable')), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        count = worker_loop.process_pending_tasks()

    assert count == 1
    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'Ace-step unavailable' in updated['error']


def test_pending_upload_task_records_mt3_results_when_enabled(data_dir, monkeypatch):
    """When MT3 is enabled, transcription metadata should be persisted on the task."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    fake_mt3 = {
        'enabled': True,
        'status': 'completed',
        'model': 'mt3',
        'output_paths': ['/srv/shank/data/mt3/song.mid'],
        'warnings': [],
        'errors': [],
        'full_mix': {'midi_path': '/srv/shank/data/mt3/song.mid', 'model': 'mt3'},
        'stems': {},
    }
    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.run_mt3_transcription', return_value=fake_mt3), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 128.0, 'key': 'A minor'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['mt3']['status'] == 'completed'
    assert updated['mt3']['full_mix']['midi_path'].endswith('.mid')


def test_pending_upload_task_caches_ace_step_url_stems_for_mt3(data_dir, monkeypatch):
    """Ace-step URL stems should be downloaded to local cache paths before MT3."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('ACE_STEP_STEMS', 'vocals,drums')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)
    task_id = task['task_id']

    class _FakeResponse:
        def __init__(self, payload: bytes):
            self._buf = BytesIO(payload)

        def read(self, size=-1):
            return self._buf.read(size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', return_value={
             'task_id': 'ace-task-1',
             'tracks': {
                 'vocals': '/v1/audio?path=/tmp/vocals.wav',
                 'drums': 'http://ace-step:8001/v1/audio?path=/tmp/drums.wav',
                 'bass': '/v1/audio?path=/tmp/bass.wav',
             },
         }), \
         patch('worker_loop.urllib.request.urlopen', side_effect=[
             _FakeResponse(b'vocals-wav-bytes'),
             _FakeResponse(b'drums-wav-bytes'),
         ]) as mock_urlopen, \
         patch('worker_loop.run_mt3_transcription', return_value={
             'enabled': True,
             'status': 'completed',
             'model': 'mt3',
             'output_paths': [],
             'warnings': [],
             'errors': [],
             'full_mix': {'midi_path': '/srv/shank/data/mt3/song.mid', 'model': 'mt3'},
             'stems': {},
         }) as mock_run_mt3, \
         patch('worker_loop.analyze_audio', return_value={'bpm': 128.0, 'key': 'A minor'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    assert mock_urlopen.call_count == 2
    stems_arg = mock_run_mt3.call_args.kwargs['stems']
    assert set(stems_arg.keys()) == {'vocals', 'drums'}
    assert stems_arg['vocals'].startswith(str(data_dir / 'stems' / task_id))
    assert stems_arg['drums'].startswith(str(data_dir / 'stems' / task_id))
    assert Path(stems_arg['vocals']).read_bytes() == b'vocals-wav-bytes'
    assert Path(stems_arg['drums']).read_bytes() == b'drums-wav-bytes'
    updated = json.loads(task_file.read_text())
    assert set(updated['stems'].keys()) == {'vocals', 'drums'}


def test_mt3_failure_is_non_fatal_by_default(data_dir, monkeypatch):
    """MT3 errors should not fail the task unless strict mode is enabled."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.delenv('MT3_FAIL_TASK_ON_ERROR', raising=False)
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.run_mt3_transcription', return_value={
             'enabled': True,
             'status': 'failed',
             'model': 'mt3',
             'output_paths': [],
             'warnings': [],
             'errors': ['full_mix: MT3 timeout'],
             'full_mix': None,
             'stems': {},
         }), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        worker_loop.process_pending_tasks()

    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['mt3']['status'] == 'failed'
    assert 'MT3 timeout' in updated['mt3']['errors'][0]


def test_mt3_failure_can_fail_task_when_strict_mode_enabled(data_dir, monkeypatch):
    """Strict mode should mark task failed if MT3 fails."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('MT3_FAIL_TASK_ON_ERROR', 'true')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.run_mt3_transcription', return_value={
             'enabled': True,
             'status': 'failed',
             'model': 'mt3',
             'output_paths': [],
             'warnings': [],
             'errors': ['full_mix: bad model load'],
             'error': 'full_mix: bad model load',
             'full_mix': None,
             'stems': {},
         }), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        worker_loop.process_pending_tasks()

    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'bad model load' in updated['error']


# ---------------------------------------------------------------------------
# transcribe_with_mt3 helper
# ---------------------------------------------------------------------------

def test_transcribe_with_mt3_calls_transcribe_with_service(tmp_path, monkeypatch):
    """transcribe_with_mt3 should call transcribe_with_service with the right args."""
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('MT3_MODEL', 'multi_instrument')
    monkeypatch.setenv('MT3_TIMEOUT', '600')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    fake_result = {
        'source': 'full_mix',
        'model': 'multi_instrument',
        'midi_path': '/srv/data/mt3/task1/task1__full_mix.mid',
        'completed_at': '2026-01-01T00:00:00+00:00',
    }
    task_id = str(uuid.uuid4())
    audio_path = str(tmp_path / f'{task_id}.wav')

    with patch('worker_loop.transcribe_with_service', return_value=fake_result) as mock_svc:
        result = worker_loop.transcribe_with_mt3(audio_path, task_id)

    mock_svc.assert_called_once()
    call_kwargs = mock_svc.call_args
    assert call_kwargs.kwargs['audio_path'] == audio_path
    assert call_kwargs.kwargs['task_id'] == task_id
    assert call_kwargs.kwargs['source'] == 'full_mix'
    assert result == fake_result


def test_transcribe_with_mt3_passes_custom_source_name(tmp_path, monkeypatch):
    """transcribe_with_mt3 should forward a custom source_name to the service."""
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    task_id = str(uuid.uuid4())
    with patch('worker_loop.transcribe_with_service', return_value={
        'source': 'vocals',
        'model': 'multi_instrument',
        'midi_path': '/tmp/vocals.mid',
        'completed_at': '2026-01-01T00:00:00+00:00',
    }) as mock_svc:
        worker_loop.transcribe_with_mt3('/tmp/vocals.wav', task_id, source_name='vocals')

    assert mock_svc.call_args.kwargs['source'] == 'vocals'


# ---------------------------------------------------------------------------
# run_mt3_transcription – error and completed_at fields
# ---------------------------------------------------------------------------

def test_run_mt3_transcription_sets_error_field_on_failure(tmp_path, monkeypatch):
    """run_mt3_transcription should populate mt3.error (singular) when transcription fails."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    task_id = str(uuid.uuid4())
    with patch('worker_loop.transcribe_with_mt3', side_effect=RuntimeError('service timeout')):
        result = worker_loop.run_mt3_transcription(task_id, '/tmp/audio.wav')

    assert result['status'] == 'failed'
    assert 'error' in result
    assert 'service timeout' in result['error']


def test_run_mt3_transcription_no_error_field_on_success(tmp_path, monkeypatch):
    """run_mt3_transcription should not set mt3.error when transcription succeeds."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    task_id = str(uuid.uuid4())
    fake_transcription = {
        'source': 'full_mix',
        'model': 'multi_instrument',
        'midi_path': '/tmp/out.mid',
        'completed_at': '2026-01-01T00:00:00+00:00',
    }
    with patch('worker_loop.transcribe_with_mt3', return_value=fake_transcription):
        result = worker_loop.run_mt3_transcription(task_id, '/tmp/audio.wav')

    assert result['status'] == 'completed'
    assert 'error' not in result


def test_run_mt3_transcription_uses_only_configured_stems(tmp_path, monkeypatch):
    """run_mt3_transcription should process only configured stem names."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', 'http://shank-mt3:8001')
    monkeypatch.setenv('ACE_STEP_STEMS', 'vocals,drums')
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    task_id = str(uuid.uuid4())
    sources_seen: list[str] = []

    def _fake_transcribe(path, _task_id, source_name='full_mix'):
        sources_seen.append(source_name)
        return {
            'source': source_name,
            'model': 'multi_instrument',
            'midi_path': f'/tmp/{source_name}.mid',
            'completed_at': '2026-01-01T00:00:00+00:00',
        }

    with patch('worker_loop.transcribe_with_mt3', side_effect=_fake_transcribe):
        result = worker_loop.run_mt3_transcription(task_id, '/tmp/audio.wav', stems={
            'vocals': '/tmp/vocals.wav',
            'drums': '/tmp/drums.wav',
            'bass': '/tmp/bass.wav',
        })

    assert result['status'] == 'completed'
    assert sources_seen == ['full_mix', 'vocals', 'drums']
    assert set(result['stems'].keys()) == {'vocals', 'drums'}


def test_resolve_ace_step_stem_file_does_not_send_auth_to_third_party(tmp_path, monkeypatch):
    """Authorization headers should only be sent to the configured Ace-Step host."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('ACE_STEP_API_KEY', 'super-secret')
    importlib.reload(worker_loop)

    class _FakeResponse:
        def __init__(self):
            self._sent = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            if not self._sent:
                self._sent = True
                return b'abc'
            return b''

    seen_auth_headers: list[str | None] = []

    def _fake_urlopen(request, timeout=60):
        seen_auth_headers.append(request.headers.get('Authorization'))
        return _FakeResponse()

    task_id = str(uuid.uuid4())
    with patch('worker_loop.urllib.request.urlopen', side_effect=_fake_urlopen):
        worker_loop._resolve_ace_step_stem_file(task_id, 'vocals', 'http://example.com/vocals.wav')

    assert seen_auth_headers == [None]


def test_resolve_ace_step_stem_file_rejects_oversized_download(tmp_path, monkeypatch):
    """Stem download should fail when the response exceeds ACE_STEP_MAX_DOWNLOAD_BYTES."""
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('ACE_STEP_MAX_DOWNLOAD_BYTES', '4')
    importlib.reload(worker_loop)

    class _FakeResponse:
        def __init__(self):
            self._chunks = [b'ab', b'cde', b'']
            self._idx = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            if self._idx >= len(self._chunks):
                return b''
            chunk = self._chunks[self._idx]
            self._idx += 1
            return chunk

    task_id = str(uuid.uuid4())
    with patch('worker_loop.urllib.request.urlopen', return_value=_FakeResponse()):
        with pytest.raises(RuntimeError, match='exceeded 4 bytes'):
            worker_loop._resolve_ace_step_stem_file(task_id, 'vocals', 'http://ace-step:8001/vocals.wav')
    assert not (tmp_path / 'stems' / task_id / 'vocals.wav').exists()


def test_full_mix_result_contains_completed_at(tmp_path):
    """transcribe_with_service should include completed_at in the result."""
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    fake_payload = {
        'midi_base64': __import__('base64').b64encode(b'MThd').decode(),
        'model': 'multi_instrument',
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        result = mt3_client.transcribe_with_service(
            service_url='http://localhost:8090',
            audio_path='/tmp/audio.wav',
            output_dir=output_dir,
            task_id=task_id,
            model='multi_instrument',
            source='full_mix',
            timeout=60,
        )

    assert 'completed_at' in result
    # Verify it's a valid ISO-8601 UTC timestamp reasonably close to the current time
    from datetime import datetime, timezone, timedelta
    ts = datetime.fromisoformat(result['completed_at'])
    assert ts.tzinfo is not None
    assert abs((ts - datetime.now(timezone.utc)).total_seconds()) < 5


def test_transcribe_with_service_includes_note_stats(tmp_path):
    """Note statistics should be included when MT3 service returns inline notes."""
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    fake_payload = {
        'midi_base64': __import__('base64').b64encode(b'MThd').decode(),
        'model': 'multi_instrument',
        'notes': [
            {'pitch': 48, 'start': 1.0, 'end': 1.5, 'program': 32},
            {'pitch': 72, 'start': 2.0, 'end': 3.5, 'program': 40},
        ],
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        result = mt3_client.transcribe_with_service(
            service_url='http://localhost:8090',
            audio_path='/tmp/audio.wav',
            output_dir=output_dir,
            task_id=task_id,
            model='multi_instrument',
            source='full_mix',
            timeout=60,
        )

    assert result['note_count'] == 2
    assert result['pitch_range'] == {'min': 48, 'max': 72}
    assert result['duration_seconds'] == 2.5
    assert result['program_count'] == 2


# ---------------------------------------------------------------------------
# run_mt3_transcription – disabled path
# ---------------------------------------------------------------------------

def test_run_mt3_transcription_returns_disabled_when_not_enabled(tmp_path, monkeypatch):
    """run_mt3_transcription should return status='disabled' when MT3_ENABLED is false."""
    monkeypatch.delenv('MT3_ENABLED', raising=False)
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    importlib.reload(worker_loop)

    task_id = str(uuid.uuid4())
    result = worker_loop.run_mt3_transcription(task_id, '/tmp/audio.wav')

    assert result['enabled'] is False
    assert result['status'] == 'disabled'
    assert result['full_mix'] is None
    assert result['stems'] == {}
    assert result['errors'] == []


def test_process_pending_task_records_mt3_disabled_status(data_dir, monkeypatch):
    """When MT3 is disabled the task must still complete as 'done' with mt3.status='disabled'."""
    monkeypatch.delenv('MT3_ENABLED', raising=False)
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['mt3']['enabled'] is False
    assert updated['mt3']['status'] == 'disabled'


# ---------------------------------------------------------------------------
# transcribe_with_service – additional edge cases
# ---------------------------------------------------------------------------

def test_transcribe_with_service_uses_midi_path_when_no_bytes(tmp_path):
    """transcribe_with_service should accept a midi_path string when no bytes are provided."""
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    midi_path_str = '/srv/shank/data/mt3/song.mid'
    fake_payload = {
        'model': 'multi_instrument',
        'midi_path': midi_path_str,
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        result = mt3_client.transcribe_with_service(
            service_url='http://localhost:8090',
            audio_path='/tmp/audio.wav',
            output_dir=output_dir,
            task_id=task_id,
            model='multi_instrument',
            source='full_mix',
            timeout=60,
        )

    assert result['midi_path'] == midi_path_str


def test_transcribe_with_service_raises_when_no_midi_output(tmp_path):
    """transcribe_with_service should raise RuntimeError when the service provides no MIDI."""
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    fake_payload = {
        'model': 'multi_instrument',
        # No midi_base64, no midi_path
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        with pytest.raises(RuntimeError, match='did not include MIDI output'):
            mt3_client.transcribe_with_service(
                service_url='http://localhost:8090',
                audio_path='/tmp/audio.wav',
                output_dir=output_dir,
                task_id=task_id,
                model='multi_instrument',
                source='full_mix',
                timeout=60,
            )


def test_transcribe_with_service_unwraps_data_envelope(tmp_path):
    """transcribe_with_service should unwrap a top-level 'data' envelope in the response."""
    import base64
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    inner = {
        'model': 'multi_instrument',
        'midi_base64': base64.b64encode(b'MThd').decode(),
    }
    wrapped = {'data': inner}

    with patch('mt3_client._post_json', return_value=wrapped):
        result = mt3_client.transcribe_with_service(
            service_url='http://localhost:8090',
            audio_path='/tmp/audio.wav',
            output_dir=output_dir,
            task_id=task_id,
            model='multi_instrument',
            source='full_mix',
            timeout=60,
        )

    assert result['model'] == 'multi_instrument'
    assert 'midi_path' in result


def test_transcribe_with_service_uses_notes_path_from_response(tmp_path):
    """transcribe_with_service should store notes_path when the service returns it as a string."""
    import base64
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    notes_path_str = '/srv/shank/data/mt3/task/full_mix.notes.json'
    fake_payload = {
        'model': 'multi_instrument',
        'midi_base64': base64.b64encode(b'MThd').decode(),
        'notes_path': notes_path_str,
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        result = mt3_client.transcribe_with_service(
            service_url='http://localhost:8090',
            audio_path='/tmp/audio.wav',
            output_dir=output_dir,
            task_id=task_id,
            model='multi_instrument',
            source='full_mix',
            timeout=60,
        )

    assert result['notes_path'] == notes_path_str
