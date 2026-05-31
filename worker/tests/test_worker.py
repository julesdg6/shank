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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Make the worker package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mt3_config import DEFAULT_MT3_SERVICE_URL  # noqa: E402
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
    fake_analysis = {
        'bpm': 128.0,
        'key': 'A minor',
        'duration_seconds': 95.75,
        'waveform': [0.1, -0.1],
        'frequency_histogram': [0.2, 0.8],
        'spectrogram_summary': [-12.0, -8.0],
        'loudness_curve': [0.3, 0.5],
        'energy_over_time': [0.09, 0.25],
    }

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
    assert updated['analysis']['full_mix']['frequency_histogram'] == [0.2, 0.8]
    assert updated['analysis']['stems'] == {}
    assert 'results' in updated
    assert Path(updated['results']['dir']).name == task['task_id']
    assert Path(updated['results']['analysis_json']).is_file()
    assert Path(updated['results']['task_json']).is_file()
    assert Path(updated['results']['mt3_json']).is_file()
    assert Path(updated['results']['artifacts_json']).is_file()
    structured_analysis = json.loads(Path(updated['results']['analysis_json']).read_text())
    assert structured_analysis == updated['analysis']
    structured_task = json.loads(Path(updated['results']['task_json']).read_text())
    assert structured_task['status'] == 'done'
    assert structured_task['bpm'] == 128.0
    assert structured_task['key'] == 'A minor'
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

    def fake_analysis_for_path(path):
        stem_name = Path(path).stem
        is_full_mix = Path(path).name == f"{task['task_id']}.wav"
        return {
            'bpm': 128.0 if stem_name != 'drums' else 110.0,
            'key': 'A minor',
            'duration_seconds': 60.0,
            'waveform': [0.1, -0.1],
            'frequency_histogram': [0.2, 0.8] if is_full_mix else [0.6, 0.4],
            'spectrogram_summary': [-12.0, -8.0],
            'loudness_curve': [0.3, 0.5],
            'energy_over_time': [0.09, 0.25],
        }

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
         patch('worker_loop.analyze_audio', side_effect=fake_analysis_for_path):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated['ace_step_task_id'] == 'ace-task-1'
    assert updated['stems']['vocals'].endswith('vocals.wav')
    assert updated['stems']['drums'].endswith('drums.wav')
    assert updated['stems']['bass'].endswith('bass.wav')
    assert updated['stems']['other'].endswith('other.wav')
    assert updated['analysis']['full_mix']['frequency_histogram'] == [0.2, 0.8]
    assert set(updated['analysis']['stems'].keys()) == {'vocals', 'drums', 'bass', 'other'}
    assert updated['analysis']['stems']['drums']['bpm'] == 110.0


def test_extract_track_files_supports_direct_stem_mappings(data_dir, monkeypatch):
    """Ace-step payloads with direct stem key/value maps should be parsed."""
    monkeypatch.setenv('ACE_STEP_STEMS', 'vocals,drums,bass,other')
    importlib.reload(worker_loop)

    tracks = worker_loop._extract_track_files({
        'vocals': '/tmp/vocals.wav',
        'nested': {
            'drums': 'http://ace-step:8001/drums.wav',
        },
        'items': [
            {'stem_name': 'bass', 'audio_url': 'http://ace-step:8001/bass.wav'},
            {'track_name': 'other', 'file_path': '/tmp/other.wav'},
        ],
    })

    assert tracks['vocals'] == '/tmp/vocals.wav'
    assert tracks['drums'] == 'http://ace-step:8001/drums.wav'
    assert tracks['bass'] == 'http://ace-step:8001/bass.wav'
    assert tracks['other'] == '/tmp/other.wav'


def test_separate_stems_with_ace_step_accepts_dict_query_shape(data_dir, monkeypatch):
    """Ace-step query payloads with a dict-wrapped task list should be accepted."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('ACE_STEP_STEMS', 'vocals,drums')
    importlib.reload(worker_loop)

    with patch('worker_loop._ace_step_post', side_effect=[
        {'data': {'task_id': 'ace-task-1'}},
        {'data': {'tasks': [{'status': 'completed', 'result': {'vocals': '/tmp/vocals.wav'}}]}},
    ]) as mock_post:
        result = worker_loop.separate_stems_with_ace_step('/tmp/input.wav')

    assert result['task_id'] == 'ace-task-1'
    assert result['tracks'] == {'vocals': '/tmp/vocals.wav'}
    assert mock_post.call_args_list[0].args == (
        '/release_task',
        {
            'task_type': 'extract',
            'src_audio_path': '/tmp/input.wav',
            'track_classes': ['vocals', 'drums'],
            'audio_format': 'wav',
        },
    )
    assert mock_post.call_args_list[1].args == ('/query_result', {'task_id_list': ['ace-task-1']})


def test_pending_upload_task_marks_failed_when_ace_step_fails_strict_mode(data_dir, monkeypatch):
    """In STEM_BACKEND=acestep strict mode, an Ace-Step failure must mark the task failed."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'acestep')
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


def test_auto_mode_falls_back_to_demucs_when_ace_step_fails(data_dir, monkeypatch):
    """In STEM_BACKEND=auto, Ace-Step failure should fall back to Demucs."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    vocals = data_dir / 'stems' / task['task_id'] / 'htdemucs' / f"{task['task_id']}" / 'vocals.wav'
    drums = vocals.parent / 'drums.wav'
    bass = vocals.parent / 'bass.wav'
    other = vocals.parent / 'other.wav'

    def fake_demucs(src_path, tid):
        for f in (vocals, drums, bass, other):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'fake-wav')
        return {'tracks': {'vocals': str(vocals), 'drums': str(drums), 'bass': str(bass), 'other': str(other)}}

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step down')), \
         patch('worker_loop.separate_stems_with_demucs', side_effect=fake_demucs), \
         patch('worker_loop._is_demucs_available', return_value=True), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated.get('stem_backend') == 'demucs'
    assert set(updated['stems'].keys()) == {'vocals', 'drums', 'bass', 'other'}


def test_auto_mode_continues_without_stems_when_both_backends_fail(data_dir, monkeypatch):
    """In auto mode, if Ace-Step fails and Demucs fails too, the task should still succeed without stems."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step down')), \
         patch('worker_loop.separate_stems_with_demucs', side_effect=RuntimeError('demucs error')), \
         patch('worker_loop._is_demucs_available', return_value=True), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert 'stems' not in updated


def test_auto_mode_continues_without_stems_when_ace_step_fails_and_demucs_unavailable(data_dir, monkeypatch):
    """In auto mode, Ace-Step failure with no Demucs available should still succeed without stems."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step down')), \
         patch('worker_loop._is_demucs_available', return_value=False), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert 'stems' not in updated


def test_demucs_backend_strict_mode_fails_task_on_error(data_dir, monkeypatch):
    """In STEM_BACKEND=demucs strict mode, a Demucs failure must mark the task failed."""
    monkeypatch.setenv('STEM_BACKEND', 'demucs')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_demucs', side_effect=RuntimeError('demucs not found')), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        count = worker_loop.process_pending_tasks()

    assert count == 1
    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'demucs not found' in updated['error']


def test_demucs_backend_strict_mode_records_stems_on_success(data_dir, monkeypatch):
    """In STEM_BACKEND=demucs, a successful run stores stems in the task."""
    monkeypatch.setenv('STEM_BACKEND', 'demucs')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    vocals = data_dir / 'stems' / task['task_id'] / 'htdemucs' / task['task_id'] / 'vocals.wav'
    drums = vocals.parent / 'drums.wav'

    def fake_demucs(src_path, tid):
        for f in (vocals, drums):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'fake-wav')
        return {'tracks': {'vocals': str(vocals), 'drums': str(drums)}}

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_demucs', side_effect=fake_demucs), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated.get('stem_backend') == 'demucs'
    assert set(updated['stems'].keys()) == {'vocals', 'drums'}


def test_acestep_strict_mode_fails_when_url_not_configured(data_dir, monkeypatch):
    """In STEM_BACKEND=acestep mode, task must fail if ACE_STEP_API_URL is not set."""
    monkeypatch.setenv('STEM_BACKEND', 'acestep')
    monkeypatch.delenv('ACE_STEP_API_URL', raising=False)
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        count = worker_loop.process_pending_tasks()

    assert count == 1
    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'ACE_STEP_API_URL' in updated['error']


def test_pending_upload_task_records_mt3_results_when_enabled(data_dir, monkeypatch):
    """When MT3 is enabled, transcription metadata should be persisted on the task."""
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    fake_mt3 = {
        'enabled': True,
        'backend': 'basic_pitch',
        'status': 'completed',
        'model': 'mt3',
        'output_paths': ['/srv/shank/data/mt3/song.mid'],
        'warnings': [],
        'errors': [],
        'full_mix': {
            'midi_path': '/srv/shank/data/mt3/song.mid',
            'model': 'mt3',
            'notes': [{'start': 0.0, 'end': 0.5, 'pitch': 60}],
        },
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
    assert updated['transcription']['enabled'] is True
    assert updated['transcription']['backend'] == 'basic_pitch'
    assert updated['transcription']['midi_file'].endswith('.mid')
    assert len(updated['transcription']['notes']) == 1


def test_pending_upload_task_caches_ace_step_url_stems_for_mt3(data_dir, monkeypatch):
    """Ace-step URL stems should be downloaded to local cache paths before MT3."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('MT3_ENABLED', 'true')
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    monkeypatch.setenv('MT3_SERVICE_URL', DEFAULT_MT3_SERVICE_URL)
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
    assert updated['transcription']['enabled'] is False
    assert updated['transcription']['status'] == 'disabled'


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


def test_transcribe_with_service_raises_when_service_reports_failed_status(tmp_path):
    """transcribe_with_service should raise RuntimeError when service status is failed."""
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    fake_payload = {
        'status': 'failed',
        'error': 'transcription produced no note events',
    }

    with patch('mt3_client._post_json', return_value=fake_payload):
        with pytest.raises(RuntimeError, match='no note events'):
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


def test_transcribe_with_service_warns_when_no_note_events(tmp_path):
    """transcribe_with_service should warn when the service returns an empty note list."""
    import base64
    import mt3_client

    task_id = str(uuid.uuid4())
    output_dir = tmp_path / 'mt3' / task_id
    output_dir.mkdir(parents=True)

    fake_payload = {
        'model': 'multi_instrument',
        'midi_base64': base64.b64encode(b'MThd').decode(),
        'notes': [],
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

    assert 'note_count' in result
    assert result['note_count'] == 0
    assert isinstance(result.get('warnings'), list)
    assert 'No note events returned; MIDI may be empty' in result['warnings']


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


# ---------------------------------------------------------------------------
# _parse_audio_separator_stem_name
# ---------------------------------------------------------------------------

def test_parse_audio_separator_stem_name_extracts_label_in_parentheses():
    """_parse_audio_separator_stem_name should extract and lowercase the parenthesised label."""
    assert worker_loop._parse_audio_separator_stem_name('song_(Vocals)_htdemucs_ft.wav') == 'vocals'
    assert worker_loop._parse_audio_separator_stem_name('song_(Drums)_htdemucs_ft.wav') == 'drums'
    assert worker_loop._parse_audio_separator_stem_name('song_(Bass)_htdemucs_ft.wav') == 'bass'
    assert worker_loop._parse_audio_separator_stem_name('song_(Other)_htdemucs_ft.wav') == 'other'
    assert worker_loop._parse_audio_separator_stem_name('song_(Guitar)_htdemucs_6s.wav') == 'guitar'
    assert worker_loop._parse_audio_separator_stem_name('song_(Piano)_htdemucs_6s.wav') == 'piano'


def test_parse_audio_separator_stem_name_falls_back_to_stem():
    """_parse_audio_separator_stem_name should fall back to the bare filename stem."""
    assert worker_loop._parse_audio_separator_stem_name('vocals.wav') == 'vocals'
    assert worker_loop._parse_audio_separator_stem_name('drums') == 'drums'


# ---------------------------------------------------------------------------
# audio_separator backend – strict mode
# ---------------------------------------------------------------------------

def test_audio_separator_backend_strict_mode_fails_task_on_error(data_dir, monkeypatch):
    """In STEM_BACKEND=audio_separator strict mode, a failure must mark the task failed."""
    monkeypatch.setenv('STEM_BACKEND', 'audio_separator')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_audio_separator',
               side_effect=RuntimeError('audio-separator error')), \
         patch('worker_loop.analyze_audio') as mock_analyze:
        count = worker_loop.process_pending_tasks()

    assert count == 1
    mock_analyze.assert_not_called()
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'failed'
    assert 'audio-separator error' in updated['error']


def test_audio_separator_backend_strict_mode_records_stems_on_success(data_dir, monkeypatch):
    """In STEM_BACKEND=audio_separator, a successful run stores stems in the task."""
    monkeypatch.setenv('STEM_BACKEND', 'audio_separator')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    stems_dir = data_dir / 'stems' / task['task_id']
    vocals = stems_dir / 'song_(Vocals)_htdemucs_ft.wav'
    drums = stems_dir / 'song_(Drums)_htdemucs_ft.wav'
    bass = stems_dir / 'song_(Bass)_htdemucs_ft.wav'
    other = stems_dir / 'song_(Other)_htdemucs_ft.wav'

    def fake_audio_separator(src_path, tid):
        for f in (vocals, drums, bass, other):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'fake-wav')
        return {
            'tracks': {
                'vocals': str(vocals),
                'drums': str(drums),
                'bass': str(bass),
                'other': str(other),
            },
        }

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_audio_separator', side_effect=fake_audio_separator), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated.get('stem_backend') == 'audio_separator'
    assert set(updated['stems'].keys()) == {'vocals', 'drums', 'bass', 'other'}


# ---------------------------------------------------------------------------
# audio_separator backend – auto mode fallback
# ---------------------------------------------------------------------------

def test_auto_mode_falls_back_to_audio_separator_when_ace_step_fails(data_dir, monkeypatch):
    """In STEM_BACKEND=auto, Ace-Step failure should fall back to audio-separator."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    stems_dir = data_dir / 'stems' / task['task_id']
    vocals = stems_dir / 'song_(Vocals)_htdemucs_ft.wav'
    drums = stems_dir / 'song_(Drums)_htdemucs_ft.wav'
    bass = stems_dir / 'song_(Bass)_htdemucs_ft.wav'
    other = stems_dir / 'song_(Other)_htdemucs_ft.wav'

    def fake_audio_separator(src_path, tid):
        for f in (vocals, drums, bass, other):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'fake-wav')
        return {
            'tracks': {
                'vocals': str(vocals),
                'drums': str(drums),
                'bass': str(bass),
                'other': str(other),
            },
        }

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step down')), \
         patch('worker_loop.separate_stems_with_audio_separator', side_effect=fake_audio_separator), \
         patch('worker_loop._is_audio_separator_available', return_value=True), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated.get('stem_backend') == 'audio_separator'
    assert set(updated['stems'].keys()) == {'vocals', 'drums', 'bass', 'other'}


def test_auto_mode_falls_back_to_demucs_when_audio_separator_also_fails(data_dir, monkeypatch):
    """In auto mode, if Ace-Step and audio-separator fail, the task should try Demucs."""
    monkeypatch.setenv('ACE_STEP_API_URL', 'http://ace-step:8001')
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    vocals = data_dir / 'stems' / task['task_id'] / 'htdemucs' / f"{task['task_id']}" / 'vocals.wav'
    drums = vocals.parent / 'drums.wav'

    def fake_demucs(src_path, tid):
        for f in (vocals, drums):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'fake-wav')
        return {'tracks': {'vocals': str(vocals), 'drums': str(drums)}}

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop.separate_stems_with_ace_step', side_effect=RuntimeError('Ace-step down')), \
         patch('worker_loop.separate_stems_with_audio_separator',
               side_effect=RuntimeError('audio-separator error')), \
         patch('worker_loop._is_audio_separator_available', return_value=True), \
         patch('worker_loop.separate_stems_with_demucs', side_effect=fake_demucs), \
         patch('worker_loop._is_demucs_available', return_value=True), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert updated.get('stem_backend') == 'demucs'
    assert set(updated['stems'].keys()) == {'vocals', 'drums'}


def test_auto_mode_continues_without_stems_when_audio_separator_unavailable(data_dir, monkeypatch):
    """In auto mode with no backends configured, the task should still succeed without stems."""
    monkeypatch.delenv('ACE_STEP_API_URL', raising=False)
    monkeypatch.setenv('STEM_BACKEND', 'auto')
    importlib.reload(worker_loop)
    task, task_file = _make_upload_task(data_dir)

    with patch('worker_loop.normalize_audio'), \
         patch('worker_loop._is_audio_separator_available', return_value=False), \
         patch('worker_loop._is_demucs_available', return_value=False), \
         patch('worker_loop.analyze_audio', return_value={'bpm': 120.0, 'key': 'C major'}):
        count = worker_loop.process_pending_tasks()

    assert count == 1
    updated = json.loads(task_file.read_text())
    assert updated['status'] == 'done'
    assert 'stems' not in updated
