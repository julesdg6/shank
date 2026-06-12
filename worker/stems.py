"""Stem separation backends for SHANK audio processing.

Supports three backends:
- **Ace-Step** (``acestep``): external REST service for stem extraction.
- **python-audio-separator** (``audio_separator``): bundled local library.
- **Demucs** (``demucs``): legacy CLI tool.

All configuration is read from environment variables at call time so that the
module does not need to be reloaded when settings change during testing.
"""
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers – Ace-Step
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = '/srv/shank/data'
_DEFAULT_ACE_STEP_STEMS = 'vocals,drums,bass,other'
_DEFAULT_ACE_STEP_POLL_INTERVAL = '2'
_DEFAULT_ACE_STEP_TIMEOUT = '300'
_DEFAULT_ACE_STEP_MAX_DOWNLOAD_BYTES = str(100 * 1024 * 1024)
_DEFAULT_AUDIO_SEPARATOR_MODEL = 'htdemucs_ft.yaml'
_DEFAULT_AUDIO_SEPARATOR_MODEL_DIR = '/srv/shank/models/separator'
_DEFAULT_DEMUCS_MODEL = 'htdemucs'


def _ace_step_post(path: str, payload: dict) -> dict:
    """POST JSON payload to the configured Ace-Step endpoint and return parsed JSON."""
    ace_step_api_url = os.getenv('ACE_STEP_API_URL', '').strip().rstrip('/')
    ace_step_api_key = os.getenv('ACE_STEP_API_KEY', '').strip()
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{ace_step_api_url}{path}',
        data=body,
        headers={
            'Content-Type': 'application/json',
            **({'Authorization': f'Bearer {ace_step_api_key}'} if ace_step_api_key else {}),
        },
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def _ace_step_response_data(response_payload: dict[str, Any]) -> Any:
    """Unwrap Ace-Step responses that use a top-level ``data`` envelope."""
    if isinstance(response_payload, dict) and 'data' in response_payload:
        return response_payload.get('data')
    return response_payload


def _extract_track_files(data: Any) -> dict[str, str]:
    """Collect ``track_name``/``file`` pairs from nested Ace-Step result payloads."""
    tracks: dict[str, str] = {}
    configured_stems = {
        stem.strip().lower()
        for stem in os.getenv('ACE_STEP_STEMS', _DEFAULT_ACE_STEP_STEMS).split(',')
        if stem.strip()
    }

    def collect(node: Any) -> None:
        """Recursively walk *node* (dict or list) and populate ``tracks``."""
        if isinstance(node, dict):
            track_name = (
                node.get('track_name')
                or node.get('track')
                or node.get('stem_name')
                or node.get('stem')
                or node.get('name')
            )
            file_url = (
                node.get('file')
                or node.get('url')
                or node.get('uri')
                or node.get('path')
                or node.get('audio_url')
                or node.get('file_path')
            )
            # Support APIs that return direct stem-key mappings: {"vocals": "...", "drums": "..."}
            if not isinstance(track_name, str) and not isinstance(file_url, str):
                recognized_stem_items = [
                    (stem_name.strip(), stem_ref)
                    for stem_name, stem_ref in node.items()
                    if (
                        isinstance(stem_name, str)
                        and isinstance(stem_ref, str)
                        and stem_name.strip().lower() in configured_stems
                    )
                ]
                for stem_name, stem_ref in recognized_stem_items:
                    tracks[stem_name] = stem_ref
            if isinstance(track_name, str) and isinstance(file_url, str):
                tracks[track_name] = file_url
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(data)
    return tracks


def _resolve_ace_step_stem_file(task_id: str, stem_name: str, stem_ref: str) -> str:
    """Return a local file path for one Ace-Step stem.

    Accepts local file paths plus ``file://``, ``http://``, and ``https://`` references.
    URL sources are cached under ``DATA_DIR/stems/<task_id>/`` and reused on retries.
    Raises ``RuntimeError`` when the stem cannot be resolved locally or downloaded.
    """
    ace_step_api_url = os.getenv('ACE_STEP_API_URL', '').strip().rstrip('/')
    ace_step_api_key = os.getenv('ACE_STEP_API_KEY', '').strip()
    ace_step_max_bytes = int(os.getenv('ACE_STEP_MAX_DOWNLOAD_BYTES', _DEFAULT_ACE_STEP_MAX_DOWNLOAD_BYTES))
    stems_cache_dir = Path(os.getenv('DATA_DIR', _DEFAULT_DATA_DIR)) / 'stems'

    parsed = urlparse(stem_ref)
    scheme = parsed.scheme.lower()
    if not scheme:
        local_candidate = Path(stem_ref)
        if local_candidate.exists():
            return str(local_candidate)
        if stem_ref.startswith('/'):
            stem_ref = urljoin(f'{ace_step_api_url}/', stem_ref.lstrip('/'))
            parsed = urlparse(stem_ref)
            scheme = parsed.scheme.lower()
    if scheme == 'file':
        local_candidate = Path(parsed.path)
        if local_candidate.exists():
            return str(local_candidate)
        raise RuntimeError(f'Ace-Step local stem file not found for {stem_name}: {stem_ref}')

    if scheme not in ('http', 'https'):
        raise RuntimeError(
            f'Ace-Step stem for {stem_name} has unsupported scheme {scheme!r}; '
            f'only http, https, and file are supported: {stem_ref}'
        )

    try:
        safe_task_id = str(uuid.UUID(task_id))
    except ValueError as exc:
        raise RuntimeError(f'Invalid task_id for Ace-Step stem cache: {task_id!r}') from exc

    ext = Path(parsed.path).suffix or '.wav'
    cache_path = stems_cache_dir / safe_task_id / f'{stem_name}{ext}'
    if cache_path.exists():
        return str(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request_headers: dict[str, str] = {}
    if ace_step_api_key and ace_step_api_url:
        ace_url = urlparse(ace_step_api_url)
        if parsed.scheme == ace_url.scheme and parsed.netloc == ace_url.netloc:
            request_headers['Authorization'] = f'Bearer {ace_step_api_key}'
    request_kwargs: dict[str, Any] = {'headers': request_headers} if request_headers else {}
    request = urllib.request.Request(stem_ref, **request_kwargs)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, cache_path.open('wb') as output_file:
            total_bytes = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                if total_bytes + len(chunk) > ace_step_max_bytes:
                    raise RuntimeError(
                        f'Ace-Step stem download exceeded {ace_step_max_bytes} bytes: {stem_ref}'
                    )
                total_bytes += len(chunk)
                output_file.write(chunk)
    except Exception as exc:
        cache_path.unlink(missing_ok=True)
        raise RuntimeError(f'Failed to download Ace-Step stem {stem_name} from {stem_ref}: {exc}') from exc
    return str(cache_path)


def _prepare_ace_step_stems_for_mt3(task_id: str, tracks: dict[str, str] | None) -> dict[str, str]:
    """Resolve Ace-Step ``tracks`` into local files for configured ``ACE_STEP_STEMS`` only."""
    if not tracks:
        return {}

    ace_step_stems = tuple(
        stem.strip()
        for stem in os.getenv('ACE_STEP_STEMS', _DEFAULT_ACE_STEP_STEMS).split(',')
        if stem.strip()
    )
    normalized = {
        stem_name.strip().lower(): stem_ref
        for stem_name, stem_ref in tracks.items()
        if isinstance(stem_name, str) and isinstance(stem_ref, str)
    }
    prepared: dict[str, str] = {}
    for configured_stem in ace_step_stems:
        stem_ref = normalized.get(configured_stem.lower())
        if not stem_ref:
            continue
        prepared[configured_stem] = _resolve_ace_step_stem_file(task_id, configured_stem, stem_ref)
    return prepared


def separate_stems_with_ace_step(src_audio_path: str) -> dict:
    """Run Ace-Step extract flow and return ``{'task_id': str, 'tracks': dict[str, str]}``."""
    ace_step_stems = tuple(
        stem.strip()
        for stem in os.getenv('ACE_STEP_STEMS', _DEFAULT_ACE_STEP_STEMS).split(',')
        if stem.strip()
    )
    ace_step_timeout = int(os.getenv('ACE_STEP_TIMEOUT', _DEFAULT_ACE_STEP_TIMEOUT))
    ace_step_poll_interval = float(os.getenv('ACE_STEP_POLL_INTERVAL', _DEFAULT_ACE_STEP_POLL_INTERVAL))

    release_payload = {
        'task_type': 'extract',
        'src_audio_path': src_audio_path,
        'track_classes': list(ace_step_stems),
        'audio_format': 'wav',
    }
    release_data = _ace_step_response_data(_ace_step_post('/release_task', release_payload))
    if not isinstance(release_data, dict) or not release_data.get('task_id'):
        raise RuntimeError('Ace-step did not return a task_id for stem separation')
    ace_task_id = release_data['task_id']
    deadline = time.time() + ace_step_timeout

    while time.time() < deadline:
        query_data = _ace_step_response_data(_ace_step_post('/query_result', {'task_id_list': [ace_task_id]}))
        task_entries: list[dict[str, Any]] = []
        if isinstance(query_data, list):
            task_entries = [entry for entry in query_data if isinstance(entry, dict)]
        elif isinstance(query_data, dict):
            if isinstance(query_data.get('tasks'), list):
                task_entries = [entry for entry in query_data['tasks'] if isinstance(entry, dict)]
            elif isinstance(query_data.get('task_list'), list):
                task_entries = [entry for entry in query_data['task_list'] if isinstance(entry, dict)]
            elif isinstance(query_data.get(ace_task_id), dict):
                task_entries = [query_data[ace_task_id]]
            elif 'status' in query_data and any(key in query_data for key in ('result', 'error', 'task_id')):
                task_entries = [query_data]

        if task_entries:
            task_data = task_entries[0]
            status = task_data.get('status')
            status_str = str(status).lower()
            if status_str in ('1', 'succeeded', 'done', 'success', 'completed'):
                result = task_data.get('result')
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        pass
                return {
                    'task_id': ace_task_id,
                    'tracks': _extract_track_files(result),
                }
            if status_str in ('2', 'failed', 'error', 'fail'):
                raise RuntimeError(task_data.get('error') or 'Ace-step stem separation failed')
        time.sleep(ace_step_poll_interval)

    raise RuntimeError('Ace-step stem separation timed out')


# ---------------------------------------------------------------------------
# python-audio-separator backend
# ---------------------------------------------------------------------------

def _is_audio_separator_available() -> bool:
    """Return True if the ``audio_separator`` package is importable."""
    return importlib.util.find_spec('audio_separator') is not None


def _parse_audio_separator_stem_name(filename: str) -> str:
    """Extract a normalised stem name from an audio-separator output filename.

    audio-separator names output files like::

        song_(Vocals)_htdemucs_ft.wav  →  vocals
        song_(Drums)_htdemucs_ft.wav   →  drums
        song_(Bass)_htdemucs_ft.wav    →  bass
        song_(Other)_htdemucs_ft.wav   →  other
        song_(Guitar)_htdemucs_ft.wav  →  guitar  (6-stem model)
        song_(Piano)_htdemucs_ft.wav   →  piano   (6-stem model)

    Falls back to the bare filename stem (without extension) when no
    parenthesised label is found.
    """
    m = re.search(r'\(([^)]+)\)', filename)
    if m:
        return m.group(1).lower()
    return Path(filename).stem.lower()


def separate_stems_with_audio_separator(
    src_audio_path: str,
    task_id: str,
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> dict:
    """Run python-audio-separator stem separation and return ``{'tracks': dict[str, str]}``.

    Output stems are stored under ``DATA_DIR/stems/<task_id>/``.
    The model is downloaded automatically on first use if not already cached in
    ``AUDIO_SEPARATOR_MODEL_DIR``.

    Raises ``RuntimeError`` if separation fails or produces no output stems.
    """
    from audio_separator.separator import Separator  # noqa: PLC0415 – lazy import

    stems_cache_dir = Path(os.getenv('DATA_DIR', _DEFAULT_DATA_DIR)) / 'stems'
    audio_separator_model = (model_name or os.getenv('AUDIO_SEPARATOR_MODEL', _DEFAULT_AUDIO_SEPARATOR_MODEL)).strip()
    if not audio_separator_model:
        audio_separator_model = _DEFAULT_AUDIO_SEPARATOR_MODEL
    audio_separator_model_dir = (
        os.getenv('AUDIO_SEPARATOR_MODEL_DIR', _DEFAULT_AUDIO_SEPARATOR_MODEL_DIR).strip()
        or _DEFAULT_AUDIO_SEPARATOR_MODEL_DIR
    )
    audio_separator_device = (device or os.getenv('AUDIO_SEPARATOR_DEVICE', 'cpu')).strip().lower() or 'cpu'
    if audio_separator_device not in ('cpu', 'cuda'):
        log.warning('Unsupported audio-separator device %r; defaulting to cpu', audio_separator_device)
        audio_separator_device = 'cpu'

    out_dir = stems_cache_dir / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=audio_separator_model_dir,
        output_dir=str(out_dir),
        output_format='wav',
        use_autocast=audio_separator_device == 'cuda',
    )
    separator.load_model(model_filename=audio_separator_model)
    output_files = separator.separate(src_audio_path)

    if not output_files:
        raise RuntimeError(f'audio-separator produced no output stems for {src_audio_path}')

    tracks: dict[str, str] = {}
    for file_path in output_files:
        stem_name = _parse_audio_separator_stem_name(Path(file_path).name)
        tracks[stem_name] = str(file_path)

    if not tracks:
        raise RuntimeError(f'audio-separator produced no usable tracks for {src_audio_path}')

    return {'tracks': tracks}


# ---------------------------------------------------------------------------
# Demucs (legacy CLI) backend
# ---------------------------------------------------------------------------

def _is_demucs_available() -> bool:
    """Return True if the ``demucs`` command-line tool is found in PATH."""
    return shutil.which('demucs') is not None


def separate_stems_with_demucs(
    src_audio_path: str,
    task_id: str,
    *,
    device: str | None = None,
) -> dict:
    """Run Demucs stem separation and return ``{'tracks': dict[str, str]}``.

    Output stems are stored under ``DATA_DIR/stems/<task_id>/`` to mirror the
    layout used for Ace-Step cached stems.
    Raises ``RuntimeError`` if Demucs exits with a non-zero status or produces
    no output stems.
    """
    stems_cache_dir = Path(os.getenv('DATA_DIR', _DEFAULT_DATA_DIR)) / 'stems'
    demucs_model = os.getenv('DEMUCS_MODEL', _DEFAULT_DEMUCS_MODEL).strip() or _DEFAULT_DEMUCS_MODEL
    demucs_device = (device or os.getenv('DEMUCS_DEVICE', 'cpu')).strip() or 'cpu'

    out_base = stems_cache_dir / task_id
    out_base.mkdir(parents=True, exist_ok=True)

    cmd = [
        'demucs',
        '--model', demucs_model,
        '--device', demucs_device,
        '--out', str(out_base),
        src_audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f'demucs failed (exit {result.returncode}): {result.stderr}')

    # Demucs writes to {out}/{model}/{track_stem}/
    track_stem = Path(src_audio_path).stem
    stem_dir = out_base / demucs_model / track_stem
    if not stem_dir.exists():
        raise RuntimeError(f'Demucs output directory not found: {stem_dir}')

    tracks: dict[str, str] = {}
    for stem_file in sorted(stem_dir.glob('*.wav')):
        tracks[stem_file.stem] = str(stem_file)

    if not tracks:
        raise RuntimeError(f'Demucs produced no output stems in {stem_dir}')

    return {'tracks': tracks}
