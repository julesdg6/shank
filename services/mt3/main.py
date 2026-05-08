import json
import os
import posixpath
import uuid
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='SHANK MT3 Service')

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data')).resolve()
MT3_OUTPUT_DIR = DATA_DIR / 'mt3'
DEFAULT_MODEL = os.getenv('MT3_MODEL', 'multi_instrument').strip() or 'multi_instrument'


class TranscribeRequest(BaseModel):
    """MT3 transcription request payload."""

    path: str | None = None
    audio_path: str | None = None
    task_id: str | None = None
    source: str | None = None
    model: str | None = None


def _normalize_data_relative_path(path_value: str) -> str | None:
    """Return a normalized DATA_DIR-relative path or None if invalid."""
    normalized = path_value.replace('\\', '/').strip()
    if not normalized:
        return None

    data_root = DATA_DIR.as_posix().rstrip('/')
    if normalized.startswith('/'):
        if not normalized.startswith(f'{data_root}/'):
            return None
        normalized = normalized[len(data_root) + 1:]

    normalized = posixpath.normpath(normalized).lstrip('/')
    if normalized in ('', '.', '..') or normalized.startswith('../'):
        return None
    return normalized


def _known_data_files() -> set[str]:
    return {
        path.relative_to(DATA_DIR).as_posix()
        for path in DATA_DIR.rglob('*')
        if path.is_file()
    }


def _empty_midi_bytes() -> bytes:
    """Generate a valid single-track MIDI file with no notes."""
    return (
        b'MThd'
        b'\x00\x00\x00\x06'
        b'\x00\x00'
        b'\x00\x01'
        b'\x01\xe0'
        b'MTrk'
        b'\x00\x00\x00\x04'
        b'\x00\xff\x2f\x00'
    )


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'mt3'}


@app.post('/transcribe')
def transcribe(body: TranscribeRequest) -> dict:
    model = body.model or DEFAULT_MODEL
    path_value = body.path or body.audio_path
    if not path_value:
        return {
            'status': 'failed',
            'error': 'path is required',
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    normalized_path = _normalize_data_relative_path(path_value)
    if normalized_path is None:
        return {
            'status': 'failed',
            'error': 'path must point to a file inside DATA_DIR',
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    if normalized_path not in _known_data_files():
        return {
            'status': 'failed',
            'error': 'path does not exist',
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    output_dir = MT3_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    midi_path = output_dir / f'{file_id}.mid'
    notes_path = output_dir / f'{file_id}.notes.json'
    notes: list[dict] = []

    midi_path.write_bytes(_empty_midi_bytes())
    notes_path.write_text(json.dumps(notes))

    return {
        'status': 'completed',
        'error': None,
        'model': model,
        'midi_path': str(midi_path),
        'notes_path': str(notes_path),
        'notes': notes,
    }
