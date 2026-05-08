import json
import os
import re
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='SHANK MT3 Service')

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data')).resolve()
MT3_OUTPUT_DIR = DATA_DIR / 'mt3'
DEFAULT_MODEL = os.getenv('MT3_MODEL', 'multi_instrument').strip() or 'multi_instrument'


class TranscribeRequest(BaseModel):
    path: str | None = None
    audio_path: str | None = None
    task_id: str | None = None
    source: str | None = None
    model: str | None = None


def _safe_name(value: str, fallback: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', value).strip('_')
    return safe or fallback


def _resolve_shared_path(path_value: str) -> Path | None:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = DATA_DIR / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(DATA_DIR)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _empty_midi_bytes() -> bytes:
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

    input_path = _resolve_shared_path(path_value)
    if input_path is None:
        return {
            'status': 'failed',
            'error': 'path must point to a file inside DATA_DIR',
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    task_name = _safe_name(body.task_id or 'mt3', 'mt3')
    source_name = _safe_name(body.source or input_path.stem, 'track')
    output_dir = MT3_OUTPUT_DIR / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_path = output_dir / f'{source_name}.mid'
    notes_path = output_dir / f'{source_name}.notes.json'
    notes: list[dict] = []

    midi_path.write_bytes(_empty_midi_bytes())
    notes_path.write_text(json.dumps(notes, indent=2))

    return {
        'status': 'completed',
        'error': None,
        'model': model,
        'midi_path': str(midi_path),
        'notes_path': str(notes_path),
        'notes': notes,
    }
