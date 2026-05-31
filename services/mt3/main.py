import json
import logging
import os
import posixpath
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from transcription import (
    BackendDependencyError,
    EmptyTranscriptionError,
    InvalidAudioError,
    TranscriptionError,
    get_backend,
)

app = FastAPI(title='SHANK MT3 Service')
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data')).resolve()
MT3_OUTPUT_DIR = DATA_DIR / 'mt3'
DEFAULT_MODEL = os.getenv('MT3_MODEL', 'multi_instrument').strip() or 'multi_instrument'
TRANSCRIPTION_BACKEND = os.getenv('TRANSCRIPTION_BACKEND', 'basic_pitch').strip() or 'basic_pitch'


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
    backend_name = TRANSCRIPTION_BACKEND
    path_value = body.path or body.audio_path
    if not path_value:
        return {
            'status': 'failed',
            'error': 'path is required',
            'backend': backend_name,
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    normalized_path = _normalize_data_relative_path(path_value)
    if normalized_path is None:
        return {
            'status': 'failed',
            'error': 'path must point to a file inside DATA_DIR',
            'backend': backend_name,
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    if normalized_path not in _known_data_files():
        return {
            'status': 'failed',
            'error': 'path does not exist',
            'backend': backend_name,
            'model': model,
            'midi_path': None,
            'notes_path': None,
        }

    audio_file = DATA_DIR / normalized_path
    output_dir = MT3_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    midi_path = output_dir / f'{file_id}.mid'
    notes_path = output_dir / f'{file_id}.notes.json'
    metadata_path = output_dir / f'{file_id}.meta.json'

    warnings: list[str] = []
    notes: list[dict] = []
    error_message: str | None = None

    try:
        backend = get_backend(backend_name)
        transcription = backend.transcribe(audio_file, model=model)
        backend_name = transcription.backend
        warnings.extend(transcription.warnings)
        notes = transcription.notes
        if not notes:
            raise EmptyTranscriptionError('transcription produced no note events')
        midi_path.write_bytes(transcription.midi_bytes)
    except BackendDependencyError as exc:
        log.warning('Transcription dependency unavailable: %s', exc)
        error_message = 'transcription backend dependency is unavailable'
    except InvalidAudioError as exc:
        log.warning('Invalid transcription input: %s', exc)
        error_message = 'invalid audio input for transcription'
    except EmptyTranscriptionError:
        error_message = 'transcription produced no note events'
    except (TranscriptionError, ValueError, OSError) as exc:
        log.warning('Transcription failed: %s', exc)
        error_message = 'transcription failed'
    except Exception as exc:  # pragma: no cover - defensive catch for backend failures
        log.exception('Unexpected transcription backend error: %s', exc)
        error_message = 'unexpected transcription backend error'

    if error_message is not None:
        if not midi_path.exists():
            midi_path.write_bytes(_empty_midi_bytes())
        notes = []
        warnings.append(error_message)
        status = 'failed'
    else:
        notes_path.write_text(json.dumps(notes, indent=2))
        status = 'completed'

    metadata = {
        'status': status,
        'error': error_message,
        'backend': backend_name,
        'model': model,
        'audio_path': str(audio_file),
        'midi_path': str(midi_path),
        'notes_path': str(notes_path) if status == 'completed' else None,
        'note_count': len(notes),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'warnings': warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        'status': status,
        'error': error_message,
        'backend': backend_name,
        'model': model,
        'midi_path': str(midi_path),
        'notes_path': str(notes_path) if status == 'completed' else None,
        'notes': notes,
        'note_count': len(notes),
        'warnings': warnings,
        'meta_path': str(metadata_path),
    }
