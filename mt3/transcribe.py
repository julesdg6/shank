"""Standalone MT3 transcription wrapper.

Transcribes a WAV file using the MT3 service and persists outputs under
``DATA_DIR/mt3/<task_id>/``.

Usage (command line)::

    python -m mt3.transcribe /path/to/audio.wav
    python -m mt3.transcribe /path/to/audio.wav --output-dir /tmp/out
    python -m mt3.transcribe /path/to/audio.wav --no-notes --json

Usage (Python API)::

    from mt3.transcribe import transcribe
    result = transcribe('/path/to/audio.wav')
    print(result['midi_path'])
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
MT3_OUTPUT_DIR = DATA_DIR / 'mt3'
MT3_SERVICE_URL = os.getenv('MT3_SERVICE_URL', '').strip().rstrip('/')
MT3_MODEL = os.getenv('MT3_MODEL', 'multi_instrument').strip() or 'multi_instrument'
MT3_TIMEOUT = int(os.getenv('MT3_TIMEOUT', '1800'))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: str | Path,
    *,
    output_dir: Path | None = None,
    save_notes: bool = True,
    model: str | None = None,
    task_id: str | None = None,
    service_url: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Transcribe a WAV file with MT3 and persist results.

    Parameters
    ----------
    wav_path:
        Path to the input WAV file.
    output_dir:
        Directory for output artifacts.  Defaults to
        ``DATA_DIR/mt3/<task_id>/``.
    save_notes:
        When *True* (default) the raw JSON note events are written next to
        the MIDI file as ``<task_id>.notes.json``.
    model:
        MT3 model name.  Defaults to the ``MT3_MODEL`` environment variable
        (``'multi_instrument'``).
    task_id:
        Identifier used to name output files.  A UUID hex string is generated
        when not provided.
    service_url:
        URL of the MT3 service endpoint.  Defaults to the ``MT3_SERVICE_URL``
        environment variable.  When empty, the local ``services.mt3.main``
        module is used in-process if importable; otherwise a minimal valid
        empty MIDI is written.
    timeout:
        HTTP request timeout in seconds.  Defaults to ``MT3_TIMEOUT``.

    Returns
    -------
    dict with the following keys:

    * ``wav_path``       – absolute input path (str)
    * ``midi_path``      – path of the written MIDI file (str)
    * ``notes_path``     – path of the notes JSON file, or *None*
    * ``note_count``     – number of note events (int)
    * ``model``          – model name used (str)
    * ``task_id``        – identifier used for output files (str)
    * ``output_dir``     – directory containing all outputs (str)
    * ``transcribed_at`` – ISO-8601 UTC timestamp (str)
    * ``warnings``       – list of warning strings

    Raises
    ------
    FileNotFoundError
        If *wav_path* does not exist.
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f'WAV file not found: {wav_path}')

    effective_model = model or MT3_MODEL
    effective_timeout = timeout if timeout is not None else MT3_TIMEOUT
    effective_url = (service_url if service_url is not None else MT3_SERVICE_URL).rstrip('/')
    effective_task_id = task_id or uuid.uuid4().hex

    out_dir = output_dir if output_dir is not None else (MT3_OUTPUT_DIR / effective_task_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    notes: list = []
    midi_bytes: bytes | None = None
    backend: str | None = None

    if effective_url:
        payload = _call_service(
            effective_url, wav_path, effective_task_id, effective_model, effective_timeout,
        )
        backend_raw = payload.get('backend')
        if isinstance(backend_raw, str) and backend_raw:
            backend = backend_raw
        status_raw = payload.get('status')
        if isinstance(status_raw, str) and status_raw.lower() in ('failed', 'error', 'low_confidence'):
            raise RuntimeError(str(payload.get('error') or f'Transcription failed with status={status_raw}'))
        effective_model = payload.get('model') or effective_model
        midi_bytes = _decode_midi(payload)

        notes_raw = payload.get('notes')
        if isinstance(notes_raw, list):
            notes = notes_raw
        elif notes_raw is not None:
            warnings.append(
                f'Unexpected notes format from service: {type(notes_raw).__name__}'
            )

        if isinstance(payload.get('warnings'), list):
            warnings.extend(str(w) for w in payload['warnings'])
    else:
        midi_bytes, notes, inline_warnings = _transcribe_inline(
            wav_path, effective_task_id, effective_model,
        )
        warnings.extend(inline_warnings)

    # ------------------------------------------------------------------
    # Persist MIDI
    # ------------------------------------------------------------------
    midi_file = out_dir / f'{effective_task_id}.mid'
    if midi_bytes:
        midi_file.write_bytes(midi_bytes)
    else:
        midi_file.write_bytes(_empty_midi_bytes())
        warnings.append('No MIDI data returned; empty MIDI written')

    # ------------------------------------------------------------------
    # Persist notes JSON (optional)
    # ------------------------------------------------------------------
    notes_file: Path | None = None
    if save_notes:
        notes_file = out_dir / f'{effective_task_id}.notes.json'
        notes_file.write_text(json.dumps(notes, indent=2))

    # ------------------------------------------------------------------
    # Persist summary metadata
    # ------------------------------------------------------------------
    note_stats = _compute_note_stats(notes)
    metadata: dict[str, Any] = {
        'wav_path': str(wav_path),
        'midi_path': str(midi_file),
        'notes_path': str(notes_file) if notes_file is not None else None,
        'note_count': note_stats['note_count'],
        'pitch_range': note_stats['pitch_range'],
        'duration_seconds': note_stats['duration_seconds'],
        'program_count': note_stats['program_count'],
        'model': effective_model,
        'backend': backend,
        'task_id': effective_task_id,
        'output_dir': str(out_dir),
        'transcribed_at': datetime.now(timezone.utc).isoformat(),
        'warnings': warnings,
    }
    meta_file = out_dir / f'{effective_task_id}.meta.json'
    meta_file.write_text(json.dumps(metadata, indent=2))

    return metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_service(
    url: str,
    wav_path: Path,
    task_id: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    """POST to the MT3 service ``/transcribe`` endpoint and return the payload."""
    import urllib.request

    body = json.dumps({
        'audio_path': str(wav_path),
        'task_id': task_id,
        'model': model,
        'source': 'full_mix',
    }).encode('utf-8')
    req = urllib.request.Request(
        f'{url}/transcribe',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode('utf-8'))

    # Unwrap optional data envelope
    if isinstance(raw, dict) and 'data' in raw and len(raw) == 1:
        raw = raw['data']

    if not isinstance(raw, dict):
        raise RuntimeError(f'MT3 service returned unexpected response type: {type(raw).__name__}')

    return raw


def _decode_midi(payload: dict[str, Any]) -> bytes | None:
    """Extract and decode base64-encoded MIDI bytes from a service payload."""
    import base64

    for key in ('midi_base64', 'midi_b64', 'midi'):
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            return base64.b64decode(raw)
    return None


def _compute_note_stats(notes: list) -> dict[str, Any]:
    pitches = [
        int(round(float(note.get('pitch'))))
        for note in notes
        if isinstance(note, dict) and isinstance(note.get('pitch'), (int, float))
    ]
    starts = [
        float(note.get('start'))
        for note in notes
        if isinstance(note, dict) and isinstance(note.get('start'), (int, float))
    ]
    ends = [
        float(note.get('end'))
        for note in notes
        if isinstance(note, dict) and isinstance(note.get('end'), (int, float))
    ]
    programs = {
        int(note.get('program'))
        for note in notes
        if isinstance(note, dict) and isinstance(note.get('program'), (int, float))
    }

    duration_seconds: float | None = None
    if starts and ends:
        start_min = min(starts)
        end_max = max(ends)
        if end_max >= start_min:
            duration_seconds = end_max - start_min

    pitch_range = None
    if pitches:
        pitch_range = {'min': min(pitches), 'max': max(pitches)}

    return {
        'note_count': len(notes),
        'pitch_range': pitch_range,
        'duration_seconds': duration_seconds,
        'program_count': len(programs),
    }


def _transcribe_inline(
    wav_path: Path,
    task_id: str,
    model: str,
) -> tuple[bytes, list, list[str]]:
    """Transcribe using ``services.mt3.main`` in-process (no HTTP required).

    Falls back to an empty MIDI stub when the services package is unavailable.
    """
    warnings: list[str] = []

    try:
        import importlib

        _repo_root = Path(__file__).resolve().parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        mt3_main = importlib.import_module('services.mt3.main')
    except ImportError as exc:
        warnings.append(f'services.mt3.main not importable, using empty MIDI stub: {exc}')
        return _empty_midi_bytes(), [], warnings

    req = mt3_main.TranscribeRequest(
        audio_path=str(wav_path),
        task_id=task_id,
        model=model,
        source='full_mix',
    )
    result = mt3_main.transcribe(req)

    notes = result.get('notes') or []
    if not isinstance(notes, list):
        notes = []

    midi_path_str = result.get('midi_path')
    if isinstance(midi_path_str, str):
        midi_p = Path(midi_path_str)
        if midi_p.exists():
            return midi_p.read_bytes(), notes, warnings

    warnings.append('Inline transcription did not produce a readable MIDI file; using empty stub')
    return _empty_midi_bytes(), notes, warnings


def _empty_midi_bytes() -> bytes:
    """Return a minimal valid single-track MIDI file with no notes."""
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for MT3 transcription."""
    parser = argparse.ArgumentParser(
        prog='mt3.transcribe',
        description='Transcribe a WAV file to MIDI using MT3.',
    )
    parser.add_argument('wav', help='Path to the input WAV file.')
    parser.add_argument(
        '--output-dir', metavar='DIR',
        help='Directory for output artifacts (default: DATA_DIR/mt3/<task_id>/).',
    )
    parser.add_argument(
        '--no-notes', action='store_true',
        help='Skip writing the JSON note events file.',
    )
    parser.add_argument(
        '--model', default=None,
        help=f'MT3 model name (default: {MT3_MODEL!r}).',
    )
    parser.add_argument(
        '--service-url', default=None, metavar='URL',
        help='MT3 service URL (overrides MT3_SERVICE_URL env var).',
    )
    parser.add_argument(
        '--task-id', default=None,
        help='Custom task / output identifier (default: generated UUID hex).',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Print the summary metadata as JSON to stdout.',
    )
    args = parser.parse_args(argv)

    result = transcribe(
        args.wav,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        save_notes=not args.no_notes,
        model=args.model,
        task_id=args.task_id,
        service_url=args.service_url,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f'MIDI:  {result["midi_path"]}')
        if result['notes_path']:
            print(f'Notes: {result["notes_path"]} ({result["note_count"]} events)')
        print(f'Meta:  {result["output_dir"]}/{result["task_id"]}.meta.json')
        if result['warnings']:
            for w in result['warnings']:
                print(f'[WARN] {w}', file=sys.stderr)


if __name__ == '__main__':
    main()
