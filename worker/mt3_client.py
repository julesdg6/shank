"""Client wrapper for MT3 transcription service."""
import base64
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _unwrap_data(payload: Any) -> Any:
    if isinstance(payload, dict) and 'data' in payload:
        return payload.get('data')
    return payload


def _safe_name(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', value).strip('_') or 'track'


def _decode_midi_bytes(payload: dict[str, Any]) -> bytes | None:
    for key in ('midi_base64', 'midi_b64', 'midi'):
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            return base64.b64decode(raw)
    return None


def _note_stats(notes: list[dict[str, Any]]) -> dict[str, Any]:
    pitches = [
        int(note.get('pitch'))
        for note in notes
        if isinstance(note.get('pitch'), (int, float))
    ]
    starts = [
        float(note.get('start'))
        for note in notes
        if isinstance(note.get('start'), (int, float))
    ]
    ends = [
        float(note.get('end'))
        for note in notes
        if isinstance(note.get('end'), (int, float))
    ]
    programs = {
        int(note.get('program'))
        for note in notes
        if isinstance(note.get('program'), (int, float))
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


def transcribe_with_service(
    service_url: str,
    audio_path: str,
    output_dir: Path,
    task_id: str,
    model: str,
    source: str,
    timeout: int,
) -> dict[str, Any]:
    """Transcribe one audio source with MT3 service and persist artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _unwrap_data(_post_json(
        f'{service_url.rstrip("/")}/transcribe',
        {
            'audio_path': audio_path,
            'task_id': task_id,
            'model': model,
            'source': source,
        },
        timeout=timeout,
    ))
    if not isinstance(payload, dict):
        raise RuntimeError('MT3 service returned an invalid response payload')

    safe_source = _safe_name(source)
    result: dict[str, Any] = {
        'source': source,
        'model': payload.get('model') or model,
    }

    midi_bytes = _decode_midi_bytes(payload)
    midi_path = payload.get('midi_path')
    if midi_bytes:
        midi_file = output_dir / f'{task_id}__{safe_source}.mid'
        midi_file.write_bytes(midi_bytes)
        result['midi_path'] = str(midi_file)
    elif isinstance(midi_path, str) and midi_path:
        result['midi_path'] = midi_path
    else:
        raise RuntimeError('MT3 service response did not include MIDI output')

    notes = payload.get('notes')
    if notes is not None:
        notes_file = output_dir / f'{task_id}__{safe_source}.notes.json'
        notes_file.write_text(json.dumps(notes, indent=2))
        result['notes_path'] = str(notes_file)
        if isinstance(notes, list):
            stats = _note_stats([note for note in notes if isinstance(note, dict)])
            result['note_count'] = stats['note_count']
            result['pitch_range'] = stats['pitch_range']
            result['duration_seconds'] = stats['duration_seconds']
            result['program_count'] = stats['program_count']
    elif isinstance(payload.get('notes_path'), str) and payload['notes_path']:
        result['notes_path'] = payload['notes_path']

    warnings = payload.get('warnings')
    if isinstance(warnings, list):
        result['warnings'] = [str(w) for w in warnings]

    result['completed_at'] = datetime.now(timezone.utc).isoformat()

    return result
