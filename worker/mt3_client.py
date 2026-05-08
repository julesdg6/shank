"""Client wrapper for MT3 transcription service."""
import base64
import json
import urllib.request
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
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value).strip('_') or 'track'


def _decode_midi_bytes(payload: dict[str, Any]) -> bytes | None:
    for key in ('midi_base64', 'midi_b64', 'midi'):
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            return base64.b64decode(raw)
    return None


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
            result['note_count'] = len(notes)
    elif isinstance(payload.get('notes_path'), str) and payload['notes_path']:
        result['notes_path'] = payload['notes_path']

    warnings = payload.get('warnings')
    if isinstance(warnings, list):
        result['warnings'] = [str(w) for w in warnings]

    return result
