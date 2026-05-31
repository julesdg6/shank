from __future__ import annotations

from io import BytesIO
from pathlib import Path

from .base import (
    BackendDependencyError,
    EmptyTranscriptionError,
    InvalidAudioError,
    TranscriptionBackend,
    TranscriptionError,
    TranscriptionResult,
)


class BasicPitchBackend(TranscriptionBackend):
    name = 'basic_pitch'

    def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
        try:
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise BackendDependencyError(
                'basic_pitch backend is not installed; install with `pip install basic-pitch`'
            ) from exc

        try:
            _model_output, midi_data, note_events = predict(str(audio_path))
        except Exception as exc:
            raise InvalidAudioError(f'basic_pitch failed to process audio: {exc}') from exc

        notes: list[dict[str, float | int]] = []
        if isinstance(note_events, list):
            for item in note_events:
                if not isinstance(item, (tuple, list)) or len(item) < 3:
                    continue
                try:
                    start = float(item[0])
                    end = float(item[1])
                    pitch = int(round(float(item[2])))
                except (TypeError, ValueError):
                    continue
                note: dict[str, float | int] = {
                    'start': start,
                    'end': end,
                    'pitch': pitch,
                    'velocity': 100,
                }
                if len(item) >= 4:
                    try:
                        note['confidence'] = float(item[3])
                    except (TypeError, ValueError):
                        pass
                notes.append(note)

        if not notes:
            raise EmptyTranscriptionError('basic_pitch produced no note events')

        if midi_data is None:
            raise EmptyTranscriptionError('basic_pitch did not produce MIDI data')

        try:
            midi_buffer = BytesIO()
            midi_data.write(midi_buffer)
            midi_bytes = midi_buffer.getvalue()
        except Exception as exc:
            raise TranscriptionError(f'failed to serialize basic_pitch MIDI output: {exc}') from exc

        if not midi_bytes:
            raise EmptyTranscriptionError('basic_pitch produced an empty MIDI file')

        return TranscriptionResult(backend=self.name, midi_bytes=midi_bytes, notes=notes)
