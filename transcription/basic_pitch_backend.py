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
            resolved_audio = audio_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InvalidAudioError(f'audio file not found: {audio_path}') from exc
        if not resolved_audio.is_file():
            raise InvalidAudioError(f'audio path is not a file: {resolved_audio}')
        if resolved_audio.suffix.lower() not in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
            raise InvalidAudioError(
                f'unsupported audio extension for basic_pitch: {resolved_audio.suffix}'
            )

        try:
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise BackendDependencyError(
                'basic_pitch backend is not installed; install with `pip install basic-pitch`'
            ) from exc

        try:
            _model_output, midi_data, note_events = predict(str(resolved_audio))
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
