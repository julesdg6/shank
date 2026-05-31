from __future__ import annotations

from pathlib import Path

from .base import (
    BackendDependencyError,
    EmptyTranscriptionError,
    InvalidAudioError,
    TranscriptionBackend,
    TranscriptionError,
    TranscriptionResult,
)
from .basic_pitch_backend import BasicPitchBackend
from .mt3_backend import MT3Backend
from .omnizart_backend import OmnizartBackend


class DisabledBackend(TranscriptionBackend):
    name = 'disabled'

    def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
        raise BackendDependencyError('transcription backend is disabled')


def get_backend(name: str) -> TranscriptionBackend:
    backend_name = (name or '').strip().lower()
    if backend_name == 'basic_pitch':
        return BasicPitchBackend()
    if backend_name == 'mt3':
        return MT3Backend()
    if backend_name == 'omnizart':
        return OmnizartBackend()
    if backend_name in ('', 'disabled', 'none', 'off'):
        return DisabledBackend()
    raise ValueError(f'unsupported transcription backend: {name}')


__all__ = [
    'BackendDependencyError',
    'EmptyTranscriptionError',
    'InvalidAudioError',
    'TranscriptionBackend',
    'TranscriptionError',
    'TranscriptionResult',
    'get_backend',
]
