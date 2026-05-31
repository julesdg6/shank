from __future__ import annotations

from pathlib import Path

from .base import BackendDependencyError, TranscriptionBackend, TranscriptionResult


class OmnizartBackend(TranscriptionBackend):
    name = 'omnizart'

    def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
        raise BackendDependencyError('omnizart backend is not implemented yet')
