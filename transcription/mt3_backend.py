from __future__ import annotations

from pathlib import Path

from .base import BackendDependencyError, TranscriptionBackend, TranscriptionResult


class MT3Backend(TranscriptionBackend):
    name = 'mt3'

    def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
        raise BackendDependencyError('mt3 backend is not implemented yet')
