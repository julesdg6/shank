from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    """Base error for transcription backends."""


class BackendDependencyError(TranscriptionError):
    """Raised when an optional backend dependency is missing."""


class InvalidAudioError(TranscriptionError):
    """Raised when audio input is invalid for transcription."""


class EmptyTranscriptionError(TranscriptionError):
    """Raised when a backend returns no usable transcription output."""


@dataclass
class TranscriptionResult:
    backend: str
    midi_bytes: bytes
    notes: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


class TranscriptionBackend(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, model: str | None = None) -> TranscriptionResult:
        raise NotImplementedError
