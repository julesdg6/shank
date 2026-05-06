"""librosa-based audio analysis: BPM and musical key extraction."""

import numpy as np
import librosa

# Krumhansl-Kessler key profiles (tonic at index 0)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _detect_key(y: np.ndarray, sr: int) -> str:
    """Return the detected musical key as a string, e.g. 'C major'.

    Uses the Krumhansl-Kessler algorithm: the mean chroma energy across time is
    correlated against all 24 major/minor key templates (12 roots × 2 modes).
    The template with the highest Pearson correlation determines the key.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_corr = -1.0  # correlation is bounded to [-1, 1]
    best_key = 'C major'

    for root in range(12):
        # np.roll(profile, root) aligns the tonic with index *root*
        major_corr = float(np.corrcoef(chroma_mean, np.roll(_MAJOR_PROFILE, root))[0, 1])
        minor_corr = float(np.corrcoef(chroma_mean, np.roll(_MINOR_PROFILE, root))[0, 1])

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = f'{_PITCH_CLASSES[root]} major'

        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f'{_PITCH_CLASSES[root]} minor'

    return best_key


def analyze_audio(file_path: str) -> dict:
    """Load audio from *file_path* and return BPM and key analysis results.

    Parameters
    ----------
    file_path:
        Absolute or relative path to an audio file supported by librosa
        (MP3, WAV, FLAC, etc.).

    Returns
    -------
    dict with keys:
        ``bpm``  – estimated beats-per-minute (float, rounded to 2 decimal places).
        ``key``  – detected musical key string, e.g. ``'A minor'``.
    """
    y, sr = librosa.load(file_path, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)

    key = _detect_key(y, sr)

    return {'bpm': bpm, 'key': key}
