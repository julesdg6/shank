"""librosa-based audio analysis: BPM, key, and chord progression extraction."""

from functools import lru_cache

import numpy as np
import librosa
from music21 import pitch as m21_pitch

# Krumhansl-Kessler key profiles (tonic at index 0)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_THIRD = 4
_MINOR_THIRD = 3
_PERFECT_FIFTH = 7
# Ignore effectively silent chroma frames so low-level noise does not create false chord segments.
_CHROMA_ACTIVITY_THRESHOLD = 1e-8


@lru_cache(maxsize=12)
def _normalize_pitch_name(pitch_class: str) -> str:
    return m21_pitch.Pitch(pitch_class).name


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


def _detect_chords(y: np.ndarray, sr: int) -> dict:
    """Return a lightweight, time-segmented chord progression summary."""
    hop_length = 1024
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    if chroma.size == 0:
        return {'segments': [], 'progression': []}

    frame_boundaries = librosa.frames_to_time(np.arange(chroma.shape[1] + 1), sr=sr, hop_length=hop_length)
    frame_labels: list[dict] = []

    for frame_idx in range(chroma.shape[1]):
        frame = chroma[:, frame_idx]
        root_idx = int(np.argmax(frame))
        if float(frame[root_idx]) <= _CHROMA_ACTIVITY_THRESHOLD:
            continue

        major_indices = [root_idx, (root_idx + _MAJOR_THIRD) % 12, (root_idx + _PERFECT_FIFTH) % 12]
        minor_indices = [root_idx, (root_idx + _MINOR_THIRD) % 12, (root_idx + _PERFECT_FIFTH) % 12]

        major_score = float(np.sum(frame[major_indices]))
        minor_score = float(np.sum(frame[minor_indices]))

        if major_score >= minor_score:
            quality = 'major'
        else:
            quality = 'minor'

        frame_labels.append({
            'root': _PITCH_CLASSES[root_idx],
            'quality': quality,
            'start_seconds': float(frame_boundaries[frame_idx]),
            'end_seconds': float(frame_boundaries[frame_idx + 1]),
        })

    if not frame_labels:
        return {'segments': [], 'progression': []}

    segments: list[dict] = []
    for frame in frame_labels:
        root_name = _normalize_pitch_name(frame['root'])
        symbol = f'{root_name}{"m" if frame["quality"] == "minor" else ""}'
        if not segments or segments[-1]['symbol'] != symbol:
            segments.append({
                'symbol': symbol,
                'root': root_name,
                'quality': frame['quality'],
                'start_seconds': round(frame['start_seconds'], 3),
                'end_seconds': round(frame['end_seconds'], 3),
            })
            continue
        segments[-1]['end_seconds'] = round(frame['end_seconds'], 3)

    return {
        'segments': segments,
        'progression': [segment['symbol'] for segment in segments],
    }


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
        ``bpm``               – estimated beats-per-minute (float, rounded to 2 decimal places).
        ``key``               – detected musical key string, e.g. ``'A minor'``.
        ``duration_seconds``  – audio duration in seconds (float, rounded to 2 decimal places).
    """
    y, sr = librosa.load(file_path, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    duration_seconds = round(float(librosa.get_duration(y=y, sr=sr)), 2)

    key = _detect_key(y, sr)
    chords = _detect_chords(y, sr)

    # Lightweight summary artifacts for UI visualizations.
    waveform_bins = 128
    histogram_bins = 64
    curve_bins = 128

    if y.size:
        waveform = [
            round(float(np.mean(chunk)), 5)
            for chunk in np.array_split(y, waveform_bins)
        ]
    else:
        waveform = [0.0] * waveform_bins

    stft_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    if stft_mag.size:
        mean_by_freq = stft_mag.mean(axis=1)
        frequency_histogram = [
            round(float(np.mean(chunk)), 5)
            for chunk in np.array_split(mean_by_freq, histogram_bins)
        ]
        max_hist = max(max(frequency_histogram, default=1.0), 1.0)
        frequency_histogram = [round(v / max_hist, 5) for v in frequency_histogram]
    else:
        frequency_histogram = [0.0] * histogram_bins

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=1024)
    if mel_spec.size:
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mean_by_band = mel_db.mean(axis=1)
        spectrogram_summary = [
            round(float(np.mean(chunk)), 2)
            for chunk in np.array_split(mean_by_band, 8)
        ]
    else:
        spectrogram_summary = [0.0] * 8

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if rms.size:
        loudness_curve = [
            round(float(np.mean(chunk)), 5)
            for chunk in np.array_split(rms, curve_bins)
        ]
    else:
        loudness_curve = [0.0] * curve_bins

    frame_energy = rms ** 2 if rms.size else np.array([])
    if frame_energy.size:
        energy_over_time = [
            round(float(np.mean(chunk)), 5)
            for chunk in np.array_split(frame_energy, curve_bins)
        ]
    else:
        energy_over_time = [0.0] * curve_bins

    return {
        'bpm': bpm,
        'key': key,
        'duration_seconds': duration_seconds,
        'chords': chords,
        'waveform': waveform,
        'frequency_histogram': frequency_histogram,
        'spectrogram_summary': spectrogram_summary,
        'loudness_curve': loudness_curve,
        'energy_over_time': energy_over_time,
    }
