"""Audio analysis helpers with optional advanced beat/downbeat/loudness detectors."""

from functools import lru_cache
import logging

import librosa
import numpy as np
from music21 import pitch as m21_pitch

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover - optional dependency
    pyln = None

log = logging.getLogger(__name__)

# Krumhansl-Kessler key profiles (tonic at index 0)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_THIRD = 4
_MINOR_THIRD = 3
_PERFECT_FIFTH = 7
# Ignore effectively silent chroma frames so low-level noise does not create false chord segments.
_CHROMA_ACTIVITY_THRESHOLD = 1e-8
_HIGH_BEAT_CONFIDENCE = 0.9
_LOW_BEAT_CONFIDENCE = 0.2
_SINGLE_BEAT_CONFIDENCE = 0.4
_KEY_CONFIDENCE_DIVISOR = 2.0
_TEMPO_WINDOW_BEATS = 4
_TEMPO_CHANGE_THRESHOLD_BPM = 6.0
_TEMPO_CHANGE_MIN_GAP_SECONDS = 1.0
_BARS_PER_SECTION = 8
_OUTRO_OFFSET_SECONDS = 8.0
_SILENT_LUFS = -70.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@lru_cache(maxsize=12)
def _normalize_pitch_name(pitch_class: str) -> str:
    return m21_pitch.Pitch(pitch_class).name


def _detect_key_and_confidence(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Return the detected musical key as a string, e.g. 'C major'.

    Uses the Krumhansl-Kessler algorithm: the mean chroma energy across time is
    correlated against all 24 major/minor key templates (12 roots × 2 modes).
    The template with the highest Pearson correlation determines the key.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_corr = -1.0  # correlation is bounded to [-1, 1]
    second_best_corr = -1.0
    best_key = 'C major'

    for root in range(12):
        # np.roll(profile, root) aligns the tonic with index *root*
        major_corr = float(np.corrcoef(chroma_mean, np.roll(_MAJOR_PROFILE, root))[0, 1])
        minor_corr = float(np.corrcoef(chroma_mean, np.roll(_MINOR_PROFILE, root))[0, 1])

        if major_corr > best_corr:
            second_best_corr = best_corr
            best_corr = major_corr
            best_key = f'{_PITCH_CLASSES[root]} major'
        elif major_corr > second_best_corr:
            second_best_corr = major_corr

        if minor_corr > best_corr:
            second_best_corr = best_corr
            best_corr = minor_corr
            best_key = f'{_PITCH_CLASSES[root]} minor'
        elif minor_corr > second_best_corr:
            second_best_corr = minor_corr

    margin = best_corr - second_best_corr
    key_confidence = _clamp(float(margin / _KEY_CONFIDENCE_DIVISOR), 0.0, 1.0)
    return best_key, round(key_confidence, 3)


def _detect_key(y: np.ndarray, sr: int) -> str:
    key, _ = _detect_key_and_confidence(y, sr)
    return key


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


def _librosa_beats(y: np.ndarray, sr: int) -> tuple[float, list[float], float]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [round(float(value), 3) for value in beat_times.tolist()]
    confidence = _HIGH_BEAT_CONFIDENCE if len(beats) >= 2 else _LOW_BEAT_CONFIDENCE
    return bpm, beats, confidence


def _madmom_beats(file_path: str) -> tuple[float | None, list[float], float] | None:
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        activations = RNNBeatProcessor()(file_path)
        tracked = DBNBeatTrackingProcessor(fps=100)(activations)
        if tracked is None:
            return None
        beat_times = np.atleast_1d(tracked).astype(float).tolist()
        beats = [round(float(value), 3) for value in beat_times if float(value) >= 0.0]
        if not beats:
            return None
        if len(beats) > 1:
            beat_intervals = np.diff(np.array(beats))
            bpm = round(float(60.0 / np.mean(beat_intervals)), 2)
            # Lower interval variation means steadier beat timing, so confidence increases.
            coefficient_of_variation = np.std(beat_intervals) / np.mean(beat_intervals)
            confidence = _clamp(float(1.0 / (1.0 + coefficient_of_variation)), 0.0, 1.0)
        else:
            bpm = None
            confidence = _SINGLE_BEAT_CONFIDENCE
        return bpm, beats, round(confidence, 3)
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug('madmom beat analysis unavailable: %s', exc)
        return None


def _beatnet_downbeats(file_path: str) -> list[float] | None:
    try:
        from BeatNet.BeatNet import BeatNet
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        detector = BeatNet(1, mode='offline', inference_model='DBN')
        prediction = detector.process(file_path)
        downbeats: list[float] = []
        for row in np.atleast_2d(prediction):
            if len(row) < 2:
                continue
            timestamp = float(row[0])
            beat_index = int(round(float(row[1])))
            if beat_index == 1:
                downbeats.append(round(timestamp, 3))
        return downbeats or None
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug('BeatNet downbeat analysis unavailable: %s', exc)
        return None


def _fallback_downbeats(beats: list[float]) -> list[float]:
    return [value for index, value in enumerate(beats) if index % 4 == 0]


def _detect_tempo_changes(beats: list[float]) -> list[dict]:
    if len(beats) < 8:
        return []

    intervals = np.diff(np.array(beats))
    if intervals.size == 0:
        return []

    reference_bpm = 60.0 / np.mean(intervals)
    tempo_changes: list[dict] = []
    window = _TEMPO_WINDOW_BEATS
    for index in range(0, len(intervals) - window + 1):
        local_interval = float(np.mean(intervals[index:index + window]))
        if local_interval <= 0:
            continue
        local_bpm = 60.0 / local_interval
        if abs(local_bpm - reference_bpm) >= _TEMPO_CHANGE_THRESHOLD_BPM:
            change_point = beats[index + 1]
            # Keep a minimum gap so we do not emit near-duplicate tempo change points.
            if tempo_changes and abs(tempo_changes[-1]['start_seconds'] - change_point) < _TEMPO_CHANGE_MIN_GAP_SECONDS:
                continue
            tempo_changes.append({
                'start_seconds': round(change_point, 3),
                'bpm': round(float(local_bpm), 2),
            })
    return tempo_changes


def _measure_lufs(y: np.ndarray, sr: int) -> float:
    if not y.size:
        return _SILENT_LUFS
    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            return round(float(meter.integrated_loudness(y.astype(np.float64))), 2)
        except Exception as exc:  # pragma: no cover - optional dependency
            log.debug('pyloudnorm loudness analysis unavailable: %s', exc)
    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms <= 0.0:
        return _SILENT_LUFS
    return round(float(20.0 * np.log10(rms)), 2)


def _derive_sections(downbeats: list[float], duration_seconds: float) -> list[dict]:
    if not downbeats:
        return [{'start_seconds': 0.0, 'end_seconds': duration_seconds, 'label': 'full_mix'}]

    sections: list[dict] = []
    for index in range(0, len(downbeats), _BARS_PER_SECTION):
        start = downbeats[index]
        end = downbeats[index + _BARS_PER_SECTION] if index + _BARS_PER_SECTION < len(downbeats) else duration_seconds
        section_number = (index // _BARS_PER_SECTION) + 1
        if end < start:
            log.debug('Section boundary corrected: end=%s < start=%s', end, start)
        end = max(end, start)
        sections.append({
            'start_seconds': round(float(start), 3),
            'end_seconds': round(float(end), 3),
            'label': f'section_{section_number}',
        })
    return sections


def _derive_cue_points(downbeats: list[float], beats: list[float], duration_seconds: float) -> list[dict]:
    cue_points: list[dict] = []
    if beats:
        cue_points.append({'name': 'intro', 'time_seconds': beats[0]})
    if downbeats:
        cue_points.append({'name': 'first_downbeat', 'time_seconds': downbeats[0]})
    if duration_seconds > 0:
        cue_points.append({'name': 'outro', 'time_seconds': round(max(0.0, duration_seconds - _OUTRO_OFFSET_SECONDS), 3)})
    unique: dict[str, dict] = {}
    for cue in cue_points:
        unique[cue['name']] = cue
    return list(unique.values())


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

    bpm, beats, bpm_confidence = _librosa_beats(y, sr)
    madmom_result = _madmom_beats(file_path)
    if madmom_result is not None:
        madmom_bpm, madmom_beats, madmom_confidence = madmom_result
        beats = madmom_beats or beats
        bpm_confidence = madmom_confidence
        if madmom_bpm is not None:
            bpm = madmom_bpm

    duration_seconds = round(float(librosa.get_duration(y=y, sr=sr)), 2)

    key, key_confidence = _detect_key_and_confidence(y, sr)
    chords = _detect_chords(y, sr)
    downbeats = _beatnet_downbeats(file_path) or _fallback_downbeats(beats)
    tempo_changes = _detect_tempo_changes(beats)
    lufs = _measure_lufs(y, sr)
    sections = _derive_sections(downbeats, duration_seconds)
    cue_points = _derive_cue_points(downbeats, beats, duration_seconds)

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
        'bpm_confidence': round(float(bpm_confidence), 3),
        'key': key,
        'key_confidence': key_confidence,
        'lufs': lufs,
        'duration_seconds': duration_seconds,
        'beats': beats,
        'downbeats': downbeats,
        'sections': sections,
        'cue_points': cue_points,
        'chords': chords,
        'tempo_changes': tempo_changes,
        'waveform': waveform,
        'frequency_histogram': frequency_histogram,
        'spectrogram_summary': spectrogram_summary,
        'loudness_curve': loudness_curve,
        'energy_over_time': energy_over_time,
    }
