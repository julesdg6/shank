"""Song metadata extraction helpers (credits + lyrics)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


_SPLIT_TOKENS = (';', '/', '&')


def _env_flag(name: str, default: str = 'true') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


def _split_people(value: str | None) -> list[str]:
    if not isinstance(value, str):
        return []
    normalized = value
    for token in _SPLIT_TOKENS:
        normalized = normalized.replace(token, ',')
    people = [item.strip() for item in normalized.split(',') if item.strip()]
    # preserve order while deduplicating
    return list(dict.fromkeys(people))


def _ffprobe_tags(audio_path: str) -> dict[str, str]:
    audio_file = Path(audio_path)
    if not audio_file.is_file():
        return {}
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format_tags',
        '-of',
        'json',
        str(audio_file),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or '{}')
    except json.JSONDecodeError:
        return {}
    tags = payload.get('format', {}).get('tags', {})
    if not isinstance(tags, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float)):
            normalized[key.strip().lower()] = str(value).strip()
    return normalized


def _first_tag(tags: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sidecar_lrc(audio_path: str) -> str | None:
    audio_file = Path(audio_path)
    if not audio_file.name:
        return None
    lrc_path = audio_file.with_suffix('.lrc')
    if not lrc_path.is_file():
        return None
    try:
        content = lrc_path.read_text()
    except OSError:
        return None
    return content.strip() or None


def _apply_lyrics_policy(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    mode = os.getenv('LYRICS_STORAGE_MODE', 'full').strip().lower()
    if mode in {'off', 'none', 'disabled'}:
        return None
    if mode in {'snippet', 'preview'}:
        max_chars_raw = os.getenv('LYRICS_SNIPPET_CHARS', '280')
        try:
            max_chars = max(32, int(max_chars_raw))
        except ValueError:
            max_chars = 280
        if len(text) <= max_chars:
            return text
        return f'{text[:max_chars].rstrip()}…'
    return text


def _provider_attribution(plain_lyrics: str | None, synced_lyrics: str | None) -> str | None:
    if plain_lyrics and synced_lyrics:
        return 'embedded tags + local sidecar'
    if plain_lyrics:
        return 'embedded tags'
    if synced_lyrics:
        return 'local sidecar'
    return None


def collect_song_metadata(task: dict[str, Any], source_audio_path: str) -> dict[str, Any]:
    tags = _ffprobe_tags(source_audio_path)
    youtube = task.get('youtube') if isinstance(task.get('youtube'), dict) else {}
    title = _first_tag(tags, 'title') or (youtube.get('title') if isinstance(youtube.get('title'), str) else None)
    artist = (
        _first_tag(tags, 'artist', 'album_artist')
        or (youtube.get('channel') if isinstance(youtube.get('channel'), str) else None)
    )
    album = _first_tag(tags, 'album')
    release_date = _first_tag(tags, 'date')
    release_year = None
    if isinstance(release_date, str) and len(release_date) >= 4 and release_date[:4].isdigit():
        release_year = int(release_date[:4])
    elif isinstance(youtube.get('release_year'), int):
        release_year = youtube['release_year']

    plain_lyrics = _first_tag(tags, 'lyrics', 'unsyncedlyrics', 'unsynchronised_lyrics')
    synced_lyrics = _sidecar_lrc(source_audio_path)

    if not _env_flag('SONG_METADATA_ENABLED', 'true'):
        plain_lyrics = None
        synced_lyrics = None

    provider_url = youtube.get('webpage_url') if isinstance(youtube.get('webpage_url'), str) else None
    credits = {
        'track_title': title,
        'artist': artist,
        'album': album,
        'release_date': release_date,
        'release_year': release_year,
        'label': _first_tag(tags, 'label', 'organization'),
        'isrc': _first_tag(tags, 'isrc'),
        'musicbrainz_recording_id': _first_tag(tags, 'musicbrainz_trackid', 'musicbrainz_recordingid'),
        'musicbrainz_release_id': _first_tag(tags, 'musicbrainz_releaseid', 'musicbrainz_albumid'),
        'discogs_release_id': _first_tag(tags, 'discogs_release_id', 'discogs_releaseid'),
        'writers': _split_people(_first_tag(tags, 'writer', 'writers')),
        'composers': _split_people(_first_tag(tags, 'composer', 'composers')),
        'lyricists': _split_people(_first_tag(tags, 'lyricist', 'lyricists')),
        'producers': _split_people(_first_tag(tags, 'producer', 'producers')),
        'remixers': _split_people(_first_tag(tags, 'remixer', 'mixartist')),
        'featured_artists': _split_people(_first_tag(tags, 'featured_artist', 'featured_artists')),
        'performers': _split_people(_first_tag(tags, 'performer', 'performers')),
        'engineers': _split_people(_first_tag(tags, 'engineer', 'engineers')),
        'mastering_engineer': _first_tag(tags, 'mastering_engineer', 'masteringengineer'),
        'publisher': _first_tag(tags, 'publisher'),
        'copyright': _first_tag(tags, 'copyright'),
        'phonographic_copyright': _first_tag(tags, 'phonographic_copyright', 'p_line'),
        'source_url': provider_url,
        'confidence_score': round(
            min(
                1.0,
                0.35
                + (0.2 if isinstance(title, str) and title else 0.0)
                + (0.2 if isinstance(artist, str) and artist else 0.0)
                + (0.15 if tags else 0.0)
                + (0.1 if provider_url else 0.0),
            ),
            3,
        ),
    }

    lyrics = {
        'plain_lyrics': _apply_lyrics_policy(plain_lyrics),
        'synced_lyrics_lrc': _apply_lyrics_policy(synced_lyrics),
        'provider_url': provider_url,
        'provider_attribution': _provider_attribution(plain_lyrics, synced_lyrics),
        'language': _first_tag(tags, 'language'),
        'confidence_score': round(
            min(
                1.0,
                (0.75 if synced_lyrics else 0.0)
                + (0.55 if plain_lyrics else 0.0)
                + (0.1 if provider_url else 0.0),
            ),
            3,
        ),
    }

    return {
        'enabled': _env_flag('SONG_METADATA_ENABLED', 'true'),
        'credits': credits,
        'lyrics': lyrics,
        'source_audio_path': source_audio_path,
    }
