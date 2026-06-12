"""YouTube audio downloader using yt-dlp."""
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import yt_dlp

log = logging.getLogger(__name__)

# Matches video IDs from:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://www.youtube.com/shorts/VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://music.youtube.com/watch?v=VIDEO_ID
_YOUTUBE_ID_RE = re.compile(
    r'(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/|music\.youtube\.com/watch\?v=)'
    r'([A-Za-z0-9_-]{11})'
)

_YTDLP_COOKIES_FILE_ENV = 'YTDLP_COOKIES_FILE'


def _resolve_cookies_file() -> Path | None:
    """Return configured yt-dlp cookies file if configured and present."""
    configured_path = os.getenv(_YTDLP_COOKIES_FILE_ENV, '').strip()
    if not configured_path:
        return None

    cookies_file = Path(configured_path)
    if cookies_file.is_file():
        return cookies_file

    log.warning('Configured %s does not exist or is not a regular file: %s', _YTDLP_COOKIES_FILE_ENV, configured_path)
    return None


def _apply_yt_dlp_cookies(ydl_opts: dict[str, Any]) -> None:
    """Set yt-dlp cookiefile option when a configured file is available."""
    cookies_file = _resolve_cookies_file()
    if cookies_file is None:
        return
    ydl_opts['cookiefile'] = str(cookies_file)


def _rewrite_blocked_youtube_error(exc: yt_dlp.utils.DownloadError) -> yt_dlp.utils.DownloadError:
    """Return a clearer error when YouTube blocks unauthenticated yt-dlp requests."""
    message = str(exc)
    lowered = message.lower()
    bot_check_markers = (
        "not a bot",
        'cookies-from-browser',
        '--cookies',
        'for the authentication',
    )
    if not any(marker in lowered for marker in bot_check_markers):
        return exc

    return yt_dlp.utils.DownloadError(
        'YouTube blocked this request. Add a valid cookies export for yt-dlp and set '
        f'{_YTDLP_COOKIES_FILE_ENV} (for Docker: mount ./config:/srv/shank/config and set '
        'YTDLP_COOKIES_FILE=/srv/shank/config/youtube-cookies.txt).'
    )


def extract_video_id(url: str) -> str | None:
    """Return the YouTube video ID extracted from *url*, or ``None`` if not found."""
    m = _YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def extract_youtube_metadata(url: str) -> dict[str, Any]:
    """Fetch YouTube video metadata without downloading the audio.

    Returns a dict with keys: ``video_id``, ``title``, ``channel``,
    ``duration``, ``thumbnail``, ``webpage_url``.  Any field that is
    unavailable will be ``None``.

    Raises ``yt_dlp.utils.DownloadError`` on network or format errors.
    """
    ydl_opts: dict = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    _apply_yt_dlp_cookies(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False) or {}

    return {
        'video_id': info.get('id') or extract_video_id(url),
        'title': info.get('title'),
        'channel': info.get('channel') or info.get('uploader'),
        'duration': info.get('duration'),
        'thumbnail': info.get('thumbnail'),
        'webpage_url': info.get('webpage_url') or url,
    }


def download_youtube(url: str, output_dir: Path, task_id: str) -> Path:
    """Download the best audio track from *url* and save it as an MP3.

    Parameters
    ----------
    url:        A validated YouTube HTTPS URL.
    output_dir: Directory where the downloaded file will be written.
    task_id:    Used as the base filename so the result is easy to find.
                Must be a valid UUID string; raises ValueError otherwise.

    Returns
    -------
    Path to the downloaded MP3 file.

    Raises
    ------
    ValueError              if task_id is not a valid UUID.
    yt_dlp.utils.DownloadError on network/format errors.
    """
    # Parse task_id as UUID to canonicalize it and guard against path traversal.
    canonical_task_id = str(uuid.UUID(task_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f'{canonical_task_id}.%(ext)s')

    ydl_opts: dict = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }
        ],
        'quiet': True,
        'no_warnings': True,
    }
    _apply_yt_dlp_cookies(ydl_opts)

    log.info('Downloading audio from %s → %s/%s.mp3', url, output_dir, canonical_task_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise _rewrite_blocked_youtube_error(exc) from exc

    return output_dir / f'{canonical_task_id}.mp3'
