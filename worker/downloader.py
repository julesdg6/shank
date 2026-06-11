"""YouTube audio downloader using yt-dlp."""
import logging
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

    log.info('Downloading audio from %s → %s/%s.mp3', url, output_dir, canonical_task_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_dir / f'{canonical_task_id}.mp3'
