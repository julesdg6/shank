"""YouTube audio downloader using yt-dlp."""
import logging
import uuid
from pathlib import Path

import yt_dlp

log = logging.getLogger(__name__)


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
