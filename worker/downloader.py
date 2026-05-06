"""YouTube audio downloader using yt-dlp."""
import logging
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

    Returns
    -------
    Path to the downloaded MP3 file.

    Raises
    ------
    yt_dlp.utils.DownloadError on network/format errors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f'{task_id}.%(ext)s')

    ydl_opts: dict = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
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

    log.info('Downloading audio from %s → %s/%s.mp3', url, output_dir, task_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_dir / f'{task_id}.mp3'
