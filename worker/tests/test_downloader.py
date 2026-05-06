"""Tests for worker/downloader.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

# Make the worker package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent))

import downloader  # noqa: E402


TASK_ID = 'abc12345-0000-0000-0000-000000000000'
YOUTUBE_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'


def test_download_youtube_returns_mp3_path(tmp_path):
    """download_youtube should return the expected .mp3 path."""
    with patch('downloader.yt_dlp.YoutubeDL') as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

    assert result == tmp_path / f'{TASK_ID}.mp3'


def test_download_youtube_calls_ydl_download(tmp_path):
    """download_youtube must call yt_dlp.YoutubeDL.download with the given URL."""
    with patch('downloader.yt_dlp.YoutubeDL') as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

        downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

        mock_ydl.download.assert_called_once_with([YOUTUBE_URL])


def test_download_youtube_uses_mp3_postprocessor(tmp_path):
    """The YoutubeDL options must request MP3 extraction via FFmpegExtractAudio."""
    captured_opts: list[dict] = []

    def fake_init(opts):
        captured_opts.append(opts)
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch('downloader.yt_dlp.YoutubeDL', side_effect=fake_init):
        downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

    opts = captured_opts[0]
    pp = opts['postprocessors'][0]
    assert pp['key'] == 'FFmpegExtractAudio'
    assert pp['preferredcodec'] == 'mp3'


def test_download_youtube_noplaylist_option(tmp_path):
    """noplaylist=True must be set to prevent downloading entire playlists."""
    captured_opts: list[dict] = []

    def fake_init(opts):
        captured_opts.append(opts)
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch('downloader.yt_dlp.YoutubeDL', side_effect=fake_init):
        downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

    assert captured_opts[0].get('noplaylist') is True


def test_download_youtube_rejects_non_uuid_task_id(tmp_path):
    """A task_id that is not a valid UUID must raise ValueError (path traversal guard)."""
    with pytest.raises(ValueError):
        downloader.download_youtube(YOUTUBE_URL, tmp_path, '../../../etc/passwd')


def test_download_youtube_creates_output_dir(tmp_path):
    """download_youtube should create output_dir if it does not exist."""
    nested = tmp_path / 'a' / 'b' / 'c'
    assert not nested.exists()

    with patch('downloader.yt_dlp.YoutubeDL') as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

        downloader.download_youtube(YOUTUBE_URL, nested, TASK_ID)

    assert nested.exists()


def test_download_youtube_propagates_download_error(tmp_path):
    """Exceptions raised by yt-dlp must bubble up to the caller."""
    with patch('downloader.yt_dlp.YoutubeDL') as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.download.side_effect = yt_dlp.utils.DownloadError('network error')
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(yt_dlp.utils.DownloadError):
            downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)
