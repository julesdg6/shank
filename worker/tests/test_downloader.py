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


def test_download_youtube_uses_configured_cookies_file(tmp_path, monkeypatch):
    """A valid YTDLP_COOKIES_FILE must be forwarded as cookiefile option."""
    captured_opts: list[dict] = []
    cookies_file = tmp_path / 'youtube-cookies.txt'
    cookies_file.write_text('# Netscape HTTP Cookie File')
    monkeypatch.setenv('YTDLP_COOKIES_FILE', str(cookies_file))

    def fake_init(opts):
        captured_opts.append(opts)
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch('downloader.yt_dlp.YoutubeDL', side_effect=fake_init):
        downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

    assert captured_opts[0].get('cookiefile') == str(cookies_file)


def test_download_youtube_ignores_missing_configured_cookies_file(tmp_path, monkeypatch):
    """A missing YTDLP_COOKIES_FILE must not add cookiefile option."""
    captured_opts: list[dict] = []
    missing = tmp_path / 'missing-cookies.txt'
    monkeypatch.setenv('YTDLP_COOKIES_FILE', str(missing))

    def fake_init(opts):
        captured_opts.append(opts)
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock())
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch('downloader.yt_dlp.YoutubeDL', side_effect=fake_init):
        downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)

    assert 'cookiefile' not in captured_opts[0]


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


def test_download_youtube_rewrites_bot_check_error_with_cookie_guidance(tmp_path):
    """YouTube bot-check errors should include actionable cookies guidance."""
    bot_check_message = (
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot. "
        'Use --cookies-from-browser or --cookies for the authentication.'
    )
    with patch('downloader.yt_dlp.YoutubeDL') as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.download.side_effect = yt_dlp.utils.DownloadError(bot_check_message)
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(yt_dlp.utils.DownloadError, match='YTDLP_COOKIES_FILE'):
            downloader.download_youtube(YOUTUBE_URL, tmp_path, TASK_ID)


# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('url,expected', [
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://youtu.be/dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://music.youtube.com/watch?v=dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://www.youtube.com/shorts/dQw4w9WgXcQ', 'dQw4w9WgXcQ'),
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123', 'dQw4w9WgXcQ'),
    ('https://example.com/not-youtube', None),
    ('', None),
])
def test_extract_video_id(url, expected):
    assert downloader.extract_video_id(url) == expected


# ---------------------------------------------------------------------------
# extract_youtube_metadata
# ---------------------------------------------------------------------------

def test_extract_youtube_metadata_returns_expected_keys():
    """extract_youtube_metadata must return the documented metadata keys."""
    fake_info = {
        'id': 'dQw4w9WgXcQ',
        'title': 'Never Gonna Give You Up',
        'channel': 'Rick Astley',
        'duration': 213,
        'thumbnail': 'https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg',
        'webpage_url': YOUTUBE_URL,
    }
    with patch('downloader.yt_dlp.YoutubeDL') as mock_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = fake_info
        mock_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)

        meta = downloader.extract_youtube_metadata(YOUTUBE_URL)

    assert meta['video_id'] == 'dQw4w9WgXcQ'
    assert meta['title'] == 'Never Gonna Give You Up'
    assert meta['channel'] == 'Rick Astley'
    assert meta['duration'] == 213
    assert meta['thumbnail'] == 'https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg'
    assert meta['webpage_url'] == YOUTUBE_URL


def test_extract_youtube_metadata_falls_back_to_url_for_video_id():
    """When yt-dlp returns no id, the video ID should be parsed from the URL."""
    with patch('downloader.yt_dlp.YoutubeDL') as mock_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {}
        mock_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)

        meta = downloader.extract_youtube_metadata(YOUTUBE_URL)

    assert meta['video_id'] == 'dQw4w9WgXcQ'


def test_extract_youtube_metadata_calls_extract_info_without_download():
    """extract_youtube_metadata must NOT trigger a download (download=False)."""
    with patch('downloader.yt_dlp.YoutubeDL') as mock_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {}
        mock_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)

        downloader.extract_youtube_metadata(YOUTUBE_URL)

        mock_ydl.extract_info.assert_called_once_with(YOUTUBE_URL, download=False)


def test_extract_youtube_metadata_uses_configured_cookies_file(tmp_path, monkeypatch):
    """Metadata extraction should also respect YTDLP_COOKIES_FILE when present."""
    cookies_file = tmp_path / 'youtube-cookies.txt'
    cookies_file.write_text('# Netscape HTTP Cookie File')
    monkeypatch.setenv('YTDLP_COOKIES_FILE', str(cookies_file))
    captured_opts: list[dict] = []

    def fake_init(opts):
        captured_opts.append(opts)
        m = MagicMock()
        inner = MagicMock()
        inner.extract_info.return_value = {}
        m.__enter__ = MagicMock(return_value=inner)
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch('downloader.yt_dlp.YoutubeDL', side_effect=fake_init):
        downloader.extract_youtube_metadata(YOUTUBE_URL)

    assert captured_opts[0].get('cookiefile') == str(cookies_file)
