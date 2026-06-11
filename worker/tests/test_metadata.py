import json
from unittest.mock import patch

import metadata


def test_collect_song_metadata_uses_youtube_and_embedded_lyrics(tmp_path):
    audio_file = tmp_path / 'song.mp3'
    audio_file.write_bytes(b'ID3')

    ffprobe_stdout = json.dumps({
        'format': {
            'tags': {
                'title': 'Test Song',
                'artist': 'Test Artist',
                'lyrics': 'line one\nline two',
                'language': 'en',
            },
        },
    })
    task = {
        'youtube': {
            'webpage_url': 'https://www.youtube.com/watch?v=test1234567a',
        },
    }

    with patch('metadata.subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ffprobe_stdout
        result = metadata.collect_song_metadata(task, str(audio_file))

    assert result['credits']['track_title'] == 'Test Song'
    assert result['credits']['artist'] == 'Test Artist'
    assert result['lyrics']['plain_lyrics'] == 'line one\nline two'
    assert result['lyrics']['provider_url'] == 'https://www.youtube.com/watch?v=test1234567a'


def test_collect_song_metadata_reads_sidecar_lrc(tmp_path):
    audio_file = tmp_path / 'song.wav'
    lrc_file = tmp_path / 'song.lrc'
    audio_file.write_bytes(b'RIFF')
    lrc_file.write_text('[00:01.00]hello world')

    with patch('metadata.subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps({'format': {'tags': {}}})
        result = metadata.collect_song_metadata({}, str(audio_file))

    assert result['lyrics']['synced_lyrics_lrc'] == '[00:01.00]hello world'
