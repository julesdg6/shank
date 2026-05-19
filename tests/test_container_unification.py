from pathlib import Path


def test_compose_uses_single_unified_service():
    compose = Path(__file__).resolve().parents[1] / 'docker-compose.yml'
    text = compose.read_text()

    assert 'shank:' in text
    assert 'shank-mt3:' in text
    assert 'build: .' in text
    assert '- "8088:8080"' in text
    assert '- DATA_DIR=/srv/shank/data' in text
    assert '- POLL_INTERVAL=10' in text
    assert '- MT3_ENABLED=${MT3_ENABLED:-false}' in text
    assert '- MT3_SERVICE_URL=${MT3_SERVICE_URL:-http://shank-mt3:8090}' in text
    assert '- MT3_MODEL=${MT3_MODEL:-multi_instrument}' in text
    assert '- MT3_DEVICE=${MT3_DEVICE:-auto}' in text
    assert '- MT3_TIMEOUT=${MT3_TIMEOUT:-1800}' in text
    assert '- ./data:/srv/shank/data' in text
    assert '- ./cache/mt3:/srv/shank/cache/mt3' in text
    assert '- ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro' in text
    assert 'profiles:' in text
    assert 'mt3' in text
    assert 'command: uvicorn services.mt3.main:app --host 0.0.0.0 --port 8090 --log-level info' in text
    assert '# Optional NVIDIA GPU runtime example:' in text
    assert '# gpus: all' in text
    assert 'capabilities: [gpu]' in text
    assert 'shank-api:' not in text
    assert 'shank-worker:' not in text
    assert 'shank-mt3-gpu:' not in text


def test_supervisord_runs_api_worker_and_mt3():
    config = Path(__file__).resolve().parents[1] / 'docker' / 'supervisord.conf'
    text = config.read_text()

    assert '[program:api]' in text
    assert '[program:worker]' in text
    assert '[program:mt3]' not in text


def test_dockerfile_installs_audio_processing_tools():
    dockerfile = Path(__file__).resolve().parents[1] / 'Dockerfile'
    text = dockerfile.read_text()

    assert 'apt-get install -y --no-install-recommends' in text
    assert 'ffmpeg' in text
    assert 'yt-dlp' in text
    assert 'supervisor' in text
