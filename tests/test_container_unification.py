from pathlib import Path


def test_compose_uses_single_unified_service():
    compose = Path(__file__).resolve().parents[1] / 'docker-compose.yml'
    text = compose.read_text()

    assert 'shank:' in text
    assert 'build: .' in text
    assert '- "8088:8080"' in text
    assert '- DATA_DIR=/srv/shank/data' in text
    assert '- POLL_INTERVAL=10' in text
    assert '- ./data:/srv/shank/data' in text
    assert 'shank-api:' not in text
    assert 'shank-worker:' not in text


def test_supervisord_runs_api_and_worker():
    config = Path(__file__).resolve().parents[1] / 'docker' / 'supervisord.conf'
    text = config.read_text()

    assert '[program:api]' in text
    assert '[program:worker]' in text
