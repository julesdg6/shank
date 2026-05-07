from pathlib import Path


def test_compose_uses_single_unified_service():
    compose = Path(__file__).resolve().parents[1] / 'docker-compose.yml'
    text = compose.read_text()

    assert 'shank:' in text
    assert 'build: .' in text
    assert 'shank-api:' not in text
    assert 'shank-worker:' not in text
