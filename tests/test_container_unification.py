from pathlib import Path

import mt3_config


def test_compose_uses_single_unified_service():
    compose = Path(__file__).resolve().parents[1] / 'docker-compose.yml'
    text = compose.read_text()
    expected_service_url = f'- MT3_SERVICE_URL=${{MT3_SERVICE_URL:-{mt3_config.DEFAULT_MT3_SERVICE_URL}}}'
    expected_model = f'- MT3_MODEL=${{MT3_MODEL:-{mt3_config.DEFAULT_MT3_MODEL}}}'
    expected_timeout = f'- MT3_TIMEOUT=${{MT3_TIMEOUT:-{mt3_config.DEFAULT_MT3_TIMEOUT}}}'
    expected_checkpoint_root = f'- MT3_CHECKPOINT_ROOT=${{MT3_CHECKPOINT_ROOT:-{mt3_config.DEFAULT_MT3_CHECKPOINT_ROOT}}}'
    expected_cache_dir = f'- MT3_CACHE_DIR=${{MT3_CACHE_DIR:-{mt3_config.DEFAULT_MT3_CACHE_DIR}}}'
    expected_output_path = f'- MT3_OUTPUT_PATH=${{MT3_OUTPUT_PATH:-{mt3_config.DEFAULT_MT3_OUTPUT_PATH}}}'

    assert 'shank:' in text
    assert 'build: .' in text
    assert '- "8088:8080"' in text
    assert '- DATA_DIR=/srv/shank/data' in text
    assert '- POLL_INTERVAL=10' in text
    assert '- MT3_ENABLED=${MT3_ENABLED:-false}' in text
    assert expected_service_url in text
    assert expected_model in text
    assert '- MT3_DEVICE=${MT3_DEVICE:-auto}' in text
    assert expected_timeout in text
    assert expected_checkpoint_root in text
    assert expected_cache_dir in text
    assert expected_output_path in text
    assert '- ./data:/srv/shank/data' in text
    assert '- ./cache/mt3:/srv/shank/cache/mt3' in text
    assert '- ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro' in text
    assert 'healthcheck:' in text
    assert "http://127.0.0.1:8080/openapi.json" in text
    assert '# Optional NVIDIA GPU runtime example:' in text
    assert '# gpus: all' in text
    assert 'capabilities: [gpu]' in text
    assert 'shank-api:' not in text
    assert 'shank-worker:' not in text
    assert 'shank-mt3:' not in text
    assert 'shank-mt3-gpu:' not in text


def test_supervisord_runs_api_worker_and_mt3():
    config = Path(__file__).resolve().parents[1] / 'docker' / 'supervisord.conf'
    text = config.read_text()

    assert '[program:api]' in text
    assert '[program:worker]' in text
    assert '[program:mt3]' in text
    assert 'command=uvicorn services.mt3.main:app --host 0.0.0.0 --port 8090 --log-level info' in text


def test_dockerfile_installs_audio_processing_tools():
    dockerfile = Path(__file__).resolve().parents[1] / 'Dockerfile'
    text = dockerfile.read_text()

    assert 'ffmpeg' in text
    assert 'yt-dlp' in text
    assert 'HEALTHCHECK' in text


def test_worker_dockerfile_installs_audio_processing_tools():
    dockerfile = Path(__file__).resolve().parents[1] / 'worker' / 'Dockerfile'
    text = dockerfile.read_text()

    assert 'ffmpeg' in text
    assert 'yt-dlp' in text


def test_env_example_documents_mt3_paths():
    env_example = Path(__file__).resolve().parents[1] / '.env.example'
    text = env_example.read_text()

    assert f'MT3_SERVICE_URL={mt3_config.DEFAULT_MT3_SERVICE_URL}' in text
    assert f'MT3_CHECKPOINT_ROOT={mt3_config.DEFAULT_MT3_CHECKPOINT_ROOT}' in text
    assert f'MT3_CACHE_DIR={mt3_config.DEFAULT_MT3_CACHE_DIR}' in text
    assert f'MT3_OUTPUT_PATH={mt3_config.DEFAULT_MT3_OUTPUT_PATH}' in text
