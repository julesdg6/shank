"""Shared MT3 configuration defaults used by runtime, tests, and deployment docs."""

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path('/srv/shank/data')
DEFAULT_MT3_SERVICE_URL = 'http://127.0.0.1:8090'
DEFAULT_MT3_MODEL = 'multi_instrument'
DEFAULT_MT3_TIMEOUT = 1800
DEFAULT_MT3_CHECKPOINT_ROOT = Path('/srv/shank/models/mt3/checkpoints')
DEFAULT_MT3_CACHE_DIR = Path('/srv/shank/cache/mt3')
DEFAULT_MT3_OUTPUT_PATH = DEFAULT_DATA_DIR / 'mt3'


def get_mt3_output_path(data_dir: Path | None = None) -> Path:
    """Return the configured MT3 output directory."""
    effective_data_dir = Path(data_dir) if data_dir is not None else Path(
        os.getenv('DATA_DIR', str(DEFAULT_DATA_DIR))
    )
    return Path(os.getenv('MT3_OUTPUT_PATH', str(effective_data_dir / 'mt3')))
