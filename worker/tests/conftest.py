"""Pytest configuration for worker tests: ensures worker/ is on sys.path."""

import os
import sys

# Add the worker package root to sys.path so tests can import analyze and worker_loop
_WORKER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)
