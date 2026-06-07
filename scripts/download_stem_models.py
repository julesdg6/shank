#!/usr/bin/env python3
"""Download Htdemucs stem-separation models for python-audio-separator.

Usage
-----
    python3 scripts/download_stem_models.py [--6stems] [--model-dir DIR]

Options
-------
--6stems        Also download the 6-stem model (htdemucs_6s.yaml) in addition
                to the default 4-stem model (htdemucs_ft.yaml).
--model-dir DIR Directory to store model files.
                Defaults to the value of the AUDIO_SEPARATOR_MODEL_DIR
                environment variable, or /srv/shank/models/separator if unset.
--help          Show this help message and exit.

The script uses python-audio-separator's built-in downloader, so
``audio-separator`` must be installed first:

    pip install audio-separator[cpu]
"""
from __future__ import annotations

import argparse
import os


_DEFAULT_MODEL_DIR = os.getenv(
    'AUDIO_SEPARATOR_MODEL_DIR',
    '/srv/shank/models/separator',
)

_MODELS_4STEM = ['htdemucs_ft.yaml']
_MODELS_6STEM = ['htdemucs_6s.yaml']


def _ensure_audio_separator() -> None:
    """Exit with a friendly message if audio_separator is not installed."""
    try:
        import audio_separator  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'ERROR: audio-separator is not installed.\n'
            'Install it with:\n'
            '    pip install audio-separator[cpu]\n'
            'or, for GPU support:\n'
            '    pip install audio-separator[gpu]'
        ) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Download Htdemucs models for python-audio-separator stem separation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--6stems',
        dest='six_stems',
        action='store_true',
        help='Also download the 6-stem model (htdemucs_6s.yaml).',
    )
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        default=_DEFAULT_MODEL_DIR,
        metavar='DIR',
        help=f'Directory to store model files (default: {_DEFAULT_MODEL_DIR}).',
    )
    args = parser.parse_args(argv)

    # Verify the dependency is available before doing anything else.
    _ensure_audio_separator()

    # Deferred import: audio_separator may not be installed in all environments.
    # _ensure_audio_separator() above already exited with a helpful message when
    # the package is missing, so this import is guaranteed to succeed here.
    from audio_separator.separator import Separator  # type: ignore[import]  # noqa: PLC0415

    models = list(_MODELS_4STEM)
    if args.six_stems:
        models += _MODELS_6STEM

    print(f'Model directory: {args.model_dir}')
    print(f'Models to download: {", ".join(models)}')
    print()

    os.makedirs(args.model_dir, exist_ok=True)
    # Create a single Separator instance and reuse it for all downloads to
    # avoid redundant initialisation overhead.
    separator = Separator(model_file_dir=args.model_dir)

    for model_name in models:
        print(f'  Downloading {model_name} …', flush=True)
        separator.download_model_files(model_name)
        print(f'  ✅  {model_name} ready.', flush=True)

    print()
    print('All models downloaded successfully.')
    print()
    print('Verify with:')
    print(f'  ls -lh {args.model_dir}')


if __name__ == '__main__':
    main()
