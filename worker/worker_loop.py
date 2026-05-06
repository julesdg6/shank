import json
import os
import subprocess
import time
from pathlib import Path

DATA_DIR = Path(os.getenv('DATA_DIR', '/srv/shank/data'))
TASKS_DIR = DATA_DIR / 'tasks'
NORMALIZED_DIR = DATA_DIR / 'normalized'

# Standard WAV output format
WAV_SAMPLE_RATE = '44100'
WAV_CHANNELS = '2'
WAV_CODEC = 'pcm_s16le'

POLL_INTERVAL = 10  # seconds


def normalize_audio(input_path: str, output_path: str) -> None:
    """Normalize an audio file to a standard WAV format using ffmpeg.

    Output: 44100 Hz, stereo, 16-bit PCM WAV.
    Raises RuntimeError if ffmpeg exits with a non-zero status.
    """
    cmd = [
        'ffmpeg',
        '-y',              # overwrite output file without prompting
        '-i', input_path,
        '-ar', WAV_SAMPLE_RATE,
        '-ac', WAV_CHANNELS,
        '-c:a', WAV_CODEC,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed (exit {result.returncode}): {result.stderr}')


def process_task(task_file: Path) -> None:
    """Read a task JSON file and, if pending, normalize its audio via ffmpeg."""
    task = json.loads(task_file.read_text())

    if task.get('status') != 'pending':
        return

    # URL tasks have no local file yet (yt-dlp not implemented); skip them.
    input_path = task.get('file_path')
    if not input_path:
        return

    task_id = task['task_id']

    # Mark as in-progress so another worker instance won't pick it up.
    task['status'] = 'processing'
    task_file.write_text(json.dumps(task, indent=2))

    try:
        NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(NORMALIZED_DIR / f'{task_id}.wav')

        normalize_audio(input_path, output_path)

        task['status'] = 'completed'
        task['normalized_path'] = output_path
    except Exception as exc:
        task['status'] = 'failed'
        task['error'] = str(exc)

    task_file.write_text(json.dumps(task, indent=2))


def run_worker() -> None:
    """Main loop: poll TASKS_DIR every POLL_INTERVAL seconds for pending tasks."""
    print('Worker running...')
    while True:
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        for task_file in sorted(TASKS_DIR.glob('*.json')):
            try:
                process_task(task_file)
            except Exception as exc:  # noqa: BLE001
                # Include task_id (stem of the JSON filename) to aid log correlation.
                print(f'[worker] error processing task {task_file.stem}: {exc}')
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    run_worker()