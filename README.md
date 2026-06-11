# 🎵 SHANK: Self-Hosted Audio Notes Kit

SHANK is a Dockerized, self-hosted toolkit for deep music analysis across uploaded audio files and YouTube videos. It produces practical outputs for creators and DJs, including BPM, key, chord progression data, separated stems, and optional MIDI transcription.

## 🎯 Project Aim
To provide users with an automated pipeline that transforms raw audio/URLs into structured musical intelligence, including tempo, key, chord progressions, and MIDI melodies.

## 🚀 Key Features
- **Multi-Source Input**: Direct audio uploads (MP3, WAV, FLAC) and YouTube URLs (via `yt-dlp`).
- **Audio Normalization**: Automatic conversion to standard 44100 Hz stereo WAV using `ffmpeg`.
- **Musical Extraction**:
    - **BPM & Tempo**: Beat tracking with confidence, beat grid, and tempo-change hints.
    - **Downbeats**: Optional deep-learning downbeat detection (BeatNet) with automatic fallback.
    - **Musical Key**: Krumhansl-Kessler key detection (e.g. `A minor`, `C major`).
    - **Chord Progressions**: Segment-level chord summaries.
    - **Loudness (LUFS)**: Optional `pyloudnorm` integrated loudness with fallback estimate.
    - **Melody to MIDI**: Optional transcription backend (`basic_pitch`, with `mt3`/`omnizart` placeholders)
    - **Song Structure**: Detection of intro, verse, chorus, etc. *(planned)*
- **Built-in Stem Separation**: [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) separates vocals, drums, bass, and other instruments (4-stem or 6-stem) — no external service required. Optional **ACE-Step** integration for comparison.
- **Optional MT3 Transcription**: Worker can call a dedicated `shank-mt3` service to generate MIDI + note metadata from normalized mix and stems.
- **Asynchronous Workflow**: A background worker polls a filesystem task queue and processes jobs independently of the API.
- **Structured Result Artifacts**: Completed tasks now write a predictable `DATA_DIR/results/<task_id>/` folder with `task.json`, `analysis.json`, `beatgrid.json`, `waveform_beats.png`, `tempo_curve.png`, `beatgraph.png`, `mt3.json`, and `artifacts.json`.
- **Web Dashboard**: A built-in UI at `/ui` to submit tasks, monitor progress, and inspect results.
- **MCP Automation Server**: Optional MCP server exposing SHANK task operations for automation clients.

### Setup

To enable stem separation, download the models:

```bash
# Download Htdemucs models
python3 scripts/download_stem_models.py

# For 6 stems (guitar, piano included):
python3 scripts/download_stem_models.py --6stems
```

Options:
- `-h` or `--help` → Show all options
- `--6stems` → Also download the 6-stem model
- `--model-dir DIR` → Custom directory for models

### Verify installation

```bash
ls /srv/shank/models/separator/
# Should show:
#   htdemucs_ft.yaml
#   (and related model files)
```

### 6-stem model (optional)

To also get guitar and piano stems:

```bash
python3 scripts/download_stem_models.py --6stems
# Downloads ~530 MB
```

The 6-stem model includes:
- Vocals
- Drums
- Bass
- Guitar
- Piano
- Other

### Troubleshooting

#### Models download fails

```bash
# Ensure audio-separator is installed
docker compose exec shank pip list | grep -i audio

# If missing:
docker compose exec shank pip install audio-separator[cpu]

# Retry
docker compose exec shank python3 scripts/download_stem_models.py
```

#### Check model files

```bash
docker compose exec shank ls -lh /srv/shank/models/separator/
```

#### GPU acceleration (optional)

If you have an NVIDIA GPU, you can accelerate model inference:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 scripts/download_stem_models.py
```

Set in `.env`:

```dotenv
AUDIO_SEPARATOR_DEVICE=cuda
```

## 🛠 Technical Stack
- **Backend**: FastAPI (Python) served by Uvicorn
- **Worker**: Python — `librosa`, `numpy`, `scipy`, `yt-dlp`, `ffmpeg`
- **Process Management**: `supervisord` runs the API server and background worker in a single container
- **Deployment**: Docker & Docker Compose
- **Orchestration**: Asynchronous task queue via filesystem polling

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Clone and configure
```bash
git clone https://github.com/julesdg6/shank.git
cd shank
cp .env.example .env   # edit as needed
```

### 2. (Optional) Pre-download stem separation models

Model weights are fetched automatically on first use, but you can pre-download them to avoid the delay:

```bash
# 4-stem model (htdemucs_ft.yaml, ~400 MB) — default
python3 scripts/download_stem_models.py

# Also download the 6-stem model (htdemucs_6s.yaml, ~530 MB)
python3 scripts/download_stem_models.py --6stems

# Options:
#   --help           Show help
#   --6stems         Also download 6-stem model
#   --model-dir DIR  Custom directory (default: /srv/shank/models/separator)
```

Requires `audio-separator` to be installed (`pip install audio-separator[cpu]`). See the [Stem Separation](#-stem-separation-python-audio-separator) section for full details.

### 3. Start the service
```bash
docker compose up --build -d
# For older standalone Compose (for example docker-compose 1.29.x):
docker-compose up --build -d
```

The API and Web UI are available at **http://localhost:8088**.

> **Port note:** The container's internal API port is **8080**. Docker Compose maps it to host port **8088** (`"8088:8080"`). The Dockerfile and `docker-compose.yml` healthchecks both probe `http://127.0.0.1:8080` (the internal port). Always use port **8088** when accessing SHANK from your browser or host tools.
>
> **Applying `.env` changes:** if environment changes are not picked up, recreate the container:
> `docker compose up -d --force-recreate`  
> `docker-compose up -d --force-recreate`

### 4. Open the dashboard
Navigate to **http://localhost:8088/** in your browser to upload audio files or submit YouTube URLs. The dashboard is also available at **http://localhost:8088/ui**.

### 5. Stop the service
```bash
docker compose down
# or: docker-compose down
```

## 🖥️ Unraid 7+ setup

An Unraid template has been added at `/unraid/shank.xml` with icon `/unraid/shank-icon.png`.

### 1. Install template + icon to the Unraid boot drive

```bash
curl -fsSL https://raw.githubusercontent.com/julesdg6/shank/master/unraid/shank.xml \
  -o /boot/config/plugins/dockerMan/templates-user/shank.xml

curl -fsSL https://raw.githubusercontent.com/julesdg6/shank/master/unraid/shank-icon.png \
  -o /boot/config/plugins/dockerMan/images/shank-icon.png
```

Then open **Docker > Add Container** and select the `shank` template.

### 2. Configure paths and launch

- Set your data path (template default): `/mnt/user/appdata/shank`
- The template maps container-internal port **8080** to host port **8088** by default. Change the host port value if 8088 is already in use.
- Save and start the container

### 3. GPU flags (optional)

In Unraid's Docker template advanced view, set:

- **Extra Parameters**: `--gpus all`
- **Environment Variables**:
  - `AUDIO_SEPARATOR_DEVICE=cuda`
  - `MT3_DEVICE=gpu`

If you are not using an NVIDIA GPU, leave these unset and keep CPU defaults.

## 👩‍💻 Local Development Workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run checks locally:

```bash
ruff check .
mypy
python -m pytest api/tests/ -v
python -m pytest worker/tests/ -v
python -m pytest tests/ -v
docker build -t shank:local .
```

`mypy` currently checks API modules (`api/main.py` and `api/mcp_server.py`). Worker modules are excluded because they rely heavily on dynamic third-party audio libraries.

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser landing page (dashboard) or JSON health check for API clients |
| `POST` | `/tasks/upload` | Upload an audio file (MP3, WAV, FLAC, max 200 MB) |
| `POST` | `/tasks/melody` | Upload audio and queue a melody-focused analysis task |
| `POST` | `/tasks/url` | Submit a YouTube URL for download and analysis |
| `GET` | `/tasks/{task_id}` | Retrieve the status and results of a task |
| `POST` | `/tasks/{task_id}/reprocess` | Requeue an existing task using current or original analysis settings |
| `GET` | `/tasks/{task_id}/chords` | Return chord detection results for a completed task |
| `GET` | `/tasks/{task_id}/beatgrid` | Return beat grid and beat detection metadata for a completed task |
| `GET` | `/tasks/{task_id}/artifacts` | List downloadable output files for a completed task |
| `GET` | `/tasks/{task_id}/artifacts/{artifact_name}` | Download a named artifact file (e.g. normalised WAV, stem) |
| `GET` | `/tasks/completed` | List all completed (`done`) tasks |
| `GET` | `/tasks/{task_id}/mt3/midi/{track_name}` | Download MT3 MIDI (`full_mix` or stem name) |
| `GET` | `/tasks/{task_id}/mt3/notes/{track_name}` | Retrieve MT3 note metadata JSON |
| `GET` | `/worker/status` | Return worker liveness and last-heartbeat timestamp |
| `GET` | `/doctor` | Return consolidated deployment health checks (worker, tooling, models, transcription, disk) |
| `GET` | `/analysis/settings` | Return current analysis defaults, backend/model availability, devices, and warnings |
| `GET` | `/stem-backend/status` | Report which stem-separation backend is active |
| `GET` | `/mt3/status` | Report MT3 transcription availability and backend configuration |
| `GET` | `/api/models/status` | Report separator model availability and download progress |
| `POST` | `/api/models/download` | Start downloading separator models (`six_stems` optional) |
| `POST` | `/api/models/cancel` | Cancel an in-progress separator model download |
| `GET` | `/ui` | Web dashboard (static HTML/JS) |

### Reprocess a task

`POST /tasks/{task_id}/reprocess` resets the existing task to `pending` using either the current page-level analysis settings or the original task snapshot. By default the current report is replaced in place; archive modes store the previous task JSON before reprocessing.

```bash
curl -X POST http://localhost:8088/tasks/<task_id>/reprocess \
     -H 'Content-Type: application/json' \
     -d '{"mode": "all", "reprocess_mode": "use_current_replace"}'
```

Request body fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"all"` | What to reprocess: `all`, `audio_analysis`, `stems`, `midi`, `metadata`, `ai_prompts` |
| `reprocess_mode` | string | `"use_current_replace"` | One of `use_current_replace`, `use_current_archive`, `reuse_original_replace`, `reuse_original_archive` |
| `enable_mt3` | bool\|null | null | Override MIDI transcription for the reprocess |
| `stem_backend` | string\|null | null | Override stem backend for this run |
| `stem_model` | string\|null | null | Override Audio Separator model |
| `stem_device` | string\|null | null | Override device (`auto`, `cpu`, `cuda`) |
| `stem_mode` | string\|null | null | Override stem output mode (`4_stem`, `6_stem`) |

Response:

```json
{"task_id": "existing-task-id", "source_task_id": "existing-task-id", "status": "pending"}
```

### Example — submit a YouTube URL
```bash
curl -X POST http://localhost:8088/tasks/url \
     -H 'Content-Type: application/json' \
     -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Example — check task status
```bash
curl http://localhost:8088/tasks/<task_id>
```

### Example — OpenAPI schema
```bash
curl http://localhost:8088/openapi.json | python -m json.tool | head -n 30
```

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "SHANK API"
  },
  "paths": {
    "/tasks/upload": {},
    "/tasks/url": {},
    "/tasks/{task_id}": {}
  }
}
```

## 🤖 MCP Automation

SHANK includes an MCP server entrypoint for automation workflows:

```bash
python -m api.mcp_server --api-url http://127.0.0.1:8088
```

Exposed MCP tools:
- `shank_health`
- `shank_submit_url`
- `shank_submit_audio`
- `shank_get_task`
- `shank_list_completed_tasks`
- `shank_list_task_artifacts`

A completed task response looks like:
```json
{
  "task_id": "...",
  "type": "url",
  "source": "https://www.youtube.com/watch?v=...",
  "status": "done",
  "bpm": 113.45,
  "bpm_confidence": 0.93,
  "key": "A minor",
  "key_confidence": 0.88,
  "lufs": -13.2,
  "beats": [0.51, 1.04, 1.57],
  "downbeats": [0.51],
  "sections": [{"start_seconds": 0.51, "end_seconds": 245.31, "label": "section_1"}],
  "structure": [{"label": "Intro", "start_seconds": 0.0, "end_seconds": 16.0, "timestamp": "00:00"}],
  "cue_points": [{"name": "intro", "time_seconds": 0.51}],
  "tempo_changes": [],
  "duration_seconds": 245.31,
  "created_at": "2025-01-01T00:00:00+00:00",
  "completed_at": "2025-01-01T00:01:00+00:00"
}
```

## ⚙️ Configuration

Environment variables (set in `.env` or `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHORD_BACKEND` | `auto` | Chord detection backend: `auto` (librosa), `madmom`, or `disabled` |
| `BEAT_DETECTION_ENGINE` | `librosa` | Beat/BPM backend: `librosa`, `auto` (try Mixxx CLI first), or `mixxx` |
| `MIXXX_BEAT_CLI` | *(empty)* | Optional Mixxx beat-analysis CLI wrapper command |
| `MIXXX_FAST_ANALYSIS` | `false` | Enable fast Mixxx analysis mode when supported by the wrapper |
| `MIXXX_ASSUME_CONSTANT_TEMPO` | `true` | Request constant-tempo beat grid mode from Mixxx wrapper |
| `MIXXX_OFFSET_CORRECTION` | `true` | Enable first-beat offset correction in Mixxx wrapper |
| `MIXXX_REANALYSE_IF_OUTDATED` | `true` | Allow Mixxx wrapper to re-analyze stale metadata |
| `DATA_DIR` | `/srv/shank/data` | Directory for uploads, task files, and normalized audio |
| `POLL_INTERVAL` | `10` | Worker polling interval in seconds |
| `STEM_BACKEND` | `auto` | Stem separation backend: `auto`, `audio_separator`, `acestep`, `demucs`, or `none` |
| `AUDIO_SEPARATOR_MODEL` | `htdemucs_ft.yaml` | python-audio-separator model (4-stem default; `htdemucs_6s.yaml` for 6-stem) |
| `AUDIO_SEPARATOR_MODEL_DIR` | `/srv/shank/models/separator` | Directory for cached model weights |
| `AUDIO_SEPARATOR_DEVICE` | `cpu` | Inference device for audio-separator: `cpu` or `cuda` |
| `ACE_STEP_API_URL` | *(empty)* | Base URL of an ACE-Step API for stem separation |
| `ACE_STEP_API_KEY` | *(empty)* | Optional ****** for the ACE-Step API |
| `ACE_STEP_STEMS` | `vocals,drums,bass,other` | Comma-separated list of stems to request from ACE-Step |
| `DEMUCS_MODEL` | `htdemucs` | Model name passed to the `demucs` CLI (legacy backend) |
| `DEMUCS_DEVICE` | `cpu` | Device flag passed to the `demucs` CLI (`cpu`, `cuda`, `mps`) |
| `MT3_ENABLED` | `false` | Enable MT3 transcription in worker |
| `MT3_SERVICE_URL` | `http://127.0.0.1:8090` | Base URL for the optional MT3 FastAPI service running inside the unified `shank` container |
| `MT3_MODEL` | `multi_instrument` | Requested model identifier to send to MT3 service |
| `TRANSCRIPTION_BACKEND` | `basic_pitch` | Transcription backend in the service: `basic_pitch`, `mt3`, `omnizart`, `disabled` |
| `MODEL_CACHE_DIR` | `/srv/shank/models/transcription` | Optional cache/model directory for transcription backends |
| `MT3_TIMEOUT` | `1800` | MT3 HTTP timeout in seconds |
| `MT3_TRANSCRIBE_STEMS` | `true` | Also transcribe separated stems when present |
| `MT3_FAIL_TASK_ON_ERROR` | `false` | If true, MT3 failure marks task as failed |
| `MT3_CHECKPOINT_ROOT` | `/srv/shank/models/mt3/checkpoints` | Mount path for MT3 checkpoints in MT3 service |
| `MT3_CACHE_DIR` | `/srv/shank/cache/mt3` | Mount path for MT3 runtime cache |
| `MT3_OUTPUT_PATH` | `/srv/shank/data/mt3` | Persisted output directory for generated MT3 MIDI and note JSON files |
| `MT3_DEVICE` | `auto` | MT3 device hint (`auto`, `cpu`, or `gpu`) |

Use `ACE_STEP_API_URL` and `ACE_STEP_API_KEY` (with underscore). Legacy names such as `ACESTEP_BASE_URL` and `ACESTEP_ENABLED` are not used by the current `docker-compose.yml`.

## 🗺 Roadmap & Implementation Plan

### Phase 1: Foundation
- [x] Infrastructure setup (Docker, Docker Compose)
- [x] Network configuration (Port 8088)
- [x] Initialized Git repository and remote link
- [x] Basic API skeleton (Health check)

### Phase 2: Core API & Worker Development
- [x] Implement FastAPI endpoints for task submission (Upload/URL)
- [x] Implement Worker loop for task polling
- [x] Integrate `yt-dlp` for YouTube processing
- [x] Implement `ffmpeg` normalization pipeline
- [x] Implement `librosa` based analysis (BPM/Key)

### Phase 3: Advanced Analysis & UI
- [x] Implement Chord progression detection
- [ ] Implement Melody -> MIDI extraction
- [x] Implement Song structure/segmentation detection
- [x] Build Web UI (Dashboard, task list, result viewing)

### Phase 4: Stem Separation & Optimization
- [x] Integrate ACE-Step for optional stem separation
- [x] Integrate python-audio-separator (4-stem and 6-stem models, CPU default)
- [x] GPU support via CUDA for audio-separator (see README instructions)

### Phase 5: Ecosystem Integration
- [ ] WordPress Build Log automation
- [ ] GitHub Repository/Issue automation
- [ ] Final Deployment & Documentation

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.

## 🤖 Automated README Updates
<!-- readme-update:start -->
- Last automated update: 2026-06-11T13:29:06Z
- Latest commit: `2aa1f6c`
- Commit message: Merge pull request #166 from julesdg6/copilot/feature-song-structure-detection  Add deterministic song-structure labeling and publish `structure.json` artifacts
<!-- readme-update:end -->

## 🥁 Beat Detection & Beat Grid

SHANK extracts BPM, beat timestamps, downbeats, and a structured beat grid from every analysed track.  The beat grid is stored in the task JSON and is also available through a dedicated API endpoint:

```bash
curl http://localhost:8088/tasks/<task_id>/beatgrid
```

Example response:
```json
{
  "beatgrid": {
    "bpm": 128.02,
    "first_beat_seconds": 0.423,
    "beats": [
      {"index": 1, "time": 0.423},
      {"index": 2, "time": 0.892},
      {"index": 3, "time": 1.361}
    ]
  },
  "beat_detection": {
    "engine": "mixxx",
    "mode": "constant_tempo",
    "first_beat_seconds": 0.423,
    "beat_count": 3,
    "confidence": 0.95
  }
}
```

Variable-tempo tracks include a `mode: "variable_tempo"` key and per-beat `local_bpm` values in the beat list.

The beat grid is also saved as a standalone `beatgrid.json` artifact under `DATA_DIR/results/<task_id>/`.

### Backends

| `BEAT_DETECTION_ENGINE` | Description |
|-------------------------|-------------|
| `librosa` *(default)* | librosa beat tracker — always available, no extra dependencies |
| `auto` | Try Mixxx CLI first (if `MIXXX_BEAT_CLI` is set), then fall back to librosa/madmom |
| `mixxx` | Use Mixxx CLI exclusively; fall back to librosa/madmom if the command fails |

### Mixxx-grade BPM detection

SHANK integrates with a Mixxx beat-analysis CLI wrapper for DJ-grade accuracy.  Mixxx uses production algorithms designed for accurate beat tracking and beat grids compatible with DJ software.

To enable it, point `MIXXX_BEAT_CLI` at a CLI wrapper that accepts the flags below and emits JSON on stdout:

```dotenv
BEAT_DETECTION_ENGINE=auto
MIXXX_BEAT_CLI=/usr/local/bin/mixxx-beat-analyzer
```

The wrapper is invoked with:

```
<MIXXX_BEAT_CLI> --input <file> --output-format json [flags…]
```

Optional flags (controlled by env vars):

| Env var | Default | Flag added when `true` |
|---------|---------|------------------------|
| `MIXXX_FAST_ANALYSIS` | `false` | `--fast-analysis` |
| `MIXXX_ASSUME_CONSTANT_TEMPO` | `true` | `--assume-constant-tempo` |
| `MIXXX_OFFSET_CORRECTION` | `true` | `--offset-correction` |
| `MIXXX_REANALYSE_IF_OUTDATED` | `true` | `--reanalyse-if-outdated` |

Expected JSON output shape:

```json
{
  "bpm": 128.02,
  "confidence": 0.95,
  "mode": "constant_tempo",
  "first_beat_seconds": 0.423,
  "beats": [
    {"time": 0.423},
    {"time": 0.892}
  ]
}
```

`beats` items may be plain numbers (seconds) or objects with `time` and an optional `local_bpm` for variable-tempo tracks.  `confidence`, `mode`, and `first_beat_seconds` are optional.

If the CLI is unavailable or exits with an error, SHANK falls back to madmom (if installed) and then to librosa so analysis always completes.

## 🎸 Chord Detection

SHANK performs automatic chord detection on every analysed track and returns timestamped chord segments alongside BPM, key, beats, and downbeats.

### Output format

Chord data is available inside the task JSON as the `chords` object, and via the dedicated endpoint:

```bash
curl http://localhost:8088/tasks/<task_id>/chords
```

Example response:
```json
{
  "segments": [
    {
      "symbol": "Am",
      "root": "A",
      "quality": "minor",
      "confidence": 0.72,
      "start_seconds": 0.0,
      "end_seconds": 3.8
    },
    {
      "symbol": "F",
      "root": "F",
      "quality": "major",
      "confidence": 0.68,
      "start_seconds": 3.8,
      "end_seconds": 7.6
    }
  ],
  "progression": ["Am", "F"]
}
```

### Backends

| `CHORD_BACKEND` | Description |
|-----------------|-------------|
| `auto` *(default)* | Librosa chroma-based chord estimation — no extra dependencies required |
| `librosa` | Explicit alias for the same librosa backend |
| `madmom` | Deep-learning chord recognition via [madmom](https://github.com/CPJKU/madmom); falls back to librosa if madmom is not installed |
| `disabled` | Skips chord detection entirely; `chords` will contain empty `segments` and `progression` |

### Enabling madmom chord recognition

Install madmom in the worker container (add to `worker/requirements.txt` or your `Dockerfile`):

```dockerfile
RUN pip install --no-cache-dir madmom
```

Then set in `.env`:

```dotenv
CHORD_BACKEND=madmom
```

If madmom cannot be imported at runtime, SHANK automatically falls back to the librosa backend so the analysis still completes.

### Disabling chord detection

```dotenv
CHORD_BACKEND=disabled
```

Chord detection is skipped entirely and `chords` will be `{"segments": [], "progression": []}`.

## 🎛 Stem Separation (python-audio-separator)

## 🎹 Transcription backend (Basic Pitch)

The transcription service now supports backend selection with `TRANSCRIPTION_BACKEND`.

```dotenv
TRANSCRIPTION_BACKEND=basic_pitch   # basic_pitch | mt3 | omnizart | disabled
MODEL_CACHE_DIR=/srv/shank/models/transcription
```

- `basic_pitch`: real audio-to-MIDI transcription (requires `basic-pitch` dependency)
- `disabled`: cleanly turns transcription off
- empty-note outputs are returned as failed transcription results (not silent success)

Enable Basic Pitch in Docker builds with:

```bash
docker build --build-arg INSTALL_BASIC_PITCH=true -t shank .
```

SHANK bundles [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) as the default stem separation backend. No external service is required — it runs entirely inside the container.

### Setup

```bash
python3 scripts/download_stem_models.py
```

### Supported stem counts

| Model | Stems produced |
|-------|---------------|
| `htdemucs_ft.yaml` *(default)* | 4 — vocals, drums, bass, other |
| `htdemucs_6s.yaml` | 6 — vocals, drums, bass, other, guitar, piano |
| `htdemucs.yaml` | 4 — vocals, drums, bass, other |

### Default CPU configuration (no GPU required)

```dotenv
STEM_BACKEND=auto                          # auto: Ace-Step (if configured) → audio-separator → Demucs
AUDIO_SEPARATOR_MODEL=htdemucs_ft.yaml     # 4-stem model (default)
AUDIO_SEPARATOR_MODEL_DIR=/srv/shank/models/separator
AUDIO_SEPARATOR_DEVICE=cpu
```

Model weights are downloaded automatically on first use and cached in `AUDIO_SEPARATOR_MODEL_DIR`. Mount a host directory to persist the cache across container restarts:

```yaml
volumes:
  - ./models/separator:/srv/shank/models/separator
```

To pre-download models before starting the container, run:

```bash
# Download the default 4-stem model (htdemucs_ft.yaml, ~400 MB)
python3 scripts/download_stem_models.py

# Also download the optional 6-stem model (htdemucs_6s.yaml, ~530 MB)
python3 scripts/download_stem_models.py --6stems

# Use a custom directory
python3 scripts/download_stem_models.py --model-dir /path/to/models
```

Requires `audio-separator` to be installed (`pip install audio-separator[cpu]`).

### GPU acceleration (CUDA)

To use a CUDA GPU for faster separation, install the GPU variant of audio-separator and set `AUDIO_SEPARATOR_DEVICE=cuda`. Update your `Dockerfile` to replace the CPU extra:

```dockerfile
RUN pip install --no-cache-dir audio-separator[gpu]
```

Then set in `.env` or `docker-compose.yml`:

```dotenv
AUDIO_SEPARATOR_DEVICE=cuda
```

Enable the NVIDIA runtime in `docker-compose.yml`:

```yaml
services:
  shank:
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Ensure `nvidia-container-toolkit` is installed on the Docker host before enabling GPU passthrough.

### Choosing a backend explicitly

```dotenv
# Use audio-separator only (fail if unavailable)
STEM_BACKEND=audio_separator

# Use Ace-Step only (requires ACE_STEP_API_URL)
STEM_BACKEND=acestep

# Use legacy Demucs CLI only (requires demucs in PATH)
STEM_BACKEND=demucs

# Disable stem separation entirely
STEM_BACKEND=none

# Auto (default): Ace-Step (if ACE_STEP_API_URL set) → audio-separator → Demucs → skip
STEM_BACKEND=auto
```

### 6-stem separation

To separate guitar and piano in addition to the standard 4 stems:

```dotenv
AUDIO_SEPARATOR_MODEL=htdemucs_6s.yaml
```

## 🎚 Optional ACE-Step Stem Separation
To enable stem separation in the worker (vocals, drums, bass, other), set `ACE_STEP_API_URL` in your environment:

```dotenv
ACE_STEP_API_URL=http://ace-step:8001
ACE_STEP_API_KEY=          # optional — Bearer token if your API requires auth
ACE_STEP_STEMS=vocals,drums,bass,other   # optional — defaults shown
```

When `ACE_STEP_API_URL` is set, each normalized track is submitted to ACE-Step (`/release_task`), the worker polls for completion (`/query_result`), and the returned stem references are stored in the task metadata.

## 🎹 Optional MT3 MIDI Transcription

SHANK can transcribe audio to MIDI using [Magenta MT3](https://github.com/magenta/mt3), running as an internal FastAPI service inside the same `shank` container.

> **Note:** MT3 is a research project by Google Magenta and is **not officially supported by Google** for production use. SHANK's integration is a best-effort wrapper around the upstream research code.

### Enabling / Disabling MT3

MT3 transcription is controlled by the `MT3_ENABLED` variable (default: `false`).

**Enabled**:
```dotenv
MT3_ENABLED=true
MT3_SERVICE_URL=http://127.0.0.1:8090
```
The worker will call the internal MT3 service after each analysis and attach MIDI artifacts to the task result.

**Disabled**:
```dotenv
MT3_ENABLED=false
```
All transcription steps are skipped entirely. The `mt3` object in the task result will show `"status": "disabled"`.

### Unified container mode

```bash
docker compose up --build -d
```

Set in `.env`:
```dotenv
MT3_ENABLED=true
MT3_SERVICE_URL=http://127.0.0.1:8090
MT3_DEVICE=gpu
```

Optional NVIDIA Docker Compose GPU reservation example in `docker-compose.yml` under `shank`:
```yaml
# gpus: all
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

Leave `MT3_ENABLED=false` to keep transcription disabled while still running the same unified container.

### Full-Mix vs Stem Transcription

| Mode | What is transcribed | Requires |
|------|---------------------|----------|
| **Full mix** | The normalized stereo WAV of the whole track | Always attempted when MT3 is enabled |
| **Stem transcription** | Each separated stem (vocals, drums, bass, other) | ACE-Step stems present **and** `MT3_TRANSCRIBE_STEMS=true` |

- The worker always attempts full-mix transcription first.
- Stem transcription is attempted afterwards when both `ACE_STEP_API_URL` is set and `MT3_TRANSCRIBE_STEMS=true`.
- MIDI outputs are stored under `MT3_OUTPUT_PATH/<task_id>/` (defaults to `DATA_DIR/mt3/<task_id>/`).
- The task JSON gains an `mt3` object with keys: `status`, `model`, `output_paths`, `full_mix`, `stems`, `warnings`, `errors`.
- MT3 failures are **non-fatal** by default. Set `MT3_FAIL_TASK_ON_ERROR=true` to mark tasks as failed on MT3 error.

### Docker Compose Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MT3_ENABLED` | `false` | Enable (`true`) or disable (`false`) MT3 transcription |
| `MT3_SERVICE_URL` | `http://127.0.0.1:8090` | Internal URL of the MT3 FastAPI service running inside the `shank` container |
| `MT3_MODEL` | `multi_instrument` | MT3 model: `multi_instrument` (all instruments) or `ismir2021` (piano-only) |
| `MT3_TIMEOUT` | `1800` | HTTP timeout (seconds) for a single transcription request |
| `MT3_TRANSCRIBE_STEMS` | `true` | Also transcribe ACE-Step stems when available |
| `MT3_FAIL_TASK_ON_ERROR` | `false` | Mark the whole task failed if MT3 errors occur |
| `MT3_CHECKPOINT_ROOT` | `/srv/shank/models/mt3/checkpoints` | Host-mounted path for MT3 model checkpoints |
| `MT3_CACHE_DIR` | `/srv/shank/cache/mt3` | Host-mounted path for MT3 runtime/compiled cache |
| `MT3_OUTPUT_PATH` | `/srv/shank/data/mt3` | Persisted output path for generated MIDI and note JSON artifacts (covered by `./data:/srv/shank/data`) |
| `MT3_DEVICE` | `auto` | Device hint: `auto`, `cpu`, or `gpu` |

Example `.env` snippet:
```dotenv
MT3_ENABLED=false
MT3_MODEL=multi_instrument
MT3_TIMEOUT=1800
MT3_TRANSCRIBE_STEMS=true
MT3_FAIL_TASK_ON_ERROR=false
MT3_CHECKPOINT_ROOT=/srv/shank/models/mt3/checkpoints
MT3_CACHE_DIR=/srv/shank/cache/mt3
MT3_OUTPUT_PATH=/srv/shank/data/mt3
MT3_DEVICE=auto
MT3_SERVICE_URL=http://127.0.0.1:8090
```

The default MT3 paths are centralized in `mt3_config.py`, and `docker-compose.yml`, `.env.example`, and the MT3 tests are kept aligned with those defaults.

### Downloading MIDI Results

```bash
# Download full-mix MIDI
curl http://localhost:8088/tasks/<task_id>/mt3/midi/full_mix --output full_mix.mid

# Download stem MIDI (e.g. vocals)
curl http://localhost:8088/tasks/<task_id>/mt3/midi/vocals --output vocals.mid

# Retrieve note metadata JSON
curl http://localhost:8088/tasks/<task_id>/mt3/notes/full_mix
```

### Troubleshooting MT3

#### Model download failure
MT3 checkpoints must be present in `MT3_CHECKPOINT_ROOT` before the container starts. The service does **not** auto-download models at runtime.

1. Ensure the host directory `./models/mt3/checkpoints` exists and contains the checkpoint files.
2. Verify the volume mount in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro
   ```
3. Restart the container after placing the checkpoints.

#### CUDA / GPU unavailable
MT3 defaults to `MT3_DEVICE=auto`, which uses CPU when no GPU is detected.

- To force CPU: set `MT3_DEVICE=cpu`.
- To enable GPU: ensure `nvidia-container-toolkit` is installed on the host, then set `MT3_DEVICE=gpu`.
- CPU inference is functional but significantly slower on tracks longer than a few minutes.

#### Transcription timeout
If the worker logs show a timeout error against `MT3_SERVICE_URL`, increase `MT3_TIMEOUT`:
```dotenv
MT3_TIMEOUT=1800   # 30 minutes — for long tracks or slow CPU inference
```

#### No MIDI generated / empty MIDI file
- Check `task['mt3']['warnings']` in the task JSON — a warning such as `"No MIDI data returned; empty MIDI written"` indicates the service returned no note events.
- This can happen when the audio is silent, very short, or contains only non-pitched content.
- Verify `MT3_SERVICE_URL` is reachable from within the container: `docker compose exec shank curl -s http://shank-mt3:8090/health`.

#### Stem files not local / stem transcription skipped
Stem transcription requires that ACE-Step has already separated the audio **and** that the stem file paths are accessible locally inside the container.

- Confirm `ACE_STEP_API_URL` is set and ACE-Step completed successfully (check `task['stems']` in the task JSON).
- Confirm `MT3_TRANSCRIBE_STEMS=true`.
- If stems are stored on a remote service rather than a local path, SHANK cannot transcribe them — the worker will skip stem transcription and log a warning.

For detailed setup, configuration, and troubleshooting, see [`docs/mt3.md`](docs/mt3.md).
For MT3 integration design notes, see [`docs/mt3-research.md`](docs/mt3-research.md).

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.
