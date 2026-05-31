# 🎵 SHANK: AI Song Analyzer

SHANK is a powerful, Dockerized, self-hosted tool designed to perform deep musical analysis on audio files and YouTube videos. It extracts technical and creative metadata from music, including BPM, musical key, and optionally separated stems.

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
    - **Melody to MIDI**: *(planned)*
    - **Song Structure**: Detection of intro, verse, chorus, etc. *(planned)*
- **Built-in Stem Separation**: [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) separates vocals, drums, bass, and other instruments (4-stem or 6-stem) — no external service required. Optional **ACE-Step** integration for comparison.
- **Optional MT3 Transcription**: Worker can call a dedicated `shank-mt3` service to generate MIDI + note metadata from normalized mix and stems.
- **Asynchronous Workflow**: A background worker polls a filesystem task queue and processes jobs independently of the API.
- **Structured Result Artifacts**: Completed tasks now write a predictable `DATA_DIR/results/<task_id>/` folder with `task.json`, `analysis.json`, `mt3.json`, and `artifacts.json`.
- **Web Dashboard**: A built-in UI at `/ui` to submit tasks, monitor progress, and inspect results.
- **MCP Automation Server**: Optional MCP server exposing SHANK task operations for automation clients.

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

### 2. Start the service
```bash
docker compose up --build -d
```

The API and Web UI are available at **http://localhost:8088**.

### 3. Open the dashboard
Navigate to **http://localhost:8088/** in your browser to upload audio files or submit YouTube URLs. The dashboard is also available at **http://localhost:8088/ui**.

### 4. Stop the service
```bash
docker compose down
```

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser landing page (dashboard) or JSON health check for API clients |
| `POST` | `/tasks/upload` | Upload an audio file (MP3, WAV, FLAC, max 200 MB) |
| `POST` | `/tasks/melody` | Upload audio and queue a melody-focused analysis task |
| `POST` | `/tasks/url` | Submit a YouTube URL for download and analysis |
| `GET` | `/tasks/{task_id}` | Retrieve the status and results of a task |
| `GET` | `/tasks/completed` | List all completed (`done`) tasks |
| `GET` | `/tasks/{task_id}/mt3/midi/{track_name}` | Download MT3 MIDI (`full_mix` or stem name) |
| `GET` | `/tasks/{task_id}/mt3/notes/{track_name}` | Retrieve MT3 note metadata JSON |
| `GET` | `/ui` | Web dashboard (static HTML/JS) |

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
| `MT3_SERVICE_URL` | `http://shank-mt3:8090` | Base URL for the optional MT3 FastAPI service |
| `MT3_MODEL` | `multi_instrument` | Requested model identifier to send to MT3 service |
| `MT3_TIMEOUT` | `1800` | MT3 HTTP timeout in seconds |
| `MT3_TRANSCRIBE_STEMS` | `true` | Also transcribe separated stems when present |
| `MT3_FAIL_TASK_ON_ERROR` | `false` | If true, MT3 failure marks task as failed |
| `MT3_CHECKPOINT_ROOT` | `/srv/shank/models/mt3/checkpoints` | Mount path for MT3 checkpoints in MT3 service |
| `MT3_CACHE_DIR` | `/srv/shank/cache/mt3` | Mount path for MT3 runtime cache |
| `MT3_DEVICE` | `auto` | MT3 device hint (`auto`, `cpu`, or `gpu`) |

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
- [ ] Implement Chord progression detection
- [ ] Implement Melody -> MIDI extraction
- [ ] Implement Song structure/segmentation detection
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
- Last automated update: 2026-05-31T18:29:42Z
- Latest commit: `62d14b6`
- Commit message: Merge pull request #87 from julesdg6/copilot/add-advanced-musical-analysis-pipeline  Add optional advanced beat/downbeat/loudness analysis outputs to worker pipeline
<!-- readme-update:end -->

## 🎛 Stem Separation (python-audio-separator)

SHANK bundles [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) as the default stem separation backend. No external service is required — it runs entirely inside the container.

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
- MIDI outputs are stored under `DATA_DIR/mt3/<task_id>/`.
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
MT3_DEVICE=auto
MT3_SERVICE_URL=http://127.0.0.1:8090
```

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
