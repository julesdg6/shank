# 🎵 SHANK: AI Song Analyzer

SHANK is a powerful, Dockerized, self-hosted tool designed to perform deep musical analysis on audio files and YouTube videos. It extracts technical and creative metadata from music, including BPM, musical key, and optionally separated stems.

## 🎯 Project Aim
To provide users with an automated pipeline that transforms raw audio/URLs into structured musical intelligence, including tempo, key, chord progressions, and MIDI melodies.

## 🚀 Key Features
- **Multi-Source Input**: Direct audio uploads (MP3, WAV, FLAC) and YouTube URLs (via `yt-dlp`).
- **Audio Normalization**: Automatic conversion to standard 44100 Hz stereo WAV using `ffmpeg`.
- **Musical Extraction**:
    - **BPM & Tempo**: Precise beat tracking via `librosa`.
    - **Musical Key**: Krumhansl-Kessler key detection (e.g. `A minor`, `C major`).
    - **Chord Progressions**: *(planned)*
    - **Melody to MIDI**: *(planned)*
    - **Song Structure**: Detection of intro, verse, chorus, etc. *(planned)*
- **Optional Stem Separation**: Integration with **ACE-Step** to separate vocals, drums, bass, and other instruments.
- **Optional MT3 Transcription**: Worker can call a dedicated `shank-mt3` service to generate MIDI + note metadata from normalized mix and stems.
- **Asynchronous Workflow**: A background worker polls a filesystem task queue and processes jobs independently of the API.
- **Web Dashboard**: A built-in UI at `/ui` to submit tasks, monitor progress, and inspect results.

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
Navigate to **http://localhost:8088/ui** in your browser to upload audio files or submit YouTube URLs.

### 4. Stop the service
```bash
docker compose down
```

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns `{"status": "online"}` |
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

A completed task response looks like:
```json
{
  "task_id": "...",
  "type": "url",
  "source": "https://www.youtube.com/watch?v=...",
  "status": "done",
  "bpm": 113.45,
  "key": "A minor",
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
| `ACE_STEP_API_URL` | *(empty)* | Base URL of an ACE-Step API for stem separation |
| `ACE_STEP_API_KEY` | *(empty)* | Optional Bearer token for the ACE-Step API |
| `ACE_STEP_STEMS` | `vocals,drums,bass,other` | Comma-separated list of stems to request |
| `MT3_ENABLED` | `false` | Enable MT3 transcription in worker |
| `MT3_SERVICE_URL` | `http://shank-mt3:8090` | Base URL for the optional MT3 FastAPI service |
| `MT3_MODEL` | `multi_instrument` | Requested model identifier to send to MT3 service |
| `MT3_TIMEOUT` | `1800` | MT3 HTTP timeout in seconds |
| `MT3_TRANSCRIBE_STEMS` | `true` | Also transcribe Ace-Step stems when present |
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
- [ ] Implement GPU support for faster processing

### Phase 5: Ecosystem Integration
- [ ] WordPress Build Log automation
- [ ] GitHub Repository/Issue automation
- [ ] Final Deployment & Documentation

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.

## 🤖 Automated README Updates
<!-- readme-update:start -->
- Last automated update: 2026-05-13T22:25:11Z
- Latest commit: `c5be2ee`
- Commit message: Merge pull request #55 from julesdg6/copilot/add-daw-style-waving-timeline  Add DAW-style scrolling waveform timeline to song detail view
<!-- readme-update:end -->

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
MT3_SERVICE_URL=http://shank-mt3:8090
```
The worker will call the internal MT3 service after each analysis and attach MIDI artifacts to the task result.

**Disabled**:
```dotenv
MT3_ENABLED=false
```
All transcription steps are skipped entirely. The `mt3` object in the task result will show `"status": "disabled"`.

### CPU/basic mode (default, MT3 off)

```bash
docker compose up --build -d
```

### GPU/MT3 mode

```bash
docker compose --profile mt3 up --build -d
```

Set in `.env`:
```dotenv
MT3_ENABLED=true
MT3_SERVICE_URL=http://shank-mt3:8090
MT3_DEVICE=gpu
```

Optional NVIDIA Docker Compose GPU reservation example in `docker-compose.yml` under `shank-mt3`:
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

### MT3 disabled mode

```bash
docker compose up --build -d
```

Set in `.env`:
```dotenv
MT3_ENABLED=false
```

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
| `MT3_SERVICE_URL` | `http://shank-mt3:8090` | Internal URL of the optional MT3 FastAPI service |
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
MT3_SERVICE_URL=http://shank-mt3:8090
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
