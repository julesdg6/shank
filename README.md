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
| `MT3_ENABLED` | `true` | Enable MT3 transcription in worker |
| `MT3_SERVICE_URL` | `http://127.0.0.1:8090` | Base URL for the MT3 FastAPI service inside the unified container |
| `MT3_MODEL` | `multi_instrument` | Requested model identifier to send to MT3 service |
| `MT3_TIMEOUT` | `900` | MT3 HTTP timeout in seconds |
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
- Last automated update: 2026-05-08T02:03:36Z
- Latest commit: `7267d93`
- Commit message: Merge pull request #37 from julesdg6/copilot/integrate-magenta-mt3-transcription  Add optional MT3 transcription pipeline (worker + API/UI + compose profiles) without impacting core BPM/key flow
<!-- readme-update:end -->

## 🎚 Optional ACE-Step Stem Separation
To enable stem separation in the worker (vocals, drums, bass, other), set `ACE_STEP_API_URL` in your environment:

```dotenv
ACE_STEP_API_URL=http://ace-step:8001
ACE_STEP_API_KEY=          # optional — Bearer token if your API requires auth
ACE_STEP_STEMS=vocals,drums,bass,other   # optional — defaults shown
```

When `ACE_STEP_API_URL` is set, each normalized track is submitted to ACE-Step (`/release_task`), the worker polls for completion (`/query_result`), and the returned stem references are stored in the task metadata.

## 🎹 Optional MT3 Transcription

SHANK runs MT3 inference through an internal FastAPI service in the same `shank` container.

### Start SHANK with MT3 enabled
```bash
docker compose up --build -d
```

Behavior:
- Worker always attempts **full-mix MT3** transcription first (normalized WAV).
- If Ace-Step stems exist and `MT3_TRANSCRIBE_STEMS=true`, worker transcribes stems second.
- Outputs are stored under `DATA_DIR/mt3/<task_id>/`.
- Task JSON gets an `mt3` object (`status`, `model`, `output_paths`, `full_mix`, `stems`, `warnings`, `errors`).
- MT3 failures are non-fatal by default. Set `MT3_FAIL_TASK_ON_ERROR=true` for strict behavior.

Limitations:
- MT3 CPU throughput can be slow on long tracks.
- The bundled service currently emits baseline MIDI/notes artifacts for integration flow validation.

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.
