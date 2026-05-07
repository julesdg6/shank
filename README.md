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
- Last automated update: 2026-05-07T12:14:34Z
- Latest commit: `e7bd274`
- Commit message: Merge pull request #21 from julesdg6/copilot/add-workflow-update-readme  [WIP] Add workflow to update README.md on commits
<!-- readme-update:end -->

## 🎚 Optional ACE-Step Stem Separation
To enable stem separation in the worker (vocals, drums, bass, other), set `ACE_STEP_API_URL` in your environment:

```dotenv
ACE_STEP_API_URL=http://ace-step:8001
ACE_STEP_API_KEY=          # optional — Bearer token if your API requires auth
ACE_STEP_STEMS=vocals,drums,bass,other   # optional — defaults shown
```

When `ACE_STEP_API_URL` is set, each normalized track is submitted to ACE-Step (`/release_task`), the worker polls for completion (`/query_result`), and the returned stem references are stored in the task metadata.

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.
