# 🎵 SHANK: AI Song Analyzer

SHANK is a powerful, Dockerized, self-hosted tool designed to perform deep musical analysis on audio files and YouTube videos. It utilizes state-of-the-art AI models to extract technical and creative metadata from music.

## 🎯 Project Aim
To provide users with an automated pipeline that transforms raw audio/URLs into structured musical intelligence, including tempo, key, chord progressions, and MIDI melodies.

## 🚀 Key Features (Planned)
- **Multi-Source Input**: Support for direct audio uploads (MP3, WAV, FLAC) and YouTube URLs (via `yt-dlp`).
- **Advanced Musical Extraction**:
    - **BPM & Tempo**: Precise beat tracking.
    - **Musical Key**: Detection of the song's key.
    - **Chord Progressions**: Identification of harmonic structure.
    - **Melody to MIDI**: Extraction of melodic lines into MIDI format.
    - **Song Structure**: Detection of intro, verse, chorus, etc., with waveform visualizations.
- **Stem Separation**: Integration with **Ace-step 1.5** to separate vocals, drums, bass, and other instruments for specialized analysis.
- **Automated Workflow**: A headless worker architecture that processes tasks asynchronously.
- **Web Interface**: A clean, user-friendly Dashboard to manage uploads and view results.

## 🛠 Technical Stack
- **Backend**: FastAPI (Python)
- **Worker**: Python (Librosa, NumPy, Pandas, Scipy, yt-dlp, ffmpeg)
- **Deployment**: Docker & Docker Compose
- **Orchestration**: Asynchronous task queue via filesystem polling
- **Runtime**: A single container runs both the FastAPI server and the background worker loop

## 🗺 Roadmap & Implementation Plan

### Phase 1: Foundation (Current State)
- [x] Infrastructure setup (Docker, Docker Compose)
- [x] Network configuration (Port 8088)
- [x] Initialized Git repository and remote link
- [x] Basic API skeleton (Health check)

### Phase 2: Core API & Worker Development
- [x] Implement FastAPI endpoints for task submission (Upload/URL)
- [x] Implement Worker loop for task polling
- [ ] Integrate `yt-dlp` for YouTube processing
- [ ] Implement `ffmpeg` normalization pipeline
- [x] Implement `librosa` based analysis (BPM/Key)

### Phase 3: Advanced Analysis & UI
- [ ] Implement Chord progression detection
- [ ] Implement Melody -> MIDI extraction
- [ ] Implement Song structure/segmentation detection
- [ ] Build Web UI (Dashboard, Progress bars, Result viewing)

### Phase 4: Stem Separation & Optimization
- [ ] Integrate Ace-step 1.5 for stem separation
- [ ] Enable the option to use Ace-step for separating vocals, drums, bass, and others.
- [ ] Implement GPU support for faster processing

### Phase 5: Ecosystem Integration
- [ ] WordPress Build Log automation
- [ ] GitHub Repository/Issue automation
- [ ] Final Deployment & Documentation

## ⚖️ Legal Note
This project is for research and personal use. Ensure you have the rights to any audio content you process.
