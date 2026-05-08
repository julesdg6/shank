# MT3 MIDI Transcription

SHANK integrates [Magenta MT3](https://github.com/magenta/mt3) to automatically transcribe audio to MIDI as part of each analysis task.

> **Disclaimer:** MT3 is a research project by Google Magenta and is **not officially supported by Google** for production use. SHANK's integration is a best-effort adapter layer built around the upstream research code.

---

## How it works

After each analysis task completes, the SHANK worker:

1. Calls the internal MT3 FastAPI service (running on port 8090 inside the container via `supervisord`).
2. Sends the normalized WAV file to the `/transcribe` endpoint.
3. Receives MIDI bytes and note event metadata in the response.
4. Writes MIDI (`.mid`) and optional note JSON (`.notes.json`) under `DATA_DIR/mt3/<task_id>/`.
5. Attaches an `mt3` object to the task JSON with keys: `status`, `model`, `output_paths`, `full_mix`, `stems`, `warnings`, `errors`.

---

## Enabled / Disabled mode

MT3 transcription is controlled by the `MT3_ENABLED` environment variable.

### Enabled (default)

```dotenv
MT3_ENABLED=true
MT3_SERVICE_URL=http://127.0.0.1:8090
```

- The worker attempts full-mix transcription for every task.
- If ACE-Step stems are present and `MT3_TRANSCRIBE_STEMS=true`, each stem is also transcribed.
- MT3 failures are non-fatal by default (`MT3_FAIL_TASK_ON_ERROR=false`).

### Disabled

```dotenv
MT3_ENABLED=false
```

- All MT3 steps are skipped entirely.
- The `mt3` field in the task result will show `"status": "disabled"`.
- No MIDI or notes files are written.

---

## Docker Compose variables

All variables can be set in `.env` or passed directly in `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MT3_ENABLED` | `true` | Enable (`true`) or disable (`false`) MT3 transcription |
| `MT3_SERVICE_URL` | `http://127.0.0.1:8090` | URL of the internal MT3 FastAPI service |
| `MT3_MODEL` | `multi_instrument` | Model to use: `multi_instrument` (all instruments) or `ismir2021` (piano-only) |
| `MT3_TIMEOUT` | `900` | HTTP timeout in seconds per transcription request |
| `MT3_TRANSCRIBE_STEMS` | `true` | Also transcribe ACE-Step stems when present |
| `MT3_FAIL_TASK_ON_ERROR` | `false` | Mark the whole task failed on MT3 error |
| `MT3_CHECKPOINT_ROOT` | `/srv/shank/models/mt3/checkpoints` | Container path for MT3 model checkpoints (volume-mounted read-only) |
| `MT3_CACHE_DIR` | `/srv/shank/cache/mt3` | Container path for MT3 compiled/runtime cache (volume-mounted) |
| `MT3_DEVICE` | `auto` | Device hint: `auto`, `cpu`, or `gpu` |

The corresponding volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - ./data:/srv/shank/data
  - ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro
  - ./cache/mt3:/srv/shank/cache/mt3
```

---

## Full-mix vs stem transcription

| Mode | What is transcribed | When it runs |
|------|---------------------|--------------|
| **Full mix** | Normalized stereo WAV of the entire track | Always, when `MT3_ENABLED=true` |
| **Stem transcription** | Each ACE-Step stem (vocals, drums, bass, other) | When stems exist locally **and** `MT3_TRANSCRIBE_STEMS=true` |

- Full-mix transcription always runs first.
- Stem transcription is best-effort: if a stem file is not accessible locally, that stem is skipped and a warning is logged.
- MIDI outputs are stored under `DATA_DIR/mt3/<task_id>/`.

### Choosing a model

| Model | Best for |
|-------|----------|
| `multi_instrument` (default) | General music — pop, rock, mixed-instrument tracks |
| `ismir2021` | Piano-only material — better velocity and timing for solo piano |

Set via `MT3_MODEL` in `.env`.

---

## Retrieving results

```bash
# Download full-mix MIDI
curl http://localhost:8088/tasks/<task_id>/mt3/midi/full_mix --output full_mix.mid

# Download stem MIDI (e.g. vocals)
curl http://localhost:8088/tasks/<task_id>/mt3/midi/vocals --output vocals.mid

# Retrieve full-mix note metadata JSON
curl http://localhost:8088/tasks/<task_id>/mt3/notes/full_mix
```

The Web UI at `http://localhost:8088/ui` also provides download links for all MT3 artifacts once a task is complete.

---

## Troubleshooting

### Model download failure

MT3 checkpoints must be placed manually in `./models/mt3/checkpoints` on the host **before** starting the container. The service does not auto-download model weights at runtime.

**Steps:**

1. Obtain the checkpoint files from the MT3 upstream repository or a trusted source.
2. Place them in `./models/mt3/checkpoints/` (relative to the project root).
3. Verify the volume mount is present in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro
   ```
4. Restart the container:
   ```bash
   docker compose down && docker compose up --build -d
   ```

If the checkpoint directory is empty or missing, the MT3 service will start but return errors or empty MIDI on transcription requests.

---

### CUDA / GPU unavailable

MT3 defaults to `MT3_DEVICE=auto`, which falls back to CPU automatically when no GPU is found.

**To force CPU:**
```dotenv
MT3_DEVICE=cpu
```

**To enable GPU acceleration:**
1. Install `nvidia-container-toolkit` on the host.
2. Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
3. Set `MT3_DEVICE=gpu` in `.env`.

CPU inference is functional but noticeably slower for tracks longer than a few minutes.

---

### Transcription timeout

If you see timeout errors in the worker logs (e.g., `Read timeout` or `Connection timed out` against `MT3_SERVICE_URL`), the inference is taking longer than `MT3_TIMEOUT` seconds.

Increase the timeout:
```dotenv
MT3_TIMEOUT=1800   # 30 minutes — recommended for long tracks or CPU-only inference
```

For very long tracks on CPU, consider running MT3 against shorter clips or enabling GPU support.

---

### No MIDI generated / empty MIDI file

When the task JSON contains `"warnings": ["No MIDI data returned; empty MIDI written"]`:

- The MT3 service responded successfully but returned no note events.
- Possible causes:
  - Audio is silent or nearly silent.
  - Audio is very short (under a few seconds).
  - Audio contains only non-pitched content (noise, speech without melody, pure percussion).
  - The selected model (`MT3_MODEL`) is a poor fit for the audio content.

**Diagnostic steps:**

1. Confirm the service is running:
   ```bash
   docker exec shank curl -s http://127.0.0.1:8090/health
   ```
2. Try transcribing a known musical WAV to isolate whether the issue is content-specific.
3. Switch model: if using `ismir2021`, try `multi_instrument`, or vice versa.

---

### Stem files not local / stem transcription skipped

Stem transcription requires that:
- ACE-Step has successfully completed (check `task['stems']` in the task JSON).
- The stem file paths returned by ACE-Step are **local paths accessible inside the container** (not remote URLs).
- `MT3_TRANSCRIBE_STEMS=true`.

If stem files are stored on a remote ACE-Step service rather than a local mount, SHANK cannot read them for transcription. The worker will log a warning and skip those stems.

**Resolution:**
- Ensure your ACE-Step setup writes stem files to a path that is volume-mounted into the `shank` container.
- Alternatively, set `MT3_TRANSCRIBE_STEMS=false` to skip stem transcription entirely and only transcribe the full mix.

---

## Further reading

- [Magenta MT3 repository](https://github.com/magenta/mt3)
- [MT3 integration design notes](mt3-research.md)
