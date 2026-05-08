# MT3 Runtime Research and Integration Recommendation

## Scope
This note evaluates how to integrate [Magenta MT3](https://github.com/magenta/mt3) into SHANK’s worker pipeline.

## Upstream MT3 findings

### Runtime/install requirements (from MT3 repo + Colab)
- MT3 is built on **T5X/JAX/TensorFlow** (`t5x`, `t5`, `seqio`, `tensorflow`, `flax`, `note-seq`, etc. from `setup.py`).
- Official quick path for inference is the **Colab notebook** (`mt3/colab/music_transcription_with_transformers.ipynb`).
- Colab setup installs system audio deps (`libfluidsynth3`, `libasound2-dev`, `libjack-dev`) and installs MT3 editable with **JAX CUDA** (`pip install jax[cuda12] ... -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`).
- Colab pulls checkpoints from `gs://mt3/checkpoints` and uses:
  - `/content/checkpoints/ismir2021/` (piano model)
  - `/content/checkpoints/mt3/` (multi-instrument model)

### Model input/output behavior
- Inference uses **16 kHz audio samples** (`SAMPLE_RATE = 16000` and spectrogram default sample rate is 16000).
- Colab uploads WAV bytes and decodes to sample arrays; in SHANK we should convert normalized audio to **mono 16 kHz float samples** before inference.
- Output is a `note_seq.NoteSequence`; notebook exports MIDI via `note_seq.sequence_proto_to_midi_file(..., '/tmp/transcribed.mid')`.

## Integration approach decision

### Options considered
1. **Use original MT3 repo directly as-is**
   - Pros: no code divergence.
   - Cons: no production-ready CLI/service entrypoint for SHANK worker jobs; repo is primarily research/training oriented.
2. **Run Colab notebook code in production**
   - Pros: proven inference flow.
   - Cons: notebook-first UX, Colab-specific assumptions (`/content`, notebook upload/download, analytics cells).
3. **Build a small SHANK wrapper module/script using Colab inference logic** ✅ **Recommended**
   - Pros: smallest practical path to production integration, keeps upstream model behavior, fits current worker architecture.
   - Cons: we own a thin adapter layer.

### Recommendation
Adopt **Option 3**: implement a thin SHANK-side wrapper that:
- loads MT3 checkpoint from a mounted model path,
- accepts local normalized audio file path,
- resamples/downmixes to 16 kHz mono,
- runs MT3 inference,
- writes `.mid` output under task artifacts,
- updates task JSON with MIDI path and model metadata.

## Checkpoint strategy decision
- Support **both checkpoints**.
- Default to **multi-instrument (`mt3`)** for general SHANK usage.
- Allow **piano (`ismir2021`)** as an explicit override for piano-only material (better piano-specific behavior and velocity modeling).

Suggested config surface:
- `MT3_MODEL=mt3` (`mt3` or `ismir2021`)
- `MT3_CHECKPOINT_ROOT=/srv/shank/models/mt3/checkpoints`

## CPU vs GPU expectations
- MT3 can run on CPU, but practical throughput is expected to be significantly slower for longer tracks.
- GPU is the preferred production path; upstream Colab setup is explicitly CUDA/JAX-oriented.
- Recommendation for SHANK:
  - **CPU**: acceptable for development and short clips.
  - **GPU**: recommended for regular batch processing and lower latency.

## SHANK format compatibility notes

### Current SHANK worker format
- SHANK already normalizes source audio to **44.1 kHz stereo WAV**.

### MT3-required runtime format
- MT3 inference should consume **16 kHz mono** sample stream.

### Proposed handoff
1. Keep existing normalization step unchanged (44.1 kHz stereo WAV).
2. Add MT3 pre-inference conversion in wrapper: 44.1k stereo WAV -> 16k mono float samples.
3. Save output MIDI (and optionally serialized NoteSequence) to task artifact directory.

## Model/cache paths and volume mounts

### Recommended container paths
- Model checkpoints (read-only): `/srv/shank/models/mt3/checkpoints`
- Runtime cache/compiled artifacts: `/srv/shank/cache/mt3`

### Example Docker Compose mounts
```yaml
services:
  shank:
    volumes:
      - ./data:/srv/shank/data
      - ./models/mt3/checkpoints:/srv/shank/models/mt3/checkpoints:ro
      - ./cache/mt3:/srv/shank/cache/mt3
```

### Suggested env vars
- `MT3_ENABLED=true|false`
- `MT3_MODEL=mt3|ismir2021`
- `MT3_CHECKPOINT_ROOT=/srv/shank/models/mt3/checkpoints`
- `MT3_CACHE_DIR=/srv/shank/cache/mt3`
- `MT3_DEVICE=cpu|gpu` (advisory routing/selection)

## Final recommendation summary
- Use a **thin wrapper script/module** in SHANK based on Colab inference logic.
- Ship with **both checkpoints**, defaulting to **multi-instrument (`mt3`)**.
- Treat **GPU as production default**, with CPU fallback.
- Keep SHANK’s current normalization, then add MT3-specific **44.1k stereo -> 16k mono** conversion before inference.
- Mount checkpoints and cache explicitly to keep model assets persistent and reproducible.
