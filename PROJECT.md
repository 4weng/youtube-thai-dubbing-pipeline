# Project: YouTube English-to-Thai Dubbing Pipeline

## Architecture
An offline-capable, 4-stage batch pipeline in Python that converts an English YouTube video into a Thai-dubbed MP4 (`output_th.mp4`) with atomic checkpointing and resume support:

```text
[YouTube URL (EN)]
       │
       ▼
Stage 1: src/ingest.py     → source.mp4, audio_en.wav (16kHz mono), segments_en.json
       │
       ▼
Stage 2: src/translate.py  → segments_th.json (context-aware EN→TH, length-fitted, Thai TTS normalized)
       │
       ▼
Stage 3: src/tts_align.py  → tts_segments/seg_XXXX.wav + incremental tts metadata in segments_th.json
       │
       ▼
Stage 4: src/mux.py        → thai_dub_full.wav + output_th.mp4 (duck or replace mode via ffmpeg)
```

Cross-cutting foundation:
- `src/state.py`: Exception hierarchy, `PipelineConfig`, `WorkspacePaths`, TypedDict JSON contracts, YouTube `video_id` extraction, `ffmpeg`-safe atomic writes (`.tmp` + `os.replace`), `ffprobe`/`soundfile` media integrity verification, `manifest.json` management, and Verify-Before-Skip.
- `pipeline.py`: CLI entrypoint (`--url`, `--mix-mode`, `--force-stage`, plus config flags) and 4-stage orchestrator with cascade invalidation and error capture into `manifest.json["last_error"]`.

## Environment & Dependencies
- **System binaries**: `ffmpeg` and `ffprobe` (install via `sudo apt-get update && sudo apt-get install -y ffmpeg`).
- **Python environment**: `/home/antigravity/.venv` (Python 3.12.16).
- **`requirements.txt`**:
  - `yt-dlp>=2026.3.0`
  - `faster-whisper>=1.2.0`
  - `deep-translator>=1.11.4`
  - `pythainlp>=5.4.0`
  - `pythaitts>=1.0.0` (spec default; pure ONNX, includes zero-download offline `makham_paradee` + `fastthaig2p`)
  - `edge-tts>=7.3.0`
  - `soundfile>=0.14.0`
  - `numpy>=2.0.0`
  - `pytest>=9.0.0`
- **Canonical test commands**:
  - Unit tests: `/home/antigravity/.venv/bin/pytest -v tests --ignore=tests/e2e`
  - E2E tests: `/home/antigravity/.venv/bin/pytest -v tests/e2e`
  - Full suite: `/home/antigravity/.venv/bin/pytest -v`

## Code Layout
```text
/home/antigravity/
├── pipeline.py              # CLI Entrypoint & Stage Orchestrator
├── requirements.txt         # Runtime & test dependencies
├── src/
│   ├── __init__.py
│   ├── state.py             # Atomic file writes, manifest & integrity checks
│   ├── ingest.py            # Stage 1: yt-dlp + Whisper / subtitle parser
│   ├── translate.py         # Stage 2: EN -> TH translation & text normalization
│   ├── tts_align.py         # Stage 3: Thai TTS generation, trimming & atempo sync
│   └── mux.py               # Stage 4: Audio timeline assembly & ffmpeg muxing
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared real-ffmpeg media fixtures
│   ├── test_state.py
│   ├── test_ingest.py
│   ├── test_translate.py
│   ├── test_tts_align.py
│   ├── test_mux.py
│   └── e2e/                 # Independent opaque-box E2E test suite (Tiers 1-4)
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_tier1_features.py
│       ├── test_tier2_boundaries.py
│       ├── test_tier3_combinations.py
│       └── test_tier4_scenarios.py
└── work/                    # Per-video runtime workspace (work/<video_id>/...)
```

## Feature Inventory
Every Survey feature is assigned to a Milestone (none unassigned).

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `--url <youtube_url>` CLI flag | Run full 4-stage pipeline for a YouTube URL; auto-skip verified completed stages/segments | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 2 | `--mix-mode <duck\|replace>` CLI flag | Select Stage 4 mixing mode (`duck` default or `replace`); persisted in `manifest.json` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 3 | `--force-stage <stage>` CLI flag | Force re-run of `ingest`/`translate`/`tts_align`/`mux` + cascade reset downstream stages | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 4 | YouTube `video_id` extraction | Parse canonical `video_id` from `youtu.be`, `watch?v=`, `/shorts/`, `/embed/`, `/live/`; reject invalid/path-traversal inputs | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 5 | Stage orchestration & error capture | Sequential stage execution updating `manifest.json` stage statuses and `last_error` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 6 | Per-video workspace isolation | Isolate all artifacts under `work/<video_id>/` and `work/<video_id>/tts_segments/` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 7 | `manifest.json` state management | Load, initialize, update, and atomically persist `manifest.json` matching Spec §4.1 | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 8 | Atomic file writes | `atomic_write_json()` and `atomic_temp_path()` (`.tmp` preserving ext for ffmpeg + `os.replace()`) | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 9 | Verify Before Skip | `is_valid_audio_file()`, `is_valid_video_file()`, `verify_stage_outputs()` checking existence, `size > 0`, `ffprobe`/`soundfile` `duration > 0` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 10 | Config persistence | Persist `mix_mode`, `original_volume` (0.18), `max_atempo` (1.25), `tts_provider` (`pythaitts`) in `manifest.json`; detect config change on re-run | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 11 | Video download (`yt-dlp`) | Download best MP4-compatible video to `work/<video_id>/source.mp4` atomically | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 12 | Audio extraction (16kHz mono WAV) | Extract `work/<video_id>/audio_en.wav` (16kHz, 1ch, PCM16) via `ffmpeg` atomically | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 13 | Subtitle extraction & parsing | Extract/parse EN VTT/SRT subtitles (strip tags, dedup rolling captions), prefer manual subs | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 14 | Whisper ASR fallback | `FasterWhisperASR` (`faster-whisper` INT8) fallback when accurate EN subtitles are unavailable | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 15 | Sentence-level segmentation (3–8s) | Merge short / split long cues into 3–8s sentence-level segments with 1-based `id`, `start < end`, `text_en` in `segments_en.json` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 16 | Context-aware EN→TH translation | Pluggable `Translator` (`OpenAICompatibleTranslator`, `DeepTranslatorProvider`) passing surrounding `prev`/`next` context window | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 17 | Length-fitting constraint | Compute target Thai char budget from `slot_dur = end - start` to keep translations concise | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 18 | Thai TTS text normalization | `normalize_thai_for_tts()` converting numbers, decimals, currencies, `%`, units, symbols, and English acronyms to Thai spoken words | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 19 | `segments_th.json` output | Atomically write `segments_th.json` preserving `id`, `start`, `end`, `text_en` + `text_th` + `tts` metadata | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 20 | TTS Provider Abstraction & Voice Locking | `TTSProvider` protocol (`PyThaiTTSProvider` default with offline `makham_paradee` + `EdgeTTSProvider`) locking one `voice_id` per video | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 21 | Leading/trailing silence trimming | Trim leading/trailing silence via `soundfile` + `numpy` energy envelope with safety pad and all-silence guard | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 22 | Duration measurement & bounded `atempo` | `tempo = 1.0` if `actual_dur <= slot_dur`, else `min(actual_dur / slot_dur, max_atempo)` with ffmpeg `atempo` chaining if `> 2.0` or `< 0.5` | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 23 | Per-segment checkpoint & incremental save | Update `seg["tts"]` and `atomic_write_json(segments_th)` after every segment; skip verified done segments on resume | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 24 | Timeline audio assembly (`thai_dub_full.wav`) | Place `seg_XXXX.wav` at `start` (`start_ms`) with silence gaps, sample-rate normalization, and overflow-safe `push` (default) / `mix` policy | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 25 | Video muxing — Duck mode | `ffmpeg` ducking with `[0:a]volume={original_volume}[orig];[1:a]volume=1.0[th];[orig][th]amix=inputs=2:duration=first:dropout_transition=2[aout]`, `-c:v copy -c:a aac -b:a 192k`; fallback if `source.mp4` has no audio stream | M1 (Core Impl) & E2E | spec_miner_survey_1 |
| 26 | Video muxing — Replace mode | `ffmpeg` replace with `-map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k` written atomically | M1 (Core Impl) & E2E | spec_miner_survey_1 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Full Pipeline Implementation & Unit Tests | Environment setup (`ffmpeg` + `.venv`), `requirements.txt`, `src/{__init__,state,ingest,translate,tts_align,mux}.py`, `pipeline.py`, unit tests (`tests/test_*.py`) | none | DONE (70/70 unit tests passed, ruff clean) |
| 2 | E2E Track: Opaque-Box E2E Test Suite (Tiers 1–4) | `TEST_INFRA.md`, `tests/e2e/test_tier{1,2,3,4}_*.py`, `TEST_READY.md` derived strictly from requirements & contracts | none (parallel) | DONE (58/58 E2E tests passed, 26/26 features) |
| 3 | Final Verification & Gate | Parallel Verification Wave (2 Reviewers, 2 Challengers, 1 Forensic Auditor) across full unit + E2E suite | M1, E2E Track | DONE — Iteration 3 Gate **PASS** (128/128 passed, ruff clean, Auditor CLEAN) |

## Interface Contracts

### `src/state.py`
- Exceptions: `PipelineError`, `ValidationError`, `MediaError`, `ProviderError`, `StageError(stage, message, cause=None)`
- Constants: `STAGE_ORDER = ("ingest", "translate", "tts_align", "mux")`
- Dataclasses:
  - `PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25, tts_provider="pythaitts", translator="auto", asr_provider="faster_whisper", voice_id=None)` with `.to_dict()` / `.from_dict()`
  - `WorkspacePaths(root, manifest, source_video, audio_en, segments_en, segments_th, tts_segments_dir, thai_dub_full, output_video)` with `.ensure_dirs()`, `.segment_wav_path(seg_id)`, `.segment_wav_relpath(seg_id)`
- Functions:
  - `extract_video_id(url: str) -> str`
  - `get_workspace_paths(work_dir: Path | str, video_id: str) -> WorkspacePaths`
  - `atomic_temp_path(target: Path | str) -> ContextManager[Path]`
  - `atomic_write_bytes(path, data)`, `atomic_write_text(path, text)`, `atomic_write_json(path, data, indent=2)`
  - `read_json(path)`, `is_valid_json_file(path, expected_type=None, allow_empty=False)`
  - `probe_media_info(path)`, `probe_media_duration(path)`, `has_audio_stream(path)`
  - `is_valid_audio_file(path) -> bool`, `is_valid_video_file(path, require_audio=False) -> bool`
  - `init_or_load_manifest(paths, video_id, source_url, config) -> dict`
  - `save_manifest(paths, manifest)`, `set_stage_status(paths, manifest, stage, status, error=None)`, `reset_stages_from(paths, manifest, start_stage)`
  - `verify_stage_outputs(paths, stage) -> bool`
  - `should_skip_stage(paths, manifest, stage, force_stage=None) -> bool`

### `src/ingest.py`
- Types: `RawCue(start, end, text)`, `DownloadResult(video_path, subtitle_path, subtitle_is_auto=False)`, `ASRProvider` protocol (`transcribe(audio_path, language="en") -> list[RawCue]`), `DownloaderFn`
- Functions/Classes: `download_video_and_subs`, `extract_audio_16k_mono`, `parse_vtt`, `parse_srt`, `parse_subtitle_file`, `segment_cues`, `FasterWhisperASR`, `create_asr_provider`, `run_ingest(paths, url, *, asr_provider=None, downloader=None, prefer_subtitles=True, min_seg_dur=3.0, max_seg_dur=8.0) -> list[SegmentENDict]`

### `src/translate.py`
- Protocol/Classes: `Translator` (`translate_segment(text_en, prev_context, next_context, slot_duration, max_chars) -> str`), `OpenAICompatibleTranslator`, `DeepTranslatorProvider`, `create_translator(name="auto", **kwargs)`
- Functions: `estimate_max_thai_chars`, `int_to_thai_words`, `number_to_thai_words`, `normalize_thai_for_tts`, `run_translate(paths, translator, *, context_window=2, chars_per_sec=14.0) -> list[SegmentTHDict]`

### `src/tts_align.py`
- Protocol/Classes: `TTSProvider` (`voice_id: str`, `synthesize(text_th, output_wav_path) -> Path`), `PyThaiTTSProvider` (default `pretrained="makham_paradee"` offline ONNX, locked voice/speaker), `EdgeTTSProvider`, `create_tts_provider(name="pythaitts", voice_id=None, **kwargs)`
- Functions: `trim_silence`, `build_atempo_filter`, `apply_atempo`, `align_segment_audio`, `run_tts_align(paths, provider, *, max_atempo=1.25, force=False) -> list[SegmentTHDict]`

### `src/mux.py`
- Functions: `resample_mono`, `assemble_timeline(paths, segments_th, *, total_duration=None, sample_rate=44100, overlap_policy="push", min_gap_sec=0.02) -> Path`, `mux_video(source_video, dub_audio, output_video, *, mix_mode="duck", original_volume=0.18) -> Path`, `run_mux(paths, *, mix_mode="duck", original_volume=0.18, overlap_policy="push") -> Path`

### `pipeline.py`
- CLI Flags:
  - `--url` (required)
  - `--mix-mode` (`duck` | `replace`, default `duck`)
  - `--force-stage` (`ingest` | `translate` | `tts_align` | `mux`, default `None`)
  - `--work-dir` (default `work`)
  - `--original-volume` (float, default `0.18`)
  - `--max-atempo` (float, default `1.25`)
  - `--tts-provider` (default `pythaitts`)
  - `--voice-id` (default `None`)
  - `--translator` (default `auto`)
  - `--asr-provider` (default `faster_whisper`)
  - `-v` / `--verbose`
- Functions: `build_arg_parser()`, `run_pipeline(url, config, *, work_dir=Path("work"), force_stage=None, downloader=None, asr_provider=None, translator=None, tts_provider=None) -> Path`, `main(argv=None) -> int`
