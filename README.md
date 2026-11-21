# YouTube English-to-Thai Dubbing Pipeline

A 4-stage offline batch pipeline that downloads a YouTube video, extracts English captions (or runs `faster-whisper` ASR fallback), translates them into natural Thai, synthesizes locked-voice Thai speech with time-aligned `ffmpeg` `atempo`, and muxes the final dubbed video (`output_th.mp4`) with resumable state and cascade invalidation.

## Features

- **Stage 1 — Ingest (`src/ingest.py`)**: `yt-dlp` download & subtitle fetch, 16 kHz mono WAV extraction, VTT/SRT parsing, 3–8 s segmentation, and `faster-whisper` ASR fallback.
- **Stage 2 — Translate (`src/translate.py`)**: Context-aware EN→TH translation (`OpenAICompatibleTranslator` or `DeepTranslatorProvider`), length fitting, and Thai TTS text normalization.
- **Stage 3 — TTS & Align (`src/tts_align.py`)**: Locked-voice Thai TTS (`PyThaiTTSProvider` offline ONNX default, or `EdgeTTSProvider`), silence trimming, bounded `atempo` alignment (with `[0.5, 2.0]` chaining), and incremental checkpoint saves.
- **Stage 4 — Mux (`src/mux.py`)**: Unified timeline assembly (`thai_dub_full.wav`) and `ffmpeg` muxing in `duck` or `replace` modes (`output_th.mp4`).
- **Resumable State (`src/state.py`)**: Atomic file writes (`fsync` + `os.replace`), media integrity verification (`ffprobe`/`soundfile`), Verify-Before-Skip, and cascade stage invalidation.

---

## Quick Start on Windows 11

### 1. Prerequisites

Install **Git**, **FFmpeg** (`ffmpeg` + `ffprobe`), and **Python 3.12** (recommended):

```powershell
winget install Git.Git
winget install Gyan.FFmpeg
winget install Python.Python.3.12
```

Close and reopen PowerShell, then verify:

```powershell
ffmpeg -version
ffprobe -version
python --version
```

### 2. Clone & Install

```powershell
git clone https://github.com/4weng/youtube-thai-dubbing-pipeline.git
cd youtube-thai-dubbing-pipeline

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the Pipeline

```powershell
# Basic usage (duck mode, offline PyThaiTTS default)
python pipeline.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Replace original audio track completely
python pipeline.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --mix-mode replace

# Force re-run from a specific stage (auto-resets downstream stages)
python pipeline.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --force-stage tts_align

# View all options
python pipeline.py --help
```

#### Optional: LLM Translation (OpenAI-compatible API)

By default (`--translator auto`), the pipeline uses `OpenAICompatibleTranslator` when `OPENAI_API_KEY` is set, and falls back to `DeepTranslatorProvider` otherwise.

```powershell
$env:OPENAI_API_KEY = "sk-..."
# Optional custom endpoint / model:
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
# $env:OPENAI_MODEL = "gpt-4o-mini"

python pipeline.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --translator openai
```

---

## CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--url` | *(required)* | YouTube video URL (`youtu.be`, `watch?v=`, `/shorts/`, `/embed/`, `/live/`) |
| `--mix-mode` | `duck` | Stage 4 audio mixing mode (`duck` \| `replace`) |
| `--force-stage` | `None` | Force re-run of `ingest`, `translate`, `tts_align`, or `mux` (cascades downstream) |
| `--work-dir` | `work` | Root directory for per-video workspaces |
| `--original-volume` | `0.18` | Original audio volume in `duck` mode (`0.0`–`1.0`) |
| `--max-atempo` | `1.25` | Maximum `ffmpeg` `atempo` speed-up factor (`>= 1.0`) |
| `--tts-provider` | `pythaitts` | Thai TTS provider (`pythaitts` \| `edge_tts`) |
| `--voice-id` | `None` | Optional locked voice/speaker ID for the TTS provider |
| `--translator` | `auto` | Translator backend (`auto` \| `openai` \| `deep_translator`) |
| `--asr-provider` | `faster_whisper` | ASR fallback backend (`faster_whisper`) |
| `-v, --verbose` | `False` | Enable verbose debug logging |

---

## Running Tests

```powershell
pytest -v
```

The test suite includes 70 unit/regression tests (`tests/test_*.py`) and 58 end-to-end tests (`tests/e2e/test_tier*.py`).

## Documentation

- Specification: [`YouTube_Thai_Dubbing_Pipeline_Spec.md`](YouTube_Thai_Dubbing_Pipeline_Spec.md)
- Architecture & Design: [`PROJECT.md`](PROJECT.md)
