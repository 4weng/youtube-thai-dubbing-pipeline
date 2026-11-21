"""Shared fixtures and deterministic helpers for the 4-tier E2E test suite.

All media fixtures are generated using real ``ffmpeg`` / ``ffprobe`` binaries
and ``soundfile`` + ``numpy`` so that every stage exercises the genuine audio
and video toolchain offline without external network flakiness.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest
import soundfile as sf

from src.ingest import DownloadResult, RawCue
from src.state import WorkspacePaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def get_python_executable() -> str:
    """Return the project virtualenv Python if present, else current interpreter."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def require_ffmpeg() -> None:
    """Skip test if ffmpeg or ffprobe is unavailable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for E2E tests")


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and raise AssertionError with diagnostics on failure."""
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def ffprobe_json(media_path: Path | str) -> dict[str, Any]:
    """Return parsed ffprobe JSON metadata for a media file."""
    require_ffmpeg()
    proc = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ]
    )
    return json.loads(proc.stdout)


def create_synthetic_video(
    output_path: Path,
    *,
    duration: float = 12.0,
    freq: float = 440.0,
    with_audio: bool = True,
    sample_rate: int = 44100,
    fps: int = 24,
) -> Path:
    """Generate a deterministic local MP4 using real ffmpeg."""
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=navy:s=320x240:r={fps}:d={duration}",
    ]
    if with_audio:
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate={sample_rate}:duration={duration},volume=5.0",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd += [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
    run_cmd(cmd)
    return output_path


def create_tone_wav(
    output_path: Path,
    *,
    duration: float,
    freq: float = 880.0,
    sample_rate: int = 22050,
    amplitude: float = 0.5,
    leading_silence: float = 0.0,
    trailing_silence: float = 0.0,
) -> Path:
    """Write a deterministic mono PCM16 WAV with optional leading/trailing silence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_tone = max(1, int(round(duration * sample_rate)))
    t = np.arange(n_tone, dtype=np.float64) / float(sample_rate)
    tone = amplitude * np.sin(2.0 * math.pi * freq * t)

    parts: list[np.ndarray] = []
    if leading_silence > 0.0:
        parts.append(np.zeros(int(round(leading_silence * sample_rate)), dtype=np.float64))
    parts.append(tone)
    if trailing_silence > 0.0:
        parts.append(np.zeros(int(round(trailing_silence * sample_rate)), dtype=np.float64))

    audio = np.concatenate(parts).astype(np.float32)
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    return output_path


def create_silent_wav(
    output_path: Path,
    *,
    duration: float = 2.0,
    sample_rate: int = 22050,
) -> Path:
    """Write an all-zero mono PCM16 WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(max(1, int(round(duration * sample_rate))), dtype=np.float32)
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    return output_path


def measure_window_rms(
    media_path: Path | str,
    start_sec: float,
    end_sec: float,
    *,
    tmp_dir: Path,
    sample_rate: int = 22050,
) -> float:
    """Extract audio from any media file via ffmpeg and return RMS in [start_sec, end_sec]."""
    require_ffmpeg()
    extracted = tmp_dir / f"rms_extract_{abs(hash((str(media_path), start_sec, end_sec)))}.wav"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(extracted),
        ]
    )
    data, sr = sf.read(str(extracted), dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    i0 = max(0, int(round(start_sec * sr)))
    i1 = min(len(data), int(round(end_sec * sr)))
    assert i1 > i0, f"Empty audio window [{start_sec}, {end_sec}] for duration {len(data)/sr:.3f}s"
    window = data[i0:i1]
    return float(np.sqrt(np.mean(np.square(window))))


def _fmt_vtt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hrs, rem = divmod(ms, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _fmt_srt_ts(seconds: float) -> str:
    return _fmt_vtt_ts(seconds).replace(".", ",")


def write_vtt(path: Path, cues: Sequence[tuple[float, float, str]]) -> Path:
    """Write a WebVTT subtitle file from (start, end, text) tuples."""
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        lines.append(f"{_fmt_vtt_ts(start)} --> {_fmt_vtt_ts(end)}")
        lines.append(text)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_srt(path: Path, cues: Sequence[tuple[float, float, str]]) -> Path:
    """Write an SRT subtitle file from (start, end, text) tuples."""
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_ts(start)} --> {_fmt_srt_ts(end)}")
        lines.append(text)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_local_downloader(
    video_src: Path,
    *,
    subtitle_src: Path | None = None,
    subtitle_is_auto: bool = False,
    call_log: list[str] | None = None,
) -> Callable[..., DownloadResult]:
    """Create a deterministic offline downloader compatible with run_ingest/run_pipeline.

    Accepts flexible positional/keyword argument ordering ``(url, paths)`` or
    ``(paths, url)`` or ``(url, out_dir)`` to remain strictly opaque-box.
    """

    def _download(*args: Any, **kwargs: Any) -> DownloadResult:
        url: str | None = kwargs.get("url")
        paths: WorkspacePaths | None = kwargs.get("paths")
        out_dir: Path | None = kwargs.get("output_dir") or kwargs.get("out_dir")
        path_args: list[Path] = []

        for arg in args:
            if isinstance(arg, WorkspacePaths):
                paths = arg
            elif isinstance(arg, Path):
                path_args.append(arg)
            elif isinstance(arg, str) and url is None:
                url = arg

        if call_log is not None:
            call_log.append(url or "")

        if paths is not None:
            paths.ensure_dirs()
            dst_video = paths.source_video
            sub_dir = paths.root / "subs"
            sub_dir.mkdir(parents=True, exist_ok=True)
        elif len(path_args) >= 2:
            dst_video = path_args[0]
            sub_dir = path_args[1]
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            sub_dir.mkdir(parents=True, exist_ok=True)
        elif len(path_args) == 1:
            target = path_args[0]
            if target.suffix == "" or target.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                dst_video = target / "source.mp4"
                sub_dir = target
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                dst_video = target
                sub_dir = dst_video.parent
        elif out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            dst_video = out_dir / "source.mp4" if out_dir.is_dir() else out_dir
            sub_dir = dst_video.parent
        else:
            raise TypeError("Downloader did not receive WorkspacePaths or output directory")

        shutil.copy2(video_src, dst_video)
        dst_sub: Path | None = None
        if subtitle_src is not None:
            dst_sub = sub_dir / subtitle_src.name
            shutil.copy2(subtitle_src, dst_sub)

        return DownloadResult(
            video_path=dst_video,
            subtitle_path=dst_sub,
            subtitle_is_auto=subtitle_is_auto,
        )

    return _download


class FakeASRProvider:
    """Deterministic ASRProvider returning pre-configured RawCue entries."""

    def __init__(self, cues: Sequence[RawCue]) -> None:
        self.cues = list(cues)
        self.calls: list[tuple[Path, str]] = []

    def transcribe(self, audio_path: Path | str, language: str = "en") -> list[RawCue]:
        p = Path(audio_path)
        self.calls.append((p, language))
        assert p.exists() and p.stat().st_size > 0, f"ASR received missing/empty audio: {p}"
        return list(self.cues)


class FakeTranslator:
    """Deterministic Translator that records context windows and length budgets."""

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        *,
        default_prefix: str = "แปลไทย: ",
    ) -> None:
        self.mapping = dict(mapping or {})
        self.default_prefix = default_prefix
        self.calls: list[dict[str, Any]] = []

    def translate_segment(
        self,
        text_en: str,
        prev_context: Sequence[str] | None = None,
        next_context: Sequence[str] | None = None,
        slot_duration: float = 0.0,
        max_chars: int = 0,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "text_en": text_en,
                "prev_context": list(prev_context or []),
                "next_context": list(next_context or []),
                "slot_duration": slot_duration,
                "max_chars": max_chars,
                "kwargs": kwargs,
            }
        )
        if text_en in self.mapping:
            return self.mapping[text_en]
        return f"{self.default_prefix}{text_en}"


class FakeTTSProvider:
    """Deterministic offline TTSProvider generating real PCM16 WAV files.

    Supports customizable duration, silence padding, and fault injection on a
    specific 1-based synthesis call index.
    """

    def __init__(
        self,
        *,
        voice_id: str = "fake_thai_voice_01",
        duration_fn: Callable[[str, int], float] | None = None,
        freq: float = 880.0,
        amplitude: float = 0.6,
        leading_silence: float = 0.0,
        trailing_silence: float = 0.0,
        all_silence: bool = False,
        fail_on_call: int | None = None,
        sample_rate: int = 22050,
    ) -> None:
        self.voice_id = voice_id
        self.duration_fn = duration_fn
        self.freq = freq
        self.amplitude = amplitude
        self.leading_silence = leading_silence
        self.trailing_silence = trailing_silence
        self.all_silence = all_silence
        self.fail_on_call = fail_on_call
        self.sample_rate = sample_rate
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, text_th: str, output_wav_path: Path | str) -> Path:
        call_idx = len(self.calls) + 1
        out_path = Path(output_wav_path)
        self.calls.append(
            {
                "call_idx": call_idx,
                "text_th": text_th,
                "output_wav_path": out_path,
                "voice_id": self.voice_id,
            }
        )
        if self.fail_on_call is not None and call_idx == self.fail_on_call:
            raise RuntimeError(f"Simulated TTS crash on call #{call_idx}")

        dur = self.duration_fn(text_th, call_idx) if self.duration_fn else 2.0
        if self.all_silence:
            return create_silent_wav(out_path, duration=dur, sample_rate=self.sample_rate)
        return create_tone_wav(
            out_path,
            duration=dur,
            freq=self.freq,
            sample_rate=self.sample_rate,
            amplitude=self.amplitude,
            leading_silence=self.leading_silence,
            trailing_silence=self.trailing_silence,
        )


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Isolated per-test pipeline work directory."""
    d = tmp_path / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def synthetic_video_12s(tmp_path: Path) -> Path:
    """12-second synthetic MP4 with 440Hz continuous tone audio."""
    return create_synthetic_video(tmp_path / "fixture_12s.mp4", duration=12.0, freq=440.0, with_audio=True)


@pytest.fixture
def synthetic_video_silent(tmp_path: Path) -> Path:
    """8-second synthetic MP4 containing only a video stream (no audio)."""
    return create_synthetic_video(tmp_path / "fixture_silent.mp4", duration=8.0, with_audio=False)


@pytest.fixture
def sample_vtt_file(tmp_path: Path) -> Path:
    """Multi-sentence English WebVTT fixture aligned within a 12-second video."""
    cues = [
        (1.5, 4.8, "Today we build a local voice pipeline."),
        (5.5, 9.0, "In 2026, AI API growth reached 50% with $100 million."),
    ]
    return write_vtt(tmp_path / "en_manual.vtt", cues)


@pytest.fixture
def sample_srt_file(tmp_path: Path) -> Path:
    """Multi-sentence English SRT fixture aligned within a 12-second video."""
    cues = [
        (1.0, 4.5, "Welcome to the Thai dubbing demo."),
        (5.2, 8.8, "This tool aligns speech automatically."),
    ]
    return write_srt(tmp_path / "en_manual.srt", cues)
