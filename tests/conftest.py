"""Shared deterministic real-ffmpeg media fixtures for unit tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def make_test_video(
    path: Path,
    duration: float = 6.0,
    freq: float = 440.0,
    with_audio: bool = True,
) -> Path:
    """Create a real H.264 (+ optional AAC sine audio) MP4 file using ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=15:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=44100:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=15:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"make_test_video ffmpeg failed: {proc.stderr}")
    return path


def make_test_wav(
    path: Path,
    duration: float = 1.5,
    freq: float = 300.0,
    sr: int = 24000,
    amplitude: float = 0.5,
    lead_silence: float = 0.0,
    trail_silence: float = 0.0,
    channels: int = 1,
) -> Path:
    """Create a real PCM16 WAV file with optional leading/trailing silence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_tone = max(1, int(round(duration * sr)))
    t = np.arange(n_tone, dtype=np.float32) / float(sr)
    tone = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    parts = []
    if lead_silence > 0:
        parts.append(np.zeros(int(round(lead_silence * sr)), dtype=np.float32))
    parts.append(tone)
    if trail_silence > 0:
        parts.append(np.zeros(int(round(trail_silence * sr)), dtype=np.float32))

    samples = np.concatenate(parts)
    if channels == 2:
        samples = np.column_stack([samples, samples])

    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


class LocalToneTTSProvider:
    """Deterministic offline TTSProvider for unit tests with locked voice_id & fault injection."""

    def __init__(
        self,
        voice_id: str = "test-tone-v1",
        sec_per_char: float = 0.08,
        lead_silence: float = 0.25,
        trail_silence: float = 0.25,
        fail_on_call: int | None = None,
        sr: int = 24000,
    ) -> None:
        self.voice_id = voice_id
        self.sec_per_char = sec_per_char
        self.lead_silence = lead_silence
        self.trail_silence = trail_silence
        self.fail_on_call = fail_on_call
        self.sr = sr
        self.calls: list[str] = []

    def synthesize(self, text_th: str, output_wav_path: Path) -> Path:
        self.calls.append(text_th)
        call_num = len(self.calls)
        if self.fail_on_call is not None and call_num == self.fail_on_call:
            raise RuntimeError(f"Injected TTS failure on call #{call_num}")
        dur = max(0.3, len(text_th) * self.sec_per_char)
        return make_test_wav(
            output_wav_path,
            duration=dur,
            freq=320.0 + (call_num * 30.0),
            sr=self.sr,
            lead_silence=self.lead_silence,
            trail_silence=self.trail_silence,
        )


@pytest.fixture
def video_factory():
    return make_test_video


@pytest.fixture
def wav_factory():
    return make_test_wav
