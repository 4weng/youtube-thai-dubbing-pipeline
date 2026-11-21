"""Stage 3: Thai TTS synthesis (locked voice), silence trimming, bounded atempo alignment, and incremental save."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from src.state import (
    MediaError,
    ProviderError,
    SegmentTHDict,
    StageError,
    ValidationError,
    WorkspacePaths,
    atomic_temp_path,
    atomic_write_json,
    is_valid_audio_file,
    is_valid_json_file,
    probe_media_duration,
    read_json,
    resolve_workspace_relpath,
)

logger = logging.getLogger(__name__)


class TTSProvider(Protocol):
    voice_id: str

    def synthesize(self, text_th: str, output_wav_path: Path) -> Path:
        ...


class PyThaiTTSProvider:
    """Offline-capable Thai TTS provider backed by pythaitts>=1.0.0 (ONNX, default makham_paradee)."""

    def __init__(
        self,
        pretrained: str = "makham_paradee",
        speaker_idx: str = "thai_som",
        device: str = "cpu",
        voice_id: str | None = None,
    ) -> None:
        if voice_id:
            if ":" in voice_id:
                model_part, spk_part = voice_id.split(":", 1)
                self.pretrained = model_part.strip() or pretrained
                self.speaker_idx = spk_part.strip() or speaker_idx
            else:
                self.pretrained = voice_id.strip()
                self.speaker_idx = speaker_idx
        else:
            self.pretrained = pretrained
            self.speaker_idx = speaker_idx

        self.device = device
        self.voice_id = f"{self.pretrained}:{self.speaker_idx}"
        self._tts: Any = None

    def _get_tts(self) -> Any:
        if self._tts is None:
            try:
                from pythaitts import TTS
            except ImportError as exc:
                raise ProviderError("pythaitts is not installed.") from exc
            try:
                self._tts = TTS(pretrained=self.pretrained, device=self.device)
            except Exception as exc:
                raise ProviderError(
                    f"Failed to initialize PyThaiTTS(pretrained={self.pretrained!r}): {exc}"
                ) from exc
        return self._tts

    def synthesize(self, text_th: str, output_wav_path: Path) -> Path:
        text = (text_th or "").strip()
        if not text:
            raise ValidationError("Cannot synthesize empty Thai text.")

        out_path = Path(output_wav_path)
        tts = self._get_tts()
        try:
            with atomic_temp_path(out_path) as tmp_wav:
                tts.tts(
                    text,
                    speaker_idx=self.speaker_idx,
                    return_type="file",
                    filename=str(tmp_wav),
                )
        except Exception as exc:
            if isinstance(exc, (ProviderError, ValidationError)):
                raise
            raise ProviderError(f"PyThaiTTS synthesis failed: {exc}") from exc

        if not is_valid_audio_file(out_path):
            raise MediaError(f"PyThaiTTS produced invalid WAV file: {out_path}")
        return out_path


class EdgeTTSProvider:
    """Online neural Thai TTS provider backed by edge-tts + ffmpeg decode to PCM16 WAV."""

    def __init__(self, voice_id: str = "th-TH-PremwadeeNeural") -> None:
        if not voice_id or not voice_id.strip():
            raise ValidationError("EdgeTTSProvider requires a non-empty voice_id.")
        self.voice_id = voice_id.strip()

    def synthesize(self, text_th: str, output_wav_path: Path) -> Path:
        text = (text_th or "").strip()
        if not text:
            raise ValidationError("Cannot synthesize empty Thai text.")

        try:
            import edge_tts
        except ImportError as exc:
            raise ProviderError("edge-tts is not installed.") from exc

        out_path = Path(output_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_mp3 = out_path.parent / f".edge.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.mp3"

        async def _save_mp3() -> None:
            comm = edge_tts.Communicate(text, self.voice_id)
            await comm.save(str(tmp_mp3))

        try:
            asyncio.run(_save_mp3())
            if not tmp_mp3.is_file() or tmp_mp3.stat().st_size <= 0:
                raise ProviderError("edge-tts produced an empty audio stream.")

            with atomic_temp_path(out_path) as tmp_wav:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_mp3),
                    "-acodec",
                    "pcm_s16le",
                    str(tmp_wav),
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                if proc.returncode != 0:
                    raise MediaError(
                        f"ffmpeg decode of edge-tts audio failed (rc={proc.returncode}): {proc.stderr.strip()}"
                    )
        except Exception as exc:
            if isinstance(exc, (ProviderError, MediaError, ValidationError)):
                raise
            raise ProviderError(f"EdgeTTSProvider synthesis failed: {exc}") from exc
        finally:
            try:
                if tmp_mp3.exists():
                    tmp_mp3.unlink()
            except OSError:
                pass

        if not is_valid_audio_file(out_path):
            raise MediaError(f"EdgeTTSProvider produced invalid WAV file: {out_path}")
        return out_path


def create_tts_provider(
    name: str = "pythaitts",
    voice_id: str | None = None,
    **kwargs: Any,
) -> TTSProvider:
    """Factory for Thai TTS providers."""
    normalized = (name or "pythaitts").strip().lower()
    if normalized in ("pythaitts", "pythai"):
        return PyThaiTTSProvider(voice_id=voice_id, **kwargs)
    if normalized in ("edge_tts", "edge-tts", "edge"):
        return EdgeTTSProvider(voice_id=voice_id or "th-TH-PremwadeeNeural", **kwargs)
    raise ProviderError(f"Unsupported TTS provider: {name!r}")


def trim_silence(
    input_wav: Path,
    output_wav: Path,
    top_db: float = 35.0,
    pad_ms: float = 30.0,
) -> Path:
    """Trim leading and trailing silence using soundfile + numpy RMS envelope with safety pad.

    All-silence guard: if input is silent or trimming would leave 0 samples, falls back to
    preserving the original audio rather than writing an invalid 0-second WAV.
    """
    in_path = Path(input_wav)
    out_path = Path(output_wav)
    if not is_valid_audio_file(in_path):
        raise MediaError(f"Invalid input WAV for silence trimming: {in_path}")

    data, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    n_samples = mono.shape[0]

    peak = float(np.max(np.abs(mono))) if n_samples > 0 else 0.0
    trimmed = data

    if peak >= 1e-6 and n_samples > 0:
        threshold = max(peak * (10.0 ** (-float(top_db) / 20.0)), 1e-5)
        frame_len = max(1, int(round(0.01 * sr)))  # 10ms frames
        n_frames = max(1, (n_samples + frame_len - 1) // frame_len)
        active_frames: list[int] = []
        for idx in range(n_frames):
            start_i = idx * frame_len
            end_i = min(n_samples, start_i + frame_len)
            frame = mono[start_i:end_i]
            rms = float(np.sqrt(np.mean(frame * frame)))
            if rms >= threshold:
                active_frames.append(idx)

        if active_frames:
            pad_samples = max(0, int(round((float(pad_ms) / 1000.0) * sr)))
            start_sample = max(0, active_frames[0] * frame_len - pad_samples)
            end_sample = min(n_samples, (active_frames[-1] + 1) * frame_len + pad_samples)
            if end_sample > start_sample:
                trimmed = data[start_sample:end_sample]

    with atomic_temp_path(out_path) as tmp_wav:
        sf.write(str(tmp_wav), trimmed, int(sr), subtype="PCM_16")

    if not is_valid_audio_file(out_path):
        raise MediaError(f"Trimmed WAV failed validation: {out_path}")
    return out_path


def _format_atempo_factor(val: float) -> str:
    formatted = f"{val:.4f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = f"{val:.1f}"
    return formatted


def build_atempo_filter(tempo: float) -> str:
    """Build ffmpeg `atempo` filter chain keeping every factor within [0.5, 2.0]."""
    if not isinstance(tempo, (int, float)) or not math.isfinite(tempo) or tempo <= 0.0:
        raise ValidationError(f"tempo must be a positive finite number, got {tempo!r}.")

    factors: list[float] = []
    rem = float(tempo)
    while rem > 2.0 + 1e-9:
        factors.append(2.0)
        rem /= 2.0
    while rem < 0.5 - 1e-9:
        factors.append(0.5)
        rem /= 0.5
    factors.append(rem)
    return ",".join(f"atempo={_format_atempo_factor(f)}" for f in factors)


def apply_atempo(input_wav: Path, output_wav: Path, tempo: float) -> Path:
    """Apply ffmpeg `atempo` time-stretching atomically and verify the resulting WAV."""
    in_path = Path(input_wav)
    out_path = Path(output_wav)
    if not is_valid_audio_file(in_path):
        raise MediaError(f"Invalid input WAV for atempo: {in_path}")

    filter_str = build_atempo_filter(tempo)
    with atomic_temp_path(out_path) as tmp_wav:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-filter:a",
            filter_str,
            "-acodec",
            "pcm_s16le",
            str(tmp_wav),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise MediaError(
                f"ffmpeg atempo failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )

    if not is_valid_audio_file(out_path):
        raise MediaError(f"Aligned WAV failed validation after atempo: {out_path}")
    return out_path


def align_segment_audio(
    raw_wav: Path,
    output_wav: Path,
    slot_duration: float,
    max_atempo: float = 1.25,
) -> tuple[float, float]:
    """Trim silence, compute bounded atempo <= max_atempo, write aligned WAV atomically.

    Returns `(final_duration, tempo_used)`.
    """
    if slot_duration <= 0.0 or not math.isfinite(slot_duration):
        raise ValidationError(f"slot_duration must be > 0, got {slot_duration!r}.")
    if max_atempo < 1.0 or not math.isfinite(max_atempo):
        raise ValidationError(f"max_atempo must be >= 1.0, got {max_atempo!r}.")

    raw_path = Path(raw_wav)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trimmed_tmp = out_path.parent / f".trim.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.wav"
    try:
        trim_silence(raw_path, trimmed_tmp)
        actual_dur = probe_media_duration(trimmed_tmp)
        if actual_dur is None or actual_dur <= 0.0:
            raise MediaError(f"Could not measure duration of trimmed WAV: {trimmed_tmp}")

        if actual_dur <= slot_duration:
            tempo_used = 1.0
            with atomic_temp_path(out_path) as final_tmp:
                shutil.copyfile(trimmed_tmp, final_tmp)
        else:
            needed_tempo = actual_dur / float(slot_duration)
            tempo_used = min(needed_tempo, float(max_atempo))
            if tempo_used <= 1.0001:
                tempo_used = 1.0
                with atomic_temp_path(out_path) as final_tmp:
                    shutil.copyfile(trimmed_tmp, final_tmp)
            else:
                apply_atempo(trimmed_tmp, out_path, tempo_used)

        final_dur = probe_media_duration(out_path)
        if final_dur is None or final_dur <= 0.0:
            raise MediaError(f"Final aligned WAV failed duration check: {out_path}")
        return (round(float(final_dur), 3), round(float(tempo_used), 4))
    finally:
        try:
            if trimmed_tmp.exists():
                trimmed_tmp.unlink()
        except OSError:
            pass


def run_tts_align(
    paths: WorkspacePaths,
    provider: TTSProvider,
    *,
    max_atempo: float = 1.25,
    force: bool = False,
) -> list[SegmentTHDict]:
    """Execute Stage 3 with verify-before-skip per segment and incremental atomic save."""
    if not is_valid_json_file(paths.segments_th, expected_type=list, allow_empty=False):
        raise StageError("tts_align", f"Missing or invalid segments_th.json: {paths.segments_th}")

    paths.ensure_dirs()
    segments: list[SegmentTHDict] = read_json(paths.segments_th)
    if force:
        for seg in segments:
            seg["tts"] = {
                "status": "pending",
                "file": None,
                "duration": None,
                "tempo_used": None,
            }
        atomic_write_json(paths.segments_th, segments)

    logger.info(
        "Stage 3: running TTS alignment for %d segments with locked voice_id=%s",
        len(segments),
        getattr(provider, "voice_id", "unknown"),
    )

    for seg in segments:
        seg_id = int(seg["id"])
        wav_path = paths.segment_wav_path(seg_id)
        rel_path = paths.segment_wav_relpath(seg_id)

        raw_tts = seg.get("tts")
        tts_meta = raw_tts if isinstance(raw_tts, dict) else {}
        existing_rel = tts_meta.get("file") or rel_path
        existing_wav = resolve_workspace_relpath(paths, existing_rel)
        if (
            not force
            and tts_meta.get("status") == "done"
            and existing_wav is not None
            and is_valid_audio_file(existing_wav)
        ):
            logger.debug("Stage 3: skipping verified segment %d", seg_id)
            continue

        raw_tmp = paths.tts_segments_dir / f".raw_{seg_id:04d}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.wav"
        try:
            text_th = str(seg.get("text_th") or "").strip()
            if not text_th:
                raise ValidationError(f"Segment {seg_id} has empty text_th.")

            provider.synthesize(text_th, raw_tmp)
            slot_dur = max(0.1, float(seg["end"]) - float(seg["start"]))
            final_dur, tempo_used = align_segment_audio(
                raw_tmp,
                wav_path,
                slot_duration=slot_dur,
                max_atempo=max_atempo,
            )

            seg["tts"] = {
                "status": "done",
                "file": rel_path,
                "duration": final_dur,
                "tempo_used": tempo_used,
            }
            atomic_write_json(paths.segments_th, segments)
        except Exception as exc:
            seg["tts"] = {
                "status": "failed",
                "file": None,
                "duration": None,
                "tempo_used": None,
                "error": str(exc),
            }
            atomic_write_json(paths.segments_th, segments)
            raise StageError(
                "tts_align",
                f"Segment {seg_id} failed: {exc}",
                cause=exc,
            ) from exc
        finally:
            try:
                if raw_tmp.exists():
                    raw_tmp.unlink()
            except OSError:
                pass

    return segments
