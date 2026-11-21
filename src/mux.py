"""Stage 4: Audio timeline assembly (thai_dub_full.wav) and ffmpeg video muxing (duck/replace)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from src.state import (
    MediaError,
    MixMode,
    SegmentTHDict,
    StageError,
    ValidationError,
    WorkspacePaths,
    atomic_temp_path,
    has_audio_stream,
    is_valid_audio_file,
    is_valid_json_file,
    is_valid_video_file,
    probe_media_duration,
    read_json,
    resolve_workspace_relpath,
)

logger = logging.getLogger(__name__)

OverlapPolicy = Literal["push", "mix"]


def resample_mono(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Convert audio array to 1-D mono float32 and resample cleanly to `target_sr`."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    elif arr.ndim != 1:
        arr = arr.reshape(-1).astype(np.float32)

    if arr.size == 0 or orig_sr == target_sr:
        return arr

    if orig_sr <= 0 or target_sr <= 0:
        raise ValidationError(f"Invalid sample rates: orig_sr={orig_sr}, target_sr={target_sr}")

    target_len = max(1, int(round(arr.size * (target_sr / float(orig_sr)))))
    old_x = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, arr).astype(np.float32)


def assemble_timeline(
    paths: WorkspacePaths,
    segments_th: list[SegmentTHDict],
    *,
    total_duration: float | None = None,
    sample_rate: int = 44100,
    overlap_policy: OverlapPolicy = "push",
    min_gap_sec: float = 0.02,
) -> Path:
    """Assemble per-segment WAV files onto a unified PCM16 mono track (`thai_dub_full.wav`).

    Overflow handling when capped `atempo` leaves a segment longer than its slot:
    - 'push' (default): place at `max(start_sample, cursor_sample + min_gap_samples)` so
      Thai speech never talks over the next sentence.
    - 'mix': place at exact `start_sample` and additively mix overlapping tails with soft limiting.
    """
    if overlap_policy not in ("push", "mix"):
        raise ValidationError(
            f"Invalid overlap_policy: {overlap_policy!r}. Expected 'push' or 'mix'."
        )

    ordered = sorted(segments_th, key=lambda s: (float(s["start"]), int(s["id"])))
    placements: list[tuple[int, np.ndarray]] = []
    min_gap_samples = max(0, int(round(min_gap_sec * sample_rate)))
    cursor_sample = 0
    max_end_sample = int(round(total_duration * sample_rate)) if (total_duration and total_duration > 0) else 0

    for seg in ordered:
        raw_tts = seg.get("tts")
        tts_info = raw_tts if isinstance(raw_tts, dict) else {}
        rel_file = tts_info.get("file") or paths.segment_wav_relpath(int(seg["id"]))
        wav_path = resolve_workspace_relpath(paths, rel_file)
        if wav_path is None or not is_valid_audio_file(wav_path):
            raise MediaError(
                f"Missing, unsafe, or invalid segment WAV for segment {seg['id']}: {rel_file}"
            )

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        mono = resample_mono(data, int(sr), sample_rate)
        if mono.size == 0:
            raise MediaError(f"Empty audio data in segment WAV: {wav_path}")

        start_ms = int(round(max(0.0, float(seg["start"])) * 1000.0))
        start_sample = int(round((start_ms / 1000.0) * sample_rate))

        if overlap_policy == "push" and cursor_sample > 0:
            min_allowed = cursor_sample + min_gap_samples
            if start_sample < min_allowed:
                logger.warning(
                    "Segment %s overflows previous slot by %.3fs; pushing start to avoid overlap",
                    seg["id"],
                    (min_allowed - start_sample) / float(sample_rate),
                )
                place_sample = min_allowed
            else:
                place_sample = start_sample
        else:
            place_sample = start_sample

        end_sample = place_sample + mono.size
        placements.append((place_sample, mono))
        cursor_sample = max(cursor_sample, end_sample)
        max_end_sample = max(max_end_sample, end_sample)

    if max_end_sample <= 0:
        max_end_sample = int(round(0.1 * sample_rate))

    timeline = np.zeros(max_end_sample, dtype=np.float32)
    for place_sample, mono in placements:
        end_sample = place_sample + mono.size
        if end_sample > timeline.size:
            extended = np.zeros(end_sample, dtype=np.float32)
            extended[: timeline.size] = timeline
            timeline = extended
        if overlap_policy == "mix":
            timeline[place_sample:end_sample] += mono
        else:
            timeline[place_sample:end_sample] = mono

    peak = float(np.max(np.abs(timeline))) if timeline.size > 0 else 0.0
    if peak > 1.0:
        timeline = np.tanh(timeline) if overlap_policy == "mix" else (timeline / peak)

    with atomic_temp_path(paths.thai_dub_full) as tmp_wav:
        sf.write(str(tmp_wav), timeline, sample_rate, subtype="PCM_16")

    if not is_valid_audio_file(paths.thai_dub_full):
        raise MediaError(f"Assembled timeline WAV failed validation: {paths.thai_dub_full}")

    return paths.thai_dub_full


def mux_video(
    source_video: Path,
    dub_audio: Path,
    output_video: Path,
    *,
    mix_mode: MixMode = "duck",
    original_volume: float = 0.18,
) -> Path:
    """Mux `source_video` and `dub_audio` into `output_video` atomically via ffmpeg.

    - duck mode (when source has audio stream): uses exact Spec §5 Stage 4 filter_complex:
      `[0:a]volume={original_volume}[orig];[1:a]volume=1.0[th];[orig][th]amix=inputs=2:duration=first:dropout_transition=2[aout]`
      Note: ffmpeg's `amix` defaults to `normalize=1` per spec command.
    - replace mode (or graceful fallback when source has no audio stream):
      `-map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k`
    """
    source_video = Path(source_video)
    dub_audio = Path(dub_audio)
    output_video = Path(output_video)

    if mix_mode not in ("duck", "replace"):
        raise ValidationError(f"Invalid mix_mode: {mix_mode!r}. Expected 'duck' or 'replace'.")
    if not is_valid_video_file(source_video):
        raise MediaError(f"Invalid source video for muxing: {source_video}")
    if not is_valid_audio_file(dub_audio):
        raise MediaError(f"Invalid dub audio for muxing: {dub_audio}")

    use_duck = mix_mode == "duck" and has_audio_stream(source_video)
    if mix_mode == "duck" and not use_duck:
        logger.warning(
            "Source video %s has no audio stream; falling back from duck to replace mode",
            source_video,
        )

    with atomic_temp_path(output_video) as tmp_mp4:
        if use_duck:
            filter_complex = (
                f"[0:a]volume={original_volume}[orig];"
                f"[1:a]volume=1.0[th];"
                f"[orig][th]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(source_video),
                "-i",
                str(dub_audio),
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(tmp_mp4),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(source_video),
                "-i",
                str(dub_audio),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(tmp_mp4),
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
                f"ffmpeg muxing failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )

    if not is_valid_video_file(output_video, require_audio=True):
        raise MediaError(f"Muxed output video failed validation: {output_video}")

    return output_video


def run_mux(
    paths: WorkspacePaths,
    *,
    mix_mode: MixMode = "duck",
    original_volume: float = 0.18,
    overlap_policy: OverlapPolicy = "push",
) -> Path:
    """Execute Stage 4: assemble `thai_dub_full.wav` and mux into `output_th.mp4`."""
    if not is_valid_json_file(paths.segments_th, expected_type=list, allow_empty=False):
        raise StageError("mux", f"Missing or invalid segments_th.json: {paths.segments_th}")
    if not is_valid_video_file(paths.source_video):
        raise StageError("mux", f"Missing or invalid source.mp4: {paths.source_video}")

    try:
        segments_th = read_json(paths.segments_th)
        src_duration = probe_media_duration(paths.source_video)
        assemble_timeline(
            paths,
            segments_th,
            total_duration=src_duration,
            overlap_policy=overlap_policy,
        )
        return mux_video(
            paths.source_video,
            paths.thai_dub_full,
            paths.output_video,
            mix_mode=mix_mode,
            original_volume=original_volume,
        )
    except StageError:
        raise
    except Exception as exc:
        raise StageError("mux", str(exc), cause=exc) from exc
