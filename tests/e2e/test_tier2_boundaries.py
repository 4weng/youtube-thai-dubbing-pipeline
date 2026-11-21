"""Tier 2 E2E Tests: Boundaries, edge cases, corrupt files, and negative error handling."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from pipeline import run_pipeline
from src.mux import run_mux
from src.state import (
    PipelineConfig,
    PipelineError,
    ValidationError,
    atomic_write_json,
    extract_video_id,
    get_workspace_paths,
    init_or_load_manifest,
    is_valid_audio_file,
    is_valid_json_file,
    is_valid_video_file,
    should_skip_stage,
    verify_stage_outputs,
)
from src.tts_align import build_atempo_filter, run_tts_align

from .conftest import (
    PROJECT_ROOT,
    FakeTranslator,
    FakeTTSProvider,
    create_tone_wav,
    get_python_executable,
    make_local_downloader,
)


# ---------------------------------------------------------------------------
# Invalid URLs & path traversal rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "   ",
        "not-a-url",
        "https://example.com/watch?v=5orlc92154E",
        "../../../etc/passwd",
        "https://youtu.be/../../evil",
        "https://www.youtube.com/watch?v=bad;rm -rf /",
        "https://www.youtube.com/watch?v=short",
    ],
)
def test_invalid_youtube_urls_and_path_traversal_rejected(bad_url: str) -> None:
    """Invalid URLs and path-traversal payloads must be rejected cleanly."""
    with pytest.raises((ValidationError, PipelineError, ValueError)):
        extract_video_id(bad_url)


# ---------------------------------------------------------------------------
# 0-byte and corrupt media/JSON rejected by Verify Before Skip
# ---------------------------------------------------------------------------
def test_zero_byte_and_corrupt_audio_video_rejected_by_verify_before_skip(
    tmp_path: Path,
    work_dir: Path,
) -> None:
    """0-byte or garbage WAV/MP4/JSON files must fail integrity verification."""
    zero_wav = tmp_path / "zero.wav"
    zero_wav.write_bytes(b"")
    garbage_wav = tmp_path / "garbage.wav"
    garbage_wav.write_bytes(b"not a real wav file at all 1234567890")
    zero_mp4 = tmp_path / "zero.mp4"
    zero_mp4.write_bytes(b"")
    garbage_mp4 = tmp_path / "garbage.mp4"
    garbage_mp4.write_bytes(b"corrupt mp4 bytes")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{broken json", encoding="utf-8")

    assert is_valid_audio_file(zero_wav) is False
    assert is_valid_audio_file(garbage_wav) is False
    assert is_valid_audio_file(tmp_path / "missing.wav") is False
    assert is_valid_video_file(zero_mp4) is False
    assert is_valid_video_file(garbage_mp4) is False
    assert is_valid_json_file(bad_json) is False

    paths = get_workspace_paths(work_dir, "corruptChk1")
    paths.ensure_dirs()
    paths.output_video.write_bytes(b"")
    assert verify_stage_outputs(paths, "mux") is False


def test_corrupt_output_triggers_regeneration_even_when_manifest_done(
    work_dir: Path,
    synthetic_video_12s: Path,
    sample_vtt_file: Path,
) -> None:
    """Verify Before Skip: corrupting output_th.mp4 forces Stage 4 re-run even if manifest=done."""
    url = "https://youtu.be/regenMux001"
    dl = make_local_downloader(synthetic_video_12s, subtitle_src=sample_vtt_file)
    tr = FakeTranslator()
    tts = FakeTTSProvider(duration_fn=lambda *_: 1.8)

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="duck"),
        work_dir=work_dir,
        downloader=dl,
        translator=tr,
        tts_provider=tts,
    )
    assert is_valid_video_file(out, require_audio=True)

    paths = get_workspace_paths(work_dir, "regenMux001")
    # Corrupt output_th.mp4 to 0 bytes while manifest still says mux=done
    paths.output_video.write_bytes(b"")
    manifest = init_or_load_manifest(paths, "regenMux001", url, PipelineConfig(mix_mode="duck"))
    assert should_skip_stage(paths, manifest, "mux") is False

    # Re-running pipeline must regenerate a valid output_th.mp4
    out2 = run_pipeline(
        url,
        PipelineConfig(mix_mode="duck"),
        work_dir=work_dir,
        downloader=dl,
        translator=tr,
        tts_provider=tts,
    )
    assert is_valid_video_file(out2, require_audio=True)


# ---------------------------------------------------------------------------
# atempo boundary conditions: <= slot_dur, >> slot_dur, and > 2.0 chaining
# ---------------------------------------------------------------------------
def test_actual_dur_leq_slot_dur_keeps_tempo_one(work_dir: Path) -> None:
    """When actual_dur <= slot_dur, Stage 3 keeps tempo_used == 1.0."""
    paths = get_workspace_paths(work_dir, "tempoOne001")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [{"id": 1, "start": 0.0, "end": 4.0, "text_en": "Short", "text_th": "สั้น"}],
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 1.5)
    segs = run_tts_align(paths, tts, max_atempo=1.25)

    assert segs[0]["tts"]["tempo_used"] == pytest.approx(1.0, abs=0.01)
    assert segs[0]["tts"]["duration"] == pytest.approx(1.5, abs=0.15)


def test_actual_dur_far_exceeds_slot_dur_clamps_to_max_atempo(work_dir: Path) -> None:
    """When actual_dur >> slot_dur, tempo is clamped to max_atempo (1.25)."""
    paths = get_workspace_paths(work_dir, "tempoClamp1")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [{"id": 1, "start": 0.0, "end": 2.0, "text_en": "Very long", "text_th": "ยาวมากเลยทีเดียว"}],
    )
    # 5.0s audio in 2.0s slot -> needed_tempo=2.5, clamped to 1.25 -> ~4.0s output
    tts = FakeTTSProvider(duration_fn=lambda *_: 5.0)
    segs = run_tts_align(paths, tts, max_atempo=1.25)

    assert segs[0]["tts"]["tempo_used"] == pytest.approx(1.25, abs=0.02)
    assert segs[0]["tts"]["duration"] == pytest.approx(5.0 / 1.25, abs=0.25)


def test_custom_max_atempo_above_2_chains_atempo_filters(work_dir: Path) -> None:
    """Custom max_atempo > 2.0 chains multiple ffmpeg atempo filters in [0.5, 2.0]."""
    filter_expr = build_atempo_filter(2.5)
    factors = [float(m) for m in re.findall(r"atempo=([0-9.]+)", filter_expr)]
    assert len(factors) >= 2, f"Expected chained atempo filters for 2.5, got: {filter_expr}"
    assert all(0.5 <= f <= 2.0 for f in factors)
    assert math.prod(factors) == pytest.approx(2.5, rel=0.02)

    # Also verify real ffmpeg execution through Stage 3 with max_atempo=2.5
    paths = get_workspace_paths(work_dir, "tempoChain1")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [{"id": 1, "start": 0.0, "end": 2.0, "text_en": "Long", "text_th": "ยาวมาก"}],
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 5.0)
    segs = run_tts_align(paths, tts, max_atempo=2.5)
    assert segs[0]["tts"]["tempo_used"] == pytest.approx(2.5, abs=0.05)
    assert is_valid_audio_file(paths.segment_wav_path(1))
    assert sf.info(str(paths.segment_wav_path(1))).duration == pytest.approx(2.0, abs=0.25)


# ---------------------------------------------------------------------------
# All-silence TTS guard & slot_dur <= 0 guard
# ---------------------------------------------------------------------------
def test_all_silence_tts_output_guarded(work_dir: Path) -> None:
    """All-silence TTS output must not produce a 0-byte or 0-duration corrupt WAV."""
    paths = get_workspace_paths(work_dir, "allSilence1")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [{"id": 1, "start": 0.0, "end": 3.0, "text_en": "Quiet", "text_th": "เงียบ"}],
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 1.5, all_silence=True)

    try:
        segs = run_tts_align(paths, tts, max_atempo=1.25)
    except PipelineError:
        # Raising a structured PipelineError is an acceptable guard
        return

    # If it succeeds, the saved WAV must still be valid with non-zero duration
    wav = paths.segment_wav_path(1)
    assert is_valid_audio_file(wav)
    assert segs[0]["tts"]["duration"] > 0.0


def test_zero_or_negative_slot_duration_guarded(work_dir: Path) -> None:
    """Segments with slot_dur <= 0 must not crash with ZeroDivisionError."""
    paths = get_workspace_paths(work_dir, "zeroSlot001")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 2.0, "end": 2.0, "text_en": "Zero", "text_th": "ศูนย์"},
        ],
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 1.0)

    try:
        segs = run_tts_align(paths, tts, max_atempo=1.25)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError raised on slot_dur == 0")
    except (PipelineError, ValueError):
        return

    assert is_valid_audio_file(paths.segment_wav_path(1))
    assert segs[0]["tts"]["tempo_used"] >= 1.0


# ---------------------------------------------------------------------------
# Silent / audio-less source.mp4 in duck mode
# ---------------------------------------------------------------------------
def test_silent_audioless_source_video_in_duck_mode(
    work_dir: Path,
    synthetic_video_silent: Path,
) -> None:
    """Feature #25 fallback: source.mp4 with no audio stream still muxes cleanly in duck mode."""
    paths = get_workspace_paths(work_dir, "noAudioSrc1")
    paths.ensure_dirs()
    paths.source_video.write_bytes(synthetic_video_silent.read_bytes())
    create_tone_wav(paths.segment_wav_path(1), duration=2.0, freq=880.0, amplitude=0.5)
    atomic_write_json(
        paths.segments_th,
        [
            {
                "id": 1,
                "start": 1.0,
                "end": 3.5,
                "text_en": "No original audio",
                "text_th": "ไม่มีเสียงต้นฉบับ",
                "tts": {
                    "status": "done",
                    "file": "tts_segments/seg_0001.wav",
                    "duration": 2.0,
                    "tempo_used": 1.0,
                },
            }
        ],
    )

    out = run_mux(paths, mix_mode="duck", original_volume=0.18)
    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)


# ---------------------------------------------------------------------------
# High segment ID formatting (5+ digits)
# ---------------------------------------------------------------------------
def test_high_segment_id_formatting_10000(work_dir: Path) -> None:
    """Segment ID 10000 formats to seg_10000.wav without truncation."""
    paths = get_workspace_paths(work_dir, "bigSegId001")
    assert paths.segment_wav_relpath(1) == "tts_segments/seg_0001.wav"
    assert paths.segment_wav_relpath(9999) == "tts_segments/seg_9999.wav"
    assert paths.segment_wav_relpath(10000) == "tts_segments/seg_10000.wav"
    assert paths.segment_wav_path(10000) == paths.tts_segments_dir / "seg_10000.wav"


# ---------------------------------------------------------------------------
# Invalid CLI flags & invalid config bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cli_args",
    [
        [],
        ["--url", "https://youtu.be/5orlc92154E", "--mix-mode", "invalid_mode"],
        ["--url", "https://youtu.be/5orlc92154E", "--force-stage", "nonexistent_stage"],
    ],
)
def test_cli_invalid_flags_and_missing_url_exit_nonzero(cli_args: list[str]) -> None:
    """CLI must reject missing --url or invalid enum flags with non-zero exit code."""
    proc = subprocess.run(
        [get_python_executable(), str(PROJECT_ROOT / "pipeline.py"), *cli_args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mix_mode": "surround"},
        {"original_volume": -0.5},
        {"max_atempo": 0.0},
    ],
)
def test_invalid_config_bounds_rejected(kwargs: dict[str, object]) -> None:
    """PipelineConfig must reject invalid mix_mode, negative volume, or non-positive max_atempo."""
    with pytest.raises((ValidationError, PipelineError, ValueError)):
        PipelineConfig(**kwargs)  # type: ignore[arg-type]
