"""Tier 3 E2E Tests: Pairwise cross-feature and cross-stage interactions."""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from pipeline import run_pipeline
from src.ingest import RawCue
from src.mux import assemble_timeline
from src.state import (
    STAGE_ORDER,
    PipelineConfig,
    StageError,
    atomic_write_json,
    get_workspace_paths,
    is_valid_audio_file,
    is_valid_video_file,
    read_json,
)
from src.tts_align import run_tts_align

from .conftest import (
    FakeASRProvider,
    FakeTranslator,
    FakeTTSProvider,
    make_local_downloader,
    measure_window_rms,
    write_vtt,
)


def _three_segment_vtt(tmp_path: Path) -> Path:
    return write_vtt(
        tmp_path / "three_segs.vtt",
        [
            (1.2, 4.4, "First segment for checkpoint testing."),
            (4.8, 8.0, "Second segment for checkpoint testing."),
            (8.4, 11.4, "Third segment for checkpoint testing."),
        ],
    )


# ---------------------------------------------------------------------------
# Mid-Stage-3 crash + incremental save + resume skips completed segments
# ---------------------------------------------------------------------------
def test_mid_stage3_crash_incremental_save_and_resume_skips_completed_segments(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """Crash on segment 2 preserves segment 1 checkpoint; resume skips seg 1 and Stages 1-2."""
    url = "https://youtu.be/crashResm01"
    vtt = _three_segment_vtt(tmp_path)
    cfg = PipelineConfig(mix_mode="duck")

    dl_calls_1: list[str] = []
    tr_1 = FakeTranslator()
    tts_crash = FakeTTSProvider(duration_fn=lambda *_: 1.6, fail_on_call=2)

    with pytest.raises(StageError):
        run_pipeline(
            url,
            cfg,
            work_dir=work_dir,
            downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt, call_log=dl_calls_1),
            translator=tr_1,
            tts_provider=tts_crash,
        )

    paths = get_workspace_paths(work_dir, "crashResm01")
    manifest_after_crash = read_json(paths.manifest)
    assert manifest_after_crash["stages"]["ingest"]["status"] == "done"
    assert manifest_after_crash["stages"]["translate"]["status"] == "done"
    assert manifest_after_crash["stages"]["tts_align"]["status"] != "done"
    assert manifest_after_crash["stages"]["mux"]["status"] == "pending"
    assert manifest_after_crash["last_error"] is not None

    segs_after_crash = read_json(paths.segments_th)
    assert segs_after_crash[0]["tts"]["status"] == "done"
    assert is_valid_audio_file(paths.segment_wav_path(1))

    # Resume with healthy providers
    dl_calls_2: list[str] = []
    tr_2 = FakeTranslator()
    tts_healthy = FakeTTSProvider(duration_fn=lambda *_: 1.6)

    out = run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt, call_log=dl_calls_2),
        translator=tr_2,
        tts_provider=tts_healthy,
    )

    assert dl_calls_2 == [], "Stage 1 should be skipped on resume"
    assert tr_2.calls == [], "Stage 2 should be skipped on resume"
    assert len(tts_healthy.calls) == 2, f"Expected only segs 2 & 3 synthesized, got {len(tts_healthy.calls)}"
    assert is_valid_video_file(out, require_audio=True)

    manifest_final = read_json(paths.manifest)
    for stage in STAGE_ORDER:
        assert manifest_final["stages"][stage]["status"] == "done"
    assert manifest_final["last_error"] is None


# ---------------------------------------------------------------------------
# Corrupting a single TTS segment regenerates only that segment on resume
# ---------------------------------------------------------------------------
def test_corrupt_single_tts_segment_regenerates_only_that_segment(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """Corrupting seg_0002.wav triggers regeneration of ONLY segment 2 and remux."""
    url = "https://youtu.be/corruptSeg2"
    vtt = _three_segment_vtt(tmp_path)
    cfg = PipelineConfig(mix_mode="duck")

    run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
        translator=FakeTranslator(),
        tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.6),
    )

    paths = get_workspace_paths(work_dir, "corruptSeg2")
    seg1_bytes_before = paths.segment_wav_path(1).read_bytes()
    seg3_bytes_before = paths.segment_wav_path(3).read_bytes()

    # Corrupt segment 2 to 0 bytes
    paths.segment_wav_path(2).write_bytes(b"")

    dl_calls_2: list[str] = []
    tr_2 = FakeTranslator()
    tts_2 = FakeTTSProvider(duration_fn=lambda *_: 1.6)

    out = run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt, call_log=dl_calls_2),
        translator=tr_2,
        tts_provider=tts_2,
    )

    assert dl_calls_2 == []
    assert tr_2.calls == []
    assert len(tts_2.calls) == 1, f"Expected only seg 2 regenerated, got {len(tts_2.calls)}"
    assert is_valid_audio_file(paths.segment_wav_path(2))
    assert paths.segment_wav_path(1).read_bytes() == seg1_bytes_before
    assert paths.segment_wav_path(3).read_bytes() == seg3_bytes_before
    assert is_valid_video_file(out, require_audio=True)


# ---------------------------------------------------------------------------
# Manual edit to segments_th.json + --force-stage tts_align
# ---------------------------------------------------------------------------
def test_manual_edit_segments_th_with_force_stage_tts_align(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """Manual text_th edit + force_stage='tts_align' preserves edit, re-runs TTS & cascades mux."""
    url = "https://youtu.be/manualEdit1"
    vtt = _three_segment_vtt(tmp_path)
    cfg = PipelineConfig(mix_mode="duck")

    run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
        translator=FakeTranslator(),
        tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.5),
    )

    paths = get_workspace_paths(work_dir, "manualEdit1")
    segs = read_json(paths.segments_th)
    edited_text = "บทแปลภาษาไทยที่แก้ไขด้วยมือ"
    segs[0]["text_th"] = edited_text
    atomic_write_json(paths.segments_th, segs)

    dl_calls_2: list[str] = []
    tr_2 = FakeTranslator()
    tts_2 = FakeTTSProvider(duration_fn=lambda *_: 1.5)

    out = run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        force_stage="tts_align",
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt, call_log=dl_calls_2),
        translator=tr_2,
        tts_provider=tts_2,
    )

    assert dl_calls_2 == [], "Stage 1 must not re-run for force_stage='tts_align'"
    assert tr_2.calls == [], "Stage 2 must not re-run for force_stage='tts_align'"
    assert len(tts_2.calls) == 3
    assert tts_2.calls[0]["text_th"] == edited_text

    persisted_segs = read_json(paths.segments_th)
    assert persisted_segs[0]["text_th"] == edited_text
    assert is_valid_video_file(out, require_audio=True)


# ---------------------------------------------------------------------------
# Changing --mix-mode duck -> replace triggers remux without re-running Stages 1-3
# ---------------------------------------------------------------------------
def test_mix_mode_change_duck_to_replace_remuxes_without_rerunning_stages_1_to_3(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """Switching mix_mode duck -> replace skips Stages 1-3 and remuxes output_th.mp4."""
    url = "https://youtu.be/mixSwitch01"
    vtt = _three_segment_vtt(tmp_path)

    run_pipeline(
        url,
        PipelineConfig(mix_mode="duck", original_volume=0.18),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
        translator=FakeTranslator(),
        tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.5),
    )

    paths = get_workspace_paths(work_dir, "mixSwitch01")
    duck_gap_rms = measure_window_rms(paths.output_video, 0.1, 0.9, tmp_dir=tmp_path)
    assert duck_gap_rms > 0.01, "In duck mode, original 440Hz tone should remain audible in gap"

    dl_calls_2: list[str] = []
    tr_2 = FakeTranslator()
    tts_2 = FakeTTSProvider(duration_fn=lambda *_: 1.5)

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="replace"),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt, call_log=dl_calls_2),
        translator=tr_2,
        tts_provider=tts_2,
    )

    assert dl_calls_2 == [], "Stage 1 should be skipped on mix_mode change"
    assert tr_2.calls == [], "Stage 2 should be skipped on mix_mode change"
    assert tts_2.calls == [], "Stage 3 should be skipped on mix_mode change"
    assert is_valid_video_file(out, require_audio=True)

    manifest = read_json(paths.manifest)
    assert manifest["config"]["mix_mode"] == "replace"

    replace_gap_rms = measure_window_rms(out, 0.1, 0.9, tmp_dir=tmp_path)
    assert replace_gap_rms < 0.005, f"Expected silenced gap in replace mode, got RMS={replace_gap_rms:.4f}"


# ---------------------------------------------------------------------------
# Capped atempo overflow + Stage 4 timeline push prevents overlap
# ---------------------------------------------------------------------------
def test_capped_atempo_overflow_plus_timeline_push_prevents_overlap(
    work_dir: Path,
    tmp_path: Path,
) -> None:
    """When seg 1 overflows its slot after max_atempo=1.25, Stage 4 push policy shifts seg 2."""
    paths = get_workspace_paths(work_dir, "overflowPsh")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 1.0, "end": 3.0, "text_en": "Long first", "text_th": "ประโยคแรกยาวมาก"},
            {"id": 2, "start": 3.2, "end": 5.2, "text_en": "Second", "text_th": "ประโยคที่สอง"},
        ],
    )
    # Seg 1: 4.5s raw -> clamped at 1.25 -> ~3.6s (ends ~4.6s, past seg 2's start=3.2s)
    # Seg 2: 1.5s raw -> tempo 1.0 -> 1.5s
    tts = FakeTTSProvider(duration_fn=lambda _txt, idx: 4.5 if idx == 1 else 1.5)
    segs = run_tts_align(paths, tts, max_atempo=1.25)

    seg1_dur = segs[0]["tts"]["duration"]
    seg2_dur = segs[1]["tts"]["duration"]
    assert seg1_dur > 3.0, "Seg 1 should overflow its 2.0s slot after clamping to 1.25x"

    full_wav = assemble_timeline(
        paths,
        segs,
        total_duration=None,
        overlap_policy="push",
        min_gap_sec=0.05,
    )
    assert is_valid_audio_file(full_wav)

    expected_min_end = 1.0 + seg1_dur + 0.05 + seg2_dur
    actual_dur = sf.info(str(full_wav)).duration
    assert actual_dur >= expected_min_end - 0.15, (
        f"Expected pushed timeline >= {expected_min_end:.2f}s, got {actual_dur:.2f}s"
    )

    # Audio should be active where seg 2 was pushed (after 1.0 + seg1_dur + 0.05)
    pushed_seg2_start = 1.0 + seg1_dur + 0.1
    pushed_rms = measure_window_rms(
        full_wav,
        pushed_seg2_start,
        pushed_seg2_start + 0.8,
        tmp_dir=tmp_path,
    )
    assert pushed_rms > 0.1


# ---------------------------------------------------------------------------
# Missing subtitles -> ASR fallback -> translate -> TTS -> mux
# ---------------------------------------------------------------------------
def test_missing_subtitles_asr_fallback_through_full_pipeline(
    work_dir: Path,
    synthetic_video_12s: Path,
) -> None:
    """Full 4-stage pipeline completes when subtitles are absent and ASR fallback supplies cues."""
    url = "https://youtu.be/asrFullPipe"
    asr = FakeASRProvider(
        [
            RawCue(1.5, 4.8, "Transcribed sentence one from audio."),
            RawCue(5.5, 8.8, "Transcribed sentence two from audio."),
        ]
    )

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="duck"),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=None),
        asr_provider=asr,
        translator=FakeTranslator(),
        tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.8),
    )

    paths = get_workspace_paths(work_dir, "asrFullPipe")
    assert len(asr.calls) == 1
    assert is_valid_video_file(out, require_audio=True)

    manifest = read_json(paths.manifest)
    for stage in STAGE_ORDER:
        assert manifest["stages"][stage]["status"] == "done"
    assert manifest["last_error"] is None
