"""Tier 4 E2E Tests: Realistic multi-feature end-to-end application scenarios."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from pipeline import run_pipeline
from src.state import (
    STAGE_ORDER,
    PipelineConfig,
    StageError,
    get_workspace_paths,
    is_valid_audio_file,
    is_valid_video_file,
    read_json,
)
from src.tts_align import create_tts_provider

from .conftest import (
    PROJECT_ROOT,
    FakeTranslator,
    FakeTTSProvider,
    create_synthetic_video,
    ffprobe_json,
    get_python_executable,
    make_local_downloader,
    measure_window_rms,
    write_vtt,
)


def _scenario_vtt(tmp_path: Path) -> Path:
    """Multi-sentence English VTT containing numbers, acronyms, percentages, and currency."""
    return write_vtt(
        tmp_path / "scenario.vtt",
        [
            (1.8, 5.2, "Today we build a local voice pipeline."),
            (6.0, 9.6, "In 2026, AI API growth reached 50% with $100 million."),
        ],
    )


# ---------------------------------------------------------------------------
# Scenario 1: Full 4-stage pipeline in duck mode with acoustic RMS verification
# ---------------------------------------------------------------------------
def test_e2e_full_pipeline_duck_mode_with_acoustic_verification(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """End-to-end duck mode: normalized Thai, all artifacts valid, original audio attenuated."""
    url = "https://youtu.be/e2eDuckScen"
    vtt = _scenario_vtt(tmp_path)
    translator = FakeTranslator(
        {
            "Today we build a local voice pipeline.": "วันนี้เราจะสร้างไปป์ไลน์เสียงแบบโลคัล",
            "In 2026, AI API growth reached 50% with $100 million.": (
                "ในปี 2026 ระบบ AI API เติบโต 50% ด้วยงบ $100 ล้าน"
            ),
        }
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 2.0, freq=880.0, amplitude=0.7)

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
        translator=translator,
        tts_provider=tts,
    )

    paths = get_workspace_paths(work_dir, "e2eDuckScen")
    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)
    assert is_valid_audio_file(paths.audio_en)
    assert is_valid_audio_file(paths.thai_dub_full)

    # No leftover .tmp files anywhere in the video workspace
    tmp_files = [p for p in paths.root.rglob("*") if ".tmp" in p.name]
    assert tmp_files == [], f"Leftover temp files found: {tmp_files}"

    # Verify manifest completion
    manifest = read_json(paths.manifest)
    for stage in STAGE_ORDER:
        assert manifest["stages"][stage]["status"] == "done"
    assert manifest["last_error"] is None
    assert manifest["config"]["mix_mode"] == "duck"

    # Verify Stage 2 Thai TTS normalization applied to numbers/symbols
    segs_th = read_json(paths.segments_th)
    assert len(segs_th) == 2
    assert "%" not in segs_th[1]["text_th"]
    assert "$" not in segs_th[1]["text_th"]
    assert not re.search(r"\d", segs_th[1]["text_th"])

    # Acoustic verification:
    # Original video has continuous 440Hz tone. In [0.2, 1.2] (before seg 1 at 1.8s),
    # duck mode should attenuate original audio (~0.18x) rather than silencing it.
    orig_gap_rms = measure_window_rms(paths.source_video, 0.2, 1.2, tmp_dir=tmp_path)
    duck_gap_rms = measure_window_rms(out, 0.2, 1.2, tmp_dir=tmp_path)
    dub_window_rms = measure_window_rms(out, 2.0, 3.5, tmp_dir=tmp_path)

    assert orig_gap_rms > 0.2
    assert 0.01 < duck_gap_rms < orig_gap_rms * 0.5, (
        f"Expected ducked gap RMS in (0.01, {orig_gap_rms * 0.5:.3f}), got {duck_gap_rms:.4f}"
    )
    assert dub_window_rms > duck_gap_rms * 1.5, (
        f"Expected dubbed speech RMS ({dub_window_rms:.4f}) > ducked gap ({duck_gap_rms:.4f})"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Full 4-stage pipeline in replace mode silences original in gaps
# ---------------------------------------------------------------------------
def test_e2e_full_pipeline_replace_mode_silences_original_in_gaps(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """End-to-end replace mode: original audio is completely silenced in non-dubbed gaps."""
    url = "https://youtu.be/e2eReplScen"
    vtt = _scenario_vtt(tmp_path)
    tts = FakeTTSProvider(duration_fn=lambda *_: 2.0, freq=880.0, amplitude=0.7)

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="replace"),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
        translator=FakeTranslator(),
        tts_provider=tts,
    )

    assert is_valid_video_file(out, require_audio=True)

    gap_rms = measure_window_rms(out, 0.2, 1.2, tmp_dir=tmp_path)
    dub_rms = measure_window_rms(out, 2.0, 3.5, tmp_dir=tmp_path)

    assert gap_rms < 0.005, f"Expected near-zero RMS in replace-mode gap, got {gap_rms:.4f}"
    assert dub_rms > 0.05, f"Expected audible Thai dub in speech window, got {dub_rms:.4f}"


# ---------------------------------------------------------------------------
# Scenario 3: Full 4-stage pipeline with REAL offline PyThaiTTSProvider
# ---------------------------------------------------------------------------
def test_e2e_real_pythaitts_offline_synthesis_and_mux(
    work_dir: Path,
    tmp_path: Path,
) -> None:
    """End-to-end offline Thai speech synthesis using real PyThaiTTSProvider + ffmpeg mux."""
    url = "https://youtu.be/e2eRealTTS1"
    video_src = create_synthetic_video(tmp_path / "real_tts_10s.mp4", duration=10.0, freq=440.0)
    vtt = write_vtt(
        tmp_path / "real_tts.vtt",
        [
            (1.0, 4.8, "Hello everyone, welcome to our Thai dubbing test."),
            (5.2, 9.0, "This pipeline runs completely offline on Linux."),
        ],
    )
    translator = FakeTranslator(
        {
            "Hello everyone, welcome to our Thai dubbing test.": (
                "สวัสดีครับ ยินดีต้อนรับสู่การทดสอบระบบพากย์เสียงภาษาไทย"
            ),
            "This pipeline runs completely offline on Linux.": (
                "ระบบนี้ทำงานแบบออฟไลน์ได้อย่างรวดเร็ว"
            ),
        }
    )
    real_tts = create_tts_provider("pythaitts")

    out = run_pipeline(
        url,
        PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25, tts_provider="pythaitts"),
        work_dir=work_dir,
        downloader=make_local_downloader(video_src, subtitle_src=vtt),
        translator=translator,
        tts_provider=real_tts,
    )

    paths = get_workspace_paths(work_dir, "e2eRealTTS1")
    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)
    assert is_valid_audio_file(paths.thai_dub_full)

    segs_th = read_json(paths.segments_th)
    assert len(segs_th) == 2
    for seg in segs_th:
        assert seg["tts"]["status"] == "done"
        assert seg["tts"]["duration"] > 0.2
        assert 1.0 <= seg["tts"]["tempo_used"] <= 1.25
        assert is_valid_audio_file(paths.root / seg["tts"]["file"])

    info = ffprobe_json(out)
    stream_types = {s["codec_type"] for s in info["streams"]}
    assert {"video", "audio"}.issubset(stream_types)
    assert float(info["format"]["duration"]) > 8.0

    manifest = read_json(paths.manifest)
    assert manifest["config"]["tts_provider"] == "pythaitts"
    for stage in STAGE_ORDER:
        assert manifest["stages"][stage]["status"] == "done"
    assert manifest["last_error"] is None


# ---------------------------------------------------------------------------
# Scenario 4: Crash-and-recover full batch workflow
# ---------------------------------------------------------------------------
def test_e2e_crash_and_recover_full_batch_workflow(
    work_dir: Path,
    tmp_path: Path,
    synthetic_video_12s: Path,
) -> None:
    """Full batch crash populates manifest last_error; resume clears it and finishes cleanly."""
    url = "https://youtu.be/e2eBatchRcv"
    vtt = write_vtt(
        tmp_path / "batch.vtt",
        [
            (1.0, 4.0, "Batch sentence one."),
            (4.4, 7.6, "Batch sentence two."),
            (8.0, 11.2, "Batch sentence three."),
        ],
    )
    cfg = PipelineConfig(mix_mode="duck")

    # Run 1: simulated failure mid-Stage-3
    with pytest.raises(StageError):
        run_pipeline(
            url,
            cfg,
            work_dir=work_dir,
            downloader=make_local_downloader(synthetic_video_12s, subtitle_src=vtt),
            translator=FakeTranslator(),
            tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.7, fail_on_call=2),
        )

    paths = get_workspace_paths(work_dir, "e2eBatchRcv")
    m1 = read_json(paths.manifest)
    assert m1["last_error"] is not None
    assert m1["stages"]["ingest"]["status"] == "done"
    assert m1["stages"]["translate"]["status"] == "done"
    assert m1["stages"]["tts_align"]["status"] != "done"

    # Run 2: resume from checkpoint
    dl_calls_2: list[str] = []
    tr_2 = FakeTranslator()
    tts_2 = FakeTTSProvider(duration_fn=lambda *_: 1.7)

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
    assert len(tts_2.calls) == 2
    assert is_valid_video_file(out, require_audio=True)

    m2 = read_json(paths.manifest)
    assert m2["last_error"] is None
    for stage in STAGE_ORDER:
        assert m2["stages"][stage]["status"] == "done"


# ---------------------------------------------------------------------------
# Scenario 5: CLI subprocess invocation (--help and invalid URL error exit)
# ---------------------------------------------------------------------------
def test_e2e_cli_subprocess_help_and_error_handling(work_dir: Path) -> None:
    """pipeline.py CLI exposes required flags in --help and exits non-zero on invalid URL."""
    py = get_python_executable()
    cli = str(PROJECT_ROOT / "pipeline.py")

    help_proc = subprocess.run(
        [py, cli, "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_proc.returncode == 0, f"--help failed: {help_proc.stderr}"
    for flag in ("--url", "--mix-mode", "--force-stage"):
        assert flag in help_proc.stdout, f"Missing {flag} in --help output"

    err_proc = subprocess.run(
        [py, cli, "--url", "not-a-valid-youtube-url", "--work-dir", str(work_dir)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert err_proc.returncode != 0
    combined = (err_proc.stdout + "\n" + err_proc.stderr).strip()
    assert combined != "", "Expected an error diagnostic message for invalid URL"
