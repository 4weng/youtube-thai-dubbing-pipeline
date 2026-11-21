"""Unit tests for Stage 4 src/mux.py and pipeline.py orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline import build_arg_parser, run_pipeline
from src.ingest import DownloadResult
from src.mux import assemble_timeline, mux_video, resample_mono
from src.state import (
    PipelineConfig,
    atomic_write_json,
    get_workspace_paths,
    is_valid_audio_file,
    is_valid_video_file,
    probe_media_info,
    read_json,
)
from tests.conftest import LocalToneTTSProvider, make_test_video, make_test_wav


def _extract_audio_array(video_path: Path, tmp_wav: Path) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "1",
        str(tmp_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    data, sr = sf.read(str(tmp_wav), dtype="float32")
    return data, int(sr)


def test_resample_mono_stereo_and_rate_conversion() -> None:
    t = np.linspace(0.0, 1.0, 22050, endpoint=False, dtype=np.float32)
    stereo = np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)])
    out = resample_mono(stereo, orig_sr=22050, target_sr=44100)
    assert out.ndim == 1
    assert out.size == 44100


def test_assemble_timeline_gaps_and_overflow_policies(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000030")
    paths.ensure_dirs()

    # Seg 1 starts at 1.0s and is 2.0s long (ends at 3.0s)
    # Seg 2 nominally starts at 2.5s (0.5s overlap with Seg 1!)
    make_test_wav(paths.segment_wav_path(1), duration=2.0, freq=300.0, sr=24000, amplitude=0.6)
    make_test_wav(paths.segment_wav_path(2), duration=1.0, freq=500.0, sr=24000, amplitude=0.6)

    segs = [
        {
            "id": 1,
            "start": 1.0,
            "end": 2.4,
            "text_en": "A",
            "text_th": "ก",
            "tts": {"status": "done", "file": "tts_segments/seg_0001.wav", "duration": 2.0, "tempo_used": 1.25},
        },
        {
            "id": 2,
            "start": 2.5,
            "end": 3.5,
            "text_en": "B",
            "text_th": "ข",
            "tts": {"status": "done", "file": "tts_segments/seg_0002.wav", "duration": 1.0, "tempo_used": 1.0},
        },
    ]

    # Push policy: leading 0.0-0.9s must be silent; seg 2 pushed past 3.0s -> total >= 4.0s
    dub_push = assemble_timeline(paths, segs, total_duration=3.8, sample_rate=44100, overlap_policy="push", min_gap_sec=0.05)
    assert is_valid_audio_file(dub_push)
    data_push, sr = sf.read(str(dub_push), dtype="float32")
    lead_rms = float(np.sqrt(np.mean(data_push[: int(0.8 * sr)] ** 2)))
    seg1_rms = float(np.sqrt(np.mean(data_push[int(1.2 * sr) : int(1.8 * sr)] ** 2)))
    pushed_seg2_rms = float(np.sqrt(np.mean(data_push[int(3.2 * sr) : int(3.8 * sr)] ** 2)))
    assert lead_rms < 1e-4
    assert seg1_rms > 0.1
    assert pushed_seg2_rms > 0.1
    assert data_push.size / sr >= 4.0

    # Mix policy: stays at 3.8s total and clips safely within [-1, 1]
    dub_mix = assemble_timeline(paths, segs, total_duration=3.8, sample_rate=44100, overlap_policy="mix")
    data_mix, _ = sf.read(str(dub_mix), dtype="float32")
    assert float(np.max(np.abs(data_mix))) <= 1.0


def test_mux_video_duck_vs_replace_and_silent_source_fallback(tmp_path: Path) -> None:
    src_video = make_test_video(tmp_path / "source.mp4", duration=5.0, freq=440.0, with_audio=True)

    # Create dub audio that is silent from 0.0-2.5s and has a tone from 2.5-4.5s
    sr = 44100
    dub = np.zeros(int(5.0 * sr), dtype=np.float32)
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    dub[int(2.5 * sr) : int(4.5 * sr)] = (0.5 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    dub_path = tmp_path / "dub.wav"
    sf.write(str(dub_path), dub, sr, subtype="PCM_16")

    # Duck mode: 0.5-2.0s window must still contain attenuated original 440Hz tone
    out_duck = mux_video(src_video, dub_path, tmp_path / "out_duck.mp4", mix_mode="duck", original_volume=0.18)
    assert is_valid_video_file(out_duck, require_audio=True)
    info_duck = probe_media_info(out_duck)
    assert info_duck is not None
    codecs = {s.get("codec_type"): s.get("codec_name") for s in info_duck["streams"]}
    assert codecs.get("video") == "h264"
    assert codecs.get("audio") == "aac"

    duck_samples, dsr = _extract_audio_array(out_duck, tmp_path / "duck_check.wav")
    duck_gap_rms = float(np.sqrt(np.mean(duck_samples[int(0.5 * dsr) : int(2.0 * dsr)] ** 2)))
    # ffmpeg sine defaults to ~0.125 amplitude; 0.18 volume + amix 2-input normalization yields ~0.0079 RMS
    assert duck_gap_rms > 0.004

    # Replace mode: 0.5-2.0s window must be nearly silent (original audio removed)
    out_rep = mux_video(src_video, dub_path, tmp_path / "out_replace.mp4", mix_mode="replace")
    assert is_valid_video_file(out_rep, require_audio=True)
    rep_samples, rsr = _extract_audio_array(out_rep, tmp_path / "rep_check.wav")
    rep_gap_rms = float(np.sqrt(np.mean(rep_samples[int(0.5 * rsr) : int(2.0 * rsr)] ** 2)))
    assert rep_gap_rms < 1e-3
    assert duck_gap_rms > rep_gap_rms * 10.0

    # Graceful fallback when source.mp4 has no audio stream in duck mode
    silent_src = make_test_video(tmp_path / "silent_source.mp4", duration=5.0, with_audio=False)
    out_fallback = mux_video(silent_src, dub_path, tmp_path / "out_fallback.mp4", mix_mode="duck")
    assert is_valid_video_file(out_fallback, require_audio=True)


def test_pipeline_orchestration_resume_and_force_stage(tmp_path: Path) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(["--url", "https://youtu.be/5orlc92154E", "--mix-mode", "replace", "--force-stage", "mux"])
    assert parsed.mix_mode == "replace"
    assert parsed.force_stage == "mux"

    url = "https://youtu.be/5orlc92154E"
    work_dir = tmp_path / "work"
    dl_calls = 0
    tr_calls = 0

    def fake_downloader(u: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        nonlocal dl_calls
        dl_calls += 1
        make_test_video(out_vid, duration=6.0)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub = sub_dir / "en.vtt"
        sub.write_text(
            "WEBVTT\n\n00:00:00.500 --> 00:00:03.500\nHello world from YouTube.\n\n"
            "00:00:03.600 --> 00:00:05.800\nThis is a local test.\n",
            encoding="utf-8",
        )
        return DownloadResult(video_path=out_vid, subtitle_path=sub, subtitle_is_auto=False)

    class FakeTranslator:
        def translate_segment(self, text_en, prev_context, next_context, slot_duration, max_chars) -> str:
            nonlocal tr_calls
            tr_calls += 1
            return "สวัสดีจากยูทูบ" if "YouTube" in text_en else "นี่คือการทดสอบ"

    tts = LocalToneTTSProvider()
    cfg = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25)

    out1 = run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts,
    )
    assert is_valid_video_file(out1, require_audio=True)
    assert dl_calls == 1 and tr_calls == 2 and len(tts.calls) == 2

    # Second run without --force-stage must skip all 4 stages
    out2 = run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts,
    )
    assert out2 == out1
    assert dl_calls == 1 and tr_calls == 2 and len(tts.calls) == 2

    # Manual edit to segments_th.json + --force-stage tts_align -> regenerates TTS + remuxes, skips ingest & translate
    paths = get_workspace_paths(work_dir, "5orlc92154E")
    segs = read_json(paths.segments_th)
    segs[0]["text_th"] = "แก้ไขบทแปลด้วยมือแล้ว"
    atomic_write_json(paths.segments_th, segs)

    run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        force_stage="tts_align",
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts,
    )
    assert dl_calls == 1 and tr_calls == 2 and len(tts.calls) == 4
    manifest = read_json(paths.manifest)
    assert manifest["last_error"] is None
    assert all(manifest["stages"][s]["status"] == "done" for s in ("ingest", "translate", "tts_align", "mux"))


def test_pipeline_verify_before_skip_repair_cascades_to_mux(tmp_path: Path) -> None:
    url = "https://youtu.be/5orlc92154E"
    work_dir = tmp_path / "work"
    dl_calls = 0
    tr_calls = 0

    def fake_downloader(u: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        nonlocal dl_calls
        dl_calls += 1
        make_test_video(out_vid, duration=6.0)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub = sub_dir / "en.vtt"
        sub.write_text(
            "WEBVTT\n\n00:00:00.500 --> 00:00:03.500\nFirst line here.\n\n"
            "00:00:03.600 --> 00:00:05.800\nSecond line here.\n",
            encoding="utf-8",
        )
        return DownloadResult(video_path=out_vid, subtitle_path=sub, subtitle_is_auto=False)

    class FakeTranslator:
        def translate_segment(self, text_en, prev_context, next_context, slot_duration, max_chars) -> str:
            nonlocal tr_calls
            tr_calls += 1
            return "ประโยคแรก" if "First" in text_en else "ประโยคสอง"

    tts = LocalToneTTSProvider()
    cfg = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25)

    run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts,
    )
    paths = get_workspace_paths(work_dir, "5orlc92154E")
    dub_mtime_before = paths.thai_dub_full.stat().st_mtime_ns
    out_mtime_before = paths.output_video.stat().st_mtime_ns

    # Corrupt seg_0001.wav while all stages in manifest are still "done"
    paths.segment_wav_path(1).write_bytes(b"corrupted")

    # Re-run WITHOUT force_stage: tts_align repairs seg_0001.wav AND cascades to mux
    run_pipeline(
        url,
        cfg,
        work_dir=work_dir,
        force_stage=None,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts,
    )
    assert dl_calls == 1
    assert tr_calls == 2
    assert len(tts.calls) == 3  # only seg 1 regenerated
    assert is_valid_audio_file(paths.segment_wav_path(1))
    assert paths.thai_dub_full.stat().st_mtime_ns > dub_mtime_before
    assert paths.output_video.stat().st_mtime_ns > out_mtime_before


def test_pipeline_tts_config_change_regenerates_segments_without_force(tmp_path: Path) -> None:
    url = "https://youtu.be/5orlc92154E"
    work_dir = tmp_path / "work"
    dl_calls = 0
    tr_calls = 0

    def fake_downloader(u: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        nonlocal dl_calls
        dl_calls += 1
        make_test_video(out_vid, duration=6.0)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub = sub_dir / "en.vtt"
        sub.write_text(
            "WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nFirst line here.\n\n"
            "00:00:03.600 --> 00:00:04.600\nSecond line here.\n",
            encoding="utf-8",
        )
        return DownloadResult(video_path=out_vid, subtitle_path=sub, subtitle_is_auto=False)

    class FakeTranslator:
        def translate_segment(self, text_en, prev_context, next_context, slot_duration, max_chars) -> str:
            nonlocal tr_calls
            tr_calls += 1
            return "ประโยคแรก" if "First" in text_en else "ประโยคสอง"

    # ~2.25s generated audio for 1.0s slots forces atempo clamping at max_atempo
    tts1 = LocalToneTTSProvider(sec_per_char=0.25, voice_id="v1")
    cfg1 = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.25, tts_provider="pythaitts", voice_id="v1")
    run_pipeline(
        url,
        cfg1,
        work_dir=work_dir,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts1,
    )
    paths = get_workspace_paths(work_dir, "5orlc92154E")
    assert len(tts1.calls) == 2
    segs_initial = read_json(paths.segments_th)
    assert all(abs(float(s["tts"]["tempo_used"]) - 1.25) < 0.05 for s in segs_initial)

    # 1) Change max_atempo (1.25 -> 1.60) with force_stage=None
    dub_mtime = paths.thai_dub_full.stat().st_mtime_ns
    out_mtime = paths.output_video.stat().st_mtime_ns
    tts2 = LocalToneTTSProvider(sec_per_char=0.25, voice_id="v1")
    cfg2 = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.60, tts_provider="pythaitts", voice_id="v1")
    run_pipeline(
        url,
        cfg2,
        work_dir=work_dir,
        force_stage=None,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts2,
    )
    assert len(tts2.calls) == 2
    segs_after_atempo = read_json(paths.segments_th)
    assert all(abs(float(s["tts"]["tempo_used"]) - 1.60) < 0.05 for s in segs_after_atempo)
    assert paths.thai_dub_full.stat().st_mtime_ns > dub_mtime
    assert paths.output_video.stat().st_mtime_ns > out_mtime

    # 2) Change voice_id (v1 -> v2) with force_stage=None
    dub_mtime = paths.thai_dub_full.stat().st_mtime_ns
    out_mtime = paths.output_video.stat().st_mtime_ns
    tts3 = LocalToneTTSProvider(sec_per_char=0.25, voice_id="v2")
    cfg3 = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.60, tts_provider="pythaitts", voice_id="v2")
    run_pipeline(
        url,
        cfg3,
        work_dir=work_dir,
        force_stage=None,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts3,
    )
    assert len(tts3.calls) == 2
    assert paths.thai_dub_full.stat().st_mtime_ns > dub_mtime
    assert paths.output_video.stat().st_mtime_ns > out_mtime

    # 3) Change tts_provider (pythaitts -> edge_tts) with force_stage=None
    dub_mtime = paths.thai_dub_full.stat().st_mtime_ns
    out_mtime = paths.output_video.stat().st_mtime_ns
    tts4 = LocalToneTTSProvider(sec_per_char=0.25, voice_id="v2")
    cfg4 = PipelineConfig(mix_mode="duck", original_volume=0.18, max_atempo=1.60, tts_provider="edge_tts", voice_id="v2")
    run_pipeline(
        url,
        cfg4,
        work_dir=work_dir,
        force_stage=None,
        downloader=fake_downloader,
        translator=FakeTranslator(),
        tts_provider=tts4,
    )
    assert len(tts4.calls) == 2
    assert paths.thai_dub_full.stat().st_mtime_ns > dub_mtime
    assert paths.output_video.stat().st_mtime_ns > out_mtime
    # Stages 1 and 2 must remain skipped throughout
    assert dl_calls == 1
    assert tr_calls == 2


def test_assemble_timeline_handles_non_dict_tts_metadata(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000033")
    paths.ensure_dirs()
    make_test_wav(paths.segment_wav_path(1), duration=1.0)
    segments = [
        {"id": 1, "start": 0.0, "end": 1.0, "text_en": "Hi", "text_th": "สวัสดี", "tts": "not-a-dict"},
    ]
    out_wav = assemble_timeline(paths, segments, total_duration=2.0)
    assert is_valid_audio_file(out_wav)


