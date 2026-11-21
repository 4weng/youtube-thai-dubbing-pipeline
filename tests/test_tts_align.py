"""Unit tests for Stage 3 src/tts_align.py including real PyThaiTTS offline synthesis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.state import (
    StageError,
    ValidationError,
    atomic_write_json,
    get_workspace_paths,
    is_valid_audio_file,
    probe_media_duration,
    read_json,
)
from src.tts_align import (
    PyThaiTTSProvider,
    align_segment_audio,
    build_atempo_filter,
    run_tts_align,
    trim_silence,
)
from tests.conftest import LocalToneTTSProvider, make_test_wav


def test_build_atempo_filter_and_chaining() -> None:
    assert build_atempo_filter(1.0) == "atempo=1.0"
    assert build_atempo_filter(1.25) == "atempo=1.25"
    assert build_atempo_filter(2.5) == "atempo=2.0,atempo=1.25"
    assert build_atempo_filter(0.25) == "atempo=0.5,atempo=0.5"
    with pytest.raises(ValidationError):
        build_atempo_filter(0.0)
    with pytest.raises(ValidationError):
        build_atempo_filter(-1.2)


def test_trim_silence_removes_padding_and_guards_all_silence(tmp_path: Path) -> None:
    padded = make_test_wav(
        tmp_path / "padded.wav",
        duration=1.0,
        lead_silence=0.6,
        trail_silence=0.6,
        sr=24000,
    )
    trimmed = trim_silence(padded, tmp_path / "trimmed.wav", top_db=35.0, pad_ms=30.0)
    orig_dur = probe_media_duration(padded)
    trim_dur = probe_media_duration(trimmed)
    assert orig_dur is not None and trim_dur is not None
    assert orig_dur > 2.0
    assert 0.95 <= trim_dur <= 1.15

    # All-silence guard must not produce a 0-second invalid file
    silent_path = tmp_path / "all_silent.wav"
    sf.write(str(silent_path), np.zeros(12000, dtype=np.float32), 24000, subtype="PCM_16")
    guarded = trim_silence(silent_path, tmp_path / "guarded.wav")
    assert is_valid_audio_file(guarded) is True
    assert (probe_media_duration(guarded) or 0.0) >= 0.45


def test_align_segment_audio_no_speedup_and_capped_atempo(tmp_path: Path) -> None:
    # Case 1: fits within slot -> tempo == 1.0
    short_raw = make_test_wav(tmp_path / "short.wav", duration=1.2, lead_silence=0.3, trail_silence=0.3)
    dur1, tempo1 = align_segment_audio(short_raw, tmp_path / "out1.wav", slot_duration=2.5, max_atempo=1.25)
    assert tempo1 == 1.0
    assert 1.1 <= dur1 <= 1.35

    # Case 2: exceeds slot moderately -> exact needed_tempo < max_atempo
    med_raw = make_test_wav(tmp_path / "med.wav", duration=2.3, lead_silence=0.0, trail_silence=0.0)
    dur2, tempo2 = align_segment_audio(med_raw, tmp_path / "out2.wav", slot_duration=2.0, max_atempo=1.25)
    assert 1.1 <= tempo2 <= 1.2
    assert 1.85 <= dur2 <= 2.15

    # Case 3: greatly exceeds slot -> capped at max_atempo=1.25
    long_raw = make_test_wav(tmp_path / "long.wav", duration=3.5, lead_silence=0.0, trail_silence=0.0)
    dur3, tempo3 = align_segment_audio(long_raw, tmp_path / "out3.wav", slot_duration=2.0, max_atempo=1.25)
    assert tempo3 == 1.25
    assert 2.6 <= dur3 <= 3.0


def test_run_tts_align_incremental_save_and_crash_resume(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000020")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text_en": "One", "text_th": "หนึ่งสองสามสี่", "tts": {"status": "pending"}},
            {"id": 2, "start": 3.0, "end": 6.0, "text_en": "Two", "text_th": "ห้าหกเจ็ดแปด", "tts": {"status": "pending"}},
            {"id": 3, "start": 6.0, "end": 9.0, "text_en": "Three", "text_th": "เก้าสิบเอ็ดสิบสอง", "tts": {"status": "pending"}},
        ],
    )

    # First run crashes on segment 3 -> segments 1 & 2 must already be saved on disk
    failing_tts = LocalToneTTSProvider(fail_on_call=3)
    with pytest.raises(StageError):
        run_tts_align(paths, failing_tts, max_atempo=1.25)

    saved = read_json(paths.segments_th)
    assert saved[0]["tts"]["status"] == "done"
    assert saved[1]["tts"]["status"] == "done"
    assert saved[2]["tts"]["status"] == "failed"
    assert is_valid_audio_file(paths.segment_wav_path(1))
    assert is_valid_audio_file(paths.segment_wav_path(2))

    # Corrupt segment 1 WAV to test Verify-Before-Skip on resume
    paths.segment_wav_path(1).write_bytes(b"")

    # Resume with healthy provider -> must regenerate seg 1 (corrupt) and seg 3 (failed), SKIP seg 2!
    resume_tts = LocalToneTTSProvider()
    final_segs = run_tts_align(paths, resume_tts, max_atempo=1.25)
    assert len(resume_tts.calls) == 2
    assert "หนึ่งสองสามสี่" in resume_tts.calls
    assert "เก้าสิบเอ็ดสิบสอง" in resume_tts.calls
    assert "ห้าหกเจ็ดแปด" not in resume_tts.calls
    assert all(s["tts"]["status"] == "done" for s in final_segs)
    assert all(is_valid_audio_file(paths.segment_wav_path(s["id"])) for s in final_segs)


def test_real_pythaitts_offline_synthesis(tmp_path: Path) -> None:
    """Verify genuine offline Thai speech synthesis via PyThaiTTS (makham_paradee ONNX)."""
    provider = PyThaiTTSProvider(pretrained="makham_paradee")
    out_wav = tmp_path / "thai_real.wav"
    result = provider.synthesize("สวัสดีครับ วันนี้เราจะมาทดสอบเสียงภาษาไทยกัน", out_wav)
    assert result == out_wav
    assert is_valid_audio_file(out_wav) is True

    data, sr = sf.read(str(out_wav), dtype="float32")
    assert sr == 24000
    assert data.size > sr // 2  # > 0.5s of genuine speech
    rms = float(np.sqrt(np.mean(data * data)))
    assert rms > 1e-3  # non-silent waveform


def test_run_tts_align_force_crash_resume_regenerates_unreached_segments(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000021")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text_en": "One", "text_th": "เก่า 1", "tts": {"status": "pending"}},
            {"id": 2, "start": 3.0, "end": 6.0, "text_en": "Two", "text_th": "เก่า 2", "tts": {"status": "pending"}},
            {"id": 3, "start": 6.0, "end": 9.0, "text_en": "Three", "text_th": "เก่า 3", "tts": {"status": "pending"}},
        ],
    )

    # Initial complete run
    run_tts_align(paths, LocalToneTTSProvider(), max_atempo=1.25)

    # Edit text_th on all 3 segments, then run with force=True crashing on seg 2
    segs = read_json(paths.segments_th)
    segs[0]["text_th"] = "ใหม่ 1"
    segs[1]["text_th"] = "ใหม่ 2"
    segs[2]["text_th"] = "ใหม่ 3"
    atomic_write_json(paths.segments_th, segs)

    with pytest.raises(StageError):
        run_tts_align(paths, LocalToneTTSProvider(fail_on_call=2), max_atempo=1.25, force=True)

    after_crash = read_json(paths.segments_th)
    assert after_crash[0]["tts"]["status"] == "done"
    assert after_crash[1]["tts"]["status"] == "failed"
    assert after_crash[2]["tts"]["status"] == "pending"

    # Resume with force=False -> skips seg 1, regenerates BOTH seg 2 and seg 3!
    resume_tts = LocalToneTTSProvider()
    final_segs = run_tts_align(paths, resume_tts, max_atempo=1.25, force=False)
    assert resume_tts.calls == ["ใหม่ 2", "ใหม่ 3"]
    assert all(s["tts"]["status"] == "done" for s in final_segs)


def test_run_tts_align_handles_non_dict_tts_metadata(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000022")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text_en": "Hi", "text_th": "สวัสดี", "tts": "not-a-dict"},
        ],
    )
    tts = LocalToneTTSProvider()
    result = run_tts_align(paths, tts, max_atempo=1.25, force=False)
    assert tts.calls == ["สวัสดี"]
    assert result[0]["tts"]["status"] == "done"
    assert is_valid_audio_file(paths.segment_wav_path(1))


