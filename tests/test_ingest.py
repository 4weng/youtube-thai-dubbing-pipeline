"""Unit tests for Stage 1 src/ingest.py."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from src.ingest import (
    DownloadResult,
    RawCue,
    extract_audio_16k_mono,
    parse_srt,
    parse_subtitle_file,
    parse_vtt,
    run_ingest,
    segment_cues,
)
from src.state import get_workspace_paths, read_json
from tests.conftest import make_test_video


def test_extract_audio_16k_mono_real_ffmpeg(tmp_path: Path) -> None:
    vid = make_test_video(tmp_path / "src.mp4", duration=2.5, freq=440.0, with_audio=True)
    wav = extract_audio_16k_mono(vid, tmp_path / "audio_en.wav")
    info = sf.info(str(wav))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert 2.3 <= info.duration <= 2.7


def test_parse_vtt_strips_tags_and_deduplicates_rolling_captions() -> None:
    vtt = """WEBVTT
Kind: captions
Language: en

NOTE comment block to ignore

00:00:00.500 --> 00:00:02.000 align:start position:0%
<c>Hello</c> &amp; welcome

00:00:02.000 --> 00:00:04.000
Hello &amp; welcome to the pipeline.

00:00:04.000 --> 00:00:06.000
<i>to the pipeline.</i> Let us begin!
"""
    cues = parse_vtt(vtt)
    assert len(cues) == 3
    assert cues[0].text == "Hello & welcome"
    assert cues[1].text == "to the pipeline."
    assert cues[2].text == "Let us begin!"


def test_parse_srt_and_dispatch(tmp_path: Path) -> None:
    srt = """1
00:00:01,000 --> 00:00:03,500
<b>First line</b> of speech.

2
00:00:03,800 --> 00:00:06,200
Second line &quot;quoted&quot;.
"""
    cues = parse_srt(srt)
    assert len(cues) == 2
    assert cues[0].text == "First line of speech."
    assert cues[1].text == 'Second line "quoted".'

    srt_file = tmp_path / "subs.srt"
    srt_file.write_text(srt, encoding="utf-8")
    assert parse_subtitle_file(srt_file) == cues


def test_segment_cues_merges_short_and_splits_long() -> None:
    cues = [
        RawCue(0.0, 1.2, "Short one."),
        RawCue(1.3, 2.5, "Short two."),
        RawCue(2.6, 3.8, "Short three."),
        RawCue(5.0, 17.0, "Sentence A is long enough here. Sentence B continues the thought. Sentence C finishes it."),
    ]
    segs = segment_cues(cues, min_dur=3.0, max_dur=8.0)
    assert len(segs) >= 3
    # First 3 short cues merged into ~3.8s segment
    assert segs[0]["id"] == 1
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 3.8
    assert "Short one." in segs[0]["text_en"] and "Short three." in segs[0]["text_en"]

    # All segments monotonic, positive duration, <= 8.0s
    for idx, seg in enumerate(segs, start=1):
        assert seg["id"] == idx
        assert seg["end"] > seg["start"]
        assert seg["end"] - seg["start"] <= 8.01
        if idx > 1:
            assert seg["start"] >= segs[idx - 2]["end"]


def test_run_ingest_uses_manual_subs_and_falls_back_to_asr(tmp_path: Path) -> None:
    # Case 1: Manual subtitles present -> ASR should NOT be called
    paths1 = get_workspace_paths(tmp_path / "w1", "vid00000001")
    asr_calls: list[Path] = []

    class DummyASR:
        def transcribe(self, audio_path: Path, language: str = "en") -> list[RawCue]:
            asr_calls.append(audio_path)
            return [RawCue(0.0, 4.0, "Transcribed by ASR.")]

    def fake_downloader_with_subs(url: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        make_test_video(out_vid, duration=4.0)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub = sub_dir / "en.vtt"
        sub.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:03.500\nManual subtitle line.\n", encoding="utf-8")
        return DownloadResult(video_path=out_vid, subtitle_path=sub, subtitle_is_auto=False)

    segs1 = run_ingest(
        paths1,
        "https://youtu.be/vid00000001",
        asr_provider=DummyASR(),
        downloader=fake_downloader_with_subs,
    )
    assert len(segs1) == 1
    assert segs1[0]["text_en"] == "Manual subtitle line."
    assert asr_calls == []
    assert read_json(paths1.segments_en) == segs1

    # Case 2: No subtitles -> ASR fallback triggered
    paths2 = get_workspace_paths(tmp_path / "w2", "vid00000002")

    def fake_downloader_no_subs(url: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        make_test_video(out_vid, duration=4.0)
        return DownloadResult(video_path=out_vid, subtitle_path=None, subtitle_is_auto=False)

    segs2 = run_ingest(
        paths2,
        "https://youtu.be/vid00000002",
        asr_provider=DummyASR(),
        downloader=fake_downloader_no_subs,
    )
    assert len(segs2) == 1
    assert segs2[0]["text_en"] == "Transcribed by ASR."
    assert len(asr_calls) == 1


def test_parse_vtt_and_parse_srt_accept_path_and_cr_line_endings(tmp_path: Path) -> None:
    vtt_file = tmp_path / "sample.vtt"
    vtt_file.write_text(
        "WEBVTT\n\n00:00:00.500 --> 00:00:02.500\nHello from VTT path.\n",
        encoding="utf-8",
    )
    vtt_cues = parse_vtt(vtt_file)
    assert len(vtt_cues) == 1
    assert vtt_cues[0].text == "Hello from VTT path."

    srt_cr = "1\r00:00:01,000 --> 00:00:03,000\rHello\r\r2\r00:00:03,500 --> 00:00:05,000\rWorld\r"
    srt_cues = parse_srt(srt_cr)
    assert len(srt_cues) == 2
    assert srt_cues[0].text == "Hello"
    assert srt_cues[1].text == "World"

    srt_file = tmp_path / "sample.srt"
    srt_file.write_bytes(srt_cr.encode("utf-8"))
    srt_path_cues = parse_srt(srt_file)
    assert len(srt_path_cues) == 2
    assert srt_path_cues[1].text == "World"


def test_run_ingest_preserves_auto_subtitle_status_on_resume(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000003")
    asr_calls: list[Path] = []

    class DummyASR:
        def transcribe(self, audio_path: Path, language: str = "en") -> list[RawCue]:
            asr_calls.append(audio_path)
            return [RawCue(0.0, 4.0, "ASR fallback transcript.")]

    def fake_downloader_auto_sub(url: str, out_vid: Path, sub_dir: Path) -> DownloadResult:
        make_test_video(out_vid, duration=4.0)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_file = sub_dir / "captions.vtt"
        sub_file.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\nAuto generated noisy captions.\n",
            encoding="utf-8",
        )
        return DownloadResult(video_path=out_vid, subtitle_path=sub_file, subtitle_is_auto=True)

    # Run 1: auto-sub rejected -> ASR fallback used
    segs1 = run_ingest(
        paths,
        "https://youtu.be/vid00000003",
        asr_provider=DummyASR(),
        downloader=fake_downloader_auto_sub,
    )
    assert segs1[0]["text_en"] == "ASR fallback transcript."
    assert len(asr_calls) == 1

    # Run 2 (intra-stage resume reusing source.mp4 & audio_en.wav):
    # cached auto-sub in subs/ must STILL be recognized as auto and trigger ASR fallback
    paths.segments_en.unlink()
    segs2 = run_ingest(
        paths,
        "https://youtu.be/vid00000003",
        asr_provider=DummyASR(),
        downloader=fake_downloader_auto_sub,
    )
    assert segs2[0]["text_en"] == "ASR fallback transcript."
    assert len(asr_calls) == 2


def test_parse_srt_splits_on_whitespace_only_blank_lines() -> None:
    srt_text = (
        "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        "   \t  \n"
        "2\n00:00:03,500 --> 00:00:05,000\nWorld\n"
    )
    cues = parse_srt(srt_text)
    assert len(cues) == 2
    assert cues[0].text == "Hello"
    assert cues[1].text == "World"


