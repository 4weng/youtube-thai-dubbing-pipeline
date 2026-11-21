"""Tier 1 E2E Tests: Isolated happy-path coverage of all 26 features in PROJECT.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import soundfile as sf

from pipeline import build_arg_parser, run_pipeline
from src.ingest import RawCue, parse_srt, parse_vtt, run_ingest, segment_cues
from src.mux import assemble_timeline, run_mux
from src.state import (
    STAGE_ORDER,
    PipelineConfig,
    atomic_temp_path,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    extract_video_id,
    get_workspace_paths,
    init_or_load_manifest,
    is_valid_audio_file,
    is_valid_video_file,
    read_json,
    set_stage_status,
    should_skip_stage,
    verify_stage_outputs,
)
from src.translate import normalize_thai_for_tts, run_translate
from src.tts_align import create_tts_provider, run_tts_align

from .conftest import (
    FakeASRProvider,
    FakeTranslator,
    FakeTTSProvider,
    create_tone_wav,
    ffprobe_json,
    make_local_downloader,
    measure_window_rms,
    write_srt,
    write_vtt,
)


# ---------------------------------------------------------------------------
# Feature #4: YouTube video_id extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/5orlc92154E",
        "https://www.youtube.com/watch?v=5orlc92154E",
        "https://youtube.com/watch?v=5orlc92154E&t=42s&list=PL123",
        "https://www.youtube.com/shorts/5orlc92154E",
        "https://www.youtube.com/embed/5orlc92154E",
        "https://www.youtube.com/live/5orlc92154E?feature=share",
    ],
)
def test_f04_extract_video_id_all_url_formats(url: str) -> None:
    """Feature #4: Canonical 11-char video_id extracted from all YouTube URL forms."""
    assert extract_video_id(url) == "5orlc92154E"


# ---------------------------------------------------------------------------
# Feature #6: Per-video workspace isolation
# ---------------------------------------------------------------------------
def test_f06_per_video_workspace_isolation(work_dir: Path) -> None:
    """Feature #6: Each video_id gets an isolated workspace under work/<video_id>/."""
    paths_a = get_workspace_paths(work_dir, "vidA1234567")
    paths_b = get_workspace_paths(work_dir, "vidB7654321")
    paths_a.ensure_dirs()
    paths_b.ensure_dirs()

    assert paths_a.root == work_dir / "vidA1234567"
    assert paths_b.root == work_dir / "vidB7654321"
    assert paths_a.root != paths_b.root
    assert paths_a.tts_segments_dir.is_dir()
    assert paths_a.manifest == paths_a.root / "manifest.json"
    assert paths_a.source_video == paths_a.root / "source.mp4"
    assert paths_a.audio_en == paths_a.root / "audio_en.wav"
    assert paths_a.segments_en == paths_a.root / "segments_en.json"
    assert paths_a.segments_th == paths_a.root / "segments_th.json"
    assert paths_a.thai_dub_full == paths_a.root / "thai_dub_full.wav"
    assert paths_a.output_video == paths_a.root / "output_th.mp4"
    assert paths_a.segment_wav_relpath(1) == "tts_segments/seg_0001.wav"
    assert paths_a.segment_wav_path(1) == paths_a.tts_segments_dir / "seg_0001.wav"


# ---------------------------------------------------------------------------
# Features #7 & #10: manifest.json state management & config persistence
# ---------------------------------------------------------------------------
def test_f07_f10_manifest_init_and_config_persistence(work_dir: Path) -> None:
    """Features #7 & #10: manifest.json matches Spec §4.1 schema and persists defaults."""
    paths = get_workspace_paths(work_dir, "5orlc92154E")
    paths.ensure_dirs()
    cfg = PipelineConfig()
    manifest = init_or_load_manifest(
        paths,
        "5orlc92154E",
        "https://youtu.be/5orlc92154E",
        cfg,
    )

    assert paths.manifest.exists()
    persisted = read_json(paths.manifest)
    assert persisted["video_id"] == "5orlc92154E"
    assert persisted["source_url"] == "https://youtu.be/5orlc92154E"
    assert persisted["config"]["mix_mode"] == "duck"
    assert persisted["config"]["original_volume"] == pytest.approx(0.18)
    assert persisted["config"]["max_atempo"] == pytest.approx(1.25)
    assert persisted["config"]["tts_provider"] == "pythaitts"
    for stage in STAGE_ORDER:
        assert stage in persisted["stages"]
        assert persisted["stages"][stage]["status"] == "pending"
    assert persisted["last_error"] is None
    assert manifest["video_id"] == persisted["video_id"]


# ---------------------------------------------------------------------------
# Feature #8: Atomic file writes
# ---------------------------------------------------------------------------
def test_f08_atomic_file_writes_leave_no_tmp_files(tmp_path: Path) -> None:
    """Feature #8: Atomic writes replace target cleanly and leave zero .tmp artifacts."""
    json_target = tmp_path / "data.json"
    bytes_target = tmp_path / "blob.bin"
    text_target = tmp_path / "note.txt"
    wav_target = tmp_path / "clip.wav"

    atomic_write_json(json_target, {"hello": "สวัสดี", "count": 3})
    atomic_write_bytes(bytes_target, b"\x00\x01\x02\x03")
    atomic_write_text(text_target, "ภาษาไทย")
    with atomic_temp_path(wav_target) as tmp_wav:
        assert tmp_wav.suffix == ".wav"
        create_tone_wav(tmp_wav, duration=0.5)

    assert json.loads(json_target.read_text(encoding="utf-8")) == {"hello": "สวัสดี", "count": 3}
    assert bytes_target.read_bytes() == b"\x00\x01\x02\x03"
    assert text_target.read_text(encoding="utf-8") == "ภาษาไทย"
    assert is_valid_audio_file(wav_target)
    leftover_tmp = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftover_tmp == []


# ---------------------------------------------------------------------------
# Feature #9: Verify Before Skip
# ---------------------------------------------------------------------------
def test_f09_verify_before_skip_valid_media_and_stages(
    work_dir: Path,
    synthetic_video_12s: Path,
) -> None:
    """Feature #9: Valid media and stage outputs pass verify-before-skip checks."""
    paths = get_workspace_paths(work_dir, "verifyVid01")
    paths.ensure_dirs()
    manifest = init_or_load_manifest(
        paths, "verifyVid01", "https://youtu.be/verifyVid01", PipelineConfig()
    )

    wav_path = create_tone_wav(paths.audio_en, duration=1.5)
    assert is_valid_audio_file(wav_path) is True
    assert is_valid_video_file(synthetic_video_12s, require_audio=True) is True

    # Populate Stage 1 outputs and mark stage done
    paths.source_video.write_bytes(synthetic_video_12s.read_bytes())
    atomic_write_json(
        paths.segments_en,
        [{"id": 1, "start": 0.5, "end": 4.0, "text_en": "Hello world."}],
    )
    set_stage_status(paths, manifest, "ingest", "done")

    assert verify_stage_outputs(paths, "ingest") is True
    assert should_skip_stage(paths, manifest, "ingest") is True
    assert should_skip_stage(paths, manifest, "ingest", force_stage="ingest") is False


# ---------------------------------------------------------------------------
# Features #11 & #12: Video download & 16kHz mono audio extraction
# ---------------------------------------------------------------------------
def test_f11_f12_ingest_video_download_and_16k_mono_audio_extraction(
    work_dir: Path,
    synthetic_video_12s: Path,
    sample_vtt_file: Path,
) -> None:
    """Features #11 & #12: Stage 1 saves source.mp4 and extracts 16kHz mono audio_en.wav."""
    paths = get_workspace_paths(work_dir, "ingestVid01")
    paths.ensure_dirs()
    dl = make_local_downloader(synthetic_video_12s, subtitle_src=sample_vtt_file)

    segments = run_ingest(paths, "https://youtu.be/ingestVid01", downloader=dl)

    assert is_valid_video_file(paths.source_video, require_audio=True)
    assert is_valid_audio_file(paths.audio_en)
    assert paths.segments_en.exists()
    assert len(segments) >= 1

    info = ffprobe_json(paths.audio_en)
    audio_streams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    assert len(audio_streams) == 1
    assert int(audio_streams[0]["sample_rate"]) == 16000
    assert int(audio_streams[0]["channels"]) == 1
    assert float(info["format"]["duration"]) > 10.0


# ---------------------------------------------------------------------------
# Feature #13: Subtitle extraction & VTT/SRT parsing
# ---------------------------------------------------------------------------
def test_f13_subtitle_vtt_and_srt_parsing_and_tag_stripping(tmp_path: Path) -> None:
    """Feature #13: parse_vtt and parse_srt strip markup tags and deduplicate rolling lines."""
    vtt_path = write_vtt(
        tmp_path / "rolling.vtt",
        [
            (0.0, 2.0, "<b>Hello</b> <c.colorE5E5E5>world</c>"),
            (2.0, 4.0, "<b>Hello</b> <c.colorE5E5E5>world</c>"),
            (4.1, 7.5, "<i>Second</i> sentence here."),
        ],
    )
    srt_path = write_srt(
        tmp_path / "clean.srt",
        [
            (0.5, 3.8, "<i>First</i> line in SRT."),
            (4.0, 7.5, "Second line in SRT."),
        ],
    )

    vtt_cues = parse_vtt(vtt_path)
    srt_cues = parse_srt(srt_path)

    assert len(vtt_cues) >= 2
    for cue in vtt_cues:
        assert "<" not in cue.text and ">" not in cue.text
        assert cue.end > cue.start
    # Rolling duplicate should not produce two identical consecutive cues
    texts = [c.text.strip() for c in vtt_cues]
    assert texts.count("Hello world") == 1

    assert len(srt_cues) == 2
    assert "<" not in srt_cues[0].text
    assert srt_cues[0].text.strip() == "First line in SRT."


# ---------------------------------------------------------------------------
# Feature #14: Whisper ASR fallback when no subtitles are available
# ---------------------------------------------------------------------------
def test_f14_whisper_asr_fallback_when_no_subtitles(
    work_dir: Path,
    synthetic_video_12s: Path,
) -> None:
    """Feature #14: Stage 1 invokes ASRProvider.transcribe when subtitles are absent."""
    paths = get_workspace_paths(work_dir, "asrFallbk01")
    paths.ensure_dirs()
    dl = make_local_downloader(synthetic_video_12s, subtitle_src=None)
    asr = FakeASRProvider(
        [
            RawCue(1.0, 4.5, "Audio transcribed via Whisper fallback."),
            RawCue(5.0, 8.8, "Second transcribed sentence."),
        ]
    )

    segments = run_ingest(
        paths,
        "https://youtu.be/asrFallbk01",
        downloader=dl,
        asr_provider=asr,
    )

    assert len(asr.calls) == 1
    assert asr.calls[0][0] == paths.audio_en
    assert len(segments) == 2
    assert segments[0]["id"] == 1
    assert "Whisper fallback" in segments[0]["text_en"]


# ---------------------------------------------------------------------------
# Feature #15: Sentence-level segmentation (3-8s)
# ---------------------------------------------------------------------------
def test_f15_sentence_level_segmentation_3_to_8s() -> None:
    """Feature #15: Short cues are merged and long cues are split toward 3-8s segments."""
    cues = [
        RawCue(0.0, 1.2, "Short part one."),
        RawCue(1.2, 2.4, "Short part two."),
        RawCue(2.4, 3.8, "Short part three."),
        RawCue(
            5.0,
            19.0,
            "This is the first long sentence in a fourteen second block. "
            "This is the second long sentence in the same block. "
            "And this is the third sentence finishing the block.",
        ),
    ]
    segments = segment_cues(cues, min_dur=3.0, max_dur=8.0)

    assert len(segments) >= 3
    for idx, seg in enumerate(segments, start=1):
        assert seg["id"] == idx
        assert seg["start"] < seg["end"]
        assert seg["text_en"].strip() != ""
    # The three short 1.2s cues should have been merged into a segment >= 3.0s
    assert (segments[0]["end"] - segments[0]["start"]) >= 3.0
    # The 14s cue should have been split so no segment is wildly over max_dur
    assert max(s["end"] - s["start"] for s in segments) <= 9.5


# ---------------------------------------------------------------------------
# Features #16, #17, #19: Context-aware translation, length fitting, segments_th.json
# ---------------------------------------------------------------------------
def test_f16_f17_f19_context_aware_translation_and_length_fitting(work_dir: Path) -> None:
    """Features #16, #17, #19: run_translate passes context + max_chars and writes segments_th.json."""
    paths = get_workspace_paths(work_dir, "transVid001")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_en,
        [
            {"id": 1, "start": 0.5, "end": 4.0, "text_en": "First sentence."},
            {"id": 2, "start": 4.2, "end": 7.8, "text_en": "Second sentence."},
            {"id": 3, "start": 8.0, "end": 11.5, "text_en": "Third sentence."},
        ],
    )
    translator = FakeTranslator(
        {
            "First sentence.": "ประโยคแรก",
            "Second sentence.": "ประโยคที่สอง",
            "Third sentence.": "ประโยคที่สาม",
        }
    )

    segments_th = run_translate(paths, translator, context_window=1)

    assert paths.segments_th.exists()
    assert len(segments_th) == 3
    assert len(translator.calls) == 3

    # Middle segment must receive surrounding context and positive length budget
    mid_call = translator.calls[1]
    assert "First sentence." in mid_call["prev_context"]
    assert "Third sentence." in mid_call["next_context"]
    assert mid_call["slot_duration"] == pytest.approx(3.6, abs=0.05)
    assert mid_call["max_chars"] > 0

    persisted = read_json(paths.segments_th)
    assert len(persisted) == 3
    assert persisted[0]["id"] == 1
    assert persisted[0]["text_en"] == "First sentence."
    assert persisted[0]["text_th"] == "ประโยคแรก"


# ---------------------------------------------------------------------------
# Feature #18: Thai TTS text normalization
# ---------------------------------------------------------------------------
def test_f18_thai_tts_text_normalization() -> None:
    """Feature #18: normalize_thai_for_tts converts numbers, %, currency, and acronyms to Thai."""
    raw = "ในปี 2026 ระบบ AI และ API โต 50% ด้วยงบ $100 และค่า 3.14"
    normalized = normalize_thai_for_tts(raw)

    assert isinstance(normalized, str) and normalized.strip() != ""
    # Raw symbols and ASCII digits should be expanded into Thai spoken words
    assert "%" not in normalized
    assert "$" not in normalized
    assert not re.search(r"\d", normalized), f"Digits left unnormalized: {normalized}"
    # Spoken Thai words for percent / acronyms should appear
    assert "เปอร์เซ็นต์" in normalized
    assert "เอไอ" in normalized


# ---------------------------------------------------------------------------
# Feature #20: TTS Provider Abstraction & Voice Locking
# ---------------------------------------------------------------------------
def test_f20_tts_provider_voice_locking(work_dir: Path) -> None:
    """Feature #20: TTSProvider locks a single voice_id across all segments of a video."""
    provider = create_tts_provider("pythaitts")
    assert hasattr(provider, "voice_id")
    assert isinstance(provider.voice_id, str) and len(provider.voice_id) > 0

    paths = get_workspace_paths(work_dir, "voiceLock01")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 0.5, "end": 3.5, "text_en": "A", "text_th": "หนึ่ง"},
            {"id": 2, "start": 4.0, "end": 7.0, "text_en": "B", "text_th": "สอง"},
        ],
    )
    fake_tts = FakeTTSProvider(voice_id="locked_speaker_42")
    run_tts_align(paths, fake_tts, max_atempo=1.25)

    assert len(fake_tts.calls) == 2
    assert {c["voice_id"] for c in fake_tts.calls} == {"locked_speaker_42"}


# ---------------------------------------------------------------------------
# Feature #21: Leading/trailing silence trimming
# ---------------------------------------------------------------------------
def test_f21_leading_trailing_silence_trimming(work_dir: Path) -> None:
    """Feature #21: Stage 3 trims leading/trailing silence before measuring duration."""
    paths = get_workspace_paths(work_dir, "trimSilnc01")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 1.0, "end": 4.0, "text_en": "Hello", "text_th": "สวัสดี"},
        ],
    )
    # 0.8s silence + 1.0s tone + 0.8s silence = 2.6s raw
    tts = FakeTTSProvider(
        duration_fn=lambda *_: 1.0,
        leading_silence=0.8,
        trailing_silence=0.8,
    )
    segs = run_tts_align(paths, tts, max_atempo=1.25)

    wav_path = paths.segment_wav_path(1)
    assert is_valid_audio_file(wav_path)
    info = sf.info(str(wav_path))
    assert 0.8 <= info.duration <= 1.6, f"Expected trimmed duration ~1.0s, got {info.duration:.3f}s"
    assert segs[0]["tts"]["tempo_used"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Feature #22: Duration measurement & bounded atempo
# ---------------------------------------------------------------------------
def test_f22_bounded_atempo_alignment(work_dir: Path) -> None:
    """Feature #22: When actual_dur > slot_dur, Stage 3 applies bounded atempo."""
    paths = get_workspace_paths(work_dir, "atempoAln01")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 1.0, "end": 3.0, "text_en": "Fast", "text_th": "พูดเร็วขึ้นนิดหน่อย"},
        ],
    )
    # 2.4s audio into 2.0s slot -> needed_tempo = 1.2 <= max_atempo(1.25)
    tts = FakeTTSProvider(duration_fn=lambda *_: 2.4)
    segs = run_tts_align(paths, tts, max_atempo=1.25)

    tts_meta = segs[0]["tts"]
    assert 1.1 <= tts_meta["tempo_used"] <= 1.25
    assert tts_meta["duration"] == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# Feature #23: Per-segment checkpoint & incremental save
# ---------------------------------------------------------------------------
def test_f23_per_segment_checkpoint_and_incremental_save(work_dir: Path) -> None:
    """Feature #23: segments_th.json records per-segment tts metadata matching Spec §4.3."""
    paths = get_workspace_paths(work_dir, "segCkpt0001")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_th,
        [
            {"id": 1, "start": 1.0, "end": 4.0, "text_en": "One", "text_th": "หนึ่ง"},
            {"id": 2, "start": 4.5, "end": 7.5, "text_en": "Two", "text_th": "สอง"},
        ],
    )
    tts = FakeTTSProvider(duration_fn=lambda *_: 1.8)
    run_tts_align(paths, tts, max_atempo=1.25)

    persisted = read_json(paths.segments_th)
    for idx, seg in enumerate(persisted, start=1):
        assert seg["tts"]["status"] == "done"
        assert seg["tts"]["file"] == f"tts_segments/seg_{idx:04d}.wav"
        assert seg["tts"]["duration"] > 0.0
        assert seg["tts"]["tempo_used"] >= 1.0
        assert is_valid_audio_file(paths.root / seg["tts"]["file"])


# ---------------------------------------------------------------------------
# Feature #24: Timeline audio assembly (thai_dub_full.wav)
# ---------------------------------------------------------------------------
def test_f24_timeline_audio_assembly(work_dir: Path, tmp_path: Path) -> None:
    """Feature #24: assemble_timeline places segments at start_ms with silence in gaps."""
    paths = get_workspace_paths(work_dir, "timeline001")
    paths.ensure_dirs()

    create_tone_wav(paths.segment_wav_path(1), duration=1.5, freq=880.0, amplitude=0.6)
    create_tone_wav(paths.segment_wav_path(2), duration=1.5, freq=660.0, amplitude=0.6)

    segments_th = [
        {
            "id": 1,
            "start": 1.5,
            "end": 3.5,
            "text_en": "One",
            "text_th": "หนึ่ง",
            "tts": {"status": "done", "file": "tts_segments/seg_0001.wav", "duration": 1.5, "tempo_used": 1.0},
        },
        {
            "id": 2,
            "start": 5.5,
            "end": 7.5,
            "text_en": "Two",
            "text_th": "สอง",
            "tts": {"status": "done", "file": "tts_segments/seg_0002.wav", "duration": 1.5, "tempo_used": 1.0},
        },
    ]

    full_wav = assemble_timeline(paths, segments_th, total_duration=9.0)
    assert full_wav == paths.thai_dub_full
    assert is_valid_audio_file(full_wav)

    gap_rms = measure_window_rms(full_wav, 0.1, 1.2, tmp_dir=tmp_path)
    seg1_rms = measure_window_rms(full_wav, 1.7, 2.8, tmp_dir=tmp_path)
    mid_gap_rms = measure_window_rms(full_wav, 3.5, 5.0, tmp_dir=tmp_path)
    seg2_rms = measure_window_rms(full_wav, 5.7, 6.8, tmp_dir=tmp_path)

    assert gap_rms < 0.005
    assert mid_gap_rms < 0.005
    assert seg1_rms > 0.1
    assert seg2_rms > 0.1


# ---------------------------------------------------------------------------
# Features #25 & #26: Video muxing — Duck mode & Replace mode
# ---------------------------------------------------------------------------
def _prepare_mux_workspace(work_dir: Path, video_id: str, source_video: Path) -> Path:
    paths = get_workspace_paths(work_dir, video_id)
    paths.ensure_dirs()
    paths.source_video.write_bytes(source_video.read_bytes())
    create_tone_wav(paths.segment_wav_path(1), duration=2.0, freq=880.0, amplitude=0.6)
    atomic_write_json(
        paths.segments_th,
        [
            {
                "id": 1,
                "start": 2.0,
                "end": 4.5,
                "text_en": "Hello",
                "text_th": "สวัสดี",
                "tts": {
                    "status": "done",
                    "file": "tts_segments/seg_0001.wav",
                    "duration": 2.0,
                    "tempo_used": 1.0,
                },
            }
        ],
    )
    return paths


def test_f25_video_mux_duck_mode(work_dir: Path, synthetic_video_12s: Path) -> None:
    """Feature #25: run_mux in duck mode produces valid output_th.mp4 with video + AAC audio."""
    paths = _prepare_mux_workspace(work_dir, "muxDuck0001", synthetic_video_12s)
    out = run_mux(paths, mix_mode="duck", original_volume=0.18)

    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)
    info = ffprobe_json(out)
    codecs = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
    assert "video" in codecs
    assert codecs["audio"] == "aac"


def test_f26_video_mux_replace_mode(work_dir: Path, synthetic_video_12s: Path) -> None:
    """Feature #26: run_mux in replace mode produces valid output_th.mp4 with video + AAC audio."""
    paths = _prepare_mux_workspace(work_dir, "muxRepl0001", synthetic_video_12s)
    out = run_mux(paths, mix_mode="replace")

    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)
    info = ffprobe_json(out)
    codecs = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
    assert "video" in codecs
    assert codecs["audio"] == "aac"


# ---------------------------------------------------------------------------
# Features #1, #2, #3, #5: CLI parser & 4-stage pipeline orchestration
# ---------------------------------------------------------------------------
def test_f01_f02_f03_f05_cli_parser_and_pipeline_orchestration(
    work_dir: Path,
    synthetic_video_12s: Path,
    sample_vtt_file: Path,
) -> None:
    """Features #1, #2, #3, #5: CLI flags parse cleanly and run_pipeline completes all 4 stages."""
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--url",
            "https://youtu.be/orchTest001",
            "--mix-mode",
            "replace",
            "--force-stage",
            "mux",
            "--work-dir",
            str(work_dir),
            "--original-volume",
            "0.2",
            "--max-atempo",
            "1.25",
            "--tts-provider",
            "pythaitts",
        ]
    )
    assert args.url == "https://youtu.be/orchTest001"
    assert args.mix_mode == "replace"
    assert args.force_stage == "mux"

    out = run_pipeline(
        "https://youtu.be/orchTest001",
        PipelineConfig(mix_mode="duck"),
        work_dir=work_dir,
        downloader=make_local_downloader(synthetic_video_12s, subtitle_src=sample_vtt_file),
        translator=FakeTranslator(),
        tts_provider=FakeTTSProvider(duration_fn=lambda *_: 1.8),
    )

    paths = get_workspace_paths(work_dir, "orchTest001")
    assert out == paths.output_video
    assert is_valid_video_file(out, require_audio=True)

    manifest = read_json(paths.manifest)
    for stage in STAGE_ORDER:
        assert manifest["stages"][stage]["status"] == "done"
    assert manifest["last_error"] is None
