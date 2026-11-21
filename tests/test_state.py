"""Unit tests for src/state.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.state import (
    PipelineConfig,
    ValidationError,
    atomic_temp_path,
    atomic_write_json,
    extract_video_id,
    get_workspace_paths,
    has_audio_stream,
    init_or_load_manifest,
    is_valid_audio_file,
    is_valid_json_file,
    is_valid_video_file,
    probe_media_duration,
    read_json,
    reset_stages_from,
    resolve_workspace_relpath,
    set_stage_status,
    should_skip_stage,
    verify_stage_outputs,
)
from tests.conftest import make_test_video, make_test_wav


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/5orlc92154E",
        "https://youtu.be/5orlc92154E?t=45",
        "https://www.youtube.com/watch?v=5orlc92154E",
        "https://www.youtube.com/watch?v=5orlc92154E&list=PL12345&index=2",
        "https://m.youtube.com/watch?v=5orlc92154E",
        "https://www.youtube.com/shorts/5orlc92154E",
        "https://www.youtube.com/embed/5orlc92154E",
        "https://www.youtube.com/live/5orlc92154E?feature=share",
    ],
)
def test_extract_video_id_valid_urls(url: str) -> None:
    assert extract_video_id(url) == "5orlc92154E"


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "not-a-url",
        "ftp://youtube.com/watch?v=5orlc92154E",
        "https://vimeo.com/123456789",
        "https://youtu.be/../etc/passwd",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=../../evil123",
        "https://www.youtube.com/shorts/",
    ],
)
def test_extract_video_id_rejects_invalid_and_traversal(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        extract_video_id(bad_url)


def test_get_workspace_paths_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        get_workspace_paths(tmp_path, "../escape")
    with pytest.raises(ValidationError):
        get_workspace_paths(tmp_path, "a/b")


def test_pipeline_config_validation_and_roundtrip() -> None:
    cfg = PipelineConfig(mix_mode="replace", original_volume=0.25, max_atempo=1.5)
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg

    with pytest.raises(ValidationError):
        PipelineConfig(mix_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PipelineConfig(original_volume=-0.1)
    with pytest.raises(ValidationError):
        PipelineConfig(original_volume=1.5)
    with pytest.raises(ValidationError):
        PipelineConfig(max_atempo=0.8)


def test_atomic_temp_path_preserves_target_on_error_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    atomic_write_json(target, {"thai": "สวัสดี", "v": 1})
    assert read_json(target) == {"thai": "สวัสดี", "v": 1}
    assert "สวัสดี" in target.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_temp_path(target) as tmp_file:
            assert tmp_file.suffix == ".json"
            tmp_file.write_text("corrupt partial", encoding="utf-8")
            raise RuntimeError("boom")

    assert read_json(target) == {"thai": "สวัสดี", "v": 1}
    leftovers = [p for p in tmp_path.iterdir() if p.name != "data.json"]
    assert leftovers == []


def test_media_validation_audio_and_video(tmp_path: Path) -> None:
    wav = make_test_wav(tmp_path / "good.wav", duration=1.2)
    assert is_valid_audio_file(wav) is True
    dur = probe_media_duration(wav)
    assert dur is not None and 1.1 <= dur <= 1.3

    zero_wav = tmp_path / "empty.wav"
    zero_wav.write_bytes(b"")
    assert is_valid_audio_file(zero_wav) is False

    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"not a real wav file header at all")
    assert is_valid_audio_file(corrupt_wav) is False

    mp4 = make_test_video(tmp_path / "good.mp4", duration=2.0, with_audio=True)
    assert is_valid_video_file(mp4, require_audio=True) is True
    assert has_audio_stream(mp4) is True

    silent_mp4 = make_test_video(tmp_path / "silent.mp4", duration=2.0, with_audio=False)
    assert is_valid_video_file(silent_mp4, require_audio=False) is True
    assert is_valid_video_file(silent_mp4, require_audio=True) is False
    assert has_audio_stream(silent_mp4) is False


def test_manifest_verify_before_skip_and_config_mismatch(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "5orlc92154E")
    cfg = PipelineConfig(mix_mode="duck", original_volume=0.18)
    manifest = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", cfg)

    assert is_valid_json_file(paths.manifest, expected_type=dict)
    assert manifest["stages"]["ingest"]["status"] == "pending"

    # Mark ingest done without files -> verify-before-skip MUST return False
    set_stage_status(paths, manifest, "ingest", "done")
    assert verify_stage_outputs(paths, "ingest") is False
    assert should_skip_stage(paths, manifest, "ingest") is False

    # Create valid ingest artifacts -> verify-before-skip returns True
    make_test_video(paths.source_video, duration=2.0)
    make_test_wav(paths.audio_en, duration=2.0, sr=16000)
    atomic_write_json(
        paths.segments_en,
        [{"id": 1, "start": 0.0, "end": 2.0, "text_en": "Hello world."}],
    )
    assert verify_stage_outputs(paths, "ingest") is True
    assert should_skip_stage(paths, manifest, "ingest") is True
    assert should_skip_stage(paths, manifest, "ingest", force_stage="ingest") is False

    # Mark mux done with valid artifacts, then change mix_mode -> must NOT skip mux
    make_test_wav(paths.thai_dub_full, duration=2.0)
    make_test_video(paths.output_video, duration=2.0, with_audio=True)
    set_stage_status(paths, manifest, "mux", "done")
    assert should_skip_stage(paths, manifest, "mux", config=cfg) is True

    new_cfg = PipelineConfig(mix_mode="replace", original_volume=0.18)
    assert should_skip_stage(paths, manifest, "mux", config=new_cfg) is False

    reloaded = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", new_cfg)
    assert reloaded["stages"]["mux"]["status"] == "pending"
    assert reloaded["stages"]["ingest"]["status"] == "done"

    # Cascade reset from translate resets translate, tts_align, mux but keeps ingest
    set_stage_status(paths, reloaded, "translate", "done")
    set_stage_status(paths, reloaded, "tts_align", "done")
    set_stage_status(paths, reloaded, "mux", "done")
    reset_stages_from(paths, reloaded, "translate")
    assert reloaded["stages"]["ingest"]["status"] == "done"
    assert reloaded["stages"]["translate"]["status"] == "pending"
    assert reloaded["stages"]["tts_align"]["status"] == "pending"
    assert reloaded["stages"]["mux"]["status"] == "pending"


def test_extract_video_id_and_workspace_reject_trailing_newline(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        extract_video_id("https://www.youtube.com/watch?v=5orlc92154E%0A")
    with pytest.raises(ValidationError):
        extract_video_id("https://www.youtube.com/watch?v=5orlc92154E%0A&t=10")
    with pytest.raises(ValidationError):
        get_workspace_paths(tmp_path, "5orlc92154E\n")


def test_pipeline_config_rejects_bool_numeric_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(original_volume=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PipelineConfig(max_atempo=True)  # type: ignore[arg-type]


def test_manifest_recovers_from_type_corrupted_fields(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "5orlc92154E")
    paths.ensure_dirs()
    cfg = PipelineConfig()

    # Case 1: config is a string instead of dict
    atomic_write_json(
        paths.manifest,
        {"video_id": "5orlc92154E", "config": "bad", "stages": {"ingest": {"status": "done"}}},
    )
    loaded = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", cfg)
    assert isinstance(loaded["config"], dict)
    assert loaded["stages"]["tts_align"]["status"] == "pending"
    assert loaded["stages"]["mux"]["status"] == "pending"

    # Case 2: original_volume is non-numeric string
    atomic_write_json(
        paths.manifest,
        {
            "video_id": "5orlc92154E",
            "config": {"original_volume": "not_a_float"},
            "stages": {"mux": {"status": "done"}},
        },
    )
    loaded2 = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", cfg)
    assert loaded2["stages"]["mux"]["status"] == "pending"

    # Case 3: should_skip_stage on corrupted manifest dict does not raise AttributeError/ValueError
    assert (
        should_skip_stage(
            paths,
            {"stages": {"ingest": "done"}, "config": "bad"},
            "ingest",
            config=cfg,
        )
        is False
    )
    assert (
        should_skip_stage(
            paths,
            {"stages": {"mux": {"status": "done"}}, "config": {"mix_mode": "duck", "original_volume": "bad"}},
            "mux",
            config=cfg,
        )
        is False
    )


def test_verify_stage_outputs_tts_align_rejects_path_traversal(tmp_path: Path) -> None:
    outside_wav = tmp_path / "outside.wav"
    make_test_wav(outside_wav, duration=1.0)

    paths = get_workspace_paths(tmp_path / "work", "5orlc92154E")
    paths.ensure_dirs()

    # Relative escape via ..
    atomic_write_json(
        paths.segments_th,
        [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "text_en": "Hi",
                "text_th": "สวัสดี",
                "tts": {"status": "done", "file": "../../outside.wav"},
            }
        ],
    )
    assert verify_stage_outputs(paths, "tts_align") is False

    # Absolute path outside workspace
    atomic_write_json(
        paths.segments_th,
        [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "text_en": "Hi",
                "text_th": "สวัสดี",
                "tts": {"status": "done", "file": str(outside_wav)},
            }
        ],
    )
    assert verify_stage_outputs(paths, "tts_align") is False


def test_config_change_resets_tts_align_and_mux(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "5orlc92154E")
    cfg1 = PipelineConfig(max_atempo=1.25, tts_provider="pythaitts", voice_id="v1")
    manifest = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", cfg1)
    for st in ("ingest", "translate", "tts_align", "mux"):
        set_stage_status(paths, manifest, st, "done")

    # Changing max_atempo, tts_provider, or voice_id must invalidate tts_align & mux
    for changed_cfg in (
        PipelineConfig(max_atempo=1.5, tts_provider="pythaitts", voice_id="v1"),
        PipelineConfig(max_atempo=1.25, tts_provider="edge_tts", voice_id="v1"),
        PipelineConfig(max_atempo=1.25, tts_provider="pythaitts", voice_id="v2"),
    ):
        assert should_skip_stage(paths, manifest, "tts_align", config=changed_cfg) is False
        reloaded = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", changed_cfg)
        assert reloaded["stages"]["ingest"]["status"] == "done"
        assert reloaded["stages"]["translate"]["status"] == "done"
        assert reloaded["stages"]["tts_align"]["status"] == "pending"
        assert reloaded["stages"]["mux"]["status"] == "pending"
        # Restore to original cfg1 for next iteration
        manifest = init_or_load_manifest(paths, "5orlc92154E", "https://youtu.be/5orlc92154E", cfg1)
        for st in ("ingest", "translate", "tts_align", "mux"):
            set_stage_status(paths, manifest, st, "done")


def test_pipeline_config_from_dict_rejects_bool_and_invalid_numeric() -> None:
    for bad_val in (True, False, "bad", None):
        with pytest.raises(ValidationError):
            PipelineConfig.from_dict({"original_volume": bad_val})
        with pytest.raises(ValidationError):
            PipelineConfig.from_dict({"max_atempo": bad_val})


def test_resolve_workspace_relpath_rejects_null_byte(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path / "work", "5orlc92154E")
    paths.ensure_dirs()
    assert resolve_workspace_relpath(paths, "tts_segments/seg_0001.wav\x00") is None

    atomic_write_json(
        paths.segments_th,
        [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "text_en": "Hi",
                "text_th": "สวัสดี",
                "tts": {"status": "done", "file": "tts_segments/seg_0001.wav\x00"},
            }
        ],
    )
    assert verify_stage_outputs(paths, "tts_align") is False


