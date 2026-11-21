"""State management, atomic file I/O, manifest handling, and media integrity verification."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict
from urllib.parse import parse_qs, urlparse


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class ValidationError(PipelineError):
    """Invalid CLI input, URL, config, or schema."""


class MediaError(PipelineError):
    """ffmpeg/ffprobe invocation error or corrupt media file."""


class ProviderError(PipelineError):
    """External/model provider failure (yt-dlp, ASR, Translator, TTS)."""


class StageError(PipelineError):
    """Wraps a stage execution failure with stage name context."""

    def __init__(self, stage: str, message: str, cause: Exception | None = None) -> None:
        self.stage = stage
        self.message = message
        self.cause = cause
        super().__init__(f"[{stage}] {message}")


STAGE_ORDER: tuple[str, ...] = ("ingest", "translate", "tts_align", "mux")
StageName = Literal["ingest", "translate", "tts_align", "mux"]
StageStatus = Literal["pending", "in_progress", "done", "failed"]
MixMode = Literal["duck", "replace"]

_YOUTUBE_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{11}\Z")
_SAFE_VIDEO_ID_RE = re.compile(r"\A[A-Za-z0-9_-]+\Z")
_YOUTUBE_HOSTS = frozenset(
    {
        "youtu.be",
        "www.youtu.be",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)
_SOUNDFILE_EXTS = frozenset({".wav", ".flac", ".ogg", ".aiff", ".aif"})


@dataclass(frozen=True)
class PipelineConfig:
    mix_mode: MixMode = "duck"
    original_volume: float = 0.18
    max_atempo: float = 1.25
    tts_provider: str = "pythaitts"
    translator: str = "auto"
    asr_provider: str = "faster_whisper"
    voice_id: str | None = None

    def __post_init__(self) -> None:
        if self.mix_mode not in ("duck", "replace"):
            raise ValidationError(
                f"Invalid mix_mode: {self.mix_mode!r}. Expected 'duck' or 'replace'."
            )
        if (
            isinstance(self.original_volume, bool)
            or not isinstance(self.original_volume, (int, float))
            or not math.isfinite(self.original_volume)
        ):
            raise ValidationError(f"Invalid original_volume: {self.original_volume!r}.")
        if self.original_volume < 0.0 or self.original_volume > 1.0:
            raise ValidationError(
                f"original_volume must be between 0.0 and 1.0, got {self.original_volume}."
            )
        if (
            isinstance(self.max_atempo, bool)
            or not isinstance(self.max_atempo, (int, float))
            or not math.isfinite(self.max_atempo)
        ):
            raise ValidationError(f"Invalid max_atempo: {self.max_atempo!r}.")
        if self.max_atempo < 1.0:
            raise ValidationError(f"max_atempo must be >= 1.0, got {self.max_atempo}.")
        if not self.tts_provider or not isinstance(self.tts_provider, str):
            raise ValidationError("tts_provider must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mix_mode": self.mix_mode,
            "original_volume": float(self.original_volume),
            "max_atempo": float(self.max_atempo),
            "tts_provider": self.tts_provider,
            "translator": self.translator,
            "asr_provider": self.asr_provider,
        }
        if self.voice_id is not None:
            data["voice_id"] = self.voice_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        if not isinstance(data, dict):
            raise ValidationError("Config data must be a dictionary.")
        raw_vol = data.get("original_volume", 0.18)
        if isinstance(raw_vol, bool):
            raise ValidationError(f"Invalid original_volume: {raw_vol!r}.")
        try:
            orig_vol = float(raw_vol)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid original_volume: {raw_vol!r}.") from exc

        raw_atempo = data.get("max_atempo", 1.25)
        if isinstance(raw_atempo, bool):
            raise ValidationError(f"Invalid max_atempo: {raw_atempo!r}.")
        try:
            max_atempo = float(raw_atempo)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid max_atempo: {raw_atempo!r}.") from exc

        return cls(
            mix_mode=data.get("mix_mode", "duck"),
            original_volume=orig_vol,
            max_atempo=max_atempo,
            tts_provider=str(data.get("tts_provider", "pythaitts")),
            translator=str(data.get("translator", "auto")),
            asr_provider=str(data.get("asr_provider", "faster_whisper")),
            voice_id=data.get("voice_id"),
        )


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    manifest: Path
    source_video: Path
    audio_en: Path
    segments_en: Path
    segments_th: Path
    tts_segments_dir: Path
    thai_dub_full: Path
    output_video: Path

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tts_segments_dir.mkdir(parents=True, exist_ok=True)

    def segment_wav_path(self, seg_id: int) -> Path:
        if seg_id < 1:
            raise ValidationError(f"Segment id must be >= 1, got {seg_id}.")
        return self.tts_segments_dir / f"seg_{seg_id:04d}.wav"

    def segment_wav_relpath(self, seg_id: int) -> str:
        if seg_id < 1:
            raise ValidationError(f"Segment id must be >= 1, got {seg_id}.")
        return f"tts_segments/seg_{seg_id:04d}.wav"


class SegmentENDict(TypedDict):
    id: int
    start: float
    end: float
    text_en: str


class TTSInfoDict(TypedDict, total=False):
    status: Literal["pending", "done", "failed"]
    file: str | None
    duration: float | None
    tempo_used: float | None
    error: str | None


class SegmentTHDict(TypedDict):
    id: int
    start: float
    end: float
    text_en: str
    text_th: str
    tts: TTSInfoDict


def extract_video_id(url: str) -> str:
    """Extract and validate an 11-character YouTube video ID from a URL.

    Supports youtu.be, watch?v=, /shorts/, /embed/, /live/, and /v/ URLs.
    Rejects non-YouTube URLs, missing IDs, and path-traversal / unsafe inputs.
    """
    if not isinstance(url, str):
        raise ValidationError("URL must be a string.")
    raw = url.strip()
    if not raw or ".." in raw or "\x00" in raw:
        raise ValidationError(f"Invalid YouTube URL: {url!r}")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(f"YouTube URL must use http or https scheme: {url!r}")

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise ValidationError(f"Unsupported host {host!r} in YouTube URL: {url!r}")

    path_parts = [p for p in parsed.path.split("/") if p]
    video_id: str | None = None

    if host in ("youtu.be", "www.youtu.be"):
        if path_parts:
            video_id = path_parts[0]
    else:
        if path_parts and path_parts[0] == "watch":
            qs = parse_qs(parsed.query)
            v_vals = qs.get("v")
            if v_vals:
                video_id = v_vals[0]
        elif len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live", "v"):
            video_id = path_parts[1]
        elif not path_parts:
            qs = parse_qs(parsed.query)
            v_vals = qs.get("v")
            if v_vals:
                video_id = v_vals[0]

    if not video_id or not _YOUTUBE_ID_RE.match(video_id):
        raise ValidationError(f"Could not extract a valid YouTube video_id from URL: {url!r}")

    return video_id


def get_workspace_paths(work_dir: Path | str, video_id: str) -> WorkspacePaths:
    """Construct isolated workspace paths under `<work_dir>/<video_id>/`."""
    if (
        not isinstance(video_id, str)
        or not video_id
        or ".." in video_id
        or "/" in video_id
        or "\\" in video_id
        or not _SAFE_VIDEO_ID_RE.match(video_id)
    ):
        raise ValidationError(f"Unsafe or invalid video_id: {video_id!r}")

    root = Path(work_dir) / video_id
    return WorkspacePaths(
        root=root,
        manifest=root / "manifest.json",
        source_video=root / "source.mp4",
        audio_en=root / "audio_en.wav",
        segments_en=root / "segments_en.json",
        segments_th=root / "segments_th.json",
        tts_segments_dir=root / "tts_segments",
        thai_dub_full=root / "thai_dub_full.wav",
        output_video=root / "output_th.mp4",
    )


@contextmanager
def atomic_temp_path(target: Path | str) -> Iterator[Path]:
    """Yield a sibling temp path preserving target file extension for ffmpeg compatibility.

    On clean exit, atomically replaces `target` via `os.replace(tmp, target)`.
    On exception, removes the temporary file if it exists and leaves `target` untouched.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix
    stem = target_path.stem
    unique = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp_path = target_path.parent / f".{stem}.{unique}.tmp{suffix}"
    try:
        yield tmp_path
        if not tmp_path.exists():
            raise PipelineError(f"Atomic write did not create temporary file: {tmp_path}")
        os.replace(tmp_path, target_path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Atomically write raw bytes to `path` with fsync."""
    with atomic_temp_path(path) as tmp_path:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to `path` with fsync."""
    with atomic_temp_path(path) as tmp_path:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Atomically write UTF-8 JSON with ensure_ascii=False and fsync."""
    serialized = json.dumps(data, ensure_ascii=False, indent=indent) + "\n"
    atomic_write_text(path, serialized, encoding="utf-8")


def read_json(path: Path | str) -> Any:
    """Read and parse a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_json_file(
    path: Path | str,
    expected_type: type | None = None,
    allow_empty: bool = False,
) -> bool:
    """Return True iff `path` exists, is non-empty, valid JSON, and matches expected_type."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size <= 0:
            return False
        data = read_json(p)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False

    if expected_type is not None and not isinstance(data, expected_type):
        return False
    if not allow_empty and isinstance(data, (list, dict, str)) and len(data) == 0:
        return False
    return True


def probe_media_info(path: Path | str) -> dict[str, Any] | None:
    """Run ffprobe to inspect streams and format metadata; return parsed dict or None."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size <= 0:
            return None
    except OSError:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(p),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    return info if isinstance(info, dict) else None


def _extract_duration_from_ffprobe(info: dict[str, Any]) -> float | None:
    fmt = info.get("format") or {}
    raw_dur = fmt.get("duration")
    if raw_dur is not None:
        try:
            dur = float(raw_dur)
            if math.isfinite(dur) and dur > 0.0:
                return dur
        except (TypeError, ValueError):
            pass

    for stream in info.get("streams") or []:
        s_dur = stream.get("duration")
        if s_dur is not None:
            try:
                dur = float(s_dur)
                if math.isfinite(dur) and dur > 0.0:
                    return dur
            except (TypeError, ValueError):
                continue
    return None


def probe_media_duration(path: Path | str) -> float | None:
    """Return positive duration in seconds via soundfile (for WAV/FLAC/OGG) or ffprobe; else None."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size <= 0:
            return None
    except OSError:
        return None

    if p.suffix.lower() in _SOUNDFILE_EXTS:
        try:
            import soundfile as sf

            info = sf.info(str(p))
            if info.frames > 0 and info.samplerate > 0 and info.duration > 0.0:
                return float(info.duration)
            return None
        except Exception:
            pass

    ff_info = probe_media_info(p)
    if ff_info is None:
        return None
    return _extract_duration_from_ffprobe(ff_info)


def has_audio_stream(path: Path | str) -> bool:
    """Return True iff ffprobe detects at least one audio stream in `path`."""
    info = probe_media_info(path)
    if info is None:
        return False
    streams = info.get("streams") or []
    return any(s.get("codec_type") == "audio" for s in streams)


def is_valid_audio_file(path: Path | str) -> bool:
    """Verify-before-skip check for audio files: exists, size > 0, readable, duration > 0."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    except OSError:
        return False

    if p.suffix.lower() in _SOUNDFILE_EXTS:
        try:
            import soundfile as sf

            info = sf.info(str(p))
            return bool(
                info.frames > 0
                and info.samplerate > 0
                and info.channels > 0
                and info.duration > 0.0
            )
        except Exception:
            return False

    ff_info = probe_media_info(p)
    if ff_info is None:
        return False
    streams = ff_info.get("streams") or []
    if not any(s.get("codec_type") == "audio" for s in streams):
        return False
    dur = _extract_duration_from_ffprobe(ff_info)
    return dur is not None and dur > 0.0


def is_valid_video_file(path: Path | str, require_audio: bool = False) -> bool:
    """Verify-before-skip check for video files: exists, size > 0, video stream, duration > 0."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    except OSError:
        return False

    ff_info = probe_media_info(p)
    if ff_info is None:
        return False

    streams = ff_info.get("streams") or []
    has_video = any(s.get("codec_type") == "video" for s in streams)
    if not has_video:
        return False
    if require_audio and not any(s.get("codec_type") == "audio" for s in streams):
        return False

    dur = _extract_duration_from_ffprobe(ff_info)
    return dur is not None and dur > 0.0


def _default_stages() -> dict[str, dict[str, str]]:
    return {stage: {"status": "pending"} for stage in STAGE_ORDER}


def _safe_float_differs(old_val: Any, new_val: float, tol: float = 1e-6) -> bool:
    """Return True if old_val is present and differs from new_val (or is corrupted/non-numeric)."""
    if old_val is None:
        return False
    if isinstance(old_val, bool):
        return True
    try:
        parsed = float(old_val)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(parsed):
        return True
    return abs(parsed - float(new_val)) > tol


def resolve_workspace_relpath(paths: WorkspacePaths, rel_file: Any) -> Path | None:
    """Resolve `rel_file` strictly inside `paths.root`, rejecting absolute paths and `..` escapes."""
    if not isinstance(rel_file, str) or not rel_file.strip() or "\x00" in rel_file:
        return None
    rel_path = Path(rel_file)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None
    try:
        root_resolved = paths.root.resolve()
        candidate = (paths.root / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root_resolved):
        return None
    return candidate


def init_or_load_manifest(
    paths: WorkspacePaths,
    video_id: str,
    source_url: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Load existing manifest.json or initialize a new one; detect config mismatches."""
    paths.ensure_dirs()
    new_cfg = config.to_dict()

    if is_valid_json_file(paths.manifest, expected_type=dict, allow_empty=False):
        manifest = read_json(paths.manifest)
        if not isinstance(manifest.get("stages"), dict):
            manifest["stages"] = _default_stages()
        else:
            for stage in STAGE_ORDER:
                if stage not in manifest["stages"] or not isinstance(
                    manifest["stages"][stage], dict
                ):
                    manifest["stages"][stage] = {"status": "pending"}

        raw_old_cfg = manifest.get("config")
        cfg_corrupted = raw_old_cfg is not None and not isinstance(raw_old_cfg, dict)
        old_cfg: dict[str, Any] = raw_old_cfg if isinstance(raw_old_cfg, dict) else {}

        old_max_atempo = old_cfg.get("max_atempo")
        old_tts_provider = old_cfg.get("tts_provider")
        old_voice_id = old_cfg.get("voice_id")
        tts_changed = (
            cfg_corrupted
            or _safe_float_differs(old_max_atempo, config.max_atempo)
            or (old_tts_provider is not None and old_tts_provider != config.tts_provider)
            or (old_voice_id != config.voice_id)
        )
        if tts_changed:
            manifest["stages"]["tts_align"]["status"] = "pending"
            manifest["stages"]["mux"]["status"] = "pending"
            if is_valid_json_file(paths.segments_th, expected_type=list, allow_empty=False):
                segments_th = read_json(paths.segments_th)
                for seg in segments_th:
                    if isinstance(seg, dict):
                        seg["tts"] = {
                            "status": "pending",
                            "file": None,
                            "duration": None,
                            "tempo_used": None,
                        }
                atomic_write_json(paths.segments_th, segments_th)

        old_mix_mode = old_cfg.get("mix_mode")
        old_orig_vol = old_cfg.get("original_volume")
        mux_changed = (
            cfg_corrupted
            or (old_mix_mode is not None and old_mix_mode != config.mix_mode)
            or _safe_float_differs(old_orig_vol, config.original_volume)
        )
        if mux_changed:
            manifest["stages"]["mux"]["status"] = "pending"

        manifest["video_id"] = video_id
        manifest["source_url"] = source_url
        manifest["config"] = new_cfg
        manifest.setdefault("last_error", None)
        save_manifest(paths, manifest)
        return manifest

    manifest = {
        "video_id": video_id,
        "source_url": source_url,
        "config": new_cfg,
        "stages": _default_stages(),
        "last_error": None,
    }
    save_manifest(paths, manifest)
    return manifest


def save_manifest(paths: WorkspacePaths, manifest: dict[str, Any]) -> None:
    """Atomically save manifest.json."""
    atomic_write_json(paths.manifest, manifest)


def set_stage_status(
    paths: WorkspacePaths,
    manifest: dict[str, Any],
    stage: StageName,
    status: StageStatus,
    error: str | None = None,
) -> None:
    """Update a stage status (and optional last_error) and persist manifest atomically."""
    if stage not in STAGE_ORDER:
        raise ValidationError(f"Unknown stage: {stage!r}")
    if not isinstance(manifest.get("stages"), dict):
        manifest["stages"] = _default_stages()
    manifest["stages"][stage] = {"status": status}
    if error is not None:
        manifest["last_error"] = error
    save_manifest(paths, manifest)


def reset_stages_from(
    paths: WorkspacePaths,
    manifest: dict[str, Any],
    start_stage: StageName,
) -> None:
    """Reset `start_stage` and all downstream stages in STAGE_ORDER to 'pending'."""
    if start_stage not in STAGE_ORDER:
        raise ValidationError(f"Unknown stage: {start_stage!r}")
    start_idx = STAGE_ORDER.index(start_stage)
    if not isinstance(manifest.get("stages"), dict):
        manifest["stages"] = _default_stages()
    for stage in STAGE_ORDER[start_idx:]:
        manifest["stages"][stage] = {"status": "pending"}
    save_manifest(paths, manifest)


def verify_stage_outputs(paths: WorkspacePaths, stage: StageName) -> bool:
    """Verify-before-skip artifact integrity check for the given stage."""
    if stage == "ingest":
        if not is_valid_video_file(paths.source_video):
            return False
        if not is_valid_audio_file(paths.audio_en):
            return False
        if not is_valid_json_file(paths.segments_en, expected_type=list, allow_empty=False):
            return False
        segs = read_json(paths.segments_en)
        return all(
            isinstance(s, dict)
            and isinstance(s.get("id"), int)
            and isinstance(s.get("start"), (int, float))
            and isinstance(s.get("end"), (int, float))
            and s["end"] > s["start"]
            and isinstance(s.get("text_en"), str)
            and bool(s["text_en"].strip())
            for s in segs
        )

    if stage == "translate":
        if not is_valid_json_file(paths.segments_th, expected_type=list, allow_empty=False):
            return False
        segs_th = read_json(paths.segments_th)
        if not all(
            isinstance(s, dict)
            and isinstance(s.get("id"), int)
            and isinstance(s.get("text_th"), str)
            and bool(s["text_th"].strip())
            for s in segs_th
        ):
            return False
        if is_valid_json_file(paths.segments_en, expected_type=list, allow_empty=False):
            segs_en = read_json(paths.segments_en)
            if [s.get("id") for s in segs_th] != [s.get("id") for s in segs_en]:
                return False
        return True

    if stage == "tts_align":
        if not is_valid_json_file(paths.segments_th, expected_type=list, allow_empty=False):
            return False
        segs_th = read_json(paths.segments_th)
        for seg in segs_th:
            if not isinstance(seg, dict):
                return False
            tts = seg.get("tts")
            if not isinstance(tts, dict) or tts.get("status") != "done":
                return False
            wav_path = resolve_workspace_relpath(paths, tts.get("file"))
            if wav_path is None or not is_valid_audio_file(wav_path):
                return False
        return True

    if stage == "mux":
        return is_valid_audio_file(paths.thai_dub_full) and is_valid_video_file(
            paths.output_video, require_audio=True
        )

    raise ValidationError(f"Unknown stage: {stage!r}")


def should_skip_stage(
    paths: WorkspacePaths,
    manifest: dict[str, Any],
    stage: StageName,
    force_stage: StageName | None = None,
    config: PipelineConfig | None = None,
) -> bool:
    """Return True iff stage is not forced, manifest says done, config matches, and outputs verify."""
    if stage not in STAGE_ORDER:
        raise ValidationError(f"Unknown stage: {stage!r}")
    if force_stage is not None:
        if force_stage not in STAGE_ORDER:
            raise ValidationError(f"Unknown force_stage: {force_stage!r}")
        if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(force_stage):
            return False

    if not isinstance(manifest, dict):
        return False
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        return False
    stage_obj = stages.get(stage)
    if not isinstance(stage_obj, dict):
        return False
    if stage_obj.get("status") != "done":
        return False

    if config is not None:
        m_cfg = manifest.get("config")
        if not isinstance(m_cfg, dict):
            return False
        if stage == "tts_align":
            if "max_atempo" in m_cfg and _safe_float_differs(m_cfg["max_atempo"], config.max_atempo):
                return False
            if "tts_provider" in m_cfg and m_cfg["tts_provider"] != config.tts_provider:
                return False
            if m_cfg.get("voice_id") != config.voice_id:
                return False
        elif stage == "mux":
            if m_cfg.get("mix_mode") != config.mix_mode:
                return False
            if "original_volume" in m_cfg and _safe_float_differs(
                m_cfg["original_volume"], config.original_volume
            ):
                return False

    return verify_stage_outputs(paths, stage)
