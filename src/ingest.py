"""Stage 1: Video download (yt-dlp), 16kHz mono WAV extraction, subtitle parsing, segmentation, and Whisper ASR."""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from src.state import (
    MediaError,
    ProviderError,
    SegmentENDict,
    StageError,
    WorkspacePaths,
    atomic_temp_path,
    atomic_write_json,
    is_valid_audio_file,
    is_valid_video_file,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DownloadResult:
    video_path: Path
    subtitle_path: Path | None
    subtitle_is_auto: bool = False


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path, language: str = "en") -> list[RawCue]:
        ...


DownloaderFn = Callable[[str, Path, Path], DownloadResult]

_VTT_TIME_RE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}\.\d{3})"
)
_SRT_TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,\.]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def _parse_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        hours, mins, secs = int(parts[0]), int(parts[1]), float(parts[2])
        return hours * 3600.0 + mins * 60.0 + secs
    if len(parts) == 2:
        mins, secs = int(parts[0]), float(parts[1])
        return mins * 60.0 + secs
    raise ValueError(f"Invalid timestamp: {ts!r}")


def _clean_caption_text(raw: str) -> str:
    text = _TAG_RE.sub("", raw)
    text = html.unescape(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _deduplicate_rolling_cues(cues: list[RawCue]) -> list[RawCue]:
    """Remove duplicate/rolling lines common in YouTube auto-generated VTT captions."""
    if not cues:
        return []

    deduped: list[RawCue] = []
    for cue in cues:
        text = cue.text.strip()
        if not text or cue.end <= cue.start:
            continue

        if deduped:
            prev = deduped[-1]
            if text == prev.text:
                # Extend previous cue end if identical text repeats consecutively
                deduped[-1] = RawCue(
                    start=prev.start,
                    end=max(prev.end, cue.end),
                    text=prev.text,
                )
                continue

            # Rolling caption: new cue starts with the entire previous cue text
            if text.startswith(prev.text + " "):
                suffix = text[len(prev.text) :].strip()
                if suffix:
                    deduped.append(RawCue(start=cue.start, end=cue.end, text=suffix))
                continue

            # Rolling caption: overlap where previous cue ends with prefix of new cue
            prev_words = prev.text.split()
            curr_words = text.split()
            max_ov = min(len(prev_words), len(curr_words))
            overlap_len = 0
            for k in range(max_ov, 0, -1):
                if prev_words[-k:] == curr_words[:k]:
                    overlap_len = k
                    break
            if overlap_len >= 2 or (overlap_len == len(curr_words) and overlap_len > 0):
                remaining = " ".join(curr_words[overlap_len:]).strip()
                if not remaining:
                    deduped[-1] = RawCue(
                        start=prev.start,
                        end=max(prev.end, cue.end),
                        text=prev.text,
                    )
                    continue
                deduped.append(RawCue(start=cue.start, end=cue.end, text=remaining))
                continue

        deduped.append(RawCue(start=cue.start, end=cue.end, text=text))

    return deduped


def parse_vtt(vtt_text: str | Path) -> list[RawCue]:
    """Parse WebVTT subtitle text (or file path) into cleaned, deduplicated RawCue entries."""
    if isinstance(vtt_text, Path):
        vtt_text = vtt_text.read_text(encoding="utf-8", errors="replace")
    lines = vtt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    raw_cues: list[RawCue] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line or line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            if line.startswith(("NOTE", "STYLE", "REGION")):
                while i < n and lines[i].strip():
                    i += 1
            i += 1
            continue

        match = _VTT_TIME_RE.search(line)
        if not match:
            i += 1
            continue

        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        i += 1
        text_lines: list[str] = []
        while i < n and lines[i].strip():
            cleaned = _clean_caption_text(lines[i])
            if cleaned and (not text_lines or cleaned != text_lines[-1]):
                text_lines.append(cleaned)
            i += 1

        text = _MULTI_SPACE_RE.sub(" ", " ".join(text_lines)).strip()
        if text and end > start:
            raw_cues.append(RawCue(start=start, end=end, text=text))

    return _deduplicate_rolling_cues(raw_cues)


def parse_srt(srt_text: str | Path) -> list[RawCue]:
    """Parse SubRip (.srt) subtitle text (or file path) into cleaned, deduplicated RawCue entries."""
    if isinstance(srt_text, Path):
        srt_text = srt_text.read_text(encoding="utf-8", errors="replace")
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n+", normalized.strip())
    raw_cues: list[RawCue] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_idx = -1
        match = None
        for idx, ln in enumerate(lines[:2]):
            m = _SRT_TIME_RE.search(ln)
            if m:
                time_idx = idx
                match = m
                break
        if match is None:
            continue

        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        text_lines = [_clean_caption_text(ln) for ln in lines[time_idx + 1 :]]
        text = _MULTI_SPACE_RE.sub(" ", " ".join(t for t in text_lines if t)).strip()
        if text and end > start:
            raw_cues.append(RawCue(start=start, end=end, text=text))

    return _deduplicate_rolling_cues(raw_cues)


def parse_subtitle_file(sub_path: Path) -> list[RawCue]:
    """Read and parse a .vtt or .srt subtitle file."""
    path = Path(sub_path)
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    content = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".vtt" or content.lstrip().startswith("WEBVTT"):
        return parse_vtt(content)
    if suffix == ".srt":
        return parse_srt(content)
    # Fallback: try VTT then SRT
    cues = parse_vtt(content)
    return cues if cues else parse_srt(content)


def _split_long_cue(cue: RawCue, max_dur: float) -> list[RawCue]:
    """Split a single cue longer than `max_dur` at sentence/clause/word boundaries."""
    dur = cue.end - cue.start
    if dur <= max_dur:
        return [cue]

    text = cue.text.strip()
    pieces = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(pieces) == 1:
        pieces = [p.strip() for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if len(pieces) == 1:
        words = text.split()
        n_chunks = max(2, int(-(-dur // max_dur)))  # ceil(dur / max_dur)
        if len(words) < n_chunks:
            return [cue]
        chunk_size = max(1, -(-len(words) // n_chunks))
        pieces = [
            " ".join(words[j : j + chunk_size]).strip()
            for j in range(0, len(words), chunk_size)
        ]
        pieces = [p for p in pieces if p]

    if len(pieces) <= 1:
        return [cue]

    total_chars = sum(len(p) for p in pieces)
    if total_chars == 0:
        return [cue]

    sub_cues: list[RawCue] = []
    cursor = cue.start
    for idx, piece in enumerate(pieces):
        if idx == len(pieces) - 1:
            end = cue.end
        else:
            frac = len(piece) / total_chars
            end = cursor + dur * frac
        if end > cursor:
            sub = RawCue(start=cursor, end=end, text=piece)
            if (sub.end - sub.start) > max_dur and len(piece.split()) > 2:
                sub_cues.extend(_split_long_cue(sub, max_dur))
            else:
                sub_cues.append(sub)
        cursor = end

    return sub_cues


def segment_cues(
    cues: list[RawCue],
    min_dur: float = 3.0,
    max_dur: float = 8.0,
    pause_threshold: float = 0.8,
) -> list[SegmentENDict]:
    """Merge short fragments and split >max_dur cues into 3-8s sentence-level segments."""
    valid = [
        RawCue(start=max(0.0, float(c.start)), end=float(c.end), text=c.text.strip())
        for c in cues
        if c.text and c.text.strip() and float(c.end) > max(0.0, float(c.start))
    ]
    if not valid:
        return []

    valid.sort(key=lambda c: (c.start, c.end))

    # 1. Split overly long cues first
    expanded: list[RawCue] = []
    for cue in valid:
        expanded.extend(_split_long_cue(cue, max_dur=max_dur))

    # 2. Merge short consecutive cues into sentence-level 3-8s segments
    merged: list[RawCue] = []
    curr = expanded[0]
    for nxt in expanded[1:]:
        curr_dur = curr.end - curr.start
        gap = nxt.start - curr.end
        combined_dur = max(curr.end, nxt.end) - curr.start
        ends_sentence = bool(curr.text) and curr.text[-1] in ".!?"

        should_merge = (
            gap <= pause_threshold
            and combined_dur <= max_dur
            and (curr_dur < min_dur or not ends_sentence)
        )
        if should_merge:
            curr = RawCue(
                start=curr.start,
                end=max(curr.end, nxt.end),
                text=f"{curr.text} {nxt.text}".strip(),
            )
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)

    # 3. Guarantee monotonic non-overlapping timestamps and 1-based IDs
    segments: list[SegmentENDict] = []
    prev_end = 0.0
    for seg_id, cue in enumerate(merged, start=1):
        start = round(max(prev_end, cue.start), 2)
        end = round(cue.end, 2)
        if end <= start:
            end = round(start + 0.05, 2)
        segments.append(
            {
                "id": seg_id,
                "start": start,
                "end": end,
                "text_en": cue.text.strip(),
            }
        )
        prev_end = end

    return segments


def download_video_and_subs(
    url: str,
    output_video: Path,
    sub_dir: Path,
) -> DownloadResult:
    """Download best MP4-compatible video via yt-dlp atomically and fetch EN subtitles."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise ProviderError("yt-dlp is not installed.") from exc

    output_video = Path(output_video)
    sub_dir = Path(sub_dir)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    sub_path: Path | None = None
    sub_is_auto = False

    try:
        with atomic_temp_path(output_video) as tmp_video:
            # yt-dlp writes to outtmpl; use tmp_video directly as output target
            ydl_opts: dict[str, Any] = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "merge_output_format": "mp4",
                "outtmpl": str(tmp_video),
                "overwrites": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
                "subtitlesformat": "vtt/srt/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            # yt-dlp may append .mp4 if tmp_video already had .mp4 or create sibling
            if not tmp_video.exists():
                alt = Path(f"{tmp_video}.mp4")
                if alt.exists():
                    os.replace(alt, tmp_video)

            # Collect any downloaded subtitles written next to tmp_video and move to sub_dir
            requested_subs = (info or {}).get("requested_subtitles") or {}
            manual_subs = (info or {}).get("subtitles") or {}
            for lang in ("en", "en-US", "en-GB"):
                if lang in requested_subs:
                    sub_info = requested_subs[lang]
                    fp = sub_info.get("filepath")
                    if fp and Path(fp).is_file():
                        is_auto = lang not in manual_subs
                        prefix = "sub_auto_" if is_auto else "sub_"
                        dest = sub_dir / f"{prefix}{lang}{Path(fp).suffix}"
                        os.replace(fp, dest)
                        sub_path = dest
                        sub_is_auto = is_auto
                        break

            # Fallback scan for any .vtt/.srt files placed next to tmp_video
            if sub_path is None:
                for candidate in sorted(tmp_video.parent.glob(f"{tmp_video.stem}*.vtt")) + sorted(
                    tmp_video.parent.glob(f"{tmp_video.stem}*.srt")
                ):
                    dest = sub_dir / f"sub_auto_{candidate.name}"
                    os.replace(candidate, dest)
                    if sub_path is None:
                        sub_path = dest
                        sub_is_auto = True

    except Exception as exc:
        raise ProviderError(f"yt-dlp download failed for {url}: {exc}") from exc

    if not is_valid_video_file(output_video):
        raise MediaError(f"Downloaded video is invalid or unreadable: {output_video}")

    return DownloadResult(
        video_path=output_video,
        subtitle_path=sub_path,
        subtitle_is_auto=sub_is_auto,
    )


def extract_audio_16k_mono(video_path: Path, output_wav: Path) -> Path:
    """Extract 16kHz mono PCM16 WAV from `video_path` using ffmpeg atomically."""
    video_path = Path(video_path)
    output_wav = Path(output_wav)
    if not is_valid_video_file(video_path):
        raise MediaError(f"Invalid source video for audio extraction: {video_path}")

    with atomic_temp_path(output_wav) as tmp_wav:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(tmp_wav),
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
                f"ffmpeg audio extraction failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )

    if not is_valid_audio_file(output_wav):
        raise MediaError(f"Extracted audio file failed validation: {output_wav}")

    return output_wav


class FasterWhisperASR:
    """CPU-friendly Whisper ASR provider backed by faster-whisper (CTranslate2 INT8)."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ProviderError("faster-whisper is not installed.") from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str = "en") -> list[RawCue]:
        if not is_valid_audio_file(audio_path):
            raise MediaError(f"Invalid audio file for ASR: {audio_path}")
        try:
            model = self._get_model()
            segments_iter, _info = model.transcribe(str(audio_path), language=language)
            cues: list[RawCue] = []
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if text and seg.end > seg.start:
                    cues.append(RawCue(start=float(seg.start), end=float(seg.end), text=text))
            return cues
        except Exception as exc:
            if isinstance(exc, (MediaError, ProviderError)):
                raise
            raise ProviderError(f"FasterWhisperASR failed: {exc}") from exc


def create_asr_provider(name: str = "faster_whisper", **kwargs: Any) -> ASRProvider:
    """Factory for ASR providers."""
    normalized = (name or "faster_whisper").strip().lower()
    if normalized in ("faster_whisper", "faster-whisper", "whisper"):
        return FasterWhisperASR(**kwargs)
    raise ProviderError(f"Unsupported ASR provider: {name!r}")


def run_ingest(
    paths: WorkspacePaths,
    url: str,
    *,
    asr_provider: ASRProvider | None = None,
    downloader: DownloaderFn | None = None,
    prefer_subtitles: bool = True,
    min_seg_dur: float = 3.0,
    max_seg_dur: float = 8.0,
) -> list[SegmentENDict]:
    """Execute Stage 1 with intra-stage resume, subtitle-first + ASR fallback, and atomic writes."""
    paths.ensure_dirs()
    sub_dir = paths.root / "subs"
    dl_fn = downloader or download_video_and_subs
    dl_result: DownloadResult | None = None

    try:
        # 1. Video download (skip if source.mp4 already valid)
        if not is_valid_video_file(paths.source_video):
            dl_result = dl_fn(url, paths.source_video, sub_dir)
            if not is_valid_video_file(paths.source_video):
                raise MediaError(f"Downloaded video failed integrity check: {paths.source_video}")
        else:
            logger.info("Stage 1: reusing verified source video %s", paths.source_video)

        # 2. Extract 16kHz mono WAV (skip if audio_en.wav already valid)
        if not is_valid_audio_file(paths.audio_en):
            extract_audio_16k_mono(paths.source_video, paths.audio_en)
        else:
            logger.info("Stage 1: reusing verified audio %s", paths.audio_en)

        # 3. Obtain raw timed cues (prefer accurate manual EN subtitles, else ASR fallback)
        cues: list[RawCue] = []
        sub_path: Path | None = None
        sub_is_auto = False

        if dl_result is not None and dl_result.subtitle_path is not None:
            sub_path = Path(dl_result.subtitle_path)
            sub_is_auto = dl_result.subtitle_is_auto
            if (
                sub_is_auto
                and sub_path.is_file()
                and sub_path.parent.resolve() == sub_dir.resolve()
                and not sub_path.name.startswith("sub_auto_")
            ):
                tagged = sub_dir / f"sub_auto_{sub_path.name}"
                os.replace(sub_path, tagged)
                sub_path = tagged
        elif sub_dir.is_dir():
            candidates = sorted(sub_dir.glob("*.vtt")) + sorted(sub_dir.glob("*.srt"))
            manual_candidates = [p for p in candidates if not p.name.startswith("sub_auto_")]
            if manual_candidates:
                sub_path = manual_candidates[0]
                sub_is_auto = False
            elif candidates:
                sub_path = candidates[0]
                sub_is_auto = True

        if prefer_subtitles and sub_path is not None and not sub_is_auto:
            cues = parse_subtitle_file(sub_path)
            logger.info("Stage 1: parsed %d cues from subtitle %s", len(cues), sub_path)

        if not cues:
            provider = asr_provider or create_asr_provider("faster_whisper")
            logger.info("Stage 1: running ASR fallback on %s", paths.audio_en)
            cues = provider.transcribe(paths.audio_en, language="en")

        # 4. Segment into 3-8s sentence-level chunks
        segments = segment_cues(cues, min_dur=min_seg_dur, max_dur=max_seg_dur)
        if not segments:
            raise StageError("ingest", "No speech segments extracted from subtitles or ASR.")

        atomic_write_json(paths.segments_en, segments)
        return segments

    except StageError:
        raise
    except Exception as exc:
        raise StageError("ingest", str(exc), cause=exc) from exc
