#!/usr/bin/env python3
"""CLI entrypoint and 4-stage orchestrator for the YouTube English-to-Thai Dubbing Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ingest import ASRProvider, DownloaderFn, create_asr_provider, run_ingest
from src.mux import run_mux
from src.state import (
    STAGE_ORDER,
    PipelineConfig,
    PipelineError,
    StageError,
    StageName,
    extract_video_id,
    get_workspace_paths,
    init_or_load_manifest,
    reset_stages_from,
    save_manifest,
    set_stage_status,
    should_skip_stage,
)
from src.translate import Translator, create_translator, run_translate
from src.tts_align import TTSProvider, create_tts_provider, run_tts_align

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="YouTube English-to-Thai Dubbing Pipeline (4-stage offline batch orchestrator)",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube video URL (youtu.be, watch?v=, /shorts/, /embed/, /live/)",
    )
    parser.add_argument(
        "--mix-mode",
        choices=("duck", "replace"),
        default="duck",
        help="Stage 4 audio mixing mode (default: duck)",
    )
    parser.add_argument(
        "--force-stage",
        choices=STAGE_ORDER,
        default=None,
        help="Force re-execution of the specified stage and cascade-reset downstream stages",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work"),
        help="Root directory for per-video workspaces (default: work)",
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=0.18,
        help="Original audio volume in duck mode, 0.0 to 1.0 (default: 0.18)",
    )
    parser.add_argument(
        "--max-atempo",
        type=float,
        default=1.25,
        help="Maximum ffmpeg atempo speed-up factor >= 1.0 (default: 1.25)",
    )
    parser.add_argument(
        "--tts-provider",
        default="pythaitts",
        help="Thai TTS provider name: pythaitts | edge_tts (default: pythaitts)",
    )
    parser.add_argument(
        "--voice-id",
        default=None,
        help="Optional locked voice/speaker ID for the TTS provider",
    )
    parser.add_argument(
        "--translator",
        default="auto",
        help="Translator backend: auto | openai | deep_translator (default: auto)",
    )
    parser.add_argument(
        "--asr-provider",
        default="faster_whisper",
        help="ASR backend: faster_whisper (default: faster_whisper)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser


def run_pipeline(
    url: str,
    config: PipelineConfig,
    *,
    work_dir: Path | str = Path("work"),
    force_stage: StageName | None = None,
    downloader: DownloaderFn | None = None,
    asr_provider: ASRProvider | None = None,
    translator: Translator | None = None,
    tts_provider: TTSProvider | None = None,
) -> Path:
    """Orchestrate Stages 1-4 with atomic checkpointing, cascade reset, and error capture."""
    video_id = extract_video_id(url)
    paths = get_workspace_paths(work_dir, video_id)
    manifest = init_or_load_manifest(paths, video_id, url, config)

    if force_stage is not None:
        logger.info("Force-stage requested: resetting from %s onward", force_stage)
        reset_stages_from(paths, manifest, force_stage)

    force_idx = STAGE_ORDER.index(force_stage) if force_stage is not None else None

    for idx, stage in enumerate(STAGE_ORDER):
        stage_name: StageName = stage  # type: ignore[assignment]
        if should_skip_stage(paths, manifest, stage_name, force_stage=force_stage, config=config):
            logger.info("Skipping verified completed stage: %s", stage_name)
            continue

        logger.info("Starting stage: %s", stage_name)
        reset_stages_from(paths, manifest, stage_name)
        set_stage_status(paths, manifest, stage_name, "in_progress")

        try:
            if stage_name == "ingest":
                asr = asr_provider or create_asr_provider(config.asr_provider)
                run_ingest(paths, url, asr_provider=asr, downloader=downloader)
            elif stage_name == "translate":
                tr = translator or create_translator(config.translator)
                run_translate(paths, tr)
            elif stage_name == "tts_align":
                tts = tts_provider or create_tts_provider(
                    config.tts_provider, voice_id=config.voice_id
                )
                force_tts = force_idx is not None and idx >= force_idx
                run_tts_align(paths, tts, max_atempo=config.max_atempo, force=force_tts)
            elif stage_name == "mux":
                run_mux(
                    paths,
                    mix_mode=config.mix_mode,
                    original_volume=config.original_volume,
                )
        except Exception as exc:
            set_stage_status(paths, manifest, stage_name, "failed", error=str(exc))
            if isinstance(exc, StageError):
                raise
            raise StageError(stage_name, str(exc), cause=exc) from exc

        set_stage_status(paths, manifest, stage_name, "done")
        logger.info("Completed stage: %s", stage_name)

    manifest["last_error"] = None
    save_manifest(paths, manifest)
    return paths.output_video


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = PipelineConfig(
            mix_mode=args.mix_mode,
            original_volume=args.original_volume,
            max_atempo=args.max_atempo,
            tts_provider=args.tts_provider,
            translator=args.translator,
            asr_provider=args.asr_provider,
            voice_id=args.voice_id,
        )
        output_path = run_pipeline(
            url=args.url,
            config=config,
            work_dir=args.work_dir,
            force_stage=args.force_stage,
        )
        logger.info("Pipeline succeeded: %s", output_path)
        return 0
    except PipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected pipeline error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
