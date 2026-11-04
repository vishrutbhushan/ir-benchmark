#!/usr/bin/env python3
"""Filter and package the shared corpus consumed by both IR pipelines."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_pipeline_config, resolve_path


def _mark_done(path: Path, **data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"completed_at": time.time(), **data}, indent=2) + "\n")


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config/pipeline.json"
    config_parser.add_argument("--config", type=Path, default=default_config)
    config_args, _ = config_parser.parse_known_args()
    config = load_pipeline_config(config_args.config)
    settings = config["filter"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument(
        "--step",
        choices=["all", "validate", "repair", "filter", "pack"],
        default="all",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=resolve_path(paths["source_archive"]),
    )
    parser.add_argument("--corpus", type=Path, default=resolve_path(paths["corpus_dir"]))
    parser.add_argument("--output-dir", type=Path, default=resolve_path(paths["filter_output_dir"]))
    parser.add_argument("--logs-dir", type=Path, default=resolve_path(paths["filter_logs_dir"]))
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument("--force", action="store_true", help="Re-run completed stages.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    from src.common.artifacts import records
    from src.common.logging import setup_logger
    from src.fix import (
        apply_exclusions,
        fix_dataset,
        validate_dataset,
        REVIEWED_EXCLUSIONS,
    )
    from src.validation.runner import run_validation
    from src.repair.repair import run_repairs
    from src.filter import run_extraction

    validation = args.output_dir / "stage_02_validation"
    corrected = (args.corpus / "metadata/corpus_state.json").exists()
    if args.step == "pack":
        from src.archive import package_corpus

        archive_path = args.output_dir / "stage_05_archive/technical_web_corpus.zip"
        marker = archive_path.with_suffix(".done.json")
        if marker.exists() and not args.force:
            print(f"Archive already complete; reusing {archive_path}. Use --force to rerun.")
            return
        package_corpus(
            args.corpus,
            archive_path,
            args.workers,
        )
        _mark_done(marker, stage="archive", archive=str(archive_path))
        return
    if corrected:
        apply_exclusions(
            args.corpus, REVIEWED_EXCLUSIONS, args.output_dir / "recovery/quality"
        )
        validate_dataset(args.corpus, args.workers)
        repairs = {}
    else:
        manifest = args.corpus / "metadata/corpus_manifest.jsonl"
        if not args.archive.exists():
            if not manifest.exists():
                raise FileNotFoundError(
                    "Provide the original ZIP or run the collect step first"
                )
            from src.archive import snapshot_downloads

            snapshot_downloads(args.corpus, args.archive)
        if manifest.exists():
            import shutil

            backup = args.output_dir / "recovery/downloaded_manifest.jsonl"
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(manifest, backup)
            manifest.unlink()
        run_validation(
            args.archive,
            args.corpus,
            validation,
            setup_logger("filter", args.logs_dir / "filter.log"),
            args.workers,
        )
        if args.step == "validate":
            return
        repairs = run_repairs(
            args.corpus, validation, args.output_dir / "step_03_repair", args.workers
        )
    if args.step in {"validate", "repair"}:
        return
    extraction = args.output_dir / "stage_04_filtering"
    marker = extraction / "stage.done.json"
    if marker.exists() and not args.force:
        print(f"Filtering already complete; reusing {extraction}. Use --force to rerun.")
    else:
        run_extraction(args.corpus, validation, extraction, repairs, args.workers)
        _mark_done(marker, stage="filtering", output=str(extraction))
    if corrected:
        exclusions = {
            r["doc_id"]: r["reason"]
            for r in records(extraction / "decisions.jsonl")
            if r["status"] == "excluded"
        }
        apply_exclusions(args.corpus, exclusions, args.output_dir / "recovery/content")
    else:
        fix_dataset(args.corpus, extraction, args.output_dir / "recovery", args.workers)
    if args.step == "filter":
        return
    from src.archive import package_corpus

    archive_path = args.output_dir / "stage_05_archive/technical_web_corpus.zip"
    archive_marker = archive_path.with_suffix(".done.json")
    if archive_marker.exists() and not args.force:
        print(f"Archive already complete; reusing {archive_path}.")
        return
    package_corpus(
        args.corpus,
        archive_path,
        args.workers,
    )
    _mark_done(archive_marker, stage="archive", archive=str(archive_path))


if __name__ == "__main__":
    main()
