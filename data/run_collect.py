#!/usr/bin/env python3
"""Collect, validate, and repair the source corpus."""

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


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("../config/pipeline.json"))
    config_args, _ = config_parser.parse_known_args()
    config = load_pipeline_config(config_args.config)
    settings = config["data"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--step", choices=["all", "collect", "validate", "repair"], default="all")
    parser.add_argument("--archive", type=Path, default=resolve_path(settings["archive"]))
    parser.add_argument("--corpus", type=Path, default=resolve_path(paths["corpus_dir"]))
    parser.add_argument("--candidates", type=Path, default=Path("runtime/common_crawl_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/collection"))
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument("--force", action="store_true", help="Re-run completed collection stages.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.step == "collect":
        marker = args.output_dir / "stage_01_collection.done.json"
        if marker.exists() and not args.force:
            print(f"Collection already complete; reusing {args.corpus}. Use --force to rerun.")
            return
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "src/collection/collect.py",
                "--candidates",
                str(args.candidates),
                "--output",
                str(args.corpus),
                "--workers",
                str(args.workers),
            ],
            check=True,
        )
        _mark_done(marker, stage="collection", corpus=str(args.corpus))
        return

    from src.common.logging import setup_logger
    from src.repair.repair import run_repairs
    from src.validation.runner import run_validation

    validation = args.output_dir / "stage_02_validation"
    validation_marker = validation / "stage.done.json"
    repair_marker = args.output_dir / "stage_03_repair" / "stage.done.json"
    if args.step == "validate" and validation_marker.exists() and not args.force:
        print(f"Validation already complete; reusing {validation}. Use --force to rerun.")
        return
    if not validation_marker.exists() or args.force:
        run_validation(
            args.archive,
            args.corpus,
            validation,
            setup_logger("data", Path("logs/data.log")),
            args.workers,
        )
        _mark_done(validation_marker, stage="validation", corpus=str(args.corpus))
    else:
        print(f"Validation already complete; reusing {validation}.")
    if args.step == "validate":
        return
    if repair_marker.exists() and not args.force:
        print(f"Repair already complete; reusing {args.corpus}. Use --force to rerun.")
        return
    repair_dir = args.output_dir / "stage_03_repair"
    run_repairs(args.corpus, validation, repair_dir, args.workers)
    _mark_done(repair_marker, stage="repair", corpus=str(args.corpus))
    if args.step == "repair":
        return
    print("Collection, validation, and repair complete. Run filter/run_filter.py next.")


if __name__ == "__main__":
    main()
