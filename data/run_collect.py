#!/usr/bin/env python3
"""Collect, validate, and repair the source corpus."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["all", "collect", "validate", "repair"], default="all")
    parser.add_argument("--archive", type=Path, default=Path("original_corpus.zip"))
    parser.add_argument("--corpus", type=Path, default=Path("technical_web_corpus"))
    parser.add_argument("--candidates", type=Path, default=Path("common_crawl_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.step == "collect":
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
        return

    from src.common.logging import setup_logger
    from src.repair.repair import run_repairs
    from src.validation.runner import run_validation

    validation = args.output_dir / "step_02_validation"
    run_validation(
        args.archive,
        args.corpus,
        validation,
        setup_logger("data", args.output_dir / "data.log"),
        args.workers,
    )
    if args.step == "validate":
        return
    run_repairs(args.corpus, validation, args.output_dir / "step_03_repair", args.workers)
    if args.step == "repair":
        return
    print("Collection, validation, and repair complete. Run filter/run_filter.py next.")


if __name__ == "__main__":
    main()
