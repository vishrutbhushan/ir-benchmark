#!/usr/bin/env python3
"""
IR Benchmark Master Pipeline Runner

A single CLI entrypoint to trigger any pipeline phase step with hierarchical
outputs and logging.

Usage examples:
    python run_pipeline.py --phase 0 --sample-size 1000
    python run_pipeline.py --phase 1 --max-docs 500
    python run_pipeline.py --phase all  # complete neural corpus/index
    python run_pipeline.py --phase serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master runner for Information Retrieval Benchmark pipeline phases."
    )
    parser.add_argument(
        "--phase",
        choices=["0", "1", "phase0", "phase1", "validate", "prepare", "extract", "chunk", "embed", "all", "neural-all", "search", "serve", "verify", "development-features", "classification-data", "audit", "weights", "classifiers", "routing", "evaluation-pool", "evaluate"],
        default="all",
        help="all=full neural build; serve/search/verify=reuse index; 0/1=legacy classic preparation.",
    )
    parser.add_argument("--archive-path", type=Path, default=Path("technical_web_corpus.zip"))
    parser.add_argument("--workers", type=int, default=8, help="Workers for full raw validation.")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("technical_web_corpus"),
        help="Path to root corpus folder containing documents/ and metadata/.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("technical_web_corpus") / "metadata" / "corpus_manifest.jsonl",
        help="Path to corpus_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for structured outputs.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory for hierarchical logs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Total sample size for Phase 0 audit (stratified across technologies).",
    )
    parser.add_argument(
        "--sample-per-tech",
        type=int,
        default=15,
        help="Max documents to sample per technology in Phase 0 audit.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to extract in Phase 1 (useful for test runs; omit for full corpus).",
    )
    parser.add_argument(
        "--min-char-length",
        type=int,
        default=200,
        help="Minimum body text character length threshold.",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="When sampling or running subset extraction, stratify evenly across all technologies.",
    )
    parser.add_argument(
        "--exclusion-file",
        type=Path,
        default=None,
        help="Optional path to a JSON file containing doc_ids to exclude from Phase 1 extraction.",
    )
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--query')
    parser.add_argument('--mode', choices=['dense','zones','routed-centroid','routed-nb'], default='routed-centroid')
    parser.add_argument('--classic-rankings', type=Path, help='Classic document rankings JSONL for the common evaluation')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--technology', default='')
    parser.add_argument('--category', default='')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))

    phase = args.phase.lower()
    if phase.startswith("phase"):
        phase = phase.replace("phase", "")

    print(f"=== Running IR Benchmark Pipeline [Phase: {phase}] ===")

    if phase == "validate":
        from src.corpus_validation.runner import run_validation
        logger = setup_logger("corpus_validation", args.logs_dir / "corpus_validation" / "validation.log")
        run_validation(args.archive_path, args.corpus_root, args.output_dir / "corpus_validation", logger, args.workers)
        return

    if phase in {'prepare','extract','chunk','embed','all','neural-all','search','serve','verify','development-features','classification-data','audit','weights','classifiers','routing','evaluation-pool','evaluate'}:
        if args.max_docs is not None:
            raise ValueError('Neural stages always process the full retained corpus; --max-docs is legacy-only')
        if args.workers < 1 or args.batch_size < 1:
            raise ValueError('Workers and batch size must be positive')
        from src.neural.pipeline import run
        # Mirror stdout/stderr to the stage log while retaining live progress.
        import contextlib
        class Tee:
            def __init__(self,*streams): self.streams=streams
            def write(self,text):
                for stream in self.streams: stream.write(text)
                return len(text)
            def flush(self):
                for stream in self.streams: stream.flush()
        logdir=args.logs_dir/'neural'; logdir.mkdir(parents=True,exist_ok=True)
        with (logdir/f'{phase}.log').open('a',encoding='utf-8') as logfile:
            with contextlib.redirect_stdout(Tee(sys.stdout,logfile)), contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
                from datetime import datetime,timezone
                print(f'Run started: {datetime.now(timezone.utc).isoformat()} | phase={phase}',flush=True)
                run(args)
        return

    # Load the legacy classical-preparation path only when explicitly requested.
    from src.phase0_audit.audit_runner import run_corpus_audit
    from src.phase1_extraction.extraction_runner import run_structured_extraction
    from src.utils.visualizer import generate_interactive_dashboard_html

    # Execute Phase 0
    if phase == "0":
        phase0_out = args.output_dir / "phase0"
        phase0_log = args.logs_dir / "phase0" / "audit.log"
        logger0 = setup_logger("phase0_audit", phase0_log)

        run_corpus_audit(
            corpus_root=args.corpus_root,
            manifest_path=args.manifest_path,
            output_dir=phase0_out,
            logger=logger0,
            sample_size_per_tech=args.sample_per_tech,
            total_sample_limit=args.sample_size,
            min_char_length_threshold=args.min_char_length,
        )

    # Execute Phase 1
    if phase == "1":
        phase1_out = args.output_dir / "phase1"
        phase1_log = args.logs_dir / "phase1" / "extraction.log"
        logger1 = setup_logger("phase1_extraction", phase1_log)

        excluded_ids: set[str] = set()
        if args.exclusion_file and args.exclusion_file.exists():
            with args.exclusion_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    excluded_ids = set(data)
                elif isinstance(data, dict) and "excluded_doc_ids" in data:
                    excluded_ids = set(data["excluded_doc_ids"])
            logger1.info(f"Loaded {len(excluded_ids)} excluded doc IDs from {args.exclusion_file}")

        run_structured_extraction(
            corpus_root=args.corpus_root,
            manifest_path=args.manifest_path,
            output_dir=phase1_out,
            logger=logger1,
            max_documents=args.max_docs,
            min_char_length=args.min_char_length,
            excluded_doc_ids=excluded_ids,
            stratified=args.stratified,
            sample_per_tech=args.sample_per_tech,
        )

    # Compile / update visual interactive dashboard
    p0_json_path = args.output_dir / "phase0" / "audit_report.json"
    p1_json_path = args.output_dir / "phase1" / "corpus_statistics.json"
    p0_data = json.loads(p0_json_path.read_text(encoding="utf-8")) if p0_json_path.exists() else None
    p1_data = json.loads(p1_json_path.read_text(encoding="utf-8")) if p1_json_path.exists() else None

    dashboard_path = args.output_dir / "dashboard.html"
    generate_interactive_dashboard_html(p0_data, p1_data, dashboard_path)
    print(f"Visual dashboard generated: {dashboard_path}")

    print("=== Pipeline Step Finished Successfully ===")


if __name__ == "__main__":
    main()

