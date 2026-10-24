#!/usr/bin/env python3
"""
IR Benchmark Master Pipeline Runner

A single CLI entrypoint to trigger any pipeline phase step with hierarchical
outputs and logging.

Usage examples:
    python run_pipeline.py --phase 0 --sample-size 1000
    python run_pipeline.py --phase 1 --max-docs 500
    python run_pipeline.py --phase all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.phase0_audit.audit_runner import run_corpus_audit
from src.phase1_extraction.extraction_runner import run_structured_extraction
from src.phase2_tokenization.tokenization_runner import run_tokenization
from src.phase3_classic_ir.classic_ir_runner import run_classic_ir
from src.utils.logging_utils import setup_logger
from src.utils.visualizer import generate_interactive_dashboard_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master runner for Information Retrieval Benchmark pipeline phases."
    )
    parser.add_argument(
        "--phase",
        choices=["0", "1", "2", "3", "phase0", "phase1", "phase2", "phase3", "all"],
        default="0",
        help="Pipeline phase to execute (0=Corpus Audit, 1=Structured Extraction, 2=Tokenization, 3=Classic IR BM25, all=all phases).",
    )
    # Auto-resolve corpus_root and manifest_path if nested inside technical_web_corpus/technical_web_corpus
    default_root = Path("technical_web_corpus")
    if not (default_root / "documents").exists() and (default_root / "technical_web_corpus" / "documents").exists():
        default_root = default_root / "technical_web_corpus"

    default_manifest = default_root / "metadata" / "corpus_manifest.jsonl"

    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=default_root,
        help="Path to root corpus folder containing documents/ and metadata/.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=default_manifest,
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
        "--sample-per-category",
        type=int,
        default=60,
        help="Target documents to sample per category in Phase 0 audit.",
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
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("queries.jsonl"),
        help="Path to queries.jsonl file for retrieval benchmarking.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
        help="BM25 term frequency saturation parameter k1 (default: 1.2).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 document length normalization parameter b (default: 0.75).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top retrieved results per query (default: 20).",
    )
    parser.add_argument(
        "--scorer",
        choices=["bm25", "bm25f", "both"],
        default="both",
        help="Retrieval engine to execute: bm25 (flat), bm25f (zone-weighted hierarchical), or both (default: both).",
    )
    parser.add_argument(
        "--hierarchical-boost",
        type=float,
        default=0.25,
        help="Taxonomy matching boost factor for BM25F (default: 0.25).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes for Phase 1 extraction (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase = args.phase.lower()
    if phase.startswith("phase"):
        phase = phase.replace("phase", "")

    print(f"=== Running IR Benchmark Pipeline [Phase: {phase}] ===")

    # Execute Phase 0
    if phase in ["0", "all"]:
        phase0_out = args.output_dir / "phase0"
        phase0_log = args.logs_dir / "phase0" / "audit.log"
        logger0 = setup_logger("phase0_audit", phase0_log)

        run_corpus_audit(
            corpus_root=args.corpus_root,
            manifest_path=args.manifest_path,
            output_dir=phase0_out,
            logger=logger0,
            sample_size_per_tech=args.sample_per_tech,
            sample_size_per_category=args.sample_per_category,
            total_sample_limit=args.sample_size,
            min_char_length_threshold=args.min_char_length,
        )

    # Execute Phase 1
    if phase in ["1", "all"]:
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
            num_workers=args.workers,
        )

    # Execute Phase 2
    if phase in ["2", "all"]:
        phase2_out = args.output_dir / "phase2"
        phase2_log = args.logs_dir / "phase2" / "tokenization.log"
        logger2 = setup_logger("phase2_tokenization", phase2_log)

        # Phase 2 reads structured_records.jsonl from Phase 1 output
        records_path = args.output_dir / "phase1" / "structured_records.jsonl"
        if not records_path.exists():
            print(
                f"ERROR: Phase 1 output not found at {records_path}. "
                "Run Phase 1 first: python run_pipeline.py --phase 1"
            )
            sys.exit(1)

        run_tokenization(
            records_path=records_path,
            output_dir=phase2_out,
            logger=logger2,
            max_documents=args.max_docs,
            num_workers=args.workers,
        )

    # Execute Phase 3
    if phase in ["3", "all"]:
        phase3_out = args.output_dir / "phase3"
        phase3_log = args.logs_dir / "phase3" / "classic_ir.log"
        logger3 = setup_logger("phase3_classic_ir", phase3_log)

        tokenized_path = args.output_dir / "phase2" / "tokenized_corpus.jsonl"
        structured_path = args.output_dir / "phase1" / "structured_records.jsonl"

        if not tokenized_path.exists():
            print(
                f"ERROR: Phase 2 output not found at {tokenized_path}. "
                "Run Phase 2 first: python run_pipeline.py --phase 2"
            )
            sys.exit(1)
        if not structured_path.exists():
            print(
                f"ERROR: Phase 1 output not found at {structured_path}. "
                "Run Phase 1 first: python run_pipeline.py --phase 1"
            )
            sys.exit(1)

        run_classic_ir(
            tokenized_path=tokenized_path,
            structured_path=structured_path,
            queries_path=args.queries_path,
            output_dir=phase3_out,
            logger=logger3,
            k1=args.k1,
            b=args.b,
            top_k=args.top_k,
            scorer_type=args.scorer,
            hierarchical_boost=args.hierarchical_boost,
        )

    # Compile / update visual interactive dashboard
    p0_json_path = args.output_dir / "phase0" / "audit_report.json"
    p1_json_path = args.output_dir / "phase1" / "corpus_statistics.json"
    p2_json_path = args.output_dir / "phase2" / "tokenization_stats.json"
    p0_data = json.loads(p0_json_path.read_text(encoding="utf-8")) if p0_json_path.exists() else None
    p1_data = json.loads(p1_json_path.read_text(encoding="utf-8")) if p1_json_path.exists() else None
    p2_data = json.loads(p2_json_path.read_text(encoding="utf-8")) if p2_json_path.exists() else None

    dashboard_path = args.output_dir / "dashboard.html"
    generate_interactive_dashboard_html(p0_data, p1_data, dashboard_path)
    print(f"Visual dashboard generated: {dashboard_path}")

    print("=== Pipeline Step Finished Successfully ===")


if __name__ == "__main__":
    main()

