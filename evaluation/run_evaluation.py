#!/usr/bin/env python3
"""Evaluate the six saved retrieval result sets against strict pooled labels."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from src.common import ROOT, write_csv
sys.path.insert(0, str(ROOT.parent))
from config import load_pipeline_config, resolve_path


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=ROOT.parent / "config/pipeline.json")
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed metric export.")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    results_dir = resolve_path(config["paths"]["evaluation_results_dir"])
    logs_dir = resolve_path(config["paths"]["evaluation_logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "evaluation.log"),
            logging.StreamHandler(),
        ],
    )
    try:
        from src.store import Store

        store = Store()
        report = store.comparison_report()
        if args.prepare:
            print(
                f"{len(store.queries)} queries, {len(store.runs)} variants, {len(store.pool)} unique judgment pairs."
            )
            print(
                f"Classical corpus: {report['classical_corpus_documents']}; neural corpus: {report['neural_corpus_documents']}."
            )
            logging.info(
                "Prepared evaluation: queries=%d variants=%d judgment_pairs=%d",
                len(store.queries),
                len(store.runs),
                len(store.pool),
            )
        else:
            marker = results_dir / ".metrics.done.json"
            if marker.exists() and not args.force:
                print("Metrics already exported; use --force to regenerate.")
                return
            for kind in (
                "metrics",
                "per_query",
                "metrics_graded",
                "per_query_graded",
            ):
                rows, columns = store.exports(kind)
                write_csv(results_dir / f"{kind}.csv", rows, columns)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"completed_at": time.time()}, indent=2) + "\n")
            from src.charts import generate_charts

            generate_charts(args.config)
            print(
                "Saved binary and graded metrics, per-query CSVs, and charts."
            )
            logging.info("Exported evaluation metrics and per-query results.")
    except Exception:
        logging.exception("Evaluation failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
