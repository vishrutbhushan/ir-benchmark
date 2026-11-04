#!/usr/bin/env python3
"""Label fixed retrieval results and export pooled evaluation metrics."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from src.common import ROOT, write_csv
sys.path.insert(0, str(ROOT.parent))
from config import load_pipeline_config


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=ROOT.parent / "config/pipeline.json")
    config_args, _ = config_parser.parse_known_args()
    settings = load_pipeline_config(config_args.config)["evaluation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--corpus-zip", type=Path, default=ROOT.parent / "filter/outputs/stage_05_archive/technical_web_corpus.zip")
    parser.add_argument("--host", default=settings["host"])
    parser.add_argument("--port", type=int, default=settings["port"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--import-classical", type=Path)
    parser.add_argument("--revision", default="origin/bm25")
    parser.add_argument("--force", action="store_true", help="Re-run completed metric export.")
    parser.add_argument("--synthetic-random", action="store_true", help="Use seeded synthetic 0/1 labels for plumbing tests.")
    parser.add_argument("--seed", type=int, default=20261102, help="Seed for synthetic random labels.")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "logs/evaluation.log"),
            logging.StreamHandler(),
        ],
    )
    try:
        if args.import_classical:
            from src.import_classical import import_results

            import_results(args.import_classical, args.revision)
            return
        from src.store import Store

        store = Store(args.corpus_zip)
        if args.synthetic_random:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            rows = store.apply_synthetic_random(args.seed)
            write_csv(
                ROOT / f"results/synthetic_random_judgments_{stamp}.csv",
                rows,
                list(rows[0]) if rows else ["query_id", "doc_id"],
            )
            logging.warning(
                "Using synthetic_random_relevance labels only: seed=%d rows=%d",
                args.seed,
                len(rows),
            )
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
        elif args.metrics:
            marker = ROOT / "results/.metrics.done.json"
            if marker.exists() and not args.force:
                print("Metrics already exported; use --force to regenerate.")
                return
            for kind in ("metrics", "per_query"):
                rows, columns = store.exports(kind)
                write_csv(ROOT / f"results/{kind}.csv", rows, columns)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"completed_at": time.time()}, indent=2) + "\n")
            print(
                "Saved results/metrics.csv and results/per_query.csv. Unready metrics remain blank."
            )
            logging.info("Exported evaluation metrics and per-query results.")
        else:
            from src.server import serve

            serve(store, args.host, args.port)
    except Exception:
        logging.exception("Evaluation failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
