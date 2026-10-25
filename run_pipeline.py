#!/usr/bin/env python3
"""Run numbered neural stages; collection and final evaluation live on other branches."""

import argparse
import json
import os
from pathlib import Path


def parse_and_run():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("config.json"))
    config_args, _ = config_parser.parse_known_args()
    os.environ["NEURAL_CONFIG"] = str(config_args.config.resolve())
    from src.common.config import load_config, default_mode

    settings = load_config()
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "audit",
        ],
        default="all",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/00_corpus/technical_web_corpus")
    )
    parser.add_argument(
        "--corpus-zip",
        type=Path,
        default=Path("data/00_corpus/technical_web_corpus.zip"),
    )
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument("--batch-size", type=int, default=settings["batch_size"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=settings["device"])
    parser.add_argument("--query")
    parser.add_argument(
        "--mode",
        choices=["dense", "zones", "routed-centroid"],
        default=default_mode(settings),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("Workers and batch size must be positive")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    run(args, parser)


def run(args, parser):
    root = Path("outputs")
    from src.common.logging import run_stage

    if args.step in {"all", "01", "02", "03"}:
        from src.step_01_extraction.step_01_extraction import (
            prepare_extraction,
        )

        run_stage(
            "01",
            prepare_extraction,
            args.corpus_zip,
            args.corpus_dir,
            root / "step_01_extraction",
            args.workers,
        )
        if args.step == "01":
            return
        from src.step_02_windows.step_02_windows import prepare_zones

        run_stage("02", prepare_zones, root, args.workers)
        if args.step == "02":
            return
        from src.step_03_embeddings.step_03_embeddings import embed_zones

        run_stage("03", embed_zones, root, args.device, args.batch_size)
        if args.step == "03":
            return
    if args.step == "audit":
        from src.step_07_verification.learning_audit import audit_learning_data

        audit_learning_data(root)
        return
    from src.step_08_search.step_08_search import SearchEngine

    learning = args.step in {"all", "04", "05", "06"}
    engine = SearchEngine(
        root,
        args.device,
        args.corpus_dir,
        require_learning=not learning,
        corpus_zip=args.corpus_zip,
    )
    try:
        if args.step in {"all", "04"}:
            from src.step_04_weights.step_04_weights import fit_weights

            run_stage("04", fit_weights, root, engine)
            if args.step == "04":
                return
        if args.step in {"all", "05"}:
            from src.step_05_classification.step_05_classification import (
                train_classifiers,
                Classifiers,
            )

            run_stage("05", train_classifiers, root, engine)
            engine.classifier = Classifiers(root)
            if args.step == "05":
                return
        if args.step in {"all", "06"}:
            from src.step_06_routing.step_06_routing import tune_routing

            run_stage("06", tune_routing, root, engine)
            if args.step == "06":
                return
        if args.step in {"all", "07"}:
            from src.step_07_verification.step_07_verification import verify

            run_stage("07", verify, engine, root)
        elif args.step == "08":
            if not args.query:
                parser.error("--query is required for step 08")
            print(
                json.dumps(
                    engine.search(
                        args.query,
                        mode=args.mode,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.step == "09":
            from src.step_09_ui.step_09_ui import serve

            serve(engine, args.host, args.port)
    finally:
        engine.close()


def main():
    import sys
    from src.common.logging import run_log

    if "--help" in sys.argv or "-h" in sys.argv:
        parse_and_run()
        return
    preview = argparse.ArgumentParser(add_help=False)
    preview.add_argument("--step", default="all")
    args, _ = preview.parse_known_args()
    step = "07" if args.step == "audit" else args.step
    if step not in {"all", *[f"{i:02}" for i in range(1, 10)]}:
        step = "all"
    with run_log(step):
        parse_and_run()


if __name__ == "__main__":
    main()
