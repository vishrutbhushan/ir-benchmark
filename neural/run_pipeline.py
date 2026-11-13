#!/usr/bin/env python3
"""Run the numbered neural indexing, retrieval, and verification stages."""

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def configured_path(config, key, fallback=None):
    value = config.get("paths", {}).get(key, fallback)
    if value is None:
        raise KeyError(f"Missing configured path: {key}")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_and_run():
    config_parser = argparse.ArgumentParser(add_help=False)
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config/pipeline.json"
    config_parser.add_argument(
        "--config", type=Path, default=default_config
    )
    config_args, _ = config_parser.parse_known_args()
    os.environ["NEURAL_CONFIG"] = str(config_args.config.resolve())
    from src.common.config import load_config

    settings = load_config()
    pipeline = json.loads(config_args.config.resolve().read_text(encoding="utf-8"))
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
        ],
        default="all",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=configured_path(pipeline, "corpus_dir", "data/00_corpus"),
    )
    parser.add_argument(
        "--corpus-zip",
        type=Path,
        default=configured_path(
            pipeline,
            "filtered_archive",
            "filter/outputs/stage_05_archive/technical_web_corpus.zip",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=configured_path(pipeline, "neural_output_dir", "neural/outputs"),
        help="Root for this run's numbered outputs and markers.",
    )
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument("--batch-size", type=int, default=settings["batch_size"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=settings["device"])
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("Workers and batch size must be positive")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    if args.device == "cuda":
        from src.common.gpu_lock import gpu_job_lock

        with gpu_job_lock():
            run(args, parser)
    else:
        run(args, parser)


def run(args, parser):
    root = args.output_dir
    from src.common.logging import run_stage
    from src.common.config import load_config, load_pipeline_config

    pipeline = load_pipeline_config()
    markers = {
        stage: root / f"step_{stage}_{name}" / "stage.done"
        for stage, name in {
            "01": "extraction",
            "02": "windows",
            "03": "embeddings",
            "04": "weights",
            "05": "clustering",
            "06": "routing",
            "07": "verification",
        }.items()
    }

    def run_if_needed(stage, function, *stage_args):
        run_stage(stage,function,*stage_args)
        markers[stage].parent.mkdir(parents=True,exist_ok=True)
        markers[stage].write_text("completed\n",encoding="utf-8")

    if args.step in {"all", "01", "02", "03"}:
        from src.extraction.step_01_extraction import (
            prepare_extraction,
        )

        run_if_needed(
            "01",
            prepare_extraction,
            args.corpus_zip,
            args.corpus_dir,
            root / "step_01_extraction",
            args.workers,
        )
        if args.step == "01":
            return
        from src.neural_preprocessing.step_02_windows import prepare_zones

        run_if_needed("02", prepare_zones, root, args.workers)
        if args.step == "02":
            return
        from src.embeddings.step_03_embeddings import embed_zones

        run_if_needed("03", embed_zones, root, args.device, args.batch_size)
        if args.step == "03":
            return
    from src.retrieval.step_08_search import SearchEngine

    learning = args.step in {"all", "04", "05", "06"}
    import time

    startup = time.perf_counter()
    engine = SearchEngine(
        root,
        args.device,
        args.corpus_dir,
        require_learning=not learning,
        corpus_zip=args.corpus_zip,
    )
    startup_seconds = time.perf_counter() - startup
    try:
        if args.step == "all" and engine.config["use_learned_zone_weights"]:
            from src.learning.step_04_weights import configure_weights

            run_if_needed("04", configure_weights, root, engine)
        if args.step == "04":
            from src.learning.step_04_weights import configure_weights

            run_if_needed("04", configure_weights, root, engine)
            return
        if args.step in {"all", "05"}:
            from src.learning.step_05_clustering import train_clusters, Clusters

            run_if_needed("05", train_clusters, root, engine)
            engine.clusters = Clusters(root)
            if args.step == "05":
                return
        if args.step in {"all", "06"}:
            from src.learning.step_06_routing import tune_routing
            if engine.clusters is None:
                from src.learning.step_05_clustering import Clusters

                engine.clusters = Clusters(root)

            run_if_needed("06", tune_routing, root, engine)
            if args.step == "06":
                return
        if args.step in {"all", "07"}:
            from src.verification.step_07_verification import verify

            run_if_needed("07", verify, engine, root)
    finally:
        engine.close()


def main():
    import sys
    from src.common.logging import run_log
    from src.common.config import load_pipeline_config

    if "--help" in sys.argv or "-h" in sys.argv:
        parse_and_run()
        return
    preview = argparse.ArgumentParser(add_help=False)
    preview.add_argument("--step", default="all")
    args, _ = preview.parse_known_args()
    step = args.step
    if step not in {"all", *[f"{i:02}" for i in range(1, 8)]}:
        step = "all"
    configured_logs = configured_path(load_pipeline_config(), "neural_logs_dir")
    with run_log(step, configured_logs):
        parse_and_run()


if __name__ == "__main__":
    main()
