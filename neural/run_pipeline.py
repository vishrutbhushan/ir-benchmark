#!/usr/bin/env python3
"""Run numbered neural stages; collection and final evaluation live on other branches."""

import argparse
import json
import os
from pathlib import Path


def parse_and_run():
    config_parser = argparse.ArgumentParser(add_help=False)
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config/pipeline.json"
    config_parser.add_argument(
        "--config", type=Path, default=default_config
    )
    config_args, _ = config_parser.parse_known_args()
    os.environ["NEURAL_CONFIG"] = str(config_args.config.resolve())
    from src.common.config import load_config, default_mode

    settings = load_config()
    pipeline = json.loads(config_args.config.resolve().read_text(encoding="utf-8"))
    paths = pipeline.get("paths", {})
    repository_root = config_args.config.resolve().parents[1]

    def configured_path(key, fallback):
        value = paths.get(key, fallback)
        path = Path(value)
        return path if path.is_absolute() else repository_root / path
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
            "export",
        ],
        default="all",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=configured_path("corpus_dir", "../data/00_corpus"),
    )
    parser.add_argument(
        "--corpus-zip",
        type=Path,
        default=configured_path(
            "filtered_archive",
            "../filter/outputs/stage_05_archive/technical_web_corpus.zip",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=configured_path("neural_output_dir", "outputs"),
        help="Root for this run's numbered outputs and markers.",
    )
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument("--batch-size", type=int, default=settings["batch_size"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=settings["device"])
    parser.add_argument("--query")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=["dense", "zones", "routed-cluster"],
        default=default_mode(settings),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force", action="store_true", help="Re-run completed stages.")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("Workers and batch size must be positive")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    run(args, parser)


def run(args, parser):
    root = args.output_dir
    from src.common.logging import run_stage
    from src.common.artifacts import digest_file
    from src.common.config import load_config

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
        complete = markers[stage].exists()
        if stage == "04":
            model_path = root / "step_04_weights/model.json"
            feature_path = Path("results/step_04_weights/features/training_features.csv")
            complete = complete and model_path.exists() and feature_path.exists()
            if complete:
                model = json.loads(model_path.read_text(encoding="utf-8"))
                repository_root = Path(__file__).resolve().parents[1]
                settings = load_config()
                inputs = {
                    "queries_csv_sha256": repository_root
                    / settings["weight_queries"],
                    "judgments_csv_sha256": repository_root
                    / settings["weight_judgments"],
                }
                complete = all(
                    model.get(key) == digest_file(path)
                    for key, path in inputs.items()
                )
        if stage == "05":
            model_path = root / "step_05_clustering/model.json"
            complete = complete and model_path.exists() and (
                root / "step_05_clustering/zone_vectors.npz"
            ).exists()
            if complete:
                model = json.loads(model_path.read_text())
                complete = (
                    model.get("schema") == "unsupervised-zone-kmeans-v1"
                )
        if stage == "06":
            model_path = root / "step_06_routing/model.json"
            complete = complete and model_path.exists()
            if complete:
                complete = json.loads(model_path.read_text()).get(
                    "schema"
                ) == "cluster-postings-v1"
        if stage == "07":
            report_path = Path("results/step_07_verification/summary.json")
            complete = complete and report_path.exists()
            if complete:
                complete = json.loads(report_path.read_text()).get(
                    "all_cluster_row_lists_checked", False
                )
        if complete and not args.force:
            print(f"Stage {stage} already complete; reusing {markers[stage].parent}.")
            return
        run_stage(stage, function, *stage_args)
        markers[stage].parent.mkdir(parents=True, exist_ok=True)
        markers[stage].write_text("completed\n", encoding="utf-8")

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
    if args.step == "audit":
        from src.verification.learning_audit import audit_learning_data

        audit_learning_data(root)
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
        if args.step == "export":
            if not all((args.queries, args.variants, args.export_dir)):
                parser.error("export requires --queries, --variants and --export-dir")
            from src.retrieval.export_results import export_results

            run_stage(
                "08",
                export_results,
                engine,
                args.queries,
                args.variants,
                args.export_dir,
                startup_seconds,
            )
            return
        if args.step == "all":
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
            from src.ui.step_09_ui import serve

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
    step = {"audit": "07", "export": "08"}.get(args.step, args.step)
    if step not in {"all", *[f"{i:02}" for i in range(1, 10)]}:
        step = "all"
    with run_log(step):
        parse_and_run()


if __name__ == "__main__":
    main()
