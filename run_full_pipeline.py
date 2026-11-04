#!/usr/bin/env python3
"""Master pipeline runner for the full technical web corpus.

Executes all 4 pipeline components in sequence:
  1. filter/run_filter.py (Validation, Repair, Quality Filtering, Deduplication, Archive Packaging)
  2. classical/run_pipeline.py (Tokenization, Inverted Index, BM25 & BM25F Scoring)
  3. neural/run_pipeline.py (SQLite Extract, Windows, 0.6B Embeddings, Zone Weights, Title+Headings Clustering, Routing, Verification)
  4. evaluation/run_evaluation.py (Metrics Calculation & Export across all 10 Benchmark Queries)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = str(REPO_ROOT / ".venv/bin/python")


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(REPO_ROOT / "full_pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_stage(name: str, cmd: list[str]) -> float:
    log(f"=== STARTING STAGE: {name} ===")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["NEURAL_CONFIG"] = str(REPO_ROOT / "config/pipeline.json")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        line_str = line.rstrip()
        if line_str:
            print(f"[{name}] {line_str}", flush=True)
    process.stdout.close()
    return_code = process.wait()
    elapsed = time.perf_counter() - t0
    if return_code != 0:
        log(f"FAILED: Stage {name} exited with code {return_code} after {elapsed:.1f}s")
        sys.exit(return_code)
    log(f"=== FINISHED STAGE: {name} in {elapsed:.1f}s ({elapsed/60:.1f}m) ===")
    return elapsed


def prepare_neural_outputs():
    """Archive previous small/label output directories to prevent fingerprint collisions, preserving downloaded model weights."""
    neural_out = REPO_ROOT / "neural/outputs"
    summary_path = neural_out / "step_01_extraction/summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            if summary.get("source_documents", 0) < 1000:
                log(f"Archiving previous {summary.get('source_documents')} doc output to neural/outputs/labels_equivalent...")
                archive_dir = neural_out / "labels_equivalent"
                archive_dir.mkdir(parents=True, exist_ok=True)
                for item in neural_out.iterdir():
                    if item.name in {"small_equivalent", "labels_equivalent"}:
                        continue
                    if item.name == "step_03_embeddings" and (item / "model").exists():
                        continue
                    dest = archive_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                # For step_03_embeddings, move non-model subdirectories
                step03 = neural_out / "step_03_embeddings"
                if step03.exists():
                    for sub in step03.iterdir():
                        if sub.name != "model":
                            dest = archive_dir / "step_03_embeddings" / sub.name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            if dest.exists():
                                if dest.is_dir():
                                    shutil.rmtree(dest)
                                else:
                                    dest.unlink()
                            shutil.move(str(sub), str(dest))
        except Exception as e:
            log(f"Warning during output prep: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "up-to-engine", "filter", "classical", "neural", "evaluation"],
        default="all",
        help="Pipeline stages to run ('up-to-engine' runs filter, classical, and neural, stopping before evaluation)"
    )
    args = parser.parse_args()

    log(f"Starting IR Benchmark Master Pipeline Execution (target: {args.stage})")
    source_archive = REPO_ROOT / "data/00_corpus/technical_web_corpus.zip"
    if not source_archive.exists():
        alt = REPO_ROOT / "technical_web_corpus.zip"
        if alt.exists():
            source_archive.parent.mkdir(parents=True, exist_ok=True)
            log(f"Found source archive at root: linking to {source_archive}")
            try:
                os.link(str(alt), str(source_archive))
            except OSError:
                shutil.copy2(str(alt), str(source_archive))
        else:
            log(f"FATAL: Source archive not found at {source_archive} or {alt}")
            log("Please place 'technical_web_corpus.zip' in 'data/00_corpus/' before running.")
            sys.exit(1)

    timings = {}
    master_start = time.perf_counter()

    # 1. Filter Stage
    if args.stage in {"all", "up-to-engine", "filter"}:
        timings["filter"] = run_stage(
            "filter",
            [PYTHON, "filter/run_filter.py", "--config", "config/pipeline.json", "--step", "all"]
        )

    # 2. Classical Stage
    if args.stage in {"all", "up-to-engine", "classical"}:
        timings["classical"] = run_stage(
            "classical",
            [PYTHON, "classical/run_pipeline.py", "--config", "config/pipeline.json", "--phase", "all"]
        )

    # 3. Neural Stage
    if args.stage in {"all", "up-to-engine", "neural"}:
        prepare_neural_outputs()
        timings["neural"] = run_stage(
            "neural",
            [PYTHON, "neural/run_pipeline.py", "--config", "config/pipeline.json", "--step", "all"]
        )

    # 4. Evaluation Stage
    if args.stage in {"all", "evaluation"}:
        timings["evaluation_prepare"] = run_stage(
            "evaluation_prepare",
            [PYTHON, "evaluation/run_evaluation.py", "--config", "config/pipeline.json", "--prepare"]
        )
        timings["evaluation_metrics"] = run_stage(
            "evaluation_metrics",
            [PYTHON, "evaluation/run_evaluation.py", "--config", "config/pipeline.json", "--metrics", "--force"]
        )

    total_time = time.perf_counter() - master_start
    log("==================================================")
    log("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    log(f"Total Elapsed Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    for name, dur in timings.items():
        log(f"  - {name}: {dur:.1f}s ({dur/60:.1f}m)")
    log("==================================================")


if __name__ == "__main__":
    main()
