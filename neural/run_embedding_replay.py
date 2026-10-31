#!/usr/bin/env python3
"""Replay full-corpus embedding into a fresh, isolated output directory."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import logging
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    root = Path(__file__).resolve().parent
    os.chdir(root)
    from src.common.logging import Tee
    from src.common.zones import ZONES

    outputs = root / "outputs"
    replay = outputs / "embedding_replay"
    model = outputs / "step_03_embeddings/model"
    inputs = outputs / "step_02_windows"
    required = [model / "config.json", model / "model.safetensors"]
    for zone in ZONES:
        required.extend(
            inputs / zone / "passages" / name
            for name in ("summary.json", "passages.sqlite")
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing existing build inputs: " + ", ".join(missing))
    if outputs.is_symlink() or replay.is_symlink():
        parser.error("Replay output paths must not be symbolic links")

    # The lock stays outside the directory cleared at the start of each run.
    with (outputs / ".embedding_replay.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another embedding replay is already running")
        if replay.exists():
            shutil.rmtree(replay)
        replay.mkdir()
        with (replay / "run.log").open("w", buffering=1, encoding="utf-8") as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(
                Tee(sys.stderr, stream)
            ):
                logger = logging.getLogger("embedding_replay")
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                started = time.perf_counter()
                try:
                    from src.embeddings.step_03_embeddings import run_embeddings

                    logger.info("Fresh replay: %s", replay)
                    logger.info("Device %s; batch size %s", args.device, args.batch_size)
                    for zone in ZONES:
                        logger.info("Embedding %s", zone)
                        run_embeddings(
                            inputs / zone / "passages",
                            replay / zone,
                            model,
                            device=args.device,
                            batch_size=args.batch_size,
                        )
                    logger.info("Finished in %.1f seconds", time.perf_counter() - started)
                except KeyboardInterrupt:
                    logger.warning("Stopped. The next replay will delete this run and start fresh.")
                    return 130
                except Exception:
                    logger.exception("Embedding replay failed")
                    return 1
                finally:
                    logger.removeHandler(handler)
                    handler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
