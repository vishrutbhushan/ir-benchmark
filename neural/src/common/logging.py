"""Capture console output and tracebacks in a timestamped run log."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import logging
from pathlib import Path
import sys
import time


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def run_log(step):
    directory = Path("logs") / ("all" if step == "all" else f"step_{step}")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (directory / "run.log").open("a", encoding="utf-8", buffering=1) as stream:
        with (
            redirect_stdout(Tee(sys.stdout, stream)),
            redirect_stderr(Tee(sys.stderr, stream)),
        ):
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                "%(asctime)s UTC | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            formatter.converter = time.gmtime
            handler.setFormatter(formatter)
            logger = logging.getLogger("pipeline")
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.propagate = False
            logger.info("Step %s started", step)
            try:
                yield
            except KeyboardInterrupt:
                logger.warning(
                    "Step %s stopped; completed checkpoints remain saved", step
                )
                raise SystemExit(130) from None
            except Exception:
                logger.exception("Step %s failed", step)
                raise SystemExit(1) from None
            else:
                logger.info(
                    "Step %s finished in %.2f seconds",
                    step,
                    time.perf_counter() - started,
                )
            finally:
                logger.removeHandler(handler)
                handler.close()


def run_stage(number, function, *args):
    logger = logging.getLogger("pipeline")
    started = time.perf_counter()
    logger.info("Stage %s: %s", number, function.__name__)
    result = function(*args)
    logger.info(
        "Stage %s finished in %.2f seconds", number, time.perf_counter() - started
    )
    return result
