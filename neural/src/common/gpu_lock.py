"""Serialize repository jobs that allocate or compute on the same GPU."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def gpu_job_lock():
    path = Path(os.environ.get("IR_GPU_LOCK_PATH", "/tmp/ir-benchmark-gpu.lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
