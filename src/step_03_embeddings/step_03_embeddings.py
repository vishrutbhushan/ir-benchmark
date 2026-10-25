"""Resumable full passage embedding into durable contiguous NumPy arrays."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.common.artifacts import (
    connect,
    MODEL_REVISION,
    digest_file,
    report,
    write_json,
)
from src.common.encoder import Encoder


def run_embeddings(passages, out, model_path, device="cuda", batch_size=512):
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((passages / "summary.json").read_text())
    signature = hashlib.sha256(
        (
            source["fingerprint"]
            + MODEL_REVISION
            + digest_file(Path(__file__).parents[1] / "common/encoder.py")
            + "last-token-l2-fp16-v1"
        ).encode()
    ).hexdigest()
    state_path = out / "checkpoint.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("fingerprint") != signature:
        state = {"fingerprint": signature, "completed_rows": 0, "complete": False}
        write_json(state_path, state)
    n = source["passages"]
    required = [out / "summary.json", out / "vectors.npy", out / "doc_indices.npy"]
    if state.get("complete") and not all(path.exists() for path in required):
        state.update(complete=False, completed_rows=0)
        write_json(state_path, state)
    if state.get("complete") and (out / "summary.json").exists():
        print(f"Embeddings complete: reusing {n} durable vectors.", flush=True)
        return json.loads((out / "summary.json").read_text())
    encoder = Encoder(model_path, device)
    torch = encoder.torch
    dimension = encoder.dimension
    path = out / "vectors.npy"
    if state["completed_rows"] and not path.exists():
        raise RuntimeError(
            "Checkpoint exists but vectors are missing; remove checkpoint to rebuild"
        )
    vectors = np.lib.format.open_memmap(
        path,
        mode="r+" if state["completed_rows"] else "w+",
        dtype=np.float16,
        shape=(n, dimension),
    )
    from contextlib import closing

    with closing(connect(passages / "passages.sqlite", True)) as db:
        doc_indices = np.lib.format.open_memmap(
            out / "doc_indices.npy", mode="w+", dtype=np.int32, shape=(n,)
        )
        for row in db.execute(
            "SELECT passage_index,doc_index FROM passages ORDER BY passage_index"
        ):
            doc_indices[row["passage_index"]] = row["doc_index"]
        doc_indices.flush()
        cursor = db.execute(
            "SELECT passage_index,embedding_text,tokens FROM passages WHERE passage_index>=? ORDER BY passage_index",
            (state["completed_rows"],),
        )
        started, initial_rows = time.perf_counter(), state["completed_rows"]
        token_budget = 100000
        while True:
            batch = cursor.fetchmany(2048)
            if not batch:
                break
            window_end = batch[-1]["passage_index"] + 1
            pending = sorted(
                [(r["passage_index"], r["embedding_text"], r["tokens"]) for r in batch],
                key=lambda item: item[2],
            )
            while pending:
                take = min(batch_size, len(pending))
                while take > 1 and pending[take - 1][2] * take > token_budget:
                    take -= 1
                portion = pending[:take]
                try:
                    encoded = encoder.encode([text for _, text, _ in portion])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, min(batch_size, take) // 2)
                    token_budget = max(512, token_budget // 2)
                    print(
                        f"CUDA memory pressure: reducing embedding batch to {batch_size}",
                        flush=True,
                    )
                    continue
                if not np.isfinite(encoded).all() or np.any(
                    np.linalg.norm(encoded, axis=1) < 0.99
                ):
                    raise RuntimeError("Invalid embedding batch")
                for (index, _, _), vector in zip(portion, encoded):
                    vectors[index] = vector
                pending = pending[len(portion) :]
            state["completed_rows"] = window_end
            vectors.flush()
            write_json(state_path, state)
            elapsed = time.perf_counter() - started
            print(
                f"Embedded {window_end}/{n}; {(window_end - initial_rows) / elapsed:.1f} passages/s; batch {batch_size}",
                flush=True,
            )
        vectors.flush()
        norms_min, norms_max = float("inf"), 0.0
        for start in range(0, n, 10000):
            values = vectors[start : start + 10000].astype(np.float32)
            norms = np.linalg.norm(values, axis=1)
            if not np.isfinite(values).all() or np.any(np.abs(norms - 1) > 0.002):
                raise RuntimeError("Full-vector integrity verification failed")
            norms_min, norms_max = (
                min(norms_min, float(norms.min())),
                max(norms_max, float(norms.max())),
            )
        stats = {
            "fingerprint": signature,
            "input_fingerprint": source["fingerprint"],
            "documents": source["documents"],
            "passages": n,
            "dimension": dimension,
            "dtype": "float16",
            "model_revision": MODEL_REVISION,
            "all_vectors_finite": True,
            "norm_min": norms_min,
            "norm_max": norms_max,
            "vector_bytes": path.stat().st_size,
            "last_session_seconds": round(time.perf_counter() - started, 2),
            "last_session_rows": n - initial_rows,
            "device": device,
            "final_batch_size": batch_size,
            "max_batch_tokens": token_budget,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else 0,
        }
    report(
        out,
        "Full-corpus embedding verification",
        stats,
        {
            "vector_coverage": {"embedded": n, "missing": 0},
            "normalization_bounds": {
                "minimum_norm": norms_min,
                "maximum_norm": norms_max,
            },
        },
    )
    state.update(complete=True, completed_rows=n)
    write_json(state_path, state)
    return stats


def embed_zones(root, device="cuda", batch_size=512):
    from src.common.zones import ZONES

    root = Path(root)
    for zone in ZONES:
        print(f"Embedding {zone} zone...", flush=True)
        run_embeddings(
            root / f"step_02_windows/{zone}/passages",
            root / f"step_03_embeddings/{zone}",
            root / "step_03_embeddings/model",
            device,
            batch_size,
        )
