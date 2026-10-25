"""Exact dense, learned-zone and category-routed retrieval over four stores."""

from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

import numpy as np

from src.common.artifacts import (
    connect,
    digest_file,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
)
from src.common.config import load_config, default_mode
from src.common.zones import ZONES
from src.common.encoder import Encoder
from src.step_08_search.scoring import aggregate_documents, weighted_zone_scores

MODES = ("dense", "zones", "routed-centroid")


class SearchEngine:
    def __init__(
        self,
        root,
        device="cuda",
        corpus_root=None,
        require_learning=True,
        corpus_zip=None,
    ):
        self.config = load_config()
        self.default_mode = default_mode(self.config)
        self.corpus_root = Path(
            corpus_root or "data/00_corpus/technical_web_corpus"
        ).resolve()
        self.root = Path(root)
        self.summary = json.loads(
            (self.root / "step_02_windows/summary.json").read_text()
        )
        source = json.loads((self.root / "step_01_extraction/summary.json").read_text())
        if source["fingerprint"] != self.summary["source_fingerprint"]:
            raise RuntimeError("Zone index is stale; run --step all")
        from src.common.corpus import CorpusSource

        self.corpus = CorpusSource(
            self.corpus_root, corpus_zip or "data/00_corpus/technical_web_corpus.zip"
        )
        if self.corpus.manifest_sha256 != source["dataset_manifest_sha256"]:
            raise RuntimeError("Dataset changed; run --step all")
        verified = json.loads(
            (self.root / "step_03_embeddings/model/verified_files.json").read_text()
        )
        if verified["revision"] != MODEL_REVISION or verified["model"] != MODEL_ID:
            raise RuntimeError(
                "Configured model differs from the index; build compatible embeddings"
            )
        if (
            self.summary.get("max_tokens", 512),
            self.summary["content_tokens"],
            self.summary["overlap"],
        ) != (
            self.config["max_tokens"],
            self.config["max_tokens"] - self.config["reserved_tokens"],
            self.config["overlap"],
        ):
            raise RuntimeError("Window settings changed; run steps 02 and 03")
        self.encoder = Encoder(self.root / "step_03_embeddings/model", device)
        self.torch, self.device = self.encoder.torch, device
        self.document_db = connect(self.root / "step_01_extraction/corpus.sqlite", True)
        self.documents = [
            dict(r)
            for r in self.document_db.execute(
                "SELECT doc_index,doc_id,title,url,technology,category FROM documents ORDER BY doc_index"
            )
        ]
        self.ids = np.array([d["doc_id"] for d in self.documents])
        if [d["doc_index"] for d in self.documents] != list(range(len(self.documents))):
            raise RuntimeError("Document indexes must be contiguous")
        self.stores, self.matrices, self.mappings, self.resident = {}, {}, {}, {}
        total_bytes = 0
        for zone in ZONES:
            directory = self.root / f"step_02_windows/{zone}"
            embedding = self.root / f"step_03_embeddings/{zone}"
            checkpoint = json.loads((embedding / "checkpoint.json").read_text())
            summary = json.loads((embedding / "summary.json").read_text())
            passages = json.loads((directory / "passages/summary.json").read_text())
            if (
                not checkpoint["complete"]
                or summary["input_fingerprint"] != passages["fingerprint"]
                or passages["input_fingerprint"] != source["fingerprint"]
            ):
                raise RuntimeError(f"{zone}: incomplete/stale embeddings")
            self.stores[zone] = connect(directory / "passages/passages.sqlite", True)
            self.matrices[zone] = np.load(embedding / "vectors.npy", mmap_mode="r")
            self.mappings[zone] = np.load(embedding / "doc_indices.npy", mmap_mode="r")
            if self.matrices[zone].shape != (
                summary["passages"],
                self.encoder.dimension,
            ) or self.mappings[zone].shape != (summary["passages"],):
                raise RuntimeError(f"{zone}: vector/mapping shape mismatch")
            total_bytes += self.matrices[zone].nbytes
        resident = (
            device == "cuda" and total_bytes < self.torch.cuda.mem_get_info()[0] * 0.55
        )
        for zone in ZONES:
            self.resident[zone] = (
                self.torch.as_tensor(np.array(self.matrices[zone]), device=device)
                if resident
                else None
            )
        self.weights = np.array(
            [self.config["zone_weights"][z] for z in ZONES], dtype=np.float32
        )
        self.classifier = None
        if (self.root / "step_05_classification/model.json").exists():
            from src.step_05_classification.step_05_classification import Classifiers

            try:
                self.classifier = Classifiers(self.root)
            except RuntimeError:
                if require_learning:
                    raise
        self.routing = (
            json.loads((self.root / "step_06_routing/model.json").read_text())
            if (self.root / "step_06_routing/model.json").exists()
            else None
        )
        self.routing_rows = {}
        if self.routing:
            valid = (
                self.routing.get("schema") == "category-postings-v2"
                and self.routing["index_fingerprint"] == self.summary["fingerprint"]
                and self.classifier is not None
                and self.routing.get("classifier_sha256")
                == digest_file(self.root / "step_05_classification/model.json")
            )
            if valid:
                with np.load(self.root / "step_06_routing/rows.npz") as saved:
                    self.routing_rows = {k: saved[k] for k in saved.files}
            else:
                self.routing = None
        if require_learning and (
            self.weights is None or self.classifier is None or self.routing is None
        ):
            raise RuntimeError("Learning stages are incomplete; run --step all")
        self.lock, self.cache, self.query_cache = (
            threading.RLock(),
            OrderedDict(),
            OrderedDict(),
        )

    def _query(self, query):
        if query not in self.query_cache:
            self.query_cache[query] = self.encoder.encode(
                [query], max_length=MAX_TOKENS, query=True
            )[0]
            if len(self.query_cache) > self.config["query_cache_size"]:
                self.query_cache.popitem(last=False)
        return self.query_cache[query]

    def zone_scores(self, vector, allowed=None, selected_rows=None):
        n = len(self.documents)
        values = np.full((n, len(ZONES)), -np.inf, dtype=np.float32)
        winners = np.full((n, len(ZONES)), -1, dtype=np.int64)
        torch = self.torch
        q = torch.as_tensor(vector, device=self.device, dtype=torch.float32)
        scanned = 0
        with torch.inference_mode():
            for column, zone in enumerate(ZONES):
                mapping, matrix = self.mappings[zone], self.matrices[zone]
                if selected_rows is not None:
                    rows = selected_rows[zone]
                    if allowed is not None:
                        rows = rows[allowed[mapping[rows]]]
                else:
                    rows = (
                        np.arange(len(matrix))
                        if allowed is None
                        else np.flatnonzero(allowed[mapping])
                    )
                scanned += len(rows)
                scores = np.empty(len(rows), dtype=np.float32)
                for start in range(0, len(rows), 32768):
                    indices = rows[start : start + 32768]
                    block = (
                        self.resident[zone][
                            torch.as_tensor(indices, device=self.device)
                        ]
                        if self.resident[zone] is not None
                        else torch.as_tensor(
                            np.array(matrix[indices]), device=self.device
                        )
                    )
                    block = block.float()
                    scores[start : start + len(indices)] = (
                        (
                            (block @ q)
                            / torch.linalg.vector_norm(block, dim=1).clamp_min(1e-12)
                        )
                        .cpu()
                        .numpy()
                    )
                if len(rows):
                    maxima, local = aggregate_documents(scores, mapping[rows], n)
                    valid = np.isfinite(maxima)
                    values[:, column] = maxima
                    winners[valid, column] = rows[local[valid]]
        return values, winners, scanned

    def rank(self, query, mode="zones", route_k=None):
        if mode not in MODES:
            raise ValueError("Unknown retrieval mode")
        if not query.strip() or len(query) > 4000:
            raise ValueError("Enter 1-4000 query characters")
        q = self._query(query.strip())
        allowed = np.ones(len(self.documents), dtype=bool)
        selected = []
        selected_rows = None
        if mode == "routed-centroid":
            if self.classifier is None or self.routing is None:
                raise RuntimeError("Train centroids and category row lists first")
            class_scores = self.classifier.query_scores(query, q)
            k = route_k if route_k is not None else self.config["route_classes"]
            if not 1 <= k <= len(class_scores):
                raise ValueError("Category count must be between 1 and the class count")
            memberships = self.classifier.document_classes
            possible = set(memberships[allowed].ravel()) - {-1}
            selected = [
                int(c)
                for c in np.argsort(-class_scores, kind="stable")
                if c in possible
            ][:k]
            allowed &= np.isin(memberships, selected).any(axis=1)
            selected_rows = {
                z: np.unique(
                    np.concatenate([self.routing_rows[f"{c}_{z}"] for c in selected])
                )
                if selected
                else np.empty(0, np.int32)
                for z in ZONES
            }
        if selected_rows is None:
            zones, winners, scanned = self.zone_scores(
                q, None if allowed.all() else allowed
            )
        else:
            zones, winners, scanned = self.zone_scores(q, allowed, selected_rows)
        if mode == "dense" or (
            mode == "routed-centroid" and not self.config["zone_weighting"]
        ):
            scores = zones.max(axis=1)
            winning_zone = np.argmax(zones, axis=1)
            contributions = None
        else:
            if self.weights is None:
                raise RuntimeError("Fit zone weights first")
            scores, contributions = weighted_zone_scores(zones, self.weights)
            winning_zone = np.argmax(
                np.where(np.isfinite(zones), contributions, -np.inf), axis=1
            )
        candidates = np.flatnonzero(allowed & np.isfinite(scores))
        order = candidates[np.lexsort((self.ids[candidates], -scores[candidates]))]
        return {
            "order": order,
            "scores": scores,
            "zones": zones,
            "winners": winners,
            "winning_zone": winning_zone,
            "contributions": contributions,
            "scanned_vectors": scanned,
            "candidate_documents": int(allowed.sum()),
            "selected_classes": selected,
        }

    def search(
        self,
        query,
        use_cache=True,
        offset=0,
        page_size=None,
        mode=None,
    ):
        mode = mode or self.default_mode
        limit = min(self.config["result_limit"], len(self.documents))
        page_size = self.config["page_size"] if page_size is None else page_size
        if offset < 0 or not 1 <= page_size:
            raise ValueError("Invalid pagination")
        started = time.perf_counter()
        key = (query.strip(), mode)
        with self.lock:
            hit = use_cache and key in self.cache
            if hit:
                ranking = self.cache.pop(key)
            else:
                ranking = self.rank(query, mode)
            if use_cache:
                self.cache[key] = ranking
                while len(self.cache) > self.config["ranking_cache_size"]:
                    self.cache.popitem(last=False)
            order = ranking["order"][:limit]
            stop = min(offset + page_size, len(order))
            results = []
            for position in range(offset, stop):
                index = int(order[position])
                column = int(ranking["winning_zone"][index])
                zone = ZONES[column]
                results.append(
                    {
                        **{
                            k: v
                            for k, v in self.documents[index].items()
                            if k != "category"
                        },
                        "content_categories": [
                            self.classifier.model["class_names"][
                                self.classifier.model["classes"][int(c)]
                            ]
                            for c in self.classifier.document_classes[index]
                            if c >= 0
                        ]
                        if self.classifier is not None
                        else [],
                        "document_url": f"/document/{self.documents[index]['doc_id']}",
                        "rank": position + 1,
                        "score": float(ranking["scores"][index]),
                        "zone_scores": {
                            z: (
                                float(ranking["zones"][index, j])
                                if np.isfinite(ranking["zones"][index, j])
                                else None
                            )
                            for j, z in enumerate(ZONES)
                        },
                        "zone_contributions": (
                            {
                                z: float(ranking["contributions"][index, j])
                                for j, z in enumerate(ZONES)
                            }
                            if ranking["contributions"] is not None
                            else None
                        ),
                        "matched_zone": zone,
                    }
                )
            return {
                "query": query,
                "mode": mode,
                "index_version": self.summary["fingerprint"],
                "results": results,
                "total_results": len(order),
                "retrieved_depth": len(order),
                "returned": len(results),
                "result_limit": self.config["result_limit"],
                "page_size": page_size,
                "cache_hit": hit,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "scanned_vectors": ranking["scanned_vectors"],
                "candidate_documents": ranking["candidate_documents"],
                "selected_classes": ranking["selected_classes"],
            }

    def warmup(self):
        start = time.perf_counter()
        self.zone_scores(self._query("technical documentation"))
        self.startup_warmup_ms = round((time.perf_counter() - start) * 1000, 2)
        return self.startup_warmup_ms

    def close(self):
        self.document_db.close()
        for db in self.stores.values():
            db.close()
        self.cache.clear()
        self.query_cache.clear()
        self.resident.clear()
