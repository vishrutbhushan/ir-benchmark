"""Import selected classical results without changing the source branch."""

import hashlib
import json
import re
import subprocess
from .common import ROOT, COLUMNS, write_csv, write_json

SELECTED = [
    "NL_01",
    "NL_02",
    "NL_03",
    "NL_04",
    "KW_01",
    "KW_02",
    "KW_03",
    "KW_04",
    "KW_05",
    "BOOL_02",
    "BOOL_03",
    "BOOL_04",
    "BOOL_05",
    "EXACT_02",
    "EXACT_03",
    "EXACT_04",
    "EXACT_05",
    "TYPO_01",
    "TYPO_03",
    "TYPO_05",
]


def import_results(repository, revision):
    def source(path):
        return subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{revision}:{path}"]
        )

    sources = {}
    raw = {}
    for name in ("bm25",):
        path = f"outputs/test_queries_run/{name}_results.json"
        value = source(path)
        raw[name] = {q["query_id"]: q for q in json.loads(value)}
        sources[path] = hashlib.sha256(value).hexdigest()
    queries = []
    for qid in SELECTED:
        q = raw["bm25"][qid]
        queries.append(
            {k: q[k] for k in ("query_id", "query_text", "query_type", "target_tech")}
        )
    (ROOT / "queries/final_queries.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries),
        encoding="utf-8",
    )
    selection = []
    for q in raw["bm25"].values():
        qid = q["query_id"]
        reason = "Retained for technical specificity and coverage across query styles"
        if qid not in SELECTED:
            if qid.startswith("SHORT_"):
                reason = "Broad or ambiguous short fragment; prefer a specified technical task"
            elif qid == "NL_05":
                reason = "Combines FastAPI tasks and an unspecified Redis queue worker"
            elif qid == "BOOL_01":
                reason = "Broad framework exclusion without a specified task"
            else:
                reason = "Duplicates an already retained JSONB or container-networking intent"
        selection.append(
            {
                "query_id": qid,
                "query_text": q["query_text"],
                "selected": int(qid in SELECTED),
                "reason": reason,
            }
        )
    write_csv(ROOT / "queries/selection.csv", selection, list(selection[0]))
    summary = source("outputs/test_queries_run/test_queries_summary.md").decode()
    stats = json.loads(source("outputs/phase3/index_stats.json"))
    parameters = json.loads(source("outputs/phase3/phase3_summary.json"))["parameters"]
    manifests = []
    for name, data in raw.items():
        rows = []
        for q in queries:
            original = data[q["query_id"]]
            if original["query_text"] != q["query_text"]:
                raise ValueError("Classical query text differs across variants")
            results = sorted(original["results"], key=lambda r: r["rank"])[:10]
            if len(results) != 10 or len({r["doc_id"] for r in results}) != 10:
                raise ValueError("Need 10 unique results per query")
            for rank, d in enumerate(results, 1):
                rows.append(
                    dict(
                        engine="classical",
                        variant=name,
                        query_id=q["query_id"],
                        query_text=q["query_text"],
                        rank=rank,
                        doc_id=d["doc_id"],
                        title=d.get("title", d.get("document_title")),
                        url=d["url"],
                        technology=d.get("technology", ""),
                        categories=json.dumps([d.get("category", "")]),
                        score=d.get("score", d.get("doc_score")),
                    )
                )
        write_csv(ROOT / f"results/classical/{name}.csv", rows, COLUMNS)
        label = {"bm25": "BM25", "bm25f": "BM25F", "multilevel": "Multi-Level"}[name]
        match = re.search(
            r"\*\*" + re.escape(label) + r" Execution Time\*\*: ([0-9.]+)s", summary
        )
        manifests.append(
            {
                "engine": "classical",
                "variant": name,
                "source_revision": revision,
                "source_files": sources,
                "corpus_documents": stats["total_documents"],
                "corpus_manifest_sha256": None,
                "category_scope": "Source taxonomy",
                "parameters_from_source_phase3_summary": parameters,
                "query_count": 20,
                "results_per_query": 10,
                "performance": {
                    "selected_query_timings_ms": None,
                    "memory_bytes": None,
                    "index_size_bytes": None,
                    "legacy_batch_seconds": float(match.group(1)) if match else None,
                    "legacy_batch_queries": 34,
                    "legacy_batch_top_k": 20,
                },
                "missing_for_comparison": [
                    "same prepared corpus and manifest hash",
                    "per-query uncached timings for these 20 queries",
                    "model/index startup time",
                    "process RSS baseline and peak",
                    "exact index bytes",
                    "cache policy and software versions",
                ],
                "query_coverage_gaps": [
                    "No explicit homonym-disambiguation pair",
                    "Version constraint is present; no before/after temporal query",
                ],
            }
        )
    timing_columns = [
        "engine",
        "variant",
        "query_id",
        "repetition",
        "retrieval_ms",
        "query_embedding_ms",
        "ranking_ms",
        "cpu_ms",
        "rss_before_bytes",
        "rss_after_bytes",
        "process_high_water_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "scanned_vectors",
        "candidate_documents",
    ]
    timing_rows = []
    for variant in raw:
        for q in queries:
            row = dict.fromkeys(timing_columns)
            row.update(engine="classical", variant=variant, query_id=q["query_id"])
            timing_rows.append(row)
    write_csv(ROOT / "results/classical/timings.csv", timing_rows, timing_columns)
    write_json(ROOT / "results/classical/manifest.json", manifests)
    print("Imported 20 queries x 10 BM25 results.")
