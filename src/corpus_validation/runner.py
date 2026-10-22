from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from src.corpus_validation.archive import initialize_worker, validate_document
from src.corpus_validation.paths import local_html_path, normalize_corpus_path
from src.corpus_validation.reporting import write_report
from src.utils.file_utils import write_json


def run_validation(archive: Path, corpus_root: Path, output_dir: Path,
                   logger: logging.Logger, workers: int = 8) -> dict:
    if (corpus_root / 'metadata/corpus_state.json').exists():
        from src.corpus_validation.fix import validate_dataset
        from src.neural.common import report
        stats = validate_dataset(corpus_root, workers)
        stats.update(integrity_gate_passed=True, source='corrected_working_dataset')
        report(output_dir.parent / 'corpus_fix', 'Corrected dataset validation', stats,
               {'categories': stats['categories'], 'technologies': stats['technologies']})
        logger.info('Corrected dataset verified: %s documents', stats['documents'])
        return stats
    if workers < 1:
        raise ValueError("workers must be positive")
    if corpus_root.is_symlink():
        raise ValueError("Raw corpus root must not be a symlink")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"completed": False, "integrity_gate_passed": False}, output_dir / "summary.json")
    with archive.open("rb") as handle:
        archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    archive_stat = archive.stat()
    logger.info("Reading full archive manifest; SHA-256 %s", archive_hash)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member paths; ambiguous input archive")
        records = [json.loads(line) for line in z.open("technical_web_corpus/metadata/corpus_manifest.jsonl") if line.strip()]
        ids = [r["doc_id"] for r in records]
        paths = [normalize_corpus_path(r["html_path"]) for r in records]
        local_paths = [local_html_path(path) for path in paths]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)) or len(local_paths) != len(set(local_paths)):
            raise ValueError("Duplicate document IDs or normalized HTML paths")
        expected = {"technical_web_corpus/" + path for path in paths}
        html_names = {name for name in names if name.lower().endswith(".html")}
        if expected != html_names:
            raise ValueError(f"ZIP/manifest coverage mismatch: missing={len(expected-html_names)}, unlisted={len(html_names-expected)}")
        non_html_bytes = 0
        for member in z.infolist():
            if not member.is_dir() and member.filename not in html_names:
                # Stream metadata/log members to EOF: ZipFile verifies their CRC as well.
                with z.open(member) as handle:
                    while chunk := handle.read(1024 * 1024):
                        non_html_bytes += len(chunk)
        original_manifest = z.read("technical_web_corpus/metadata/corpus_manifest.jsonl")
    corpus_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = corpus_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    original_path = metadata_dir / "corpus_manifest.original.jsonl"
    if original_path.exists() and original_path.read_bytes() != original_manifest:
        raise ValueError("Refusing to overwrite a different original manifest")
    original_path.write_bytes(original_manifest)

    status_counts = collections.Counter()
    reasons = collections.Counter()
    flags = collections.Counter()
    languages = collections.Counter()
    code_counts = collections.Counter()
    content_hash_groups = collections.defaultdict(list)
    examples = []
    example_counts = collections.Counter()
    category_status = collections.defaultdict(collections.Counter)
    technology_status = collections.defaultdict(collections.Counter)
    integrity_failures = 0
    html_bytes = 0
    processed = 0
    names_to_files = {"integrity": "integrity_manifest.jsonl", "eligible": "eligible_manifest.jsonl",
                      "review": "review_manifest.jsonl", "exclude": "excluded_manifest.jsonl"}
    with ExitStack() as stack:
        checks = stack.enter_context((output_dir / "document_checks.jsonl").open("w", encoding="utf-8"))
        handles = {key: stack.enter_context((output_dir / filename).open("w", encoding="utf-8"))
                   for key, filename in names_to_files.items()}
        pool = stack.enter_context(ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_worker,
            initargs=(str(archive.resolve()), str(corpus_root.resolve())),
        ))
        for result, normalized in pool.map(validate_document, records, chunksize=16):
            processed += 1
            checks.write(json.dumps(result, ensure_ascii=False) + "\n")
            status_counts[result["status"]] += 1
            category_status[result["category"]][result["status"]] += 1
            technology_status[result["technology"]][result["status"]] += 1
            reasons.update(result["reasons"])
            flags.update(result["flags"])
            if not result["integrity_ok"]:
                integrity_failures += 1
            else:
                html_bytes += result["html_bytes"]
                handles["integrity"].write(json.dumps(normalized, ensure_ascii=False) + "\n")
                handles[result["status"]].write(json.dumps(normalized, ensure_ascii=False) + "\n")
                languages[result.get("detected_language", "unknown")] += 1
                code_counts["with_code" if result.get("code_chars", 0) else "without_code"] += 1
                if content_hash := result.get("visible_content_sha256"):
                    content_hash_groups[content_hash].append(result["doc_id"])
            example_key = (result["status"], result["technology"])
            if result["status"] != "eligible" and len(examples) < 1000 and example_counts[example_key] < 5:
                examples.append(result)
                example_counts[example_key] += 1
            if processed % 1000 == 0:
                logger.info("Validated %s/%s documents (%.1f docs/s); statuses %s", processed, len(records),
                            processed / (time.perf_counter() - started), dict(status_counts))
    if archive.stat().st_mtime_ns != archive_stat.st_mtime_ns or archive.stat().st_size != archive_stat.st_size:
        raise RuntimeError("Archive changed during validation")
    summary = {
        "completed": processed == len(records),
        "integrity_gate_passed": processed == len(records) and integrity_failures == 0,
        "processed_documents": processed, "integrity_failures": integrity_failures,
        "status_counts": dict(status_counts), "quality_reasons": dict(reasons + flags),
        "exclusion_reasons": dict(reasons), "flags": dict(flags),
        "languages": dict(languages), "code_prevalence": dict(code_counts),
        "categories": dict(collections.Counter(r["category"] for r in records)),
        "category_status": {key: dict(value) for key, value in category_status.items()},
        "technology_status": {key: dict(value) for key, value in technology_status.items()},
        "technologies_without_eligible_documents": sorted(key for key, value in technology_status.items() if not value["eligible"]),
        "inventory": {"manifest_documents": len(records), "html_members": len(html_names),
                      "zip_members": len(names), "html_bytes_verified": html_bytes,
                      "non_html_bytes_crc_verified": non_html_bytes,
                      "windows_paths_normalized": sum("\\" in r["html_path"] for r in records),
                      "duplicate_urls_excess": len(records) - len({r["url"] for r in records}),
                      "archive_sha256": archive_hash},
        "elapsed_seconds": round(time.perf_counter() - started, 2), "workers": workers,
        "neural_preprocessing_started": False,
        "eligibility_frozen": False,
        "identical_visible_content_groups": sum(len(ids) > 1 for ids in content_hash_groups.values()),
        "documents_in_identical_visible_content_groups": sum(len(ids) for ids in content_hash_groups.values() if len(ids) > 1),
        "policy": "v1: byte/CRC/SHA integrity; conservative HTML/language/error triage, uncertain content held for review",
    }
    write_json({key: ids for key, ids in content_hash_groups.items() if len(ids) > 1},
               output_dir / "identical_visible_content_candidates.json")
    # A normalized default manifest contains ALL verified raw documents, not silent exclusions.
    if summary["integrity_gate_passed"]:
        normalized_default = metadata_dir / "corpus_manifest.jsonl"
        normalized_bytes = (output_dir / "integrity_manifest.jsonl").read_bytes()
        if normalized_default.exists() and normalized_default.read_bytes() != normalized_bytes:
            raise ValueError("Existing normalized manifest differs; refusing to replace it silently")
        normalized_default.write_bytes(normalized_bytes)
    write_json(summary, output_dir / "summary.json")
    write_report(summary, examples, output_dir)
    logger.info("Full validation complete: %s", summary["status_counts"])
    if not summary["integrity_gate_passed"]:
        raise RuntimeError("Corpus integrity failed; see document_checks.jsonl")
    return summary
