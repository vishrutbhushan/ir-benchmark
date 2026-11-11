from __future__ import annotations

import collections
import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from src.phase1_extraction.corpus_statistics import generate_and_save_statistics
from src.phase1_extraction.html_parser import parse_html_to_structured_record
from src.utils.file_utils import stream_jsonl, write_json
from src.utils.simhash import CorpusDeduplicator


def sample_manifest_documents(
    manifest_path: Path | str,
    sample_size_per_tech: int | None = 15,
    total_sample_limit: int | None = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Selects a stratified sample with balanced representation across technologies."""
    random.seed(seed)
    cat_tech_buckets: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for record in stream_jsonl(manifest_path):
        cat = record.get("category", "unknown")
        tech = record.get("technology", "Unknown")
        cat_tech_buckets[cat][tech].append(record)

    sampled_records: list[dict[str, Any]] = []
    for cat, techs in cat_tech_buckets.items():
        for tech, records in techs.items():
            k = min(len(records), sample_size_per_tech or 15)
            sampled_records.extend(random.sample(records, k))

    random.shuffle(sampled_records)
    if total_sample_limit is not None and len(sampled_records) > total_sample_limit:
        sampled_records = sampled_records[:total_sample_limit]
    return sampled_records


def _parse_manifest_worker(item: tuple[str, dict[str, Any], int]) -> dict[str, Any] | None:
    """Worker function for parallel HTML extraction."""
    full_path_str, manifest_record, min_char_length = item
    try:
        p = Path(full_path_str)
        if not p.exists():
            return None
        html_content = p.read_text(encoding="utf-8", errors="replace")
        record = parse_html_to_structured_record(html_content, manifest_record)
        if len(record.get("body_text", "")) < min_char_length:
            return None
        return record
    except Exception:
        return None


def run_structured_extraction(
    corpus_root: Path,
    manifest_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    max_documents: int | None = None,
    min_char_length: int = 200,
    excluded_doc_ids: set[str] | None = None,
    stratified: bool = False,
    sample_per_tech: int | None = None,
    num_workers: int = 8,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """
    Executes Phase 1 structured extraction: parses HTML pages into structured records
    using parallel worker processes, removes exact and SimHash near-duplicate pages,
    and produces corpus statistics.
    """
    logger.info(
        f"Starting Phase 1 Structured Extraction (workers={num_workers}, deduplicate={deduplicate})..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_output_path = output_dir / "structured_records.jsonl"

    excluded_ids = excluded_doc_ids or set()
    deduplicator = CorpusDeduplicator(max_hamming_distance=3) if deduplicate else None
    start_time = time.time()
    processed_count = 0
    extracted_count = 0
    skipped_short_or_invalid = 0
    skipped_duplicates = 0
    missing_file_count = 0

    manifest_stream: Iterable[dict[str, Any]]
    if stratified:
        logger.info(
            f"Using stratified sample across technologies (sample_per_tech={sample_per_tech}, max={max_documents})..."
        )
        manifest_stream = sample_manifest_documents(
            manifest_path=manifest_path,
            sample_size_per_tech=sample_per_tech or 15,
            total_sample_limit=max_documents,
        )
    else:
        manifest_stream = stream_jsonl(manifest_path)

    batch_tasks: list[tuple[str, dict[str, Any], int]] = []
    batch_size = 500

    def _flush_batch(executor: ProcessPoolExecutor, tasks: list[tuple[str, dict[str, Any], int]], out_f: Any) -> None:
        nonlocal extracted_count, skipped_short_or_invalid, skipped_duplicates
        for record in executor.map(_parse_manifest_worker, tasks, chunksize=25):
            if not record:
                skipped_short_or_invalid += 1
                continue
            if deduplicator is not None:
                is_dup, _ = deduplicator.check_and_register(record)
                if is_dup:
                    skipped_duplicates += 1
                    continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            extracted_count += 1

    with records_output_path.open("w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
            for manifest_record in manifest_stream:
                doc_id = manifest_record.get("doc_id", "")
                if max_documents is not None and not stratified and extracted_count >= max_documents:
                    break

                processed_count += 1
                if doc_id in excluded_ids:
                    continue

                rel_html_path = manifest_record.get("html_path", "")
                normalized_rel_path = Path(str(rel_html_path).replace("\\", "/"))
                full_html_path = corpus_root / normalized_rel_path

                if not full_html_path.exists():
                    missing_file_count += 1
                    continue

                batch_tasks.append((str(full_html_path), manifest_record, min_char_length))

                if len(batch_tasks) >= batch_size:
                    _flush_batch(executor, batch_tasks, out_f)
                    batch_tasks = []
                    elapsed = time.time() - start_time
                    rate = extracted_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Extracted {extracted_count:,} unique records "
                        f"(processed {processed_count:,}, duplicates skipped: {skipped_duplicates:,}, "
                        f"{elapsed:.1f}s, {rate:.1f} docs/s)..."
                    )

            # Process remaining tasks
            if batch_tasks:
                _flush_batch(executor, batch_tasks, out_f)

    total_time = time.time() - start_time
    dedup_summary = deduplicator.get_summary() if deduplicator is not None else {}
    if deduplicator is not None:
        write_json(dedup_summary, output_dir / "duplicate_removal_report.json")

    logger.info(
        f"Extraction pass finished in {total_time:.1f}s. Total saved unique records: {extracted_count:,} "
        f"(duplicates removed: {skipped_duplicates:,}, skipped short/invalid: {skipped_short_or_invalid:,}, "
        f"missing files: {missing_file_count:,})."
    )

    # Compute and save corpus statistics
    logger.info("Computing corpus statistics report...")
    stats = generate_and_save_statistics(records_output_path, output_dir)
    logger.info(f"Phase 1 extraction & statistics completed. Saved to {output_dir}")

    return {
        "processed_manifest_records": processed_count,
        "extracted_valid_records": extracted_count,
        "skipped_duplicates": skipped_duplicates,
        "deduplication_summary": dedup_summary,
        "skipped_short_or_invalid": skipped_short_or_invalid,
        "missing_files": missing_file_count,
        "elapsed_seconds": round(total_time, 2),
        "statistics": stats,
    }
