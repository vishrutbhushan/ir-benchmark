from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from src.phase0_audit.sampler import sample_manifest_documents
from src.phase1_extraction.corpus_statistics import generate_and_save_statistics
from src.phase1_extraction.html_parser import parse_html_to_structured_record
from src.utils.file_utils import stream_jsonl


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
) -> dict[str, Any]:
    """
    Executes Phase 1 structured extraction: parses HTML pages into structured records
    and produces corpus statistics.
    """
    logger.info("Starting Phase 1 Structured Extraction...")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_output_path = output_dir / "structured_records.jsonl"

    excluded_ids = excluded_doc_ids or set()
    start_time = time.time()
    processed_count = 0
    extracted_count = 0
    skipped_short_count = 0
    skipped_excluded_count = 0
    missing_file_count = 0

    manifest_stream: Iterable[dict[str, Any]]
    if stratified:
        logger.info(f"Using stratified sample across technologies (sample_per_tech={sample_per_tech}, max={max_documents})...")
        manifest_stream = sample_manifest_documents(
            manifest_path=manifest_path,
            sample_size_per_tech=sample_per_tech or 15,
            total_sample_limit=max_documents,
        )
    else:
        manifest_stream = stream_jsonl(manifest_path)

    with records_output_path.open("w", encoding="utf-8") as out_f:
        for manifest_record in manifest_stream:
            doc_id = manifest_record.get("doc_id", "")
            if max_documents is not None and not stratified and extracted_count >= max_documents:
                break

            processed_count += 1

            if doc_id in excluded_ids:
                skipped_excluded_count += 1
                continue

            rel_html_path = manifest_record.get("html_path", "")
            full_html_path = corpus_root / rel_html_path

            if not full_html_path.exists():
                missing_file_count += 1
                continue

            try:
                html_content = full_html_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.error(f"Error reading {full_html_path}: {e}")
                continue

            # Parse to structured record
            record = parse_html_to_structured_record(html_content, manifest_record)

            # Filter short/empty bodies
            if len(record["body_text"]) < min_char_length:
                skipped_short_count += 1
                continue

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            extracted_count += 1

            if extracted_count % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Extracted {extracted_count:,} records (processed {processed_count:,}, {elapsed:.1f}s, {extracted_count / elapsed:.1f} docs/s)..."
                )

    total_time = time.time() - start_time
    logger.info(
        f"Extraction pass finished in {total_time:.1f}s. Total saved records: {extracted_count:,} (skipped short: {skipped_short_count:,}, skipped excluded: {skipped_excluded_count:,})."
    )

    # Compute and save corpus statistics
    logger.info("Computing corpus statistics report...")
    stats = generate_and_save_statistics(records_output_path, output_dir)
    logger.info(f"Phase 1 extraction & statistics completed. Saved to {output_dir}")

    return {
        "processed_manifest_records": processed_count,
        "extracted_valid_records": extracted_count,
        "skipped_short_or_empty": skipped_short_count,
        "skipped_pre_excluded": skipped_excluded_count,
        "missing_files": missing_file_count,
        "elapsed_seconds": round(total_time, 2),
        "statistics": stats,
    }
