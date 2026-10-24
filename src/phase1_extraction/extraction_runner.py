from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from src.phase0_audit.sampler import sample_manifest_documents
from src.phase1_extraction.corpus_statistics import generate_and_save_statistics
from src.phase1_extraction.html_parser import parse_html_to_structured_record
from src.utils.file_utils import stream_jsonl


def _parse_manifest_worker(item: tuple[str, dict[str, Any], int]) -> str | None:
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
        return json.dumps(record, ensure_ascii=False)
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
) -> dict[str, Any]:
    """
    Executes Phase 1 structured extraction: parses HTML pages into structured records
    using parallel worker processes and produces corpus statistics.
    """
    logger.info(f"Starting Phase 1 Structured Extraction (workers={num_workers})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_output_path = output_dir / "structured_records.jsonl"

    excluded_ids = excluded_doc_ids or set()
    start_time = time.time()
    processed_count = 0
    extracted_count = 0
    skipped_short_or_invalid = 0
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

    batch_tasks: list[tuple[str, dict[str, Any], int]] = []
    batch_size = 500

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
                    for line_json in executor.map(_parse_manifest_worker, batch_tasks, chunksize=25):
                        if line_json:
                            out_f.write(line_json + "\n")
                            extracted_count += 1
                        else:
                            skipped_short_or_invalid += 1

                    batch_tasks = []
                    elapsed = time.time() - start_time
                    rate = extracted_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Extracted {extracted_count:,} records (processed {processed_count:,}, {elapsed:.1f}s, {rate:.1f} docs/s)..."
                    )

            # Process remaining tasks
            if batch_tasks:
                for line_json in executor.map(_parse_manifest_worker, batch_tasks, chunksize=25):
                    if line_json:
                        out_f.write(line_json + "\n")
                        extracted_count += 1
                    else:
                        skipped_short_or_invalid += 1

    total_time = time.time() - start_time
    logger.info(
        f"Extraction pass finished in {total_time:.1f}s. Total saved records: {extracted_count:,} (skipped short/invalid: {skipped_short_or_invalid:,}, missing files: {missing_file_count:,})."
    )

    # Compute and save corpus statistics
    logger.info("Computing corpus statistics report...")
    stats = generate_and_save_statistics(records_output_path, output_dir)
    logger.info(f"Phase 1 extraction & statistics completed. Saved to {output_dir}")

    return {
        "processed_manifest_records": processed_count,
        "extracted_valid_records": extracted_count,
        "skipped_short_or_invalid": skipped_short_or_invalid,
        "missing_files": missing_file_count,
        "elapsed_seconds": round(total_time, 2),
        "statistics": stats,
    }
