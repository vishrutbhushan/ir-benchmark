#!/usr/bin/env python3
"""
Single-step, self-contained Common Crawl data fetch and corpus creation pipeline.

Directly produces `data/00_corpus/technical_web_corpus.zip`, fully ready for the
main pipeline (`run_full_pipeline.py`) without requiring any copying or manual repair.

Usage:
    python data/run_collect.py
    python data/run_collect.py --workers 8 --max-docs 1000
    python data/run_collect.py --candidates path/to/candidates.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import zipfile

import requests

DATA_ROOT = "https://data.commoncrawl.org/"
USER_AGENT = "technical-ir-corpus-builder/3.0"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS_DIR = REPO_ROOT / "data/00_corpus"
DEFAULT_ARCHIVE_PATH = DEFAULT_CORPUS_DIR / "technical_web_corpus.zip"
DEFAULT_MANIFEST_PATH = DEFAULT_CORPUS_DIR / "metadata/corpus_manifest.jsonl"


# -----------------------------------------------------------------------------
# WARC HTTP Body Extraction
# -----------------------------------------------------------------------------

def extract_http_body(warc_gzip: bytes) -> bytes:
    """Decompress WARC record and extract uncompressed HTTP payload bytes."""
    raw = gzip.decompress(warc_gzip)

    warc_sep = raw.find(b"\r\n\r\n")
    if warc_sep < 0:
        raise ValueError("Invalid WARC record: missing header separator")

    warc_headers = raw[:warc_sep].decode("latin-1", "replace")
    body = raw[warc_sep + 4:]

    content_len_m = re.search(r"(?im)^Content-Length:\s*(\d+)\s*$", warc_headers)
    if content_len_m:
        body = body[:int(content_len_m.group(1))]

    http_sep = body.find(b"\r\n\r\n")
    if body.startswith((b"HTTP/", b"ICY ")) and http_sep >= 0:
        http_headers = body[:http_sep].decode("latin-1", "replace")
        body = body[http_sep + 4:]

        http_len_m = re.search(r"(?im)^Content-Length:\s*(\d+)\s*$", http_headers)
        if http_len_m:
            body = body[:int(http_len_m.group(1))]

        enc_m = re.search(r"(?im)^Content-Encoding:\s*([^\s;]+)", http_headers)
        if enc_m and enc_m.group(1).lower() == "gzip":
            body = gzip.decompress(body)

    return body


def fetch_warc(record: dict, session: requests.Session, retries: int = 3) -> tuple[bytes, int]:
    """Fetch a single record using HTTP Range from Common Crawl."""
    start = int(record["warc_record_offset"])
    length = int(record["warc_record_length"])
    url = DATA_ROOT + str(record["warc_filename"]).lstrip("/")

    headers = {
        "User-Agent": USER_AGENT,
        "Range": f"bytes={start}-{start + length - 1}",
        "Accept-Encoding": "identity",
    }

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, timeout=(15, 60), stream=True)
            if resp.status_code != 206:
                resp.close()
                raise RuntimeError(f"Range request failed with HTTP {resp.status_code}")

            payload = bytearray()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    payload.extend(chunk)
            resp.close()

            if len(payload) != length:
                raise RuntimeError(f"Payload size mismatch: expected {length}, got {len(payload)}")

            return extract_http_body(bytes(payload)), len(payload)
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 ** attempt)

    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


# -----------------------------------------------------------------------------
# Filename & Path Utilities
# -----------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\0/\\:*?"<>|]', "_", name).strip()
    return re.sub(r"\s+", " ", cleaned)[:180] or "document"


def get_destination_path(corpus_dir: Path, record: dict) -> tuple[Path, str]:
    """Determine relative and absolute file paths for an HTML document."""
    tech = record.get("technology", "General")
    doc_id = record.get("doc_id", "doc_unknown")
    
    if "filename" in record and record["filename"]:
        filename = record["filename"]
    else:
        parsed = urlparse(record["url"])
        title = record.get("title", parsed.path.strip("/").replace("/", "_") or "index")
        safe_title = sanitize_filename(f"{parsed.netloc}__{title}__{doc_id}")
        filename = f"{safe_title}.html" if not safe_title.endswith(".html") else safe_title

    rel_path = f"documents/html/{tech}/{filename}"
    abs_path = corpus_dir / rel_path
    return abs_path, rel_path


# -----------------------------------------------------------------------------
# Core Collection Runner
# -----------------------------------------------------------------------------

def download_one(record: dict, corpus_dir: Path, session: requests.Session) -> dict | None:
    """Download, validate, and store a single document. Skips if already on disk."""
    abs_path, rel_path = get_destination_path(corpus_dir, record)

    if abs_path.exists() and abs_path.stat().st_size > 50:
        return {
            **record,
            "html_path": rel_path,
            "status": "cached",
        }

    try:
        body, fetched_warc_bytes = fetch_warc(record, session)
        if len(body) < 50:
            return None

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(body)

        return {
            **record,
            "html_path": rel_path,
            "html_bytes": len(body),
            "fetched_warc_bytes": fetched_warc_bytes,
            "sha256": hashlib.sha256(body).hexdigest(),
            "status": "downloaded",
        }
    except Exception as exc:
        print(f"  [ERROR] {record.get('doc_id')}: {exc}", file=sys.stderr)
        return None


def package_corpus_archive(corpus_dir: Path, archive_path: Path, manifest_records: list[dict]) -> None:
    """Package the documents and metadata directly into technical_web_corpus.zip."""
    print(f"\nPackaging corpus into {archive_path}...")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = archive_path.with_suffix(".zip.tmp")

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for r in manifest_records:
            rel = r["html_path"]
            src_file = corpus_dir / rel
            if src_file.exists():
                zf.write(src_file, f"technical_web_corpus/{rel}")

        # Write metadata members
        manifest_file = corpus_dir / "metadata/corpus_manifest.jsonl"
        if manifest_file.exists():
            zf.write(manifest_file, "technical_web_corpus/metadata/corpus_manifest.jsonl")

        state_file = corpus_dir / "metadata/corpus_state.json"
        if state_file.exists():
            zf.write(state_file, "technical_web_corpus/metadata/corpus_state.json")

    temp_zip.replace(archive_path)
    zip_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive packaged successfully: {archive_path} ({zip_mb:.1f} MB)")


def run_collection(
    candidates_path: Path,
    corpus_dir: Path,
    archive_path: Path,
    workers: int = 8,
    max_docs: int | None = None,
    force_pack: bool = False,
) -> None:
    print("=" * 70)
    print("TECHNICAL WEB CORPUS CREATION PIPELINE")
    print("=" * 70)
    print(f"Candidates  : {candidates_path}")
    print(f"Corpus Dir  : {corpus_dir}")
    print(f"Target ZIP  : {archive_path}")
    print(f"Workers     : {workers}")
    print("-" * 70)

    # 1. Load candidates
    if not candidates_path.exists():
        if archive_path.exists():
            print(f"Candidates file not found, but production archive exists: {archive_path}")
            print("Corpus is already ready for the main pipeline!")
            return
        raise FileNotFoundError(f"Candidate records not found at: {candidates_path}")

    records = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if max_docs and len(records) >= max_docs:
                break

    print(f"Loaded {len(records):,} target document records.")

    # 2. Check if already complete
    if archive_path.exists() and not force_pack:
        print(f"\nCorpus archive already exists at: {archive_path}")
        print("Directly usable by: python run_full_pipeline.py")
        print("Use --force-pack to re-download or repackage.")
        return

    # 3. Download in parallel
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    downloaded = 0
    cached = 0
    failed = 0

    t0 = time.perf_counter()
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_one, r, corpus_dir, session) for r in records]
            for idx, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if res:
                    manifest_records.append(res)
                    if res.get("status") == "cached":
                        cached += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1

                if idx % 500 == 0 or idx == len(futures):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"Progress: {idx}/{len(futures)} | "
                        f"Downloaded: {downloaded} | Cached: {cached} | Failed: {failed} "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )

    # 4. Save metadata
    meta_dir = corpus_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = meta_dir / "corpus_manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for r in manifest_records:
            clean_r = {k: v for k, v in r.items() if k != "status"}
            f.write(json.dumps(clean_r) + "\n")

    state_out = meta_dir / "corpus_state.json"
    state_out.write_text(json.dumps({
        "status": "ready",
        "created_at": time.time(),
        "total_documents": len(manifest_records),
        "source": "common_crawl",
    }, indent=2) + "\n", encoding="utf-8")

    # 5. Automatically package directly into technical_web_corpus.zip
    package_corpus_archive(corpus_dir, archive_path, manifest_records)

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("SUCCESS: CORPUS CREATION COMPLETE!")
    print(f"Total Documents: {len(manifest_records):,}")
    print(f"Output Archive : {archive_path}")
    print(f"Elapsed Time   : {total_time:.1f}s")
    print("Output is directly usable by: python run_full_pipeline.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to candidates or manifest JSONL",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Destination corpus folder (default: data/00_corpus)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Target ZIP archive path (default: data/00_corpus/technical_web_corpus.zip)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to fetch (default: all)",
    )
    parser.add_argument(
        "--force-pack",
        action="store_true",
        help="Force re-download or re-packaging even if archive exists",
    )
    args = parser.parse_args()

    run_collection(
        candidates_path=args.candidates,
        corpus_dir=args.corpus_dir,
        archive_path=args.archive,
        workers=args.workers,
        max_docs=args.max_docs,
        force_pack=args.force_pack,
    )


if __name__ == "__main__":
    main()
