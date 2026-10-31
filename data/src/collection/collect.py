#!/usr/bin/env python3
"""
Download the webpages listed in an existing Common Crawl candidate JSONL file.

Corpus creation ONLY:
- Uses the existing common_crawl_candidates.jsonl as the download manifest.
- Downloads the WARC record for each candidate using HTTP Range.
- Extracts and saves the ORIGINAL HTML body.
- Uses the HTML <title> only to create a readable filename.
- Writes one metadata record per saved document.
- Does NOT create parsed JSON, chunks, embeddings, vectors, or any retrieval index.
- Safe to rerun: existing manifest entries are skipped and unfinished candidates are retried.
- No byte limit by default; pass --max-bytes only if you deliberately want one.

Default layout:

technical_web_corpus/
├── documents/
│   └── html/
│       ├── Java/
│       ├── Python/
│       └── ...
├── metadata/
│   ├── common_crawl_candidates.jsonl
│   ├── corpus_manifest.jsonl
│   ├── download_failures.jsonl
│   └── corpus_statistics.json
└── logs/
    └── corpus_build.log
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html as html_lib
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse

import requests

DATA_ROOT = "https://data.commoncrawl.org/"
DEFAULT_CANDIDATES = (
    Path("technical_web_corpus") / "metadata" / "common_crawl_candidates.jsonl"
)
DEFAULT_OUTPUT = Path("technical_web_corpus")

DEFAULT_WORKERS = 8
DEFAULT_MAX_DOCUMENTS = 200_000
DEFAULT_MAX_BYTES = (
    None  # Unlimited by default; continue until all candidates are attempted
)
HARD_MAX_BYTES = 10 * 1024**3
USER_AGENT = "technical-ir-corpus-builder/3.0"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(root: Path) -> logging.Logger:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("technical_web_corpus")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        logs_dir / "corpus_build.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(value)
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{value} B"


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def sanitize_filename(value: str, max_chars: int = 120) -> str:
    value = html_lib.unescape(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip(" ._")
    if not value:
        value = "untitled"
    return value[:max_chars].rstrip(" ._")


def title_from_html(html_bytes: bytes) -> str:
    # This is ONLY for naming/identifying the downloaded raw HTML file.
    # No cleaned text is stored or generated.
    text = html_bytes.decode("utf-8", "replace")
    match = re.search(
        r"<title\b[^>]*>(.*?)</title\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return "untitled"

    raw = re.sub(r"<[^>]+>", " ", match.group(1))
    raw = re.sub(r"\s+", " ", html_lib.unescape(raw)).strip()
    return raw or "untitled"


def fallback_name(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    return sanitize_filename(f"{parsed.netloc}_{path.replace('/', '_')}")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# WARC / Common Crawl
# ---------------------------------------------------------------------------


def extract_http_body(warc_gzip: bytes) -> bytes:
    raw = gzip.decompress(warc_gzip)

    warc_separator = raw.find(b"\r\n\r\n")
    if warc_separator < 0:
        raise ValueError("Invalid WARC record: missing WARC header separator")

    warc_headers = raw[:warc_separator].decode("latin-1", "replace")
    body = raw[warc_separator + 4 :]

    content_length = re.search(
        r"(?im)^Content-Length:\s*(\d+)\s*$",
        warc_headers,
    )
    if content_length:
        body = body[: int(content_length.group(1))]

    http_separator = body.find(b"\r\n\r\n")
    if body.startswith((b"HTTP/", b"ICY ")) and http_separator >= 0:
        http_headers = body[:http_separator].decode("latin-1", "replace")
        body = body[http_separator + 4 :]

        http_length = re.search(
            r"(?im)^Content-Length:\s*(\d+)\s*$",
            http_headers,
        )
        if http_length:
            body = body[: int(http_length.group(1))]

        content_encoding = re.search(
            r"(?im)^Content-Encoding:\s*([^\s;]+)",
            http_headers,
        )
        if content_encoding and content_encoding.group(1).lower() == "gzip":
            body = gzip.decompress(body)

    return body


def fetch_warc(
    record: dict,
    session: requests.Session,
    retries: int = 4,
) -> tuple[bytes, int]:
    start = int(record["warc_record_offset"])
    length = int(record["warc_record_length"])
    url = DATA_ROOT + str(record["warc_filename"])

    headers = {
        "User-Agent": USER_AGENT,
        "Range": f"bytes={start}-{start + length - 1}",
        "Accept-Encoding": "identity",
    }

    last_error: Exception | None = None

    for attempt in range(retries):
        response = None
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=(20, 120),
                stream=True,
            )

            # IMPORTANT:
            # If Common Crawl ignores Range and returns HTTP 200, do not
            # download an entire multi-GB WARC file into memory.
            if response.status_code != 206:
                raise RuntimeError(
                    f"Range request not honored: HTTP {response.status_code}"
                )

            expected = length
            payload = bytearray()

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    payload.extend(chunk)

            if len(payload) != expected:
                raise RuntimeError(
                    f"Wrong WARC payload size: expected {expected:,} bytes, "
                    f"received {len(payload):,}"
                )

            return extract_http_body(bytes(payload)), len(payload)

        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        finally:
            if response is not None:
                response.close()

    raise RuntimeError(f"WARC fetch failed after {retries} attempts: {last_error}")


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# ---------------------------------------------------------------------------
# Document download
# ---------------------------------------------------------------------------


def download_one(
    candidate_line: int,
    record: dict,
) -> dict:
    session = get_session()

    url = str(record["url"])
    body, fetched_bytes = fetch_warc(record, session)

    if not body:
        raise ValueError("Empty HTML body")

    content_sha256 = hashlib.sha256(body).hexdigest()
    title = title_from_html(body)

    if title == "untitled":
        title = fallback_name(url)

    return {
        "candidate_line": candidate_line,
        "url": url,
        "technology": str(record.get("technology", "unknown")),
        "category": str(record.get("category", "unknown")),
        "host": str(record.get("host", "")),
        "title": title,
        "html_bytes": len(body),
        "fetched_warc_bytes": fetched_bytes,
        "sha256": content_sha256,
        "body": body,
        "source_record": record,
    }


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


def load_resume_state(manifest_path: Path) -> tuple[set[int], int, int]:
    completed_lines: set[int] = set()
    saved_docs = 0
    stored_bytes = 0

    if not manifest_path.exists():
        return completed_lines, saved_docs, stored_bytes

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed_lines.add(int(row["candidate_line"]))
            saved_docs += 1
            stored_bytes += int(row.get("html_bytes", 0))

    return completed_lines, saved_docs, stored_bytes


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    candidates_path = Path(args.candidates).resolve()

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidates_path}")

    if args.max_bytes is not None and (
        args.max_bytes <= 0 or args.max_bytes > HARD_MAX_BYTES
    ):
        raise ValueError(
            f"--max-bytes must be > 0 and <= {HARD_MAX_BYTES:,}, or omit it for unlimited"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "documents" / "html").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output)

    manifest_path = output / "metadata" / "corpus_manifest.jsonl"
    failures_path = output / "metadata" / "download_failures.jsonl"
    stats_path = output / "metadata" / "corpus_statistics.json"

    completed_lines, saved_docs, stored_bytes = load_resume_state(manifest_path)

    # Count candidates once. This is cheap and gives a real denominator.
    total_candidates = sum(1 for _ in read_jsonl(candidates_path))

    if total_candidates == 0:
        raise RuntimeError("Candidate file contains no documents.")

    logger.info("Candidate file: %s", candidates_path)
    logger.info("Total candidate lines: %s", f"{total_candidates:,}")
    logger.info("Already completed: %s", f"{len(completed_lines):,}")
    logger.info("Output directory: %s", output)
    logger.info("HTML directory: %s", output / "documents" / "html")
    logger.info("Workers: %d", args.workers)
    logger.info(
        "Corpus byte budget: %s",
        "UNLIMITED" if args.max_bytes is None else format_bytes(args.max_bytes),
    )

    lock = threading.Lock()
    started = time.time()

    # This run resumes from the existing manifest. Previously saved documents
    # are skipped; failed candidates are retried automatically.
    attempted_this_run = 0
    failed = 0
    skipped = len(completed_lines)
    fetched_bytes = 0
    remaining_at_start = total_candidates - len(completed_lines)

    logger.info(
        "Already saved: %s | Remaining candidates: %s",
        f"{len(completed_lines):,}",
        f"{remaining_at_start:,}",
    )

    # Stable IDs are based on candidate line numbers.
    # This means resume runs never rename existing documents.
    def make_doc_id(candidate_line: int) -> str:
        return f"document_{candidate_line:06d}"

    def save_result(result: dict) -> tuple[bool, str]:
        nonlocal saved_docs, stored_bytes

        source_record = result["source_record"]
        candidate_line = result["candidate_line"]
        doc_id = make_doc_id(candidate_line)

        technology = sanitize_filename(result["technology"], 50)
        host = sanitize_filename(result["host"] or urlparse(result["url"]).netloc, 80)
        title = sanitize_filename(result["title"], 120)

        filename = f"{host}__{title}__{doc_id}.html"
        html_dir = output / "documents" / "html" / technology
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / filename

        with lock:
            if saved_docs >= args.max_documents:
                return False, "document limit reached"

            if (
                args.max_bytes is not None
                and stored_bytes + len(result["body"]) > args.max_bytes
            ):
                return False, "byte budget reached"

        tmp_path = html_path.with_suffix(".html.part")
        tmp_path.write_bytes(result["body"])
        tmp_path.replace(html_path)

        metadata = {
            "doc_id": doc_id,
            "candidate_line": candidate_line,
            "filename": filename,
            "html_path": str(html_path.relative_to(output)),
            "url": result["url"],
            "title": result["title"],
            "technology": result["technology"],
            "category": result["category"],
            "host": result["host"],
            "html_bytes": len(result["body"]),
            "fetched_warc_bytes": result["fetched_warc_bytes"],
            "sha256": result["sha256"],
            "crawl": source_record.get("crawl"),
            "content_digest": source_record.get("content_digest"),
            "content_languages": source_record.get("content_languages"),
            "content_mime_type": source_record.get("content_mime_type"),
            "warc_filename": source_record.get("warc_filename"),
            "warc_record_offset": source_record.get("warc_record_offset"),
            "warc_record_length": source_record.get("warc_record_length"),
        }

        with lock:
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()

            saved_docs += 1
            stored_bytes += len(result["body"])

        return True, str(html_path)

    # We only keep a bounded number of requests in flight.
    # This avoids creating 80k queued futures and keeps progress meaningful.
    candidate_iter = iter(read_jsonl(candidates_path))
    pending = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Prime the pool.
        while len(pending) < args.workers:
            try:
                line_no, record = next(candidate_iter)
            except StopIteration:
                break

            if line_no in completed_lines:
                continue

            future = executor.submit(download_one, line_no, record)
            pending[future] = (line_no, record)

        last_progress = 0.0

        while pending:
            done, _ = wait(
                pending,
                timeout=0.25,
                return_when=FIRST_COMPLETED,
            )

            now = time.time()

            if not done:
                if now - last_progress >= 1.0:
                    rate = attempted_this_run / max(0.001, now - started)
                    remaining = max(0, remaining_at_start - attempted_this_run)
                    eta = remaining / rate if rate > 0 else 0
                    line = (
                        f"\rThis run {attempted_this_run:,}/{remaining_at_start:,} "
                        f"({attempted_this_run / max(1, remaining_at_start) * 100:6.2f}%) | "
                        f"total_saved={saved_docs:,} | failed={failed:,} | "
                        f"already_saved={len(completed_lines):,} | "
                        f"fetched={format_bytes(fetched_bytes)} | "
                        f"stored={format_bytes(stored_bytes)} | "
                        f"{rate:.2f} attempts/s | ETA={format_elapsed(eta)}"
                    )
                    print(line, end="", flush=True)
                    last_progress = now
                continue

            for future in done:
                line_no, record = pending.pop(future)
                attempted_this_run += 1

                try:
                    result = future.result()
                    fetched_bytes += int(result["fetched_warc_bytes"])

                    saved, detail = save_result(result)

                    if saved:
                        print(
                            f"\nSAVED {make_doc_id(line_no)} | "
                            f"{result['title'][:90]} | {result['url']}",
                            flush=True,
                        )
                    else:
                        # Limit/budget reached: don't pretend it was downloaded
                        # as part of the final corpus.
                        print(f"\nSTOPPING | {detail}", flush=True)
                        pending.clear()
                        break

                except Exception as exc:
                    failed += 1
                    failure = {
                        "candidate_line": line_no,
                        "url": record.get("url"),
                        "technology": record.get("technology"),
                        "host": record.get("host"),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }

                    with failures_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                failure,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        handle.flush()

                # Refill one slot.
                if len(pending) < args.workers:
                    while len(pending) < args.workers:
                        try:
                            next_line_no, next_record = next(candidate_iter)
                        except StopIteration:
                            break

                        if next_line_no in completed_lines:
                            skipped += 1
                            continue

                        pending[
                            executor.submit(
                                download_one,
                                next_line_no,
                                next_record,
                            )
                        ] = (next_line_no, next_record)

            if now - last_progress >= 1.0:
                rate = attempted_this_run / max(0.001, now - started)
                remaining = max(0, remaining_at_start - attempted_this_run)
                eta = remaining / rate if rate > 0 else 0

                print(
                    "\r"
                    + " " * 220
                    + "\r"
                    + (
                        f"This run {attempted_this_run:,}/{remaining_at_start:,} "
                        f"({attempted_this_run / max(1, remaining_at_start) * 100:6.2f}%) | "
                        f"total_saved={saved_docs:,} | failed={failed:,} | "
                        f"already_saved={len(completed_lines):,} | "
                        f"fetched={format_bytes(fetched_bytes)} | "
                        f"stored={format_bytes(stored_bytes)} | "
                        f"{rate:.2f} attempts/s | ETA={format_elapsed(eta)}"
                    ),
                    end="",
                    flush=True,
                )
                last_progress = now

    elapsed = time.time() - started
    actual_output_bytes = 0

    for path in output.rglob("*"):
        if path.is_file():
            try:
                actual_output_bytes += path.stat().st_size
            except OSError:
                pass

    statistics = {
        "candidate_file": str(candidates_path),
        "total_candidates": total_candidates,
        "attempted_this_run": attempted_this_run,
        "remaining_at_start": remaining_at_start,
        "remaining_after_run": max(0, remaining_at_start - attempted_this_run),
        "saved_documents": saved_docs,
        "failed": failed,
        "skipped": skipped,
        "fetched_warc_bytes": fetched_bytes,
        "stored_html_bytes": stored_bytes,
        "output_directory_bytes": actual_output_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "workers": args.workers,
    }

    stats_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "\n\nCORPUS BUILD COMPLETE\n"
        f"Documents saved : {saved_docs:,}\n"
        f"Failed          : {failed:,}\n"
        f"Skipped         : {skipped:,}\n"
        f"HTML stored     : {format_bytes(stored_bytes)}\n"
        f"Total output    : {format_bytes(actual_output_bytes)}\n"
        f"Elapsed         : {format_elapsed(elapsed)}\n"
        f"HTML directory  : {output / 'documents' / 'html'}\n"
        f"Manifest        : {manifest_path}\n"
        f"Failures        : {failures_path}\n",
        flush=True,
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download raw technical webpages from an existing Common Crawl candidate JSONL."
    )
    parser.add_argument(
        "--candidates",
        default=str(DEFAULT_CANDIDATES),
        help="Existing common_crawl_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Corpus output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent downloads (1-16)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=DEFAULT_MAX_DOCUMENTS,
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Optional maximum total HTML corpus size; omit for unlimited",
    )

    args = parser.parse_args()

    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    try:
        return build(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
