from __future__ import annotations

import collections
import random
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.utils.file_utils import stream_jsonl


JS_INDICATORS = [
    r"you need to enable javascript",
    r"javascript is required",
    r"please enable javascript",
    r"loading\.\.\.",
    r"<div id=[\"'](root|app|__next)[\"']>\s*</div>",
]


def extract_body_text_metrics(html_content: str) -> dict[str, Any]:
    """Extracts raw body text and calculates length and JS-render indicators."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    body = soup.find("body")
    body_text = body.get_text(separator=" ", strip=True) if body else ""
    char_length = len(body_text)
    words = body_text.split()
    word_count = len(words)

    # Check for empty / JS-only indicators
    js_stub_detected = False
    lower_html = html_content.lower()
    for pattern in JS_INDICATORS:
        if re.search(pattern, lower_html):
            js_stub_detected = True
            break

    return {
        "body_text": body_text,
        "char_length": char_length,
        "word_count": word_count,
        "has_body": body is not None,
        "js_stub_detected": js_stub_detected,
    }


def sample_manifest_documents(
    manifest_path: Path | str,
    sample_size_per_tech: int | None = 10,
    total_sample_limit: int | None = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Selects a stratified sample of documents across all technologies from the manifest."""
    random.seed(seed)
    tech_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for record in stream_jsonl(manifest_path):
        tech = record.get("technology", "Unknown")
        tech_buckets[tech].append(record)

    sampled_records: list[dict[str, Any]] = []

    for tech, records in tech_buckets.items():
        if sample_size_per_tech is not None:
            k = min(len(records), sample_size_per_tech)
            sampled_records.extend(random.sample(records, k))
        else:
            sampled_records.extend(records)

    random.shuffle(sampled_records)

    if total_sample_limit is not None and len(sampled_records) > total_sample_limit:
        sampled_records = sampled_records[:total_sample_limit]

    return sampled_records
