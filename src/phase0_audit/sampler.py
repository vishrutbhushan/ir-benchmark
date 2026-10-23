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
    """
    Extracts deep DOM and content metrics:
    body text length, code blocks & density, headings, tables, lists, and boilerplate ratio.
    """
    raw_bytes = len(html_content.encode("utf-8", errors="replace"))
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Structural counts before mutation
    tables_count = len(soup.find_all("table"))
    lists_count = len(soup.find_all(["ul", "ol"]))
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    headings_count = len(heading_tags)
    heading_levels = [int(h.name[1]) for h in heading_tags]
    max_heading_depth = max(heading_levels) if heading_levels else 0

    # 2. Code snippet isolation
    code_snippets: list[str] = []
    for pre in soup.find_all("pre"):
        t = pre.get_text(strip=True)
        if t:
            code_snippets.append(t)
    for code in soup.find_all("code"):
        if code.find_parent("pre") is None:
            t = code.get_text(strip=True)
            if len(t) > 30 or "\n" in t:
                code_snippets.append(t)

    code_blocks_count = len(code_snippets)
    code_text = "\n".join(code_snippets)
    code_chars = len(code_text)
    has_code = code_blocks_count > 0

    # 3. Clean body text extraction
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    body = soup.find("body")
    body_text = body.get_text(separator=" ", strip=True) if body else ""
    char_length = len(body_text)
    words = body_text.split()
    word_count = len(words)
    text_bytes = len(body_text.encode("utf-8", errors="replace"))

    # Boilerplate ratio: fraction of raw HTML consumed by markup/scripts/chrome
    boilerplate_ratio = round(1.0 - (text_bytes / max(1, raw_bytes)), 4)

    # 4. Check for JS stub indicators
    js_stub_detected = False
    lower_html = html_content.lower()
    for pattern in JS_INDICATORS:
        if re.search(pattern, lower_html):
            js_stub_detected = True
            break

    code_density = round(code_chars / max(1, char_length), 4)

    return {
        "body_text": body_text,
        "char_length": char_length,
        "word_count": word_count,
        "raw_bytes": raw_bytes,
        "text_bytes": text_bytes,
        "boilerplate_ratio": boilerplate_ratio,
        "has_body": body is not None,
        "js_stub_detected": js_stub_detected,
        # Code metrics
        "has_code": has_code,
        "code_blocks_count": code_blocks_count,
        "code_chars": code_chars,
        "code_density": code_density,
        # DOM structure metrics
        "headings_count": headings_count,
        "max_heading_depth": max_heading_depth,
        "tables_count": tables_count,
        "lists_count": lists_count,
    }


def sample_manifest_documents(
    manifest_path: Path | str,
    sample_size_per_tech: int | None = 15,
    sample_size_per_category: int | None = None,
    total_sample_limit: int | None = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Selects a stratified sample with balanced representation across categories and technologies.
    If sample_size_per_category is provided, all 11 categories receive equal target quotas.
    """
    random.seed(seed)

    # Group by category -> technology -> records
    cat_tech_buckets: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for record in stream_jsonl(manifest_path):
        cat = record.get("category", "unknown")
        tech = record.get("technology", "Unknown")
        cat_tech_buckets[cat][tech].append(record)

    sampled_records: list[dict[str, Any]] = []

    if sample_size_per_category is not None:
        # Category-balanced sampling: equal quota per category
        for cat, techs in cat_tech_buckets.items():
            cat_sample: list[dict[str, Any]] = []
            tech_count = max(1, len(techs))
            per_tech_quota = max(1, sample_size_per_category // tech_count)

            for tech, records in techs.items():
                k = min(len(records), per_tech_quota)
                cat_sample.extend(random.sample(records, k))

            # If small technologies didn't fill the category quota, fill from other techs in this category
            if len(cat_sample) < sample_size_per_category:
                remaining_needed = sample_size_per_category - len(cat_sample)
                all_cat_records = [r for t_recs in techs.values() for r in t_recs if r not in cat_sample]
                if all_cat_records:
                    k_fill = min(len(all_cat_records), remaining_needed)
                    cat_sample.extend(random.sample(all_cat_records, k_fill))

            sampled_records.extend(cat_sample)
    else:
        # Technology-stratified sampling: equal quota per technology
        for cat, techs in cat_tech_buckets.items():
            for tech, records in techs.items():
                k = min(len(records), sample_size_per_tech or 15)
                sampled_records.extend(random.sample(records, k))

    random.shuffle(sampled_records)

    if total_sample_limit is not None and len(sampled_records) > total_sample_limit:
        sampled_records = sampled_records[:total_sample_limit]

    return sampled_records
