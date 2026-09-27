from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from src.phase1_extraction.boilerplate import extract_clean_body_and_code

# Regex patterns for date detection
DATE_PATTERNS = [
    # ISO 8601: 2026-09-27 or 2026/09/27
    r"\b(20\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01]))\b",
    # Textual: September 27, 2026 or 27 Sep 2026
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2})\b",
    r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2})\b",
]


def extract_normalized_dates(text: str, max_dates: int = 10) -> list[str]:
    """Finds date-like strings in text and normalizes them to ISO format (YYYY-MM-DD)."""
    detected_iso_dates: set[str] = set()

    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match if isinstance(match, str) else match[0]
            try:
                dt = date_parser.parse(date_str, fuzzy=True)
                # Keep reasonable year range for technical documentation
                if 1990 <= dt.year <= 2030:
                    detected_iso_dates.add(dt.strftime("%Y-%m-%d"))
            except Exception:
                continue

            if len(detected_iso_dates) >= max_dates:
                break
        if len(detected_iso_dates) >= max_dates:
            break

    return sorted(list(detected_iso_dates))


def extract_headings_and_sections(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extracts heading hierarchy (h1/h2/h3) and sections grouping text under headings.
    """
    headings: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []

    # Find all headings in document order
    heading_tags = soup.find_all(re.compile(r"^h[1-6]$"))
    for h in heading_tags:
        level = int(h.name[1])
        text = h.get_text(separator=" ", strip=True)
        if text:
            headings.append({"level": level, "text": text})

    # Group sections: walk elements between headings
    body = soup.find("body") or soup
    current_heading = "Introduction"
    current_level = 1
    current_content: list[str] = []

    for elem in body.descendants:
        if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Commit previous section if it has content
            sec_text = " ".join(current_content).strip()
            if sec_text:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "content": sec_text[:2000],  # cap section content
                })
            current_heading = elem.get_text(separator=" ", strip=True) or current_heading
            current_level = int(elem.name[1])
            current_content = []
        elif elem.name in ["p", "li", "td", "pre"]:
            t = elem.get_text(separator=" ", strip=True)
            if t and len(t) > 20:
                current_content.append(t)

    # Commit last section
    final_text = " ".join(current_content).strip()
    if final_text:
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "content": final_text[:2000],
        })

    return headings, sections


def extract_anchor_texts(soup: BeautifulSoup, max_anchors: int = 50) -> list[str]:
    """Extracts meaningful anchor texts (excluding generic links like 'home', 'back')."""
    generic_stopwords = {"here", "link", "read more", "more", "click here", "back", "home"}
    anchors: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if text and len(text) > 3 and text.lower() not in generic_stopwords:
            if text.lower() not in seen:
                seen.add(text.lower())
                anchors.append(text)
                if len(anchors) >= max_anchors:
                    break
    return anchors


def parse_html_to_structured_record(
    html_content: str,
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Parses an HTML document into a structured record conforming to the project schema:
    doc_id, url, title, headings[], sections[], body_text, code_text, dates[], metadata.
    """
    doc_id = manifest_record.get("doc_id", "unknown_id")
    url = manifest_record.get("url", "")
    manifest_title = manifest_record.get("title", "")

    soup = BeautifulSoup(html_content, "html.parser")

    # Determine title (from <title> tag, or <h1>, or manifest)
    html_title_tag = soup.find("title")
    title = html_title_tag.get_text(strip=True) if html_title_tag else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else manifest_title

    # Extract headings and section hierarchy
    headings, sections = extract_headings_and_sections(soup)

    # Extract boilerplate-free body and isolated code blocks
    body_and_code = extract_clean_body_and_code(html_content, url=url)
    body_text = body_and_code["body_text"]
    code_text = body_and_code["code_text"]

    # Extract anchors
    anchor_texts = extract_anchor_texts(soup)

    # Extract dates
    combined_text_for_dates = f"{title}\n{body_text[:4000]}"
    dates = extract_normalized_dates(combined_text_for_dates)

    # Metadata
    metadata = {
        "technology": manifest_record.get("technology", "Unknown"),
        "category": manifest_record.get("category", "Unknown"),
        "host": manifest_record.get("host") or urlparse(url).netloc,
        "html_bytes": manifest_record.get("html_bytes", len(html_content)),
        "content_languages": manifest_record.get("content_languages", ""),
        "crawl": manifest_record.get("crawl", ""),
        "code_blocks_count": body_and_code["code_blocks_count"],
        "num_headings": len(headings),
        "num_sections": len(sections),
        "num_anchors": len(anchor_texts),
    }

    return {
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "headings": headings,
        "sections": sections,
        "body_text": body_text,
        "code_text": code_text,
        "anchor_texts": anchor_texts,
        "dates": dates,
        "metadata": metadata,
    }
