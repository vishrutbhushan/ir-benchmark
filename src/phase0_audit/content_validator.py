from __future__ import annotations

import re
from typing import Any

import langdetect
from langdetect import DetectorFactory

# Enforce deterministic language detection
DetectorFactory.seed = 42


ERROR_PATTERNS = [
    r"\b404\b.*(?:not found|error|page)",
    r"\bpage not found\b",
    r"\b403\b.*(?:forbidden|access denied)",
    r"\baccess denied\b",
    r"\b500\b.*(?:internal server error)",
    r"\bservice unavailable\b",
    r"\bthis page does not exist\b",
    r"\bdocument not found\b",
]

HTML_TAG_PATTERN = re.compile(r"<\s*(html|body|head|div|p|span|table|article)[^>]*>", re.IGNORECASE)


def is_non_html(html_content: str, mime_type: str | None = None) -> bool:
    """Checks if the document content is not valid HTML (e.g., pure JSON, binary, or plain text)."""
    if mime_type and "html" not in mime_type.lower():
        return True

    stripped = html_content.strip()
    if not stripped:
        return True

    # Check for raw JSON or XML without HTML structure
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        return True

    # Check for basic HTML tags
    if not HTML_TAG_PATTERN.search(html_content):
        return True

    return False


def detect_error_page(title: str, body_text: str) -> tuple[bool, str]:
    """Detects HTTP 200 responses that are actually error or redirect pages."""
    combined_target = f"{title}\n{body_text[:1000]}".lower()

    for pattern in ERROR_PATTERNS:
        match = re.search(pattern, combined_target)
        if match:
            return True, match.group(0)

    return False, ""


def detect_language(body_text: str, manifest_languages: str | None = None) -> tuple[str, bool]:
    """
    Detects language using langdetect and cross-references manifest content_languages.
    Returns (detected_language, is_english).
    """
    # Quick text check
    sample_text = body_text.strip()[:2000]

    if len(sample_text) < 50:
        # If text is too short, trust manifest if available
        if manifest_languages:
            is_eng = "eng" in manifest_languages.lower()
            return manifest_languages, is_eng
        return "unknown", False

    try:
        lang = langdetect.detect(sample_text)
        is_eng = (lang == "en")
        return lang, is_eng
    except Exception:
        # Fallback to manifest languages
        if manifest_languages:
            is_eng = "eng" in manifest_languages.lower()
            return manifest_languages, is_eng
        return "undetermined", False
