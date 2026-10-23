from __future__ import annotations

import re
from typing import Any

import langdetect
from langdetect import DetectorFactory

DetectorFactory.seed = 42

LANGUAGE_NAMES = {
    "en": "English",
    "eng": "English",
    "ja": "Japanese",
    "jpn": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zho": "Chinese",
    "da": "Danish",
    "dan": "Danish",
    "de": "German",
    "deu": "German",
    "fr": "French",
    "fra": "French",
    "es": "Spanish",
    "spa": "Spanish",
    "ru": "Russian",
    "rus": "Russian",
    "ko": "Korean",
    "kor": "Korean",
    "pt": "Portuguese",
    "por": "Portuguese",
    "it": "Italian",
    "ita": "Italian",
    "nl": "Dutch",
    "nld": "Dutch",
}

ERROR_PATTERNS = [
    (r"\b404\b.*(?:not found|error|page)", "404_not_found"),
    (r"\bpage not found\b", "404_not_found"),
    (r"\b403\b.*(?:forbidden|access denied)", "403_forbidden"),
    (r"\baccess denied\b", "403_access_denied"),
    (r"\b500\b.*(?:internal server error)", "500_server_error"),
    (r"\bservice unavailable\b", "503_service_unavailable"),
    (r"\bthis page does not exist\b", "404_not_found"),
    (r"\bdocument not found\b", "404_not_found"),
]

HTML_TAG_PATTERN = re.compile(r"<\s*(html|body|head|div|p|span|table|article)[^>]*>", re.IGNORECASE)


def is_non_html(html_content: str, mime_type: str | None = None) -> bool:
    """Checks if the document content is not valid HTML (e.g. pure JSON, binary, or plain text)."""
    if mime_type and "html" not in mime_type.lower():
        return True

    stripped = html_content.strip()
    if not stripped:
        return True

    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        return True

    if not HTML_TAG_PATTERN.search(html_content):
        return True

    return False


def detect_error_page(title: str, body_text: str) -> tuple[bool, str, str]:
    """
    Detects HTTP 200 responses that are actually error or redirect pages.
    Returns (is_error, error_type, matched_snippet).
    """
    combined_target = f"{title}\n{body_text[:1000]}".lower()

    for pattern, err_type in ERROR_PATTERNS:
        match = re.search(pattern, combined_target)
        if match:
            return True, err_type, match.group(0)

    return False, "none", ""


def detect_language(body_text: str, manifest_languages: str | None = None) -> dict[str, Any]:
    """
    Detects language with probabilities and maps to human-readable names.
    Cross-references manifest content_languages.
    """
    sample_text = body_text.strip()[:2000]

    if len(sample_text) < 50:
        if manifest_languages:
            is_eng = "eng" in manifest_languages.lower()
            code = manifest_languages.split(",")[0].strip()
            name = LANGUAGE_NAMES.get(code.lower(), code)
            return {
                "lang_code": code,
                "lang_name": name,
                "is_english": is_eng,
                "confidence": 0.5,
                "source": "manifest",
            }
        return {
            "lang_code": "unknown",
            "lang_name": "Unknown",
            "is_english": False,
            "confidence": 0.0,
            "source": "none",
        }

    try:
        langs = langdetect.detect_langs(sample_text)
        top_lang = langs[0]
        code = top_lang.lang
        conf = round(top_lang.prob, 3)
        is_eng = (code == "en")
        name = LANGUAGE_NAMES.get(code, code.upper())
        return {
            "lang_code": code,
            "lang_name": name,
            "is_english": is_eng,
            "confidence": conf,
            "source": "langdetect",
        }
    except Exception:
        if manifest_languages:
            is_eng = "eng" in manifest_languages.lower()
            code = manifest_languages.split(",")[0].strip()
            name = LANGUAGE_NAMES.get(code.lower(), code)
            return {
                "lang_code": code,
                "lang_name": name,
                "is_english": is_eng,
                "confidence": 0.5,
                "source": "manifest_fallback",
            }
        return {
            "lang_code": "undetermined",
            "lang_name": "Undetermined",
            "is_english": False,
            "confidence": 0.0,
            "source": "fallback",
        }
