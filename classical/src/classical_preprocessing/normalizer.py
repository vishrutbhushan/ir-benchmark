"""
Normalizer — Handles special token normalization for technical content.

Responsibilities:
- Version string detection and atomic preservation (e.g., '3.11', 'v2.4')
- Technical identifier splitting (e.g., 'react-router-dom' → both split and unsplit forms)
- CamelCase splitting (e.g., 'getElementById' → split components + full form)
- Date token handling (dates already extracted in Phase 1 as ISO fields)
"""

from __future__ import annotations

import re


# ── Version string patterns ──────────────────────────────────────────────
# Matches: v2.4, 3.11, 1.0.0, v10.15.3, 2.0-beta, ES2015, JDK21, etc.
VERSION_PATTERN = re.compile(
    r"""
    ^(?:
        [vV]?\d+(?:\.\d+)+(?:[-_.]\w+)?   # v2.4, 3.11.0, 1.0.0-beta
        | [vV]\d+(?:\.\d+)*               # v2, v10
        | (?:ES|JDK|JRE|SDK)\d+            # ES2015, JDK21
    )$
    """,
    re.VERBOSE,
)


def is_version_string(token: str) -> bool:
    """Check if a token is a version string that should be kept atomic."""
    return bool(VERSION_PATTERN.match(token))


# ── CamelCase splitter ───────────────────────────────────────────────────
# Splits on camelCase / PascalCase boundaries.
# "getElementById"  → ["get", "element", "by", "id"]
# "XMLHttpRequest"  → ["xml", "http", "request"]
# "FastAPI"         → ["fast", "api"]
CAMEL_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z])       # camelCase boundary: lower→Upper
    | (?<=[A-Z])(?=[A-Z][a-z])  # acronym boundary: UPPER→Upperlower (e.g., XMLHttp → XML|Http)
    """,
    re.VERBOSE,
)


def split_camel_case(token: str) -> list[str]:
    """
    Split a camelCase/PascalCase identifier into its constituent words.

    Returns lowercase components. Returns empty list if the token has no
    camelCase boundaries (i.e., is already a single word).
    """
    parts = CAMEL_SPLIT_RE.split(token)
    if len(parts) <= 1:
        return []
    return [p.lower() for p in parts if p]


# ── Hyphen / underscore / dot identifier splitter ────────────────────────
DELIMITER_SPLIT_RE = re.compile(r"[-_.]")


def split_technical_identifier(token: str) -> list[str]:
    """
    Split a hyphenated, underscored, or dotted technical identifier into
    its sub-tokens.

    E.g., 'react-router-dom' → ['react', 'router', 'dom']
          'find_unused_parameters' → ['find', 'unused', 'parameters']

    Returns empty list if no delimiter is found.
    """
    parts = DELIMITER_SPLIT_RE.split(token)
    if len(parts) <= 1:
        return []
    return [p.lower() for p in parts if p]


def expand_technical_token(token: str) -> list[str]:
    """
    Given a single raw token, return a list of all index-able forms.

    The strategy: keep the original (lowercased) form AND add all split
    components from camelCase or delimiter-based splitting.

    Examples:
        'react-router-dom' → ['react-router-dom', 'react', 'router', 'dom']
        'getElementById'   → ['getelementbyid', 'get', 'element', 'by', 'id']
        'BM25F'            → ['bm25f', 'bm', '25', 'f']  (camel split)
        'python'           → ['python']  (no expansion)
        '3.11'             → ['3.11']    (version: atomic, no split)
        'v2.4'             → ['v2.4']    (version: atomic, no split)
    """
    lower = token.lower()
    result = [lower]

    # Version strings: keep atomic, do NOT split
    if is_version_string(token):
        return result

    # Try delimiter-based split (hyphens, underscores, dots)
    delim_parts = split_technical_identifier(token)
    if delim_parts:
        result.extend(delim_parts)
        # Also try camelCase split on each sub-part
        for part in delim_parts:
            camel_parts = split_camel_case(part)
            if camel_parts:
                result.extend(camel_parts)
        return _dedupe_preserve_order(result)

    # Try camelCase split
    camel_parts = split_camel_case(token)
    if camel_parts:
        result.extend(camel_parts)
        return _dedupe_preserve_order(result)

    return result


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates from a list while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
