"""
Tokenizer — Core text processing pipeline for the IR Benchmark.

This module is the shared tokenizer used by both the Classic IR (BM25/BM25F)
and Neural IR (query-side) engines. It processes raw text through:

    1. Initial tokenization (regex-based word splitting)
    2. Case folding (lowercase)
    3. Stopword removal (standard English + domain boilerplate terms)
    4. Technical token expansion (split + unsplit forms for identifiers)
    5. Version string preservation (atomic tokens like 3.11, v2.4)
    6. Porter stemming

The tokenizer operates on individual text fields (zones) and produces
per-zone token lists ready for inverted index construction.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.phase2_tokenization.normalizer import expand_technical_token, is_version_string
from src.phase2_tokenization.porter_stemmer import porter_stem


# ── Standard English stopwords ───────────────────────────────────────────
# Sourced from the classic SMART/Snowball stopword list (≈174 terms).
STANDARD_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "d", "did", "didn't", "do",
    "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "get", "got", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his",
    "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let's",
    "ll", "m", "me", "might", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "t", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "ve", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "will", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})

# ── Domain / boilerplate stopwords ───────────────────────────────────────
# Common navigation/boilerplate terms found in technical web pages (from Phase 0 audit).
DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "click", "here", "read", "more", "menu", "sidebar", "footer",
    "header", "navigation", "nav", "breadcrumb", "cookie", "cookies",
    "accept", "privacy", "policy", "terms", "conditions", "subscribe",
    "newsletter", "sign", "login", "logout", "register", "account",
    "search", "skip", "content", "main", "toggle", "close", "open",
    "previous", "next", "page", "pages", "home", "back", "top",
    "copyright", "rights", "reserved", "disclaimer", "contact",
    "feedback", "share", "tweet", "facebook", "twitter", "linkedin",
    "github", "edit", "improve", "suggest", "report", "issue",
    "loading", "please", "wait", "advertisement", "ads", "sponsor",
})

# Combined stopword set
ALL_STOPWORDS: frozenset[str] = STANDARD_STOPWORDS | DOMAIN_STOPWORDS


# ── Tokenization regex ───────────────────────────────────────────────────
# This regex extracts meaningful tokens from text, preserving:
#   - Version strings with dots: 3.11, v2.4, 1.0.0-beta
#   - Hyphenated/underscored identifiers: react-router-dom, find_unused_parameters
#   - CamelCase identifiers: getElementById
#   - Operators/symbols with alphanumeric context: @>, --, =>
#   - Plain words and numbers
#
# The pattern uses alternation ordered by specificity.
TOKEN_PATTERN = re.compile(
    r"""
    (?:[vV]?\d+(?:\.\d+)+(?:[-_.]\w+)?)   # version strings: v2.4, 3.11.0, 1.0-beta
    | (?:(?:ES|JDK|JRE|SDK)\d+)           # tech version IDs: ES2015, JDK21
    | (?:\w+(?:[-_.]\w+)+)                # hyphenated/dotted/underscored identifiers
    | (?:[A-Za-z]\w*)                     # regular words (starts with letter)
    | (?:\d+)                             # standalone numbers
    | (?:@>|=>|->|::|\$\w+|--\w+)         # operators and CLI flags
    """,
    re.VERBOSE,
)


def tokenize_raw(text: str) -> list[str]:
    """
    Extract raw tokens from text using regex.

    This is the first step — no normalization, no stemming, no stopword removal.
    Preserves version strings, identifiers, and technical operators.
    """
    return TOKEN_PATTERN.findall(text)


def tokenize(
    text: str,
    *,
    stem: bool = True,
    remove_stopwords: bool = True,
    expand_identifiers: bool = True,
    extra_stopwords: frozenset[str] | None = None,
) -> list[str]:
    """
    Full tokenization pipeline: tokenize → case fold → expand → stop → stem.

    Args:
        text: Raw input text to tokenize.
        stem: Whether to apply Porter stemming.
        remove_stopwords: Whether to remove stopwords.
        expand_identifiers: Whether to expand technical identifiers
            (e.g., 'react-router-dom' → both split and unsplit forms).
        extra_stopwords: Additional stopwords beyond the built-in sets.

    Returns:
        List of processed tokens ready for indexing or querying.
    """
    if not text:
        return []

    # Step 1: Raw tokenization
    raw_tokens = tokenize_raw(text)

    # Step 2: Expand technical identifiers and case-fold
    expanded: list[str] = []
    for token in raw_tokens:
        if expand_identifiers:
            forms = expand_technical_token(token)
            expanded.extend(forms)
        else:
            expanded.append(token.lower())

    # Step 3: Stopword removal
    stopwords = ALL_STOPWORDS
    if extra_stopwords:
        stopwords = stopwords | extra_stopwords

    if remove_stopwords:
        expanded = [t for t in expanded if t not in stopwords]

    # Step 4: Porter stemming (skip version strings and very short tokens)
    if stem:
        result: list[str] = []
        for token in expanded:
            if is_version_string(token) or len(token) <= 1:
                result.append(token)
            else:
                result.append(porter_stem(token))
        return result

    return expanded


def tokenize_zones(record: dict) -> dict[str, list[str]]:
    """
    Tokenize all zones of a structured document record.

    Produces per-zone token lists ready for inverted index construction:
        title_tokens, heading_tokens, body_tokens, code_tokens, section_tokens

    Args:
        record: A structured record from Phase 1 extraction with fields:
            doc_id, title, headings[], sections[], body_text, code_text, ...

    Returns:
        Dictionary with doc_id and per-zone token lists.
    """
    doc_id = record.get("doc_id", "unknown")

    # Title tokens
    title_text = record.get("title", "")
    title_tokens = tokenize(title_text)

    # Heading tokens — concatenate all heading texts
    headings = record.get("headings", [])
    heading_text = " ".join(h.get("text", "") for h in headings if isinstance(h, dict))
    heading_tokens = tokenize(heading_text)

    # Body tokens
    body_text = record.get("body_text", "")
    body_tokens = tokenize(body_text)

    # Code tokens — lighter processing for code (no stemming, keep identifiers)
    code_text = record.get("code_text", "")
    code_tokens = tokenize(
        code_text,
        stem=False,             # Don't stem code tokens (preserves function/class names)
        remove_stopwords=False, # Code has meaningful keywords like 'import', 'return'
        expand_identifiers=True,
    )

    # Section tokens — concatenate all section content
    sections = record.get("sections", [])
    section_text = " ".join(
        s.get("content", "") for s in sections if isinstance(s, dict)
    )
    section_tokens = tokenize(section_text)

    return {
        "doc_id": doc_id,
        "title_tokens": title_tokens,
        "heading_tokens": heading_tokens,
        "body_tokens": body_tokens,
        "code_tokens": code_tokens,
        "section_tokens": section_tokens,
        "total_token_count": (
            len(title_tokens) + len(heading_tokens) + len(body_tokens)
            + len(code_tokens) + len(section_tokens)
        ),
    }
