"""Conservative raw-HTML checks. No model tokenization or cleaned text is saved."""
from __future__ import annotations

import hashlib
import re

from langdetect import DetectorFactory, detect_langs
from lxml import etree, html

DetectorFactory.seed = 42

_ERROR_TITLE = re.compile(
    r"^(?:(?:404|403|500|503)\s*[-:|]?\s*)?"
    r"(?:page not found|not found|access denied|forbidden|internal server error|"
    r"service unavailable|just a moment(?:\.\.\.)?|attention required!?)"
    r"(?:\s*[-|:].*)?$", re.I,
)


def inspect_html(raw: bytes, record: dict) -> dict:
    flags: list[str] = []
    rejected: list[str] = []
    try:
        raw.decode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        utf8_valid = False  # Another declared encoding is valid HTML, not file corruption.
        flags.append("non_utf8_bytes")

    if raw.lstrip().startswith((b"WARC/", b"HTTP/")):
        rejected.append("transport_headers_in_html")
    parser = html.HTMLParser(recover=True, no_network=True, encoding="utf-8" if utf8_valid else None)
    try:
        tree = html.document_fromstring(raw, parser=parser)
    except (etree.ParserError, ValueError) as error:
        return {"status": "exclude", "reasons": ["html_parse_failed"],
                "flags": flags, "parse_error": str(error), "utf8_valid": utf8_valid}

    has_markup = bool(re.search(rb"<\s*(?:html|head|body|main|article|div|p|table|pre|code|span|h[1-6]|ul|ol|li|dl|blockquote|section|a)\b", raw, re.I))
    if not has_markup:
        rejected.append("not_html_markup")
    if not re.search(rb"</html\s*>", raw, re.I):
        flags.append("no_closing_html_tag")  # HTML permits omitted closing tags.
    html_lang = (tree.get("lang") or tree.get("{http://www.w3.org/XML/1998/namespace}lang") or "").lower()
    declared_lang = html_lang.split("-")[0].split("_")[0]
    title = " ".join(tree.xpath("//title//text()"))[:1000].strip()
    h1 = " ".join(tree.xpath("//h1[1]//text()"))[:1000].strip()
    preformatted_document = record.get("technology") == "RFCs"
    code_nodes = tree.xpath("//code[not(ancestor::pre)]" if preformatted_document else "//pre | //code[not(ancestor::pre)]")
    code_chars = sum(len(node.text_content()) for node in code_nodes)
    heading_count = len(tree.xpath("//h1|//h2|//h3|//h4|//h5|//h6"))
    table_count = len(tree.xpath("//table"))
    main_nodes = tree.xpath("//main | //article | //*[@role='main']")
    region = max(main_nodes, key=lambda node: len(node.text_content()), default=tree)
    # This disposable DOM is used only for quality evidence; original bytes stay intact.
    etree.strip_elements(region, "script", "style", "nav", "footer", "header", "aside", "noscript", with_tail=False)
    visible = " ".join(region.text_content().split())
    visible_hash = hashlib.sha256((title + "\n" + visible).encode()).hexdigest()
    if not preformatted_document:
        etree.strip_elements(region, "pre", "code", with_tail=False)
    prose = " ".join(region.text_content().split())

    # A troubleshooting article mentioning 404 is not itself a 404 page.
    if _ERROR_TITLE.fullmatch(title) and len(prose) < 2000:
        rejected.append("error_or_challenge_page")
    if len(visible) < 200:
        flags.append("short_visible_text")
        if len(visible) < 30 and code_chars < 30:
            rejected.append("empty_rendered_content")
        elif re.search(r"(?:enable|requires?) javascript|javascript (?:is )?required", visible, re.I):
            rejected.append("javascript_shell")
    if not title:
        flags.append("missing_title")

    manifest_languages = str(record.get("content_languages") or "").lower()
    language, confidence, language_source = "unknown", 0.0, "insufficient_prose"
    if declared_lang == "en" and manifest_languages == "eng":
        language, confidence, language_source = "en", None, "html_and_manifest_agree"
    elif len(prose) >= 200:
        try:
            result = detect_langs(prose[:4000])[0]
            language, confidence, language_source = result.lang, round(result.prob, 5), "prose_detector"
        except Exception:
            flags.append("language_detection_failed")
    if language not in {"en", "unknown"} and confidence is not None and confidence >= 0.95:
        if declared_lang == language or ("eng" not in manifest_languages and confidence >= 0.99):
            rejected.append("confirmed_non_english")
        else:
            flags.append("language_evidence_conflict")
    elif declared_lang and declared_lang != "en":
        flags.append("non_english_html_language")
    if language == "unknown":
        flags.append("language_uncertain")
    review_flags = {"language_evidence_conflict", "non_english_html_language", "language_uncertain", "short_visible_text"}
    status = "exclude" if rejected else ("review" if review_flags.intersection(flags) else "eligible")
    return {
        "status": status, "reasons": rejected, "flags": flags,
        "html_lang": html_lang, "detected_language": language,
        "language_confidence": confidence, "language_source": language_source,
        "visible_chars": len(visible), "prose_chars": len(prose),
        "code_chars": code_chars, "headings": heading_count, "tables": table_count,
        "title": title, "first_heading": h1, "utf8_valid": utf8_valid,
        "visible_content_sha256": visible_hash,
        "preformatted_document": preformatted_document,
    }
