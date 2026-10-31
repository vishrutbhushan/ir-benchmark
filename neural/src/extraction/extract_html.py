"""Ordered, heading-aware HTML extraction; independent of lexical tokenization."""

from __future__ import annotations
import hashlib
import re
from datetime import date
from lxml import etree, html
import trafilatura

SPACE = re.compile(r"\s+")
# Only document main-content selectors. Prefer the innermost article to its outer main.
SELECTORS = [
    "//article",
    "//*[@role='main']",
    "//main",
    "//*[@id='docContent']",
    "//*[@id='docs-content']",
    "//*[@id='mw-content-text']",
    "//div[contains(concat(' ',normalize-space(@class),' '),' body ')]",
    "//div[contains(concat(' ',normalize-space(@class),' '),' document ')]",
    "//div[contains(concat(' ',normalize-space(@class),' '),' refentry ')]",
    "//*[@id='content']",
]
BLOCK_TAGS = {
    "p",
    "li",
    "dt",
    "dd",
    "pre",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
}


def clean(text):
    return SPACE.sub(" ", text or "").strip()


def dates_from(text):
    result = []
    for match in re.finditer(r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})\b", text):
        try:
            value = date(*map(int, match.groups())).isoformat()
        except ValueError:
            continue
        if value not in {item["value"] for item in result}:
            result.append(
                {
                    "value": value,
                    "kind": "mention",
                    "evidence": match.group(),
                    "offset": match.start(),
                }
            )
    return result


def extract_document(raw, record):
    try:
        raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = None
    tree = html.document_fromstring(
        raw, parser=html.HTMLParser(encoding=encoding, recover=True, no_network=True)
    )
    title = (
        clean(" ".join(tree.xpath("//title//text()")))
        or (record.get("title") if record.get("title") != record["url"] else "")
        or ""
    )
    selected = None
    method = "trafilatura_fallback"
    for selector in SELECTORS:
        candidates = [
            n for n in tree.xpath(selector) if len(clean(n.text_content())) >= 60
        ]
        if candidates:
            selected = max(candidates, key=lambda n: len(n.text_content()))
            method = selector
            break
    if record.get("technology") == "RFCs":
        selected = tree.find("body")
        if selected is None:
            selected = tree
        method = "rfc_original_preformatted_body"
    if selected is None:
        # Trafilatura is the boilerplate fallback; DOM order/code remain authoritative below.
        extracted = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="xml",
            favor_recall=True,
        )
        if extracted:
            selected = html.fromstring(extracted)
            for node in selected.xpath(".//code"):
                node.tag = "pre"
        else:
            selected = tree.find("body")
            if selected is None:
                selected = tree
            method = "body_fallback"
    etree.strip_elements(
        selected,
        "script",
        "style",
        "nav",
        "footer",
        "aside",
        "noscript",
        "form",
        "button",
        "svg",
        "iframe",
        with_tail=False,
    )
    # Remove sidebar/menu widgets without stripping legitimate article headers or API headings.
    for node in list(
        selected.xpath(
            ".//*[@role='navigation' or @aria-hidden='true'] | .//*[contains(concat(' ',normalize-space(@class),' '),' sidebar ')] | .//*[contains(concat(' ',normalize-space(@class),' '),' toc ')]"
        )
    ):
        if node.getparent() is not None:
            node.drop_tree()
    headings, sections, blocks, stack = [], [], [], []
    current = {"heading_path": [], "anchor": "", "blocks": []}
    sections.append(current)
    pre_is_prose = record.get("technology") == "RFCs"

    def emit(node):
        nonlocal current, stack
        tag = str(node.tag).lower()
        if re.fullmatch("h[1-6]", tag):
            text = clean(node.text_content())
            if not text:
                return
            level = int(tag[1])
            stack = [(l, t) for l, t in stack if l < level] + [(level, text)]
            headings.append(text)
            current = {
                "heading_path": [t for _, t in stack],
                "anchor": node.get("id") or "",
                "blocks": [],
            }
            sections.append(current)
            return
        if tag == "pre" or (tag == "code" and "\n" in node.text_content()):
            text = node.text_content().replace("\r\n", "\n").strip("\n")
            kind = "prose" if pre_is_prose else "code"
        elif tag == "table":
            text, kind = clean(" ".join(node.itertext())), "prose"
        else:
            text, kind = (
                clean(node.text_content()),
                "list" if tag in {"li", "dt", "dd"} else "prose",
            )
        if not text.strip():
            return
        block = {
            "block_id": len(blocks),
            "kind": kind,
            "text": text,
            "anchor": node.get("id") or current["anchor"],
        }
        blocks.append(block)
        current["blocks"].append(block)

    def walk(node):
        tag = str(node.tag).lower()
        if tag in {"p", "li", "dd", "blockquote"} and (
            node.xpath(".//pre|.//table|.//h1|.//h2|.//h3")
            or any("\n" in c.text_content() for c in node.xpath(".//code"))
        ):
            tag = "container"
        if tag in BLOCK_TAGS or (tag == "code" and "\n" in node.text_content()):
            # Nested lists are emitted exactly once as part of their parent item.
            emit(node)
            return
        children = [c for c in node if isinstance(c.tag, str)]
        if not children:
            text = clean(node.text_content())
            if text:
                emit(node)
            return
        if node.text and node.text.strip():
            fake = html.Element("p")
            fake.text = node.text
            emit(fake)
        for child in children:
            walk(child)
            if child.tail and child.tail.strip():
                fake = html.Element("p")
                fake.text = child.tail
                emit(fake)

    walk(selected)
    sections = [s for s in sections if s["blocks"]]
    body = "\n\n".join(b["text"] for b in blocks if b["kind"] != "code")
    code = "\n\n".join(b["text"] for b in blocks if b["kind"] == "code")
    return {
        "doc_id": record["doc_id"],
        "url": record["url"],
        "title": title,
        "headings": headings,
        "sections": sections,
        "body_text": body,
        "code_text": code,
        "dates": dates_from(title + "\n" + body),
        "metadata": {
            **record,
            "extraction_method": method,
            "block_count": len(blocks),
            "content_sha256": hashlib.sha256((body + "\n" + code).encode()).hexdigest(),
            "date_semantics": "mentions, not publication dates",
        },
    }
