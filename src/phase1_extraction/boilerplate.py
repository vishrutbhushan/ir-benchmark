from __future__ import annotations

import logging
from typing import Any

from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)


def extract_clean_body_and_code(html_content: str, url: str | None = None) -> dict[str, Any]:
    """
    Extracts boilerplate-free body text using trafilatura, with a robust BeautifulSoup
    fallback. Also isolates code blocks from <pre> and <code> elements.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Extract explicit code blocks
    code_snippets: list[str] = []
    for pre in soup.find_all("pre"):
        code_text = pre.get_text(strip=True)
        if code_text:
            code_snippets.append(code_text)

    # If few or no <pre> blocks, check standalone <code> blocks that are substantial
    if len(code_snippets) < 3:
        for code in soup.find_all("code"):
            # Only include code if it is not inside a pre already captured
            if code.find_parent("pre") is None:
                snippet = code.get_text(strip=True)
                if len(snippet) > 40 or "\n" in snippet:
                    code_snippets.append(snippet)

    combined_code_text = "\n\n".join(code_snippets)

    # 2. Extract main text using trafilatura
    trafilatura_text = None
    try:
        trafilatura_text = trafilatura.extract(
            html_content,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
    except Exception as e:
        logger.debug(f"Trafilatura extraction exception: {e}")

    # 3. Fallback if trafilatura returns nothing or fails
    if trafilatura_text and len(trafilatura_text.strip()) > 50:
        main_body = trafilatura_text.strip()
    else:
        # Fallback to BeautifulSoup clean text extraction
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
            element.decompose()
        body = soup.find("body") or soup
        main_body = body.get_text(separator="\n", strip=True)

    return {
        "body_text": main_body,
        "code_text": combined_code_text,
        "code_blocks_count": len(code_snippets),
    }
