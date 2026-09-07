"""Shared HTML-to-text helper for collectors.

Both Greenhouse and Lever return job descriptions as HTML. This is the one
shared, minimal conversion so the same few lines aren't duplicated between
collectors/greenhouse.py and collectors/lever.py.
"""

import re
from html.parser import HTMLParser
from typing import Optional


class _HTMLTextExtractor(HTMLParser):
    """Collects the visible text out of an HTML document, discarding tags."""

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def text(self) -> str:
        return " ".join(chunk.strip() for chunk in self._chunks if chunk.strip())


def html_to_text(html_content: Optional[str]) -> Optional[str]:
    """Convert an HTML job description into readable plain text.

    Tags are stripped with the standard library's HTMLParser rather than a
    full HTML library like BeautifulSoup. A small regex cleanup afterwards
    removes stray whitespace left by tag boundaries (e.g. a space before a
    period).
    """
    if not html_content:
        return None

    parser = _HTMLTextExtractor()
    parser.feed(html_content)
    text = parser.text()
    if not text:
        return None

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
