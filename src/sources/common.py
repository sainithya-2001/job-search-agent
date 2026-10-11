"""Helpers shared by the ATS source modules."""
import html
import re
from datetime import datetime, timezone
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(raw: str | None) -> str:
    """Strip HTML to readable plain text. Greenhouse double-escapes its HTML, so unescape first."""
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(raw))
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def parse_datetime(value) -> datetime | None:
    """Accept ISO-8601 strings or epoch milliseconds (Lever); return a tz-aware UTC datetime."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError):
        return None
