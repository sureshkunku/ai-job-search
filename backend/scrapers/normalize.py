"""
normalize.py — turn messy source data into clean, comparable values.
Module 3: Scrapers

WHY a separate module:
    Every job API is messy in its own way (we checked the real responses):
      - RemoteOK sends mojibake: "â€”" instead of "—" (UTF-8 bytes that were
        decoded as Latin-1 somewhere upstream).
      - Arbeitnow sometimes double-escapes HTML: "&lt;p&gt;" instead of "<p>".
      - Dates arrive as Unix epochs (RemoteOK, Arbeitnow, Himalayas) or
        ISO strings (The Muse).
      - Salaries are 0, null, monthly, hourly, or in PHP/EUR/USD.
    Fixing these once, here, means every scraper stays tiny and every job
    reaches the database (and later the LLM) in the same clean shape.
    Clean text also matters for cost: HTML tags are wasted LLM tokens.
"""
from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from html.parser import HTMLParser

# ── Text ─────────────────────────────────────────────────────────────────────

# Characters that show up when UTF-8 text was wrongly decoded as Latin-1/cp1252
_MOJIBAKE_MARKERS = ("Ã", "Â", "â\x80", "â€")


def fix_mojibake(text: str) -> str:
    """
    Repair text like 'Engineer â€” Remote' → 'Engineer — Remote'.

    How: re-encode the broken string back to the bytes it came from
    (Latin-1 / cp1252), then decode those bytes properly as UTF-8.
    If that fails, the text wasn't mojibake after all — return it unchanged.
    """
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text  # fast path: most text is fine
    for wrong_codec in ("latin-1", "cp1252"):
        try:
            return text.encode(wrong_codec).decode("utf-8")
        except UnicodeError:
            continue
    return text


class _TextExtractor(HTMLParser):
    """Collects the visible text from HTML, keeping paragraph/list breaks."""

    BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "section",
                  "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # turns &nbsp; &amp; etc. into chars
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("• ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(raw: str | None) -> str:
    """
    HTML (possibly double-escaped) → readable plain text.

    WHY not BeautifulSoup: the stdlib HTMLParser is enough for job
    descriptions and keeps our dependency list short.
    """
    if not raw:
        return ""
    text = raw
    # Double-escaped feeds: "&lt;p&gt;Hello&lt;/p&gt;" → "<p>Hello</p>".
    # At most twice, so a JD that legitimately *shows* "&lt;" stays readable.
    for _ in range(2):
        if "&lt;" in text or "&amp;" in text or "&quot;" in text:
            text = html.unescape(text)
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t  ‑]+", " ", text)   # odd spaces → normal space
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()          # max one blank line


def clean_text(raw: str | None) -> str:
    """The one call scrapers use: fix encoding, strip HTML, tidy whitespace."""
    return html_to_text(fix_mojibake(raw)) if raw else ""


# ── Dates ────────────────────────────────────────────────────────────────────

def from_epoch(value: int | float | str | None) -> datetime | None:
    """Unix seconds → timezone-aware UTC datetime. 0/None/'' → None."""
    if value in (None, "", 0, "0"):
        return None
    try:
        return datetime.fromtimestamp(int(float(value)), tz=UTC)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def from_iso(value: str | None) -> datetime | None:
    """ISO-8601 string (with 'Z' or an offset) → UTC datetime, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ── Salary ───────────────────────────────────────────────────────────────────

def salary_text(
    low: int | float | None,
    high: int | float | None,
    currency: str | None = "USD",
    period: str | None = None,
) -> str | None:
    """
    Build the human-readable `RawJob.salary_raw` string, e.g. 'USD 120,000–150,000 /year'.

    WHY a string and not numbers: sources mix currencies and periods
    (monthly PHP, hourly USD...). Converting correctly needs judgement, so
    the Enrichment Agent (Module 6) parses this into salary_min/max later.
    APIs use 0 for "not given" — treat 0 the same as missing.
    """
    lo = int(low) if low else None
    hi = int(high) if high else None
    if lo is None and hi is None:
        return None
    if lo and hi:
        amount = f"{lo:,}–{hi:,}" if lo != hi else f"{lo:,}"
    else:
        amount = f"{(lo or hi):,}+" if lo else f"up to {hi:,}"
    per = {"annual": "/year", "yearly": "/year", "monthly": "/month",
           "hourly": "/hour", "weekly": "/week", "daily": "/day"}.get((period or "").lower(), "")
    return f"{currency or 'USD'} {amount} {per}".strip()


# ── Keyword filter ───────────────────────────────────────────────────────────

def matches_keywords(text: str, keywords: list[str]) -> bool:
    """
    True if any keyword appears in `text` as a whole word/phrase (case-insensitive).

    WHY whole-word: plain substring matching makes "ml" match "HTML" and
    "ai" match "maintenance". The look-arounds require a non-letter/digit
    (or string edge) on both sides. An empty keyword list keeps everything.
    """
    if not keywords:
        return True
    haystack = text.lower()
    for keyword in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            return True
    return False
