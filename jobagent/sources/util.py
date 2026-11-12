from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

UA = "job-agent/0.2 (personal job search)"


def html_to_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)  # Greenhouse double-escapes its HTML
    text = BeautifulSoup(s, "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def matches(job, query: str) -> bool:
    """Loose keyword match for feeds that return every job (company boards)."""
    words = [w for w in re.split(r"\W+", (query or "").lower()) if len(w) > 2]
    blob = f"{job.title} {job.description}".lower()
    return not words or any(w in blob for w in words)
