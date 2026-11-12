"""Jooble aggregator - free key from https://jooble.org/api/about. Covers Sri Lanka.

Free keys have a limited lifetime request budget and return description snippets only.
"""
from __future__ import annotations

from ..models import Job
from .util import html_to_text


def fetch(query, location, settings, http) -> list[Job]:
    key = settings.get("api_key")
    if not key:
        raise RuntimeError("Jooble needs an API key (set JOOBLE_API_KEY). Use another source.")
    r = http.post(f"https://jooble.org/api/{key}", json={"keywords": query, "location": location or "", "page": "1"})
    r.raise_for_status()
    return [
        Job(
            source="jooble",
            title=html_to_text(j.get("title", "")),
            company=j.get("company", ""),
            url=j.get("link", ""),
            location=j.get("location", ""),
            description=html_to_text(j.get("snippet", "")),
            posted_at=j.get("updated", ""),
            salary=j.get("salary", ""),
            extra={"snippet_only": True},
        )
        for j in r.json().get("jobs", [])
    ]
