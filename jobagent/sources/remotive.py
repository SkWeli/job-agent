"""Remotive public API - free, no key. Remotive asks clients to poll only a few times a day."""
from __future__ import annotations

from ..models import Job
from .util import html_to_text

URL = "https://remotive.com/api/remote-jobs"


def fetch(query, location, settings, http) -> list[Job]:
    r = http.get(URL, params={"search": query, "limit": settings.get("limit", 50)})
    r.raise_for_status()
    return [
        Job(
            source="remotive",
            title=j.get("title", ""),
            company=j.get("company_name", ""),
            url=j.get("url", ""),
            location=j.get("candidate_required_location") or "Remote",
            description=html_to_text(j.get("description", "")),
            posted_at=j.get("publication_date", ""),
            salary=j.get("salary", "") or "",
            remote=True,
        )
        for j in r.json().get("jobs", [])
    ]
