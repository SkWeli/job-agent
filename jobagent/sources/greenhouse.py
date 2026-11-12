"""Greenhouse job boards - public, no key. Token = slug in boards.greenhouse.io/<token>."""
from __future__ import annotations

from ..models import Job
from .util import html_to_text, matches


def fetch(query, location, settings, http) -> list[Job]:
    out = []
    for token in settings.get("companies", []):
        r = http.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs", params={"content": "true"})
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for j in r.json().get("jobs", []):
            job = Job(
                source=f"greenhouse:{token}",
                title=j.get("title", ""),
                company=j.get("company_name") or token,
                url=j.get("absolute_url", ""),
                location=(j.get("location") or {}).get("name", ""),
                description=html_to_text(j.get("content", "")),
                posted_at=j.get("updated_at", ""),
            )
            if matches(job, query):
                out.append(job)
    return out
