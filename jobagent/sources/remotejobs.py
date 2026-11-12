"""remotejobs.org - free, no key. Docs: https://remotejobs.org/developers/api"""
from __future__ import annotations

from ..models import Job
from .util import html_to_text

URL = "https://remotejobs.org/api/v1/jobs"


def fetch(query, location, settings, http) -> list[Job]:
    params = {"q": query, "limit": settings.get("limit", 50), "sort": "recent"}
    r = http.get(URL, params=params)
    r.raise_for_status()
    out = []
    for j in r.json().get("data", []):
        out.append(Job(
            source="remotejobs",
            title=j.get("title", ""),
            company=(j.get("company") or {}).get("name", ""),
            url=j.get("apply_url") or j.get("url", ""),
            location=j.get("location") or "Remote",
            description=html_to_text(j.get("description", "")),
            posted_at=j.get("posted_at", ""),
            salary=j.get("salary_text") or "",
            remote=True,
            extra={"applicant_location_requirements": j.get("applicant_location_requirements")},
        ))
    return out
