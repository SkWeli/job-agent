"""Lever postings - public, no key. Slug from jobs.lever.co/<slug>."""
from __future__ import annotations

from datetime import datetime, timezone

from ..models import Job
from .util import matches


def fetch(query, location, settings, http) -> list[Job]:
    out = []
    for slug in settings.get("companies", []):
        r = http.get(f"https://api.lever.co/v0/postings/{slug}", params={"mode": "json"})
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for j in r.json():
            cats = j.get("categories") or {}
            created = j.get("createdAt")
            lists = "\n".join(f"{l.get('text', '')}:\n{l.get('content', '')}" for l in j.get("lists") or [])
            job = Job(
                source=f"lever:{slug}",
                title=j.get("text", ""),
                company=slug,
                url=j.get("hostedUrl", ""),
                location=cats.get("location", ""),
                description=(j.get("descriptionPlain", "") + "\n\n" + lists).strip(),
                posted_at=datetime.fromtimestamp(created / 1000, timezone.utc).isoformat() if created else "",
                remote=(j.get("workplaceType") == "remote") or None,
            )
            if matches(job, query):
                out.append(job)
    return out
