"""Job feeds. Each module exposes:

    fetch(query: str, location: str, settings: dict, http) -> list[Job]

Rules: return only what the feed returned (never invent jobs); raise on failure with a
short message - the search tool turns that into a readable error for the agent.
"""
from __future__ import annotations

from . import greenhouse, jooble, lever, remotejobs, remotive

REGISTRY = {
    "remotejobs": remotejobs,    # remote, no key
    "remotive": remotive,        # remote, no key
    "jooble": jooble,            # 60+ countries incl. Sri Lanka, free key
    "company_boards": None,      # greenhouse + lever companies from config (handled below)
}


def fetch(name: str, query: str, location: str, settings: dict, http, cfg_sources: dict):
    if name == "company_boards":
        jobs = []
        gh = cfg_sources.get("greenhouse") or {}
        lv = cfg_sources.get("lever") or {}
        if gh.get("companies"):
            jobs += greenhouse.fetch(query, location, gh, http)
        if lv.get("companies"):
            jobs += lever.fetch(query, location, lv, http)
        return jobs
    return REGISTRY[name].fetch(query, location, settings, http)
