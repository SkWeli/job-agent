"""Discovery tools: search_jobs, get_job_details."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .. import sources
from ..models import Job
from .context import ToolContext, ToolError

SourceName = Literal["remotejobs", "remotive", "jooble", "company_boards"]


class SearchJobsInput(BaseModel):
    source: SourceName = Field(description="Which feed to search. remotejobs/remotive = remote jobs worldwide; "
                                           "jooble = local jobs by city/country (incl. Sri Lanka); "
                                           "company_boards = the candidate's watch-list of companies.")
    query: str = Field(min_length=2, max_length=80, description="Keywords, e.g. 'python developer'.")
    location: str = Field("", max_length=60, description="City or country. Used by jooble only.")
    limit: int = Field(15, ge=1, le=25, description="Max new jobs to list in the result.")


class JobIdInput(BaseModel):
    job_id: str = Field(min_length=8, max_length=8, description="8-character id from search_jobs or list_jobs.")


def _date(s: str):
    try:
        d = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def rule_filter(job: Job, filters: dict, max_age_days: int) -> str | None:
    """Free, deterministic filter from config. Returns why a job is dropped, or None."""
    title = job.title.lower()
    if not job.url:
        return "no apply link"
    for w in filters.get("exclude_titles") or []:
        if w.lower() in title:
            return f"title contains '{w}'"
    locs = filters.get("locations_any") or []
    loc = (job.location or "").lower()
    if locs and not job.remote and not any(x in loc for x in ("remote", "anywhere", "worldwide")) \
            and not any(l.lower() in loc for l in locs):
        return f"location '{job.location}'"
    d = _date(job.posted_at)
    if d and max_age_days and d < datetime.now(timezone.utc) - timedelta(days=max_age_days):
        return f"older than {max_age_days} days"
    return None


def build(ctx: ToolContext) -> list[StructuredTool]:
    cfg, db = ctx.cfg, ctx.db

    def search_jobs(source: str, query: str, location: str = "", limit: int = 15) -> str:
        enabled = [n for n in sources.REGISTRY
                   if (n == "company_boards" and (cfg.sources.get("greenhouse") or cfg.sources.get("lever")))
                   or (cfg.sources.get(n) or {}).get("enabled")]
        if source not in enabled:
            raise ToolError(f"Source '{source}' is not enabled. Enabled sources: {', '.join(enabled) or 'none'}.")
        try:
            found = sources.fetch(source, query, location, cfg.sources.get(source) or {}, ctx.http, cfg.sources)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status else f"{type(e).__name__}: {str(e)[:100]}"
            raise ToolError(f"{source} is unavailable right now ({detail}). "
                            "Try a different source; don't retry this one in this run.")
        new, seen, dropped = [], 0, 0
        for job in found:
            why = rule_filter(job, cfg.filters, cfg.limits["max_age_days"])
            if not db.add(job, status="filtered" if why else "new", filter_reason=why):
                seen += 1
            elif why:
                dropped += 1
            else:
                new.append(job)
        head = (f"{source} '{query}'{' @ ' + location if location else ''}: {len(found)} results, "
                f"{len(new)} new, {seen} already seen, {dropped} removed by the candidate's filters.")
        if not new:
            return head + " Nothing new to review here."
        rows = [f"[{j.uid}] {j.title} | {j.company} | {j.location or '?'} | {(j.posted_at or '')[:10]} | "
                f"{' '.join(j.description.split())[:160]}" for j in new[:limit]]
        more = f"\n(+{len(new) - limit} more new jobs saved; see list_jobs with status ['new'])" if len(new) > limit else ""
        return head + "\n" + "\n".join(rows) + more

    def get_job_details(job_id: str) -> str:
        row = db.get(job_id)
        if not row:
            raise ToolError(f"No job with id {job_id}. Use an id exactly as shown by search_jobs or list_jobs.")
        job = db.to_job(row)
        desc = job.description.strip() or "(no description)"
        cap = cfg.limits["tool_result_chars"] - 600
        if len(desc) > cap:
            desc = desc[:cap] + " …[truncated]"
        elig = job.extra.get("applicant_location_requirements")
        snippet = " NOTE: only a snippet is available from this source; open the link for full details." \
            if job.extra.get("snippet_only") else ""
        return (f"[{job_id}] {job.title} at {job.company}\nLocation: {job.location}\n"
                + (f"Applicant location requirements: {elig}\n" if elig else "")
                + f"Salary: {job.salary or 'not stated'}\nPosted: {job.posted_at or '?'}\nApply: {job.url}\n"
                f"Status: {row['status']}{snippet}\n\n{desc}")

    return [
        StructuredTool.from_function(
            search_jobs, name="search_jobs", args_schema=SearchJobsInput,
            description="Search one live job feed and save new results to the tracker. Returns only jobs not seen "
                        "in earlier runs, as '[job_id] title | company | location | posted | snippet'. Jobs that "
                        "fail the candidate's own rules (excluded titles, locations, too old) are removed "
                        "automatically. Call once per source+query; repeating gives nothing new."),
        StructuredTool.from_function(
            get_job_details, name="get_job_details", args_schema=JobIdInput,
            description="Full posting for one job: requirements, applicant-location rules, salary, apply link. "
                        "Read this before scoring a job 4 or 5."),
    ]
