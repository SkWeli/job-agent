"""Judgement + memory tools: save_assessment, list_jobs, write_note."""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..store import STATUSES
from .context import ToolContext, ToolError


class AssessmentInput(BaseModel):
    job_id: str = Field(min_length=8, max_length=8)
    score: int = Field(ge=1, le=5, description="1-5 fit score, using the scale in your instructions.")
    verdict: Literal["apply", "maybe", "skip"]
    reasons: str = Field(min_length=10, max_length=400, description="Why, in one or two sentences.")
    gaps: list[str] = Field(default_factory=list, max_length=8, description="Hard requirements the CV lacks.")
    eligible_location: bool = Field(description="False if the candidate cannot work from where they live.")


class ListJobsInput(BaseModel):
    status: list[Literal[tuple(STATUSES)]] | None = Field(  # type: ignore[valid-type]
        None, description="Filter by status, e.g. ['new'] or ['assessed','drafted']. Omit for all.")
    min_score: int = Field(0, ge=0, le=5)
    limit: int = Field(10, ge=1, le=30)


class NoteInput(BaseModel):
    note: str = Field(min_length=5, max_length=300, description="One lesson for future runs.")


def build(ctx: ToolContext) -> list[StructuredTool]:
    db = ctx.db

    def save_assessment(job_id, score, verdict, reasons, gaps=None, eligible_location=True) -> str:
        row = db.get(job_id)
        if not row:
            raise ToolError(f"No job with id {job_id}. Only assess jobs returned by search_jobs/list_jobs.")
        if not eligible_location and score > 2:
            score = 2  # enforced, not just requested in the prompt
        if verdict == "apply" and score < 4:
            verdict = "maybe"
        db.update(job_id, status="assessed", score=score, verdict=verdict, reasons=reasons,
                  gaps=json.dumps(gaps or []))
        can_draft = score >= ctx.cfg.limits["min_score_to_draft"]
        return (f"Saved: {job_id} {row['title']} at {row['company']} = {score}/5 ({verdict})."
                + (" Eligible for draft_application." if can_draft else ""))

    def list_jobs(status=None, min_score=0, limit=10) -> str:
        rows = db.query(status, min_score, limit)
        if not rows:
            return "No matching jobs in the tracker."
        return "\n".join(
            f"[{r['id']}] {r['status']:<9} {str(r['score'] or '-')}/5  {r['title']} | {r['company']} | {r['location']}"
            for r in rows) + f"\nTotals by status: {db.counts()}"

    def write_note(note: str) -> str:
        ctx.notes.add(note)
        return "Noted. It will appear in your instructions on future runs."

    return [
        StructuredTool.from_function(
            save_assessment, name="save_assessment", args_schema=AssessmentInput,
            description="Record your fit judgement for one job in the tracker. Call it once per job you have "
                        "evaluated. Ineligible locations are capped at 2/5 automatically."),
        StructuredTool.from_function(
            list_jobs, name="list_jobs", args_schema=ListJobsInput,
            description="Look up jobs already in the tracker, from this run or earlier ones (best score first). "
                        "Useful to see what is still 'new' or what was assessed before."),
        StructuredTool.from_function(
            write_note, name="write_note", args_schema=NoteInput,
            description="Save a short lesson for future runs: a source that keeps failing, a query that works "
                        "well, a company to avoid. Don't use it for per-job details (save_assessment holds those)."),
    ]
