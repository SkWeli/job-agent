"""Action tools: draft_application, read_web_page, add_job_from_page, send_application_email."""
from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, EmailStr, Field, HttpUrl

from ..export import write_draft
from ..models import Job
from ..sources.util import UA, html_to_text
from .context import ToolContext, ToolError
from .search import JobIdInput


class UrlInput(BaseModel):
    url: HttpUrl = Field(description="A job-posting URL, usually one the candidate gave you.")


class AddJobInput(BaseModel):
    url: HttpUrl = Field(description="The page you just read with read_web_page.")
    title: str = Field(min_length=2, max_length=120)
    company: str = Field(min_length=1, max_length=80)
    location: str = Field("", max_length=80)
    description: str = Field(min_length=50, max_length=8000, description="Requirements and duties, copied from the page.")


class AddTextJobInput(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    company: str = Field(min_length=1, max_length=80)
    location: str = Field("", max_length=80)
    how_to_apply: str = Field("", max_length=300, description="Apply link/email exactly as written in the ad.")
    description: str = Field(min_length=50, max_length=8000, description="Requirements and duties from the ad.")


class EmailInput(BaseModel):
    job_id: str = Field(min_length=8, max_length=8)
    to_email: EmailStr = Field(description="Address from the posting's 'how to apply' text. Never guess it.")
    subject: str = Field(min_length=5, max_length=150)
    body: str = Field(min_length=50, max_length=3000, description="Short email body. The CV and letter are attached.")


def robots_allows(http, url: str) -> bool:
    rp = RobotFileParser()
    try:
        r = http.get(urljoin(url, "/robots.txt"))
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except Exception:
        return True
    return rp.can_fetch(UA, url)


class DraftInput(JobIdInput):
    feedback: str = Field("", max_length=500, description="Optional: the candidate's instructions for a redraft, "
                                                          "e.g. 'shorter letter, lead with SQL'. Leave empty normally.")


def tailor(cfg, db, drafter, job_id: str, feedback: str = ""):
    """Run the tailoring subgraph for one job and write its folder. Shared by the tool and `jobagent tailor`."""
    row = db.get(job_id)
    job = db.to_job(row)
    job_text = f"{job.title} at {job.company}\nLocation: {job.location}\nApply: {job.url}\n\n{job.description[:9000]}"
    lw = cfg.raw.get("cover_letter_words") or [220, 320]
    out = drafter({"cv": cfg.cv_text, "job": job_text, "feedback": feedback, "letter_words": tuple(lw),
                   "assessment": f"Score {row['score']}/5. {row['reasons']} Gaps: {row['gaps']}",
                   "max_rounds": cfg.limits.get("revision_rounds", 2)})
    d = write_draft(cfg.output_dir, row, out, Path(row["draft_dir"]) if row["draft_dir"] else None)
    db.update(job_id, status="drafted", draft_dir=str(d))
    return d, out


def summarize_draft(d: Path, out: dict) -> str:
    from ..drafting import is_clean, remaining_problems
    cov = out["coverage"]
    must = cov["must_have"]
    strong = sum(v == "strong" for v in must.values())
    missing = [k for k, v in must.items() if v == "none"]
    check = "fact-check PASSED" if is_clean(out) else "fact-check FLAGGED: " + "; ".join(remaining_problems(out)[:3])
    return (f"Drafted tailored CV + cover letter in {d}. Must-haves met {strong}/{len(must)}"
            + (f" (missing: {', '.join(missing[:4])})" if missing else "")
            + f"; ATS keywords {cov['keywords_before']}→{cov['keywords_after']}/{cov['keywords_total']}; "
            f"{out.get('rounds', 0)} revision round(s); {check}.")


def build(ctx: ToolContext) -> list[StructuredTool]:
    cfg, db = ctx.cfg, ctx.db

    def draft_application(job_id: str, feedback: str = "") -> str:
        row = db.get(job_id)
        if not row:
            raise ToolError(f"No job with id {job_id}.")
        if row["score"] is None:
            raise ToolError("Assess this job with save_assessment first.")
        if row["score"] < cfg.limits["min_score_to_draft"]:
            raise ToolError(f"Score {row['score']}/5 is below the drafting threshold "
                            f"({cfg.limits['min_score_to_draft']}). Don't draft this one.")
        if row["draft_dir"] and not feedback:
            return f"Already drafted earlier: {row['draft_dir']}. Pass feedback to redraft."
        if ctx.drafts_this_run >= cfg.limits["max_drafts_per_run"]:
            raise ToolError(f"Draft limit for this run reached ({cfg.limits['max_drafts_per_run']}). "
                            "List the remaining good jobs under 'Worth a look' in your report.")
        if ctx.drafter is None:
            raise ToolError("Drafting isn't configured.")
        try:
            d, out = tailor(cfg, db, ctx.drafter, job_id, feedback)
        except Exception as e:
            raise ToolError(f"Drafting model failed ({type(e).__name__}: {str(e)[:150]}). Skip this job for now "
                            "and mention it in the report.")
        ctx.drafts_this_run += 1
        return summarize_draft(d, out)

    def read_web_page(url) -> str:
        url = str(url)
        if not robots_allows(ctx.http, url):
            raise ToolError("This site's robots.txt forbids automated reading. Ask the candidate to paste the "
                            "posting text, or to save it as a file.")
        try:
            r = ctx.http.get(url)
            r.raise_for_status()
        except Exception as e:
            raise ToolError(f"Couldn't open the page ({type(e).__name__}: {str(e)[:100]}).")
        ctx.pages_read.add(url)
        text = html_to_text(r.text)
        cap = cfg.limits["tool_result_chars"]
        return text[:cap] + (" …[truncated]" if len(text) > cap else "")

    def add_job_from_page(url, title, company, location="", description="") -> str:
        url = str(url)
        if url not in ctx.pages_read:
            raise ToolError("You can only add a job from a page you read in this run with read_web_page.")
        job = Job(source="web", title=title, company=company, url=url, location=location, description=description)
        added = db.add(job)
        return f"{'Added' if added else 'Already tracked'}: [{job.uid}] {title} at {company}. Assess it next."

    def add_job_from_text(title, company, location="", how_to_apply="", description="") -> str:
        task = " ".join(ctx.task_text.lower().split())
        if title.lower() not in task or company.lower() not in task:
            raise ToolError("Title and company must appear word-for-word in the job ad the candidate pasted. "
                            "Copy them exactly; this tool is only for ads in the candidate's message.")
        job = Job(source="pasted", title=title, company=company, url=how_to_apply or "see ad",
                  location=location, description=description + (f"\n\nHow to apply: {how_to_apply}" if how_to_apply else ""))
        added = db.add(job)
        return f"{'Added' if added else 'Already tracked'}: [{job.uid}] {title} at {company}. Assess it next."

    def send_application_email(job_id, to_email, subject, body) -> str:
        # Only reachable after the human approved this exact call (see graph.approval node).
        row = db.get(job_id)
        if not row or not row["draft_dir"]:
            raise ToolError("Draft the application first (draft_application), then email it.")
        folder = Path(row["draft_dir"])
        msg = EmailMessage()
        e = cfg.email
        msg["From"], msg["To"], msg["Subject"] = e.get("from", ""), str(to_email), subject
        msg.set_content(body)
        for name in ("cv.docx", "cover_letter.docx"):
            p = folder / name
            if p.exists():
                cand = (cfg.candidate_name or "candidate").replace(" ", "_")
                msg.add_attachment(p.read_bytes(), maintype="application",
                                   subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   filename=f"{cand}_{name}")
        if not (e.get("smtp_host") and e.get("from") and e.get("password")):
            out = folder / "application_email.eml"
            out.write_bytes(bytes(msg))
            db.update(job_id, notes=f"email saved to {out}")
            return f"SMTP isn't configured, so the email was saved to {out} for the candidate to send by hand."
        with smtplib.SMTP_SSL(e["smtp_host"], int(e.get("smtp_port", 465)), timeout=30) as s:
            s.login(e.get("username") or e["from"], e["password"])
            s.send_message(msg)
        db.update(job_id, status="emailed", notes=json.dumps({"to": str(to_email), "subject": subject}))
        return f"Email sent to {to_email} with CV and cover letter attached."

    return [
        StructuredTool.from_function(
            draft_application, name="draft_application", args_schema=DraftInput,
            description="Tailor the candidate's CV to one assessed job (score at or above the threshold) and write "
                        "a cover letter. It analyses the posting's requirements, matches them to evidence in the "
                        "CV, rewrites the CV around them in the employer's terms, writes the letter, and "
                        "fact-checks both (code + model). Saves Markdown, Word and a requirements-match report. "
                        "Returns a one-line summary. Slow: use only for the best matches."),
        StructuredTool.from_function(
            read_web_page, name="read_web_page", args_schema=UrlInput,
            description="Read the text of one web page, e.g. a job link the candidate gave you. Respects "
                        "robots.txt. To track the job, follow up with add_job_from_page."),
        StructuredTool.from_function(
            add_job_from_page, name="add_job_from_page", args_schema=AddJobInput,
            description="Save a job you just read with read_web_page into the tracker, and get its job_id. Copy "
                        "the details from the page; don't make them up."),
        StructuredTool.from_function(
            add_job_from_text, name="add_job_from_text", args_schema=AddTextJobInput,
            description="Save a job ad that the candidate pasted into their message (e.g. from topjobs.lk), and "
                        "get its job_id. Copy the title, company and details exactly from the pasted text."),
        StructuredTool.from_function(
            send_application_email, name="send_application_email", args_schema=EmailInput,
            description="IRREVERSIBLE. Email an application (drafted CV + letter attached) to an employer. The "
                        "human must approve every call. Use only when the task asks you to email an "
                        "application and the posting says to apply by email."),
    ]
