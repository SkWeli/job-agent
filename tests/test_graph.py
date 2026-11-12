"""Graph mechanics and guardrails, tested with a scripted model. No LLM and no network: runs in seconds.
    python -m pytest -q
"""
from __future__ import annotations

import json
import shutil

import pytest
import yaml
from langchain_core.messages import ToolMessage

from jobagent.config import load_config
from jobagent.graph import run_agent
from jobagent.models import Job
from jobagent.store import Tracker

from .fakes import FIXTURES, ScriptedModel, ai, call, job_feed

ACME = Job(source="", title="Python Backend Engineer", company="Acme Analytics", url="", location="Remote - APAC").uid
USCO = Job(source="", title="Python Developer", company="US Health Co", url="", location="Remote").uid


def make_cfg(tmp_path, **limit_overrides):
    raw = yaml.safe_load((FIXTURES / "config.yaml").read_text())
    raw["limits"].update(limit_overrides)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    shutil.copy(FIXTURES / "cv.md", tmp_path / "cv.md")
    return load_config(tmp_path / "config.yaml")


def model(script, **kw):
    return ScriptedModel(script=script, seen=[], draft_calls=[], cvs=[], letters=[], structured={}, **kw)


def deny(request):
    return {c["id"]: {"approved": False, "reason": "not this one"} for c in request["calls"]}


def approve(request):
    return {c["id"]: {"approved": True} for c in request["calls"]}


def events(res):
    return [json.loads(l) for l in open(res["log"])]


def tool_results(m):
    """Tool results in order, each once (every prompt repeats the history)."""
    out, ids = [], set()
    for msgs in m.seen:
        for x in msgs:
            if isinstance(x, ToolMessage) and x.tool_call_id + x.content[:20] not in ids:
                ids.add(x.tool_call_id + x.content[:20])
                out.append(x)
    return out


# --------------------------------------------------------------------------------------------- happy path
def test_full_run(tmp_path):
    cfg = make_cfg(tmp_path)
    m = model([
        ai(call("search_jobs", source="remotejobs", query="python")),
        ai(call("search_jobs", source="remotive", query="data engineer")),
        ai(call("get_job_details", job_id=ACME), call("get_job_details", job_id=USCO)),
        ai(call("save_assessment", job_id=ACME, score=5, verdict="apply", reasons="FastAPI + Postgres, APAC remote",
                gaps=["AWS"], eligible_location=True),
           call("save_assessment", job_id=USCO, score=4, verdict="apply", reasons="Python but US-only",
                eligible_location=False)),
        ai(call("draft_application", job_id=ACME)),
        ai(call("write_note", note="remotejobs has APAC-eligible Python roles")),
        ai(text="## Ready to apply\n- [5/5] Python Backend Engineer, Acme Analytics"),
    ])
    res = run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert res["stop_reason"] == "done"

    db = Tracker(cfg.data_dir / "jobs.db")
    assert db.get(ACME)["status"] == "drafted"
    us = db.get(USCO)
    assert us["score"] == 2 and us["verdict"] == "maybe"          # location cap enforced in the tool
    c = db.counts()
    assert c["filtered"] == 3                                     # senior staff, sales, too old
    folder = tmp_path / db.get(ACME)["draft_dir"]
    for f in ("cv.md", "cv.docx", "cover_letter.md", "cover_letter.docx", "fit.md", "review.md", "job.md"):
        assert (folder / f).exists()
    assert "remotejobs has APAC" in (cfg.data_dir / "notes.md").read_text()
    results = [t.content for t in tool_results(m)]
    assert any("1 already seen" in r for r in results)            # Acme duplicate from remotive
    ev = [e["event"] for e in events(res)]
    assert ev[0] == "run_start" and ev[-1] == "run_end" and ev.count("model_call") == 7
    assert list(cfg.output_dir.rglob("report-*.md"))


def test_notes_reach_next_run(tmp_path):
    cfg = make_cfg(tmp_path)
    run_agent(cfg, model=model([ai(call("write_note", note="Jooble key is missing")), ai(text="done")]),
              http=job_feed(), approver=deny)
    m = model([ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert "Jooble key is missing" in m.seen[0][0].content       # system prompt of the second run


# --------------------------------------------------------------------------------------------- guardrails
def test_repeated_call_is_blocked(tmp_path):
    cfg = make_cfg(tmp_path)
    same = call("list_jobs", min_score=0)
    m = model([ai({**same, "id": "a"}), ai({**same, "id": "b"}), ai({**same, "id": "c"}), ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert tool_results(m)[-1].content.startswith("BLOCKED")


def test_invalid_arguments_return_readable_error(tmp_path):
    cfg = make_cfg(tmp_path)
    m = model([ai(call("save_assessment", job_id=ACME, score=9, verdict="yes", reasons="x")), ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    err = tool_results(m)[-1].content
    assert err.startswith("ERROR: invalid arguments for save_assessment") and "Traceback" not in err


def test_hallucinated_job_id_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    m = model([ai(call("get_job_details", job_id="deadbeef")), ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert "No job with id deadbeef" in tool_results(m)[-1].content


def test_failing_source_is_survivable(tmp_path):
    cfg = make_cfg(tmp_path)
    m = model([ai(call("search_jobs", source="remotive", query="python")),
               ai(call("search_jobs", source="remotejobs", query="python")), ai(text="done")])
    res = run_agent(cfg, model=m, http=job_feed(fail={"remotive"}), approver=deny)
    r = tool_results(m)
    assert "unavailable right now" in r[0].content and "4 results" in r[1].content
    assert res["stop_reason"] == "done"


def test_consecutive_errors_stop_the_run(tmp_path):
    cfg = make_cfg(tmp_path, max_consecutive_errors=3)
    m = model([ai(call("get_job_details", job_id=f"bad0000{i}")) for i in range(10)] + [ai(text="report")])
    res = run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert res["stop_reason"] == "too_many_errors" and res["steps"] <= 5


def test_step_limit(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=4)
    def step(msgs):  # keeps calling tools until told to stop
        if "Stop now" in str(msgs[-1].content):
            return ai(text="Final report after budget.")
        return ai(call("list_jobs", min_score=0, limit=len(msgs) % 30 + 1))
    m = model([step] * 20)
    res = run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert res["stop_reason"] == "max_steps" and res["steps"] == 5   # 4 + the tool-less wrap-up call
    assert res["report"] == "Final report after budget."


def test_draft_requires_score_and_respects_limit(tmp_path):
    cfg = make_cfg(tmp_path, max_drafts_per_run=0)
    m = model([ai(call("search_jobs", source="remotejobs", query="python")),
               ai(call("draft_application", job_id=ACME)),
               ai(call("save_assessment", job_id=ACME, score=5, verdict="apply", reasons="great fit here",
                       eligible_location=True)),
               ai(call("draft_application", job_id=ACME)), ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    r = tool_results(m)
    assert "Assess this job" in r[1].content and "Draft limit" in r[3].content


# --------------------------------------------------------------------------------------------- human in the loop
def _email_script():
    return [ai(call("search_jobs", source="remotejobs", query="python")),
            ai(call("save_assessment", job_id=ACME, score=5, verdict="apply", reasons="great fit here",
                    eligible_location=True)),
            ai(call("draft_application", job_id=ACME)),
            ai(call("send_application_email", _id="mail1", job_id=ACME, to_email="hr@acme.example",
                    subject="Application: Python Backend Engineer", body="Dear team, " + "please find attached. " * 4)),
            ai(text="done")]


def test_email_declined_by_human(tmp_path):
    cfg = make_cfg(tmp_path)
    asked = []
    m = model(_email_script())
    run_agent(cfg, model=m, http=job_feed(), approver=lambda r: (asked.append(r), deny(r))[1])
    assert asked and asked[0]["calls"][0]["args"]["to_email"] == "hr@acme.example"
    assert tool_results(m)[-1].content.startswith("DECLINED")
    assert not list(tmp_path.rglob("*.eml"))


def test_email_approved_without_smtp_saves_eml(tmp_path):
    cfg = make_cfg(tmp_path)
    m = model(_email_script())
    run_agent(cfg, model=m, http=job_feed(), approver=approve)
    assert list(tmp_path.rglob("application_email.eml"))


# --------------------------------------------------------------------------------------------- memory
def test_old_messages_get_summarised(tmp_path):
    cfg = make_cfg(tmp_path, max_context_messages=6, keep_recent_messages=3)
    m = model([ai(call("list_jobs", limit=i + 1)) for i in range(5)] + [ai(text="SUMMARY: searched nothing")] * 0)
    # the model's 6th reply is used by the summariser, then the run continues to "done"
    m.script.insert(3, ai(text="- listed jobs a few times"))
    m.script.append(ai(text="done"))
    res = run_agent(cfg, model=m, http=job_feed(), approver=deny)
    assert any(e["event"] == "summarised" for e in events(res))
    later = m.seen[-1]
    assert not isinstance(later[1], ToolMessage)                  # never starts with an orphan tool result
    assert "Summary of earlier steps" in later[0].content


# --------------------------------------------------------------------------------------------- web tools
def test_web_page_flow_and_robots(tmp_path):
    cfg = make_cfg(tmp_path)
    url = "https://jobs.example.lk/ceylon-tech/se-python"
    m = model([
        ai(call("add_job_from_page", url=url, title="XY", company="Y", description="z" * 60)),
        ai(call("read_web_page", url="https://blocked.example/job/1")),
        ai(call("read_web_page", url=url)),
        ai(call("add_job_from_page", url=url, title="Software Engineer (Python)", company="Ceylon Tech",
                location="Colombo", description="Python, Django/FastAPI, SQL, Git. Email careers@ceylontech.example")),
        ai(text="done")])
    run_agent(cfg, model=m, http=job_feed(), approver=deny)
    r = [t.content for t in tool_results(m)]
    assert "only add a job from a page you read" in r[0]
    assert "robots.txt" in r[1]
    assert "Ceylon Tech" in r[2] and r[3].startswith("Added")


def test_pasted_ad_must_match_message(tmp_path):
    cfg = make_cfg(tmp_path)
    ad = ("Assess this ad: Junior Python Engineer at Serendib Systems, Colombo. Requirements: Python, SQL, "
          "Git, 1 year experience. Apply: jobs@serendib.example")
    desc = "Python, SQL, Git, 1 year experience. Junior role in Colombo office, full time."
    m = model([ai(call("add_job_from_text", title="Senior Rust Engineer", company="Serendib Systems", description=desc)),
               ai(call("add_job_from_text", title="Junior Python Engineer", company="Serendib Systems",
                       location="Colombo", how_to_apply="jobs@serendib.example", description=desc)),
               ai(text="done")])
    run_agent(cfg, ad, model=m, http=job_feed(), approver=deny)
    r = [t.content for t in tool_results(m)]
    assert "word-for-word" in r[0] and r[1].startswith("Added")


def test_resume_after_crash_reasks_approval(tmp_path):
    """Run stops at the approval pause (approver raises, like a closed terminal); `resume` re-asks."""
    cfg = make_cfg(tmp_path)

    def crash(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_agent(cfg, "email it", model=model(_email_script()), http=job_feed(), approver=crash, thread_id="t1")
    asked = []
    res = run_agent(cfg, model=model([ai(text="resumed and done")]), http=job_feed(),
                    approver=lambda r: (asked.append(r), deny(r))[1], thread_id="t1", resume=True)
    assert asked and res["report"] == "resumed and done"
