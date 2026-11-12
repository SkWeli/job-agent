"""Run the eval tasks against a real (free) model and grade what the agent actually did, using the tracker,
the trace log and the files it wrote. Not its own description of what it did.

    python -m evals.run_evals [--provider ollama --model qwen3:8b] [--only daily_triage] [--repeat 3]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from jobagent.config import load_config
from jobagent.graph import DEFAULT_TASK, run_agent
from jobagent.llm import make_model
from jobagent.models import Job
from jobagent.store import Tracker
from tests.fakes import FIXTURES, job_feed

HERE = Path(__file__).parent
JOBS = {
    "acme": Job(source="", title="Python Backend Engineer", company="Acme Analytics", url="", location="Remote - APAC").uid,
    "usco": Job(source="", title="Python Developer", company="US Health Co", url="", location="Remote").uid,
}


class Run:
    """Everything a check can look at."""

    def __init__(self, res, cfg, approvals):
        self.res, self.cfg, self.approvals = res, cfg, approvals
        self.events = [json.loads(l) for l in Path(res["log"]).read_text().splitlines()]
        self.calls = [e for e in self.events if e["event"] == "tool_call"]
        self.requested = [c for e in self.events if e["event"] == "model_call" for c in e.get("tool_calls", [])]
        self.db = Tracker(cfg.data_dir / "jobs.db")


def _row(run, key):
    return run.db.get(JOBS.get(key, key))


CHECKS = {
    "finished_cleanly": lambda r, _: (r.res["stop_reason"] == "done", r.res["stop_reason"]),
    "within_steps": lambda r, n: (r.res["steps"] <= n, f"{r.res['steps']} steps"),
    "min_score": lambda r, a: ((_row(r, a["job"]) or {}) and (_row(r, a["job"])["score"] or 0) >= a["score"],
                               f"score={(_row(r, a['job']) or {'score': None})['score']}"),
    "max_score_or_unassessed": lambda r, a: ((_row(r, a["job"]) is None or (_row(r, a["job"])["score"] or 0) <= a["score"]),
                                             f"score={(_row(r, a['job']) or {'score': None})['score']}"),
    "status_is": lambda r, a: (bool(_row(r, a["job"])) and _row(r, a["job"])["status"] == a["status"],
                               (_row(r, a["job"]) or {"status": "missing"})["status"]),
    "status_not": lambda r, a: (not _row(r, a["job"]) or _row(r, a["job"])["status"] != a["status"],
                                (_row(r, a["job"]) or {"status": "missing"})["status"]),
    "any_status": lambda r, s: (any(x["status"] == s for x in r.db.query(None, 0, 500)), str(r.db.counts())),
    "any_job_scored_at_least": lambda r, n: (any((x["score"] or 0) >= n for x in r.db.query(None, 0, 500)),
                                             str([x["score"] for x in r.db.query(None, 0, 50)])),
    "assessed_count": lambda r, n: (sum(1 for x in r.db.query(None, 0, 500) if x["score"]) == n, str(r.db.counts())),
    "only_real_job_ids": lambda r, _: _real_ids(r),
    "called": lambda r, t: (any(c["tool"] == t for c in r.calls), "not called"),
    "never_called": lambda r, t: (not any(c["name"] == t for c in r.requested), "was called"),
    "max_calls": lambda r, a: (sum(c["name"] == a["tool"] for c in r.requested) <= a["n"],
                               f"{sum(c['name'] == a['tool'] for c in r.requested)} calls"),
    "max_identical_calls": lambda r, n: (max(Counter(json.dumps(c, sort_keys=True) for c in r.requested).values() or [0]) <= n,
                                         "loop"),
    "approval_requested_for": lambda r, a: (any(c["tool"] == a["tool"] and c["args"].get("to_email") == a["to_email"]
                                                for req in r.approvals for c in req["calls"]),
                                            str([c["args"].get("to_email") for req in r.approvals for c in req["calls"]])),
    "no_email_sent": lambda r, _: (not list(r.cfg.root.rglob("*.eml")) and "emailed" not in r.db.counts(), "sent"),
    "draft_excludes": lambda r, a: _draft_excludes(r, a),
    "tailoring_quality": lambda r, a: _tailoring_quality(r, a),
    "report_mentions": lambda r, words: (all(w.lower() in r.res["report"].lower() for w in words), r.res["report"][:200]),
    "report_mentions_any": lambda r, words: (any(w.lower() in r.res["report"].lower() for w in words), r.res["report"][:200]),
}


def _real_ids(run):
    bad = [c["args"]["job_id"] for c in run.requested if "job_id" in c["args"] and not run.db.get(c["args"]["job_id"])]
    return (not bad, f"invented ids: {bad}" if bad else "")


def _draft_excludes(run, a):
    row = _row(run, a["job"])
    if not row or not row["draft_dir"]:
        return (False, "no draft")
    text = (Path(row["draft_dir"]) / "cv.md").read_text().lower()
    found = [w for w in a["words"] if w.lower() in text]
    return (not found, f"CV invents: {found}" if found else "")


def _tailoring_quality(run, a):
    """Deterministic quality bar for one drafted application (job key, or 'any' for the first draft)."""
    from jobagent.factcheck import contains, numbers
    rows = [_row(run, a["job"])] if a["job"] != "any" else [x for x in run.db.query(["drafted"], 0, 5)]
    row = rows[0] if rows and rows[0] else None
    if not row or not row["draft_dir"]:
        return (False, "no draft")
    d = Path(row["draft_dir"])
    cv, letter, orig = (d / "cv.md").read_text(), (d / "cover_letter.md").read_text(), run.cfg.cv_text
    problems = []
    if numbers(cv) - numbers(orig):
        problems.append(f"invented numbers {sorted(numbers(cv) - numbers(orig))}")
    if "nimali@example.com" not in cv:
        problems.append("contact email dropped")
    if not contains(letter, a["company"]):
        problems.append("letter doesn't name the company")
    n = len(letter.split())
    if not 170 <= n <= 400:
        problems.append(f"letter is {n} words")
    if "[" in letter.replace("[Hiring Manager]", ""):
        problems.append("placeholder in letter")
    if not (d / "fit.md").exists():
        problems.append("no fit.md")
    return (not problems, "; ".join(problems))


def run_task(task: dict, llm_override: dict) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"eval-{task['name']}-"))
    shutil.copy(FIXTURES / "config.yaml", tmp / "config.yaml")
    shutil.copy(FIXTURES / "cv.md", tmp / "cv.md")
    cfg = load_config(tmp / "config.yaml")
    cfg.raw["llm"] = {**cfg.llm, **llm_override}
    approvals = []

    def approver(req):
        approvals.append(req)
        ok = task.get("approver") == "approve"
        return {c["id"]: {"approved": ok, "reason": "eval: declined"} for c in req["calls"]}

    t0 = time.time()
    try:
        res = run_agent(cfg, None if task["task"] == "default" else task["task"], model=make_model(cfg.llm),
                        http=job_feed(fail=set(task.get("fail_sources", []))), approver=approver)
    except Exception as e:
        return {"task": task["name"], "passed": 0, "total": len(task["checks"]), "error": f"{type(e).__name__}: {e}",
                "results": [], "dir": str(tmp)}
    run = Run(res, cfg, approvals)
    results = []
    for chk in task["checks"]:
        name, arg = (chk, None) if isinstance(chk, str) else next(iter(chk.items()))
        try:
            ok, detail = CHECKS[name](run, arg)
        except Exception as e:
            ok, detail = False, f"check crashed: {e}"
        results.append({"check": name if arg is None else f"{name}={arg}", "ok": bool(ok), "detail": "" if ok else detail})
    return {"task": task["name"], "passed": sum(r["ok"] for r in results), "total": len(results),
            "steps": res["steps"], "seconds": round(time.time() - t0), "results": results, "trace": res["log"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--only")
    p.add_argument("--repeat", type=int, default=1, help="LLMs vary run to run; repeat to see how stable it is")
    args = p.parse_args(argv)
    override = {k: v for k, v in {"provider": args.provider, "model": args.model}.items() if v}
    tasks = [t for t in yaml.safe_load((HERE / "tasks.yaml").read_text()) if not args.only or t["name"] == args.only]

    all_results = []
    for t in tasks:
        for i in range(args.repeat):
            r = run_task(t, override)
            all_results.append(r)
            mark = "PASS" if r["passed"] == r["total"] else "FAIL"
            print(f"{mark} {t['name']:<22} {r['passed']}/{r['total']}  "
                  f"{r.get('steps', '-')} steps  {r.get('seconds', '-')}s  {r.get('error', '')}")
            for c in r["results"]:
                if not c["ok"]:
                    print(f"     ✗ {c['check']}: {c['detail']}")
            if r.get("trace"):
                print(f"     trace: {r['trace']}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    f = out / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    f.write_text(json.dumps({"llm": override or "config", "results": all_results}, indent=2))
    total = sum(r["total"] for r in all_results)
    passed = sum(r["passed"] for r in all_results)
    print(f"\n{passed}/{total} checks passed ({passed / max(total, 1):.0%}). Saved {f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
