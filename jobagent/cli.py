"""python -m jobagent <command>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .store import STATUSES, Tracker


def cli_approver(request: dict) -> dict:
    """Shows each irreversible action and asks the human to approve it."""
    decisions = {}
    for c in request.get("calls", []):
        print("\n" + "=" * 70 + f"\nAPPROVAL NEEDED: {c['tool']}")
        for k, v in c["args"].items():
            print(f"  {k}: {v}")
        ans = input("Approve? [y/N] ").strip().lower()
        reason = "" if ans == "y" else input("Reason (optional, shown to the agent): ").strip()
        decisions[c["id"]] = {"approved": ans == "y", "reason": reason or "declined"}
    return decisions


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="jobagent", description="LangGraph job-search agent (free models).")
    p.add_argument("-c", "--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="Daily run with the default task")
    a = sub.add_parser("ask", help="Run with your own task, e.g. 'assess this job: <url>'")
    a.add_argument("task")
    r = sub.add_parser("resume", help="Resume an interrupted run from its checkpoint")
    r.add_argument("thread_id")
    ls = sub.add_parser("list", help="Show tracked jobs")
    ls.add_argument("--status", nargs="*", default=["assessed", "drafted", "emailed", "applied", "interview", "offer"])
    ls.add_argument("--min-score", type=int, default=0)
    s = sub.add_parser("status", help="Update a job after you act on it")
    s.add_argument("job_id")
    s.add_argument("status", choices=STATUSES)
    s.add_argument("--note")
    tl = sub.add_parser("tailor", help="(Re)draft the tailored CV + cover letter for one job, optionally with feedback")
    tl.add_argument("job_id")
    tl.add_argument("--feedback", default="", help="e.g. 'shorter letter, lead with the SQL work'")
    sub.add_parser("export", help="Write tracker.xlsx")
    t = sub.add_parser("trace", help="Show what the agent did in a run")
    t.add_argument("run_id", nargs="?", default="latest")
    t.add_argument("--full", action="store_true")
    sub.add_parser("graph", help="Print the graph as a Mermaid diagram")
    sub.add_parser("check", help="Check that the configured model is reachable and can call tools")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "tailor":
        from .drafting import build_drafter
        from .llm import make_model
        from .tools.apply import summarize_draft, tailor
        db = Tracker(cfg.data_dir / "jobs.db")
        row = db.get(args.job_id)
        if not row or row["score"] is None:
            print("Unknown job id, or the job hasn't been assessed yet (run the agent first).")
            return 1
        drafter = build_drafter(make_model(cfg.llm, cfg.llm.get("draft_model")))
        print("Tailoring… (several model calls)")
        d, out = tailor(cfg, db, drafter.invoke, args.job_id, args.feedback)
        print(summarize_draft(d, out))
        print(f"\nOpen: {d / 'cv.docx'}\n      {d / 'cover_letter.docx'}\n      {d / 'fit.md'}  (requirements match)")
        return 0

    if args.cmd in ("run", "ask", "resume", "graph", "check"):
        import httpx

        from .graph import draw, run_agent
        from .llm import make_model
        from .sources.util import UA
        model = make_model(cfg.llm)
        if args.cmd == "graph":
            print(draw(cfg, model))
            return 0
        if args.cmd == "check":
            from langchain_core.tools import tool

            @tool
            def ping(word: str) -> str:
                """Echo a word back."""
                return word

            ai = model.bind_tools([ping]).invoke("Call the ping tool with the word 'ok'.")
            print("tool call:", ai.tool_calls or "NONE - this model can't call tools reliably; pick another")
            return 0 if ai.tool_calls else 1
        draft_model = make_model(cfg.llm, cfg.llm.get("draft_model")) if cfg.llm.get("draft_model") else None
        http = httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": UA})
        res = run_agent(cfg, getattr(args, "task", None), model=model, draft_model=draft_model, http=http,
                        approver=cli_approver, thread_id=getattr(args, "thread_id", None),
                        resume=args.cmd == "resume")
        print("\n" + res["report"])
        print(f"\n— run {res['thread_id']} · {res['stop_reason']} · {res['steps']} model calls · "
              f"{res['tool_calls']} tool calls · trace: python -m jobagent trace {res['thread_id']}")
        return 0

    db = Tracker(cfg.data_dir / "jobs.db")
    if args.cmd == "list":
        for r in db.query(args.status, args.min_score, 200):
            print(f"{r['id']}  {r['status']:<9} {r['score'] or '-'}/5  {r['title'][:45]:<45} {(r['company'] or '')[:22]:<22} {r['url']}")
    elif args.cmd == "status":
        if not db.get(args.job_id):
            print("No such job id.")
            return 1
        db.update(args.job_id, status=args.status, **({"notes": args.note} if args.note else {}))
        print("Updated.")
    elif args.cmd == "export":
        from .export import export_xlsx
        rows = db.conn.execute("SELECT * FROM jobs ORDER BY first_seen DESC").fetchall()
        print("Wrote", export_xlsx(rows, cfg.root / "tracker.xlsx"))
    elif args.cmd == "trace":
        from .tracing import show
        logs = sorted((cfg.root / "logs").glob("*.jsonl"), key=lambda q: q.stat().st_mtime)
        path = logs[-1] if args.run_id == "latest" and logs else cfg.root / "logs" / f"{args.run_id}.jsonl"
        if not path.exists():
            print("No trace found.")
            return 1
        print(show(path, args.full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
