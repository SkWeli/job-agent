"""Observability: every model call, tool call, guardrail decision and result goes to
logs/<run_id>.jsonl, one JSON object per line. Read them back with `python -m jobagent trace`.

Optional extra: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY (free developer plan) and LangGraph
also sends a visual trace to LangSmith. No code change is needed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    def __init__(self, log_dir: Path, run_id: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.jsonl"
        self.run_id = run_id

    def log(self, event: str, **data) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run": self.run_id,
               "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")

    class timer:
        def __enter__(self):
            self.t = time.perf_counter()
            return self

        def __exit__(self, *a):
            self.ms = round((time.perf_counter() - self.t) * 1000)


class NullTracer(Tracer):
    def __init__(self):
        self.run_id = "null"

    def log(self, event, **data):
        pass


def show(path: Path, full: bool = False) -> str:
    """Readable timeline of one run."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        t, e = r["ts"][11:19], r["event"]
        if e == "model_call":
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])[:80]})" for c in r.get("tool_calls", []))
            lines.append(f"{t} 🧠 step {r['step']}  {r.get('ms')}ms  tokens={r.get('usage')}  "
                         + (f"→ {calls}" if calls else f"→ answer: {r.get('text', '')[:120]}"))
        elif e == "tool_call":
            mark = "❌" if r.get("error") else "🔧"
            res = r.get("result", "") if full else r.get("result", "")[:140].replace("\n", " ")
            lines.append(f"{t} {mark} {r['tool']}({json.dumps(r.get('args'))[:100]})  {r.get('ms')}ms  {res}")
        elif e == "guard":
            lines.append(f"{t} 🛑 guard: {r['tool']} — {r['reason']}")
        elif e == "approval":
            lines.append(f"{t} 🙋 approval for {r['tool']}: {r['decision']}")
        elif e == "summarised":
            lines.append(f"{t} 🗜  summarised {r['removed']} old messages")
        elif e in ("run_start", "run_end"):
            lines.append(f"{t} ── {e} {json.dumps({k: v for k, v in r.items() if k not in ('ts', 'run', 'event')})[:200]}")
        else:
            lines.append(f"{t} {e} {json.dumps(r)[:160]}")
    return "\n".join(lines)
