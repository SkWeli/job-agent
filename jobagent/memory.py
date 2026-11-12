"""Long-term memory, part 2: a notes file the agent reads at the start of every run.
Plus the helper that decides which old messages can be summarised safely.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


class Notes:
    """memory/notes.md: dated lessons, e.g. 'Remotive rarely has Sri Lanka-eligible roles'."""

    def __init__(self, path: Path, max_lines: int = 40):
        self.path = path
        self.max_lines = max_lines

    def read(self) -> str:
        if not self.path.exists():
            return ""
        lines = [l for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return "\n".join(lines[-self.max_lines:])  # only the most recent: keeps the prompt small

    def add(self, note: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- {date.today().isoformat()}: {note.strip()}\n")


def split_point(messages: list[AnyMessage], keep: int) -> int:
    """Index where the 'recent' part starts. Never split an AI tool call from its ToolMessages:
    a ToolMessage without its matching call is rejected by most model APIs."""
    i = max(0, len(messages) - keep)
    while 0 < i < len(messages) and isinstance(messages[i], ToolMessage):
        i -= 1  # walk back to the AIMessage that made these calls; it is kept with them
    return i


def render_for_summary(messages: list[AnyMessage], max_chars: int = 600) -> str:
    out = []
    for m in messages:
        if isinstance(m, AIMessage):
            calls = "; ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            out.append(f"AGENT: {str(m.content)[:max_chars]} {('CALLS: ' + calls) if calls else ''}")
        elif isinstance(m, ToolMessage):
            out.append(f"TOOL {m.name}: {str(m.content)[:max_chars]}")
        else:
            out.append(f"{m.type.upper()}: {str(m.content)[:max_chars]}")
    return "\n".join(out)
