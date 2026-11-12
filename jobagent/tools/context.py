"""Everything tools need, passed in once (no globals - which makes tools easy to test)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Config
from ..memory import Notes
from ..store import Tracker
from ..tracing import NullTracer, Tracer


class ToolError(Exception):
    """Raise this inside a tool for an expected failure. The message, including a hint,
    goes back to the model instead of a stack trace."""


@dataclass
class ToolContext:
    cfg: Config
    db: Tracker
    http: Any
    notes: Notes
    drafter: Callable | None = None           # compiled drafting subgraph runner
    tracer: Tracer = field(default_factory=NullTracer)
    pages_read: set = field(default_factory=set)   # URLs read this run (add_job_from_page guard)
    drafts_this_run: int = 0
    task_text: str = ""                         # the human's request (add_job_from_text guard)
