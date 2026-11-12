"""Plain data shapes shared by tools, sources and the tracker."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class Job:
    source: str
    title: str
    company: str
    url: str
    location: str = ""
    description: str = ""
    posted_at: str = ""
    salary: str = ""
    remote: bool | None = None
    extra: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Short stable id. The same job from two sources gets the same id (de-duplication)."""
        key = f"{_norm(self.company)}|{_norm(self.title)}|{_norm(self.location)}"
        return hashlib.sha1(key.encode()).hexdigest()[:8]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
