"""Load config.yaml + CV. Every knob the agent has lives here, with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_LIMITS = {
    "max_steps": 40,              # model calls per run (hard stop)
    "max_tool_calls": 80,         # tool executions per run (hard stop)
    "max_same_call": 2,           # identical tool+args allowed this many times (loop guard)
    "max_consecutive_errors": 4,  # tool errors in a row before the run is stopped
    "max_drafts_per_run": 3,
    "min_score_to_draft": 4,
    "revision_rounds": 2,         # CV/letter fix-and-recheck loops in the tailoring subgraph
    "max_context_messages": 40,   # above this, old messages are summarised
    "keep_recent_messages": 14,   # kept verbatim after summarising
    "tool_result_chars": 4000,    # tool output is truncated to this
    "max_age_days": 21,
}


@dataclass
class Config:
    root: Path
    raw: dict
    cv_text: str

    def _section(self, name: str, defaults: dict) -> dict:
        return {**defaults, **(self.raw.get(name) or {})}

    @property
    def llm(self) -> dict:
        return self._section("llm", {"provider": "groq", "model": "llama-3.3-70b-versatile",
                                     "temperature": 0, "requests_per_minute": 20})

    @property
    def limits(self) -> dict:
        return self._section("limits", DEFAULT_LIMITS)

    @property
    def filters(self) -> dict:
        return self._section("filters", {"exclude_titles": [], "locations_any": []})

    @property
    def preferences(self) -> str:
        return (self.raw.get("preferences") or "").strip()

    @property
    def searches(self) -> list[dict]:
        return self.raw.get("searches") or []

    @property
    def sources(self) -> dict:
        return self.raw.get("sources") or {}

    @property
    def candidate_name(self) -> str:
        return (self.raw.get("candidate") or {}).get("name", "")

    @property
    def approval_required(self) -> set[str]:
        return set(self.raw.get("approval_required") or ["send_application_email"])

    @property
    def email(self) -> dict:
        e = dict(self.raw.get("email") or {})
        e.setdefault("password", os.environ.get("SMTP_PASSWORD", ""))
        return e

    def path(self, key: str, default: str) -> Path:
        p = self.root / (self.raw.get(key) or default)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        d = self.root / (self.raw.get("data_dir") or "data")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self.root / (self.raw.get("output_dir") or "output")
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"Config not found: {path}\nCopy config.example.yaml to config.yaml and edit it.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cv_path = path.parent / raw.get("cv_path", "profile/cv.md")
    if not cv_path.exists():
        raise SystemExit(f"CV not found: {cv_path}\nPut your CV there as Markdown or plain text.")
    src = raw.setdefault("sources", {})
    if (src.get("jooble") or {}).get("enabled") and not src["jooble"].get("api_key"):
        src["jooble"]["api_key"] = os.environ.get("JOOBLE_API_KEY", "")
    return Config(root=path.parent, raw=raw, cv_text=cv_path.read_text(encoding="utf-8"))
