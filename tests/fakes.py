"""Test doubles: a scripted chat model and a fake job-feed HTTP server."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

FIXTURES = Path(__file__).parent.parent / "evals" / "fixtures"


def call(name, _id=None, **args):
    return {"name": name, "args": args, "id": _id or f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 10**6}"}


GOOD_CV = """# Nimali Perera
Colombo, Sri Lanka · +94 77 000 0000 · nimali@example.com · github.com/nimali-example

## Summary
Backend engineer with 2 years of Python, building REST APIs with FastAPI and PostgreSQL.

## Experience
### Software Engineer — Kandy Software Solutions (Jan 2024 – Present)
- Built and maintained 12 REST API endpoints in FastAPI serving a logistics dashboard
- Cut a slow PostgreSQL report query from 9s to 0.8s with indexing

## Skills
Python, FastAPI, PostgreSQL, Docker, Git"""

GOOD_LETTER = ("Dear Hiring Manager,\n\nI'd like to apply for the Python Backend Engineer role at Acme Analytics. "
               + "At Kandy Software Solutions I built and maintained 12 REST API endpoints in FastAPI and tuned "
               "PostgreSQL queries for a logistics dashboard. " * 11
               + "\n\nSincerely,\nNimali Perera")


def default_structured():
    from jobagent.drafting import EvidenceMap, JobAnalysis, Match, Review
    return {
        "JobAnalysis": JobAnalysis(company_name="Acme Analytics", role_title="Python Backend Engineer",
                                   must_have=["1-3 years Python", "FastAPI", "PostgreSQL", "AWS"],
                                   keywords=["Python", "FastAPI", "PostgreSQL", "REST API", "Docker", "AWS"],
                                   company_hook="analytics for APAC logistics", tone="neutral"),
        "EvidenceMap": EvidenceMap(section_order=["Summary", "Experience", "Skills"], matches=[
            Match(requirement="1-3 years Python", cv_quotes=["2 years of Python backend experience"], strength="strong"),
            Match(requirement="FastAPI", cv_quotes=["REST API endpoints in FastAPI"], strength="strong"),
            Match(requirement="PostgreSQL", cv_quotes=["Designed PostgreSQL schemas"], strength="strong"),
            Match(requirement="AWS", cv_quotes=["Senior AWS architect"], strength="strong")]),  # fake quote
        "Review": Review(unsupported_claims=[], approved=True),
    }


class ScriptedModel(BaseChatModel):
    """Plays back a script for the main agent. Each entry is an AIMessage, or a function(messages) -> AIMessage
    that can look at earlier tool results. Calls from the tailoring subgraph are answered separately:
    structured outputs from `structured`, CV/letter text from the `cvs`/`letters` queues (last one repeats)."""

    script: list = []
    structured: dict = {}
    cvs: list = []
    letters: list = []
    seen: list = []
    draft_calls: list = []
    fallback: str = "Final report: nothing more to do."

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def with_structured_output(self, schema, **kw):
        def answer(_):
            self.draft_calls.append(schema.__name__)
            return (self.structured or default_structured())[schema.__name__]
        return RunnableLambda(answer)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        system = str(messages[0].content) if messages else ""
        if system.startswith(("Write the tailored CV", "Revise the CV")):
            self.draft_calls.append("cv")
            msg = AIMessage(_next(self.cvs, GOOD_CV))
        elif system.startswith(("Write a cover letter", "Revise the cover letter")):
            self.draft_calls.append("letter")
            msg = AIMessage(_next(self.letters, GOOD_LETTER))
        else:
            self.seen.append(messages)
            step = self.script.pop(0) if self.script else AIMessage(self.fallback)
            msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _next(queue: list, default: str) -> str:
    if not queue:
        return default
    return queue.pop(0) if len(queue) > 1 else queue[0]


def ai(*calls, text=""):
    return AIMessage(content=text, tool_calls=list(calls))


def job_feed(overrides: dict | None = None, fail: set | None = None) -> httpx.Client:
    """Serves the fixture feeds. `fail` = sources that return 500."""
    data = {"remotejobs": json.loads((FIXTURES / "remotejobs.json").read_text()),
            "remotive": json.loads((FIXTURES / "remotive.json").read_text()),
            **(overrides or {})}
    fail = fail or set()

    def handler(req: httpx.Request):
        u = str(req.url)
        if u.endswith("robots.txt"):
            if "blocked.example" in u:
                return httpx.Response(200, text="User-agent: *\nDisallow: /")
            return httpx.Response(404)
        for name in ("remotejobs", "remotive"):
            if name in u:
                return httpx.Response(500) if name in fail else httpx.Response(200, json=data[name])
        if "jobs.example.lk" in u:
            return httpx.Response(200, text=(FIXTURES / "job_page.html").read_text())
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))
