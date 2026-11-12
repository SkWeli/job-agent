"""Tailoring subgraph: CV adjusted to the job's requirements + a cover letter, both fact-checked.

  START → analyze_job → map_evidence ─┬─► write_cv ─────┬─► check ──(problems & rounds left)──► revise ─► check
                                      └─► write_letter ─┘      └──(clean, or out of rounds)──► END

analyze_job   what the employer screens for: must-haves, nice-to-haves, ATS keywords, company hook, tone
map_evidence  each requirement → exact quotes from YOUR CV (quotes not found in the CV are thrown away by code)
write_cv      reorders, re-summarises and rephrases your CV around the matched requirements, in the job's terms
write_letter  role + company opening, two evidence paragraphs, an honest gap line if needed, close
              (write_cv and write_letter run in parallel: two edges leave map_evidence)
check         code checks (factcheck.py) + an LLM fact-checker that confirms or dismisses them
revise        fixes only what was flagged, in the CV and/or the letter

Each stage is a small, focused call, which is what makes this work on free 8B-70B models. The CV and letter
are written as plain Markdown/text, not JSON: small models are much more reliable that way.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import factcheck as fc
from .prompts import (ANALYZE_PROMPT, CV_PROMPT, EVIDENCE_PROMPT, LETTER_PROMPT, REVIEWER_PROMPT,
                      REVISE_PROMPT)


# ----------------------------------------------------------------------------- structured outputs
class JobAnalysis(BaseModel):
    company_name: str
    role_title: str
    must_have: list[str] = Field(max_length=15)
    nice_to_have: list[str] = Field(default_factory=list, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=25)
    responsibilities: list[str] = Field(default_factory=list, max_length=8)
    company_hook: str = ""
    tone: Literal["formal", "neutral", "casual"] = "neutral"


class Match(BaseModel):
    requirement: str
    cv_quotes: list[str] = Field(default_factory=list, description="Verbatim text from the CV.")
    strength: Literal["strong", "partial", "none"]


class EvidenceMap(BaseModel):
    matches: list[Match]
    section_order: list[str] = Field(default_factory=list)


class Review(BaseModel):
    unsupported_claims: list[str] = Field(description="Claims in either draft that the original CV doesn't back.")
    cv_problems: list[str] = Field(default_factory=list)
    letter_problems: list[str] = Field(default_factory=list)
    approved: bool


# ----------------------------------------------------------------------------- state
class DraftState(TypedDict, total=False):
    cv: str                     # original CV (source of truth)
    job: str                    # posting text
    assessment: str             # the main agent's screening notes
    feedback: str               # optional instructions from the human, e.g. "shorter, lead with SQL"
    letter_words: tuple         # (min, max)
    max_rounds: int
    analysis: JobAnalysis
    evidence: EvidenceMap
    cv_draft: str
    letter_draft: str
    cv_issues: list[str]
    letter_issues: list[str]
    review: Review
    rounds: int
    coverage: dict


def _evidence_block(ev: EvidenceMap) -> str:
    lines = []
    for m in ev.matches:
        tag = {"strong": "MATCHED", "partial": "PARTIAL", "none": "MISSING"}[m.strength]
        q = " | ".join(f'"{x}"' for x in m.cv_quotes) or "-"
        lines.append(f"- [{tag}] {m.requirement}: {q}")
    return "\n".join(lines)


def build_drafter(model):
    analyzer = model.with_structured_output(JobAnalysis)
    mapper = model.with_structured_output(EvidenceMap)
    checker = model.with_structured_output(Review)

    def text_call(system: str, user: str) -> str:
        return fc.clean_output(model.invoke([("system", system), ("human", user)]).content)

    # ------------------------------------------------------------------ nodes
    def analyze_job(s: DraftState):
        return {"analysis": analyzer.invoke([("system", ANALYZE_PROMPT), ("human", s["job"])]), "rounds": 0}

    def map_evidence(s: DraftState):
        a = s["analysis"]
        reqs = "\n".join(f"- {r}" for r in a.must_have + a.nice_to_have)
        ev = mapper.invoke([("system", EVIDENCE_PROMPT),
                            ("human", f"<cv>\n{s['cv']}\n</cv>\n\n<requirements>\n{reqs}\n</requirements>")])
        # Grounding in code: drop quotes that aren't really in the CV, and downgrade unsupported matches.
        for m in ev.matches:
            m.cv_quotes = fc.verify_quotes(m.cv_quotes, s["cv"])
            if not m.cv_quotes:
                m.strength = "none"
        return {"evidence": ev}

    def _context(s: DraftState) -> str:
        a, ev = s["analysis"], s["evidence"]
        fb = f"\n\n<candidate_feedback>\n{s['feedback']}\n</candidate_feedback>" if s.get("feedback") else ""
        return (f"<original_cv>\n{s['cv']}\n</original_cv>\n\n<job>\n{s['job'][:6000]}\n</job>\n\n"
                f"ROLE: {a.role_title} at {a.company_name}\nCOMPANY HOOK: {a.company_hook or '-'}\n"
                f"KEYWORDS: {', '.join(a.keywords)}\nSECTION ORDER: {', '.join(ev.section_order) or 'keep original'}\n"
                f"EVIDENCE (requirement: CV quotes):\n{_evidence_block(ev)}{fb}")

    def write_cv(s: DraftState):
        return {"cv_draft": text_call(CV_PROMPT, _context(s))}

    def write_letter(s: DraftState):
        lo, hi = s.get("letter_words") or (220, 320)
        prompt = LETTER_PROMPT.format(min_words=lo, max_words=hi, tone=s["analysis"].tone)
        return {"letter_draft": text_call(prompt, _context(s))}

    def check(s: DraftState):
        a = s["analysis"]
        lo, hi = s.get("letter_words") or (220, 320)
        cv_issues = fc.check_cv(s["cv_draft"], s["cv"], s["job"], a.keywords + a.must_have)
        letter_issues = fc.check_letter(s["letter_draft"], a.company_name, a.role_title, lo, hi)
        auto = "\n".join(f"- CV: {i}" for i in cv_issues) + "\n" + "\n".join(f"- Letter: {i}" for i in letter_issues)
        review = checker.invoke([("system", REVIEWER_PROMPT), (
            "human", f"<original_cv>\n{s['cv']}\n</original_cv>\n\n<job>\n{s['job'][:3000]}\n</job>\n\n"
                     f"<draft_cv>\n{s['cv_draft']}\n</draft_cv>\n\n<draft_letter>\n{s['letter_draft']}\n</draft_letter>\n\n"
                     f"<automated_checks>\n{auto.strip() or 'none'}\n</automated_checks>")])
        hit, miss = fc.keyword_coverage(a.keywords, s["cv_draft"])
        before, _ = fc.keyword_coverage(a.keywords, s["cv"])
        coverage = {"keywords_before": len(before), "keywords_after": len(hit), "keywords_total": len(a.keywords),
                    "missing_keywords": miss,
                    "must_have": {m.requirement: m.strength for m in s["evidence"].matches if m.requirement in a.must_have}}
        return {"cv_issues": cv_issues, "letter_issues": letter_issues, "review": review, "coverage": coverage}

    def _todo(s: DraftState):
        r = s["review"]
        unsupported = r.unsupported_claims
        cv_fix = s["cv_issues"] + r.cv_problems + [f"Unsupported: {c}" for c in unsupported]
        letter_fix = s["letter_issues"] + r.letter_problems + [f"Unsupported (if in letter): {c}" for c in unsupported]
        return cv_fix, letter_fix

    def revise(s: DraftState):
        cv_fix, letter_fix = _todo(s)
        out = {"rounds": s.get("rounds", 0) + 1}
        ctx = _context(s)
        if s["cv_issues"] or s["review"].cv_problems or s["review"].unsupported_claims:
            out["cv_draft"] = text_call(
                REVISE_PROMPT.format(what="CV", instructions=CV_PROMPT),
                f"{ctx}\n\n<draft>\n{s['cv_draft']}\n</draft>\n\n<problems>\n- " + "\n- ".join(cv_fix) + "\n</problems>")
        if s["letter_issues"] or s["review"].letter_problems or s["review"].unsupported_claims:
            lo, hi = s.get("letter_words") or (220, 320)
            out["letter_draft"] = text_call(
                REVISE_PROMPT.format(what="cover letter",
                                     instructions=LETTER_PROMPT.format(min_words=lo, max_words=hi, tone=s["analysis"].tone)),
                f"{ctx}\n\n<draft>\n{s['letter_draft']}\n</draft>\n\n<problems>\n- " + "\n- ".join(letter_fix) + "\n</problems>")
        return out

    def route(s: DraftState):
        clean = s["review"].approved and not s["cv_issues"] and not s["letter_issues"]
        if clean or s.get("rounds", 0) >= s.get("max_rounds", 2):
            return END
        return "revise"

    # ------------------------------------------------------------------ wiring
    g = StateGraph(DraftState)
    for name, fn in [("analyze_job", analyze_job), ("map_evidence", map_evidence), ("write_cv", write_cv),
                     ("write_letter", write_letter), ("check", check), ("revise", revise)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze_job")
    g.add_edge("analyze_job", "map_evidence")
    g.add_edge("map_evidence", "write_cv")          # fan-out: both writers run in parallel
    g.add_edge("map_evidence", "write_letter")
    g.add_edge(["write_cv", "write_letter"], "check")  # fan-in: check waits for both
    g.add_conditional_edges("check", route, ["revise", END])
    g.add_edge("revise", "check")
    return g.compile()


def is_clean(out: dict) -> bool:
    return out["review"].approved and not out["cv_issues"] and not out["letter_issues"]


def remaining_problems(out: dict) -> list[str]:
    r = out["review"]
    return ([f"CV: {i}" for i in out["cv_issues"] + r.cv_problems] + [f"Letter: {i}" for i in out["letter_issues"] + r.letter_problems]
            + [f"Unsupported: {c}" for c in r.unsupported_claims])
