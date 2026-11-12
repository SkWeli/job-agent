"""CV tailoring + cover letter subgraph: grounding, fact-checks, revise loop, feedback redrafts."""
from __future__ import annotations

import json

from jobagent.drafting import build_drafter
from jobagent.factcheck import check_cv, clean_output, tech_terms, verify_quotes
from jobagent.models import Job
from jobagent.store import Tracker
from jobagent.tools.apply import summarize_draft, tailor

from .fakes import FIXTURES, GOOD_CV, GOOD_LETTER, ScriptedModel
from .test_graph import make_cfg

CV = (FIXTURES / "cv.md").read_text()
JOB = Job(source="remotejobs", title="Python Backend Engineer", company="Acme Analytics", url="https://acme.example/42",
          location="Remote - APAC", description="1-3 years of Python, FastAPI and PostgreSQL. REST APIs, Docker. "
                                                "Bonus: AWS, Kubernetes.")


def setup(tmp_path, cvs=None, letters=None):
    cfg = make_cfg(tmp_path)
    db = Tracker(cfg.data_dir / "jobs.db")
    db.add(JOB)
    db.update(JOB.uid, status="assessed", score=5, verdict="apply", reasons="strong match", gaps=json.dumps(["AWS"]))
    m = ScriptedModel(script=[], seen=[], draft_calls=[], structured={}, cvs=list(cvs or []), letters=list(letters or []))
    return cfg, db, m, build_drafter(m).invoke


def test_clean_draft_files_and_grounding(tmp_path):
    cfg, db, m, drafter = setup(tmp_path)
    d, out = tailor(cfg, db, drafter, JOB.uid)
    # the fake "Senior AWS architect" quote isn't in the CV, so code drops it and AWS becomes a gap
    aws = next(x for x in out["evidence"].matches if x.requirement == "AWS")
    assert aws.strength == "none" and aws.cv_quotes == []
    fit = (d / "fit.md").read_text()
    assert "❌ **AWS**" in fit and "If you genuinely have any of them" in fit and "AWS" in fit.split("Keywords not")[1]
    assert out["rounds"] == 0 and "PASSED" in (d / "review.md").read_text()
    assert sorted(m.draft_calls) == sorted(["JobAnalysis", "EvidenceMap", "cv", "letter", "Review"])
    s = summarize_draft(d, out)
    assert "Must-haves met 3/4 (missing: AWS)" in s and "PASSED" in s
    assert db.get(JOB.uid)["status"] == "drafted"
    for f in ("cv.docx", "cover_letter.docx"):
        assert (d / f).stat().st_size > 0


def test_invented_skill_and_number_get_revised_out(tmp_path):
    bad = GOOD_CV + "\n- Deployed services on AWS and Kubernetes, improving uptime by 40%"
    cfg, db, m, drafter = setup(tmp_path, cvs=[bad, GOOD_CV])
    d, out = tailor(cfg, db, drafter, JOB.uid)
    assert out["rounds"] == 1 and out["cv_issues"] == []
    assert "AWS" not in (d / "cv.md").read_text()
    assert m.draft_calls.count("cv") == 2 and m.draft_calls.count("letter") == 1   # only the CV was redone


def test_letter_placeholders_and_cliches_get_revised(tmp_path):
    bad = "Dear [Company] team,\n\nI am passionate about this role. " + "I build APIs. " * 60
    cfg, db, m, drafter = setup(tmp_path, letters=[bad, GOOD_LETTER])
    _, out = tailor(cfg, db, drafter, JOB.uid)
    assert out["rounds"] == 1 and out["letter_issues"] == []
    assert m.draft_calls.count("letter") == 2 and m.draft_calls.count("cv") == 1


def test_stops_after_max_rounds_and_flags(tmp_path):
    bad = GOOD_CV + "\n- Led a team of 30 engineers on AWS"
    cfg, db, m, drafter = setup(tmp_path, cvs=[bad])          # always bad
    d, out = tailor(cfg, db, drafter, JOB.uid)
    assert out["rounds"] == 2
    assert "NEEDS YOUR ATTENTION" in (d / "review.md").read_text()
    assert "FLAGGED" in summarize_draft(d, out)


def test_feedback_redraft_keeps_previous_version(tmp_path):
    cfg, db, m, drafter = setup(tmp_path)
    d1, _ = tailor(cfg, db, drafter, JOB.uid)
    prompts = []
    orig = m._generate

    def spy(messages, *a, **k):
        prompts.append(str(messages[-1].content))
        return orig(messages, *a, **k)

    object.__setattr__(m, "_generate", spy)
    d2, _ = tailor(cfg, db, drafter, JOB.uid, feedback="Shorter letter, lead with the SQL tuning work")
    assert d1 == d2 and (d2 / "previous" / "cv.md").exists()
    assert any("lead with the SQL tuning work" in p for p in prompts)


# ------------------------------------------------------------------------- deterministic checks
def test_factcheck_units():
    assert verify_quotes(["12 REST API endpoints in FastAPI", "AWS expert"], CV) == ["12 REST API endpoints in FastAPI"]
    terms = tech_terms("Python, FastAPI and PostgreSQL. Bonus: AWS, Kubernetes, CI/CD.", [])
    assert {"AWS", "Kubernetes", "CI/CD", "FastAPI", "PostgreSQL"} <= terms
    issues = check_cv("# Nimali Perera\nnimali@example.com\n- 12 APIs on AWS, 99.9% uptime", CV,
                      "Python, AWS", ["AWS"])
    assert any("99.9%" in i for i in issues) and any("AWS" in i for i in issues)
    assert check_cv(GOOD_CV, CV, JOB.description, ["Python", "FastAPI"]) == []
    assert clean_output("<think>hmm</think>\n```markdown\n# CV\n```") == "# CV"
