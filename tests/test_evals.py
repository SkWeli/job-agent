"""The eval harness itself is tested with a scripted model: a good run passes, a bad run fails."""
import yaml

import evals.run_evals as ev
from .fakes import GOOD_CV, ScriptedModel, ai, call

A, U = ev.JOBS["acme"], ev.JOBS["usco"]
TASK = next(t for t in yaml.safe_load(open(ev.HERE / "tasks.yaml")) if t["name"] == "daily_triage")


def fake_factory(script, cv=GOOD_CV):
    def make(*_a, **_k):
        return ScriptedModel(seen=[], draft_calls=[], script=list(script), structured={}, cvs=[cv], letters=[])
    return make


def test_good_run_passes(monkeypatch):
    monkeypatch.setattr(ev, "make_model", fake_factory([
        ai(call("search_jobs", source="remotejobs", query="python")),
        ai(call("save_assessment", job_id=A, score=5, verdict="apply", reasons="fits well here", eligible_location=True)),
        ai(call("draft_application", job_id=A)), ai(text="Ready: Acme Analytics")]))
    r = ev.run_task(TASK, {})
    assert r["passed"] == r["total"], r["results"]


def test_bad_run_is_caught(monkeypatch):
    monkeypatch.setattr(ev, "make_model", fake_factory([
        ai(call("search_jobs", source="remotejobs", query="python")),
        ai(call("get_job_details", job_id="feedf00d")),                       # invented id
        ai(call("save_assessment", job_id=A, score=5, verdict="apply", reasons="fits well here", eligible_location=True)),
        ai(call("draft_application", job_id=A)), ai(text="All done")],     # report forgets Acme
        cv=GOOD_CV + "\n- AWS, Kubernetes"))                                  # invented skills
    r = ev.run_task(TASK, {})
    failed = {c["check"].split("=")[0] for c in r["results"] if not c["ok"]}
    assert {"only_real_job_ids", "draft_excludes", "report_mentions"} <= failed
