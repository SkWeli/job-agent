"""Deterministic checks on drafts: no LLM, so they work the same with any model, even a weak free one.

They catch the classic tailoring failures:
  * a number in the CV draft that isn't in the original (invented metrics, wrong dates)
  * a job term (e.g. "AWS") that appears in the draft CV but not in the original CV
  * placeholders, clichés, wrong/missing company name, contact details dropped, wrong length
"""
from __future__ import annotations

import re

CLICHES = ["passionate", "dynamic", "team player", "fast-paced", "go-getter", "synergy", "think outside the box",
           "i am writing to express", "hard-working", "results-driven", "self-starter", "detail-oriented"]
COMMON_CAPS = {"we", "you", "our", "the", "and", "for", "with", "remote", "team", "role", "job", "dear", "hiring",
               "manager", "about", "us", "apply", "requirements", "responsibilities", "benefits", "bonus", "salary",
               "experience", "skills", "full", "time", "senior", "junior", "mid", "years", "year", "month", "plus",
               "asia", "apac", "europe", "usa", "us", "uk", "eu", "lkr", "usd", "monday", "friday", "i", "a"}


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def contains(text: str, term: str) -> bool:
    t = re.escape(norm(term))
    return re.search(rf"(?<![a-z0-9]){t}(?![a-z0-9])", norm(text)) is not None


def verify_quotes(quotes: list[str], cv: str) -> list[str]:
    """Keep only evidence quotes that really occur in the CV (case/space-insensitive, ignoring bullets)."""
    base = norm(re.sub(r"[*_`#>-]", " ", cv))
    out = []
    for q in quotes:
        qn = norm(re.sub(r"[*_`#>-]", " ", q)).strip(" .")
        if len(qn) >= 3 and qn in base:
            out.append(q.strip())
    return out


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", text or "")}


def tech_terms(job_text: str, keywords: list[str]) -> set[str]:
    """Terms from the posting that look like skills/tools: the analysed keywords plus tokens like AWS,
    PostgreSQL, Node.js, C#, CI/CD, GPT-4 (all-caps, mixed case, or containing digits/symbols)."""
    terms = {k.strip() for k in keywords if k.strip()}
    text = job_text or ""
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9+#./-]*[A-Za-z0-9+#]", text):
        core = m.group().strip(".-/")
        if core.lower() in COMMON_CAPS or len(core) < 2:
            continue
        before = text[max(0, m.start() - 3):m.start()]
        mid_sentence_cap = core[0].isupper() and not re.search(r"(^|[.!?:\n•*-])\s*$", before) and m.start() > 0
        if core.isupper() or re.search(r"[a-z][A-Z]", core) or re.search(r"[0-9+#]|\.\w", core) or mid_sentence_cap:
            terms.add(core)
    return terms


def check_cv(draft: str, original: str, job_text: str, keywords: list[str]) -> list[str]:
    issues = []
    new_nums = sorted(numbers(draft) - numbers(original))
    if new_nums:
        issues.append(f"Numbers not in the original CV: {', '.join(new_nums)}. Remove or correct them.")
    invented = sorted(t for t in tech_terms(job_text, keywords) if contains(draft, t) and not contains(original, t))
    if invented:
        issues.append(f"Job terms in the CV that the original CV doesn't have: {', '.join(invented)}. Remove them.")
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", original)
    if emails and emails[0] not in draft:
        issues.append(f"Contact email {emails[0]} is missing; keep the original contact line.")
    if not re.search(r"^#\s+\S", draft, re.M):
        issues.append("Missing the '# Name' heading at the top.")
    issues += _common(draft)
    if len(draft.split()) > 750:
        issues.append(f"Too long ({len(draft.split())} words); aim for at most ~600.")
    return issues


def check_letter(draft: str, company: str, role: str, min_words: int, max_words: int) -> list[str]:
    issues = []
    n = len(draft.split())
    if not min_words <= n <= max_words + 40:
        issues.append(f"Length is {n} words; it should be {min_words}-{max_words}.")
    if company and not contains(draft, company) and not contains(draft, company.split()[0]):
        issues.append(f"The company name '{company}' isn't mentioned.")
    role_words = [w for w in re.findall(r"[a-z]+", role.lower()) if len(w) > 3]
    if role_words and sum(w in draft.lower() for w in role_words) < max(1, len(role_words) // 2):
        issues.append(f"The role title '{role}' isn't clearly mentioned.")
    lower = draft.lower()
    found = [c for c in CLICHES if c in lower]
    if found:
        issues.append(f"Clichés to replace with specifics: {', '.join(found)}.")
    return issues + _common(draft)


def _common(text: str) -> list[str]:
    ph = [p for p in re.findall(r"\[[^\]]{2,40}\]", text) if p.lower() != "[hiring manager]"]
    ph += re.findall(r"\{[^}]{1,40}\}|lorem ipsum|XXX", text, re.I)
    return [f"Placeholder text left in: {', '.join(sorted(set(ph)))}."] if ph else []


def keyword_coverage(keywords: list[str], text: str) -> tuple[list[str], list[str]]:
    hit = [k for k in keywords if contains(text, k)]
    return hit, [k for k in keywords if k not in hit]


def clean_output(text: str) -> str:
    """Models often wrap output in code fences or leak <think> blocks; strip both."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, re.S)
    return (m.group(1) if m else text).strip()
