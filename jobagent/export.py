"""Files the human uses: draft folders (Markdown + Word), the run report, tracker.xlsx."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def slug(s: str, n: int = 30) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:n] or "x"


def write_draft(out_root: Path, row, out: dict, folder: Path | None = None) -> Path:
    """Write one application folder. Re-drafting (with feedback) moves the old version to previous/."""
    d = folder or out_root / date.today().isoformat() / f"{row['score']}-{slug(row['company'], 22)}-{slug(row['title'])}-{row['id']}"
    d.mkdir(parents=True, exist_ok=True)
    old = [p for p in d.iterdir() if p.is_file() and p.name != "job.md"]
    if old:
        prev = d / "previous"
        prev.mkdir(exist_ok=True)
        for p in old:
            p.replace(prev / p.name)
    a, ev, cov, rv = out["analysis"], out["evidence"], out["coverage"], out["review"]
    (d / "job.md").write_text(
        f"# {row['title']} — {row['company']}\n\n**Apply:** {row['url']}\n\n**Location:** {row['location']}\n\n"
        f"**Fit:** {row['score']}/5 ({row['verdict']}) — {row['reasons']}\n\n---\n\n{row['description']}\n", encoding="utf-8")
    (d / "cv.md").write_text(out["cv_draft"] + "\n", encoding="utf-8")
    (d / "cover_letter.md").write_text(out["letter_draft"] + "\n", encoding="utf-8")

    icon = {"strong": "✅", "partial": "🟡", "none": "❌"}
    lines = [f"# How your CV matches: {a.role_title} at {a.company_name}", "",
             f"**ATS keywords in your CV:** {cov['keywords_before']}/{cov['keywords_total']} before tailoring → "
             f"**{cov['keywords_after']}/{cov['keywords_total']}** after", "", "## Requirements", ""]
    for m in ev.matches:
        kind = "must-have" if m.requirement in a.must_have else "nice-to-have"
        lines.append(f"- {icon[m.strength]} **{m.requirement}** ({kind})" +
                     (f": _{'; '.join(m.cv_quotes)}_" if m.cv_quotes else ""))
    if cov["missing_keywords"]:
        lines += ["", "## Keywords not in your CV",
                  "The agent never adds these for you. **If you genuinely have any of them, add them to "
                  "`profile/cv.md`** and redraft:", "", ", ".join(cov["missing_keywords"])]
    (d / "fit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from .drafting import is_clean, remaining_problems
    probs = remaining_problems(out)
    (d / "review.md").write_text(
        f"# Fact-check: {'PASSED' if is_clean(out) else 'NEEDS YOUR ATTENTION'} ({out.get('rounds', 0)} revision round(s))\n\n"
        + ("".join(f"- {p}\n" for p in probs) or "No problems left.\n")
        + "\nRead the CV and letter yourself before sending. You are the final check.\n", encoding="utf-8")
    md_to_docx(out["cv_draft"], d / "cv.docx")
    md_to_docx(out["letter_draft"], d / "cover_letter.docx")
    return d


def md_to_docx(md: str, path: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            level = len(m.group(1))
            doc.add_heading(re.sub(r"\*\*|`", "", m.group(2)), level=0 if level == 1 else min(level, 3))
            continue
        b = re.match(r"^\s*[-*•]\s+(.*)", s)
        p = doc.add_paragraph(style="List Bullet") if b else doc.add_paragraph()
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", b.group(1) if b else s)):
            if part:
                p.add_run(part.replace("`", "")).bold = i % 2 == 1
    doc.save(path)


def write_report(out_root: Path, run_id: str, text: str, stop_reason: str) -> Path:
    d = out_root / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"report-{run_id}.md"
    p.write_text(f"{text}\n\n---\n_run {run_id} · stopped: {stop_reason}_\n", encoding="utf-8")
    return p


def export_xlsx(rows, path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    from .store import STATUSES

    wb = Workbook()
    ws = wb.active
    ws.title = "Tracker"
    ws.append(["ID", "Status", "Score", "Verdict", "Title", "Company", "Location", "Source", "Apply link",
               "Reasons", "Gaps", "First seen", "Drafts folder", "Notes"])
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F4E78")
    for r in rows:
        ws.append([r["id"], r["status"], r["score"], r["verdict"], r["title"], r["company"], r["location"],
                   r["source"], r["url"], r["reasons"], r["gaps"], (r["first_seen"] or "")[:10], r["draft_dir"],
                   r["notes"] or r["filter_reason"]])
        if r["url"]:
            cell = ws.cell(row=ws.max_row, column=9)
            cell.hyperlink, cell.font = r["url"], Font(color="0563C1", underline="single")
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"')
    ws.add_data_validation(dv)
    dv.add(f"B2:B{max(ws.max_row, 2)}")
    for i, w in enumerate([10, 10, 6, 8, 38, 24, 22, 16, 40, 50, 30, 11, 40, 30], 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    ws.freeze_panes, ws.auto_filter.ref = "A2", ws.dimensions
    wb.save(path)
    return path
