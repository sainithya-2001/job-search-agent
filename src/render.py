"""Render a verified TailoredResume to HTML (and PDF when a PDF engine is available), plus the verification report.

Facts that must never drift — name/contact, titles, companies, dates, degrees, GPA, skill and course names — are
filled in here from master_resume.yaml by id. The model's output contributes only the selected/reworded bullets,
the summary, and which ids to show.
"""
import re
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models import TailoredResume
from src.resume import _v

ROOT = Path(__file__).resolve().parent.parent
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


class RenderError(RuntimeError):
    pass


def _month(ym) -> str:
    if ym in (None, "", "none"):
        return ""
    if str(ym).lower() == "present":
        return "Present"
    y, m = str(ym).split("-")[:2]
    return f"{_MONTHS[int(m) - 1]} {y}"


def _dates(d) -> str:
    if not d or d == "none":
        return ""
    return f"{_month(_v(d.get('start')))} – {_month(_v(d.get('end')))}".strip(" –")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def fallback_summary(master: dict) -> str:
    """Used when the model's summary was removed by verification. Built only from master facts:
    total years (as stated by Nithya), a title she has held ("Analyst"), and the master's degree."""
    ms = next(ed for ed in master["education"] if ed.get("coursework"))
    return f"Analyst with {_v(master['years_experience'])} of experience. {_v(ms['degree'])}, {ms['institution']}."


def view_model(tr: TailoredResume, master: dict) -> dict:
    skills = {s["id"]: s["name"] for s in master["skills"]}
    exp = {e["id"]: e for e in master["experience"]}
    proj = {p["id"]: p for p in master["projects"]}
    c = master["contact"]
    head = [tr.headline.role_label] if tr.headline.role_label else []
    head += [skills[s] for s in tr.headline.skill_ids if s in skills]
    education = []
    for ed in master["education"]:
        cw = ed.get("coursework") or {}
        courses = {x["id"]: x["name"] for x in cw.get("core", []) + (cw.get("electives") or [])}
        loc = _v(ed.get("location"))
        end = _v((ed.get("dates") or {}).get("end")) if isinstance(ed.get("dates"), dict) else None
        education.append({"degree": _v(ed["degree"]), "institution": ed["institution"], "location": loc,
                          "date": _month(end), "gpa": _v(ed.get("gpa")),
                          "coursework": [courses[i] for i in tr.coursework_ids if i in courses]})
    return {
        "name": _v(c["name"]), "email": _v(c["email"]), "phone": _v(c["phone"]), "location": _v(c["location"]),
        "headline": " | ".join(head), "summary": tr.summary.text or fallback_summary(master),
        "experience": [{"title": _v(exp[s.id]["title"]), "company": exp[s.id]["company"],
                        "dates": _dates(exp[s.id].get("dates")), "bullets": [b.text for b in s.bullets]}
                       for s in tr.experience if s.id in exp and s.bullets],
        "projects": [{"name": _v(proj[s.id]["name"]), "tech": ", ".join(t["name"] for t in proj[s.id].get("tech", [])),
                      "bullets": [b.text for b in s.bullets]} for s in tr.projects if s.id in proj and s.bullets],
        "skill_groups": [{"label": g.label, "skills": [skills[i] for i in g.skill_ids if i in skills]}
                         for g in tr.skill_groups],
        "education": education,
    }


def render_html(tr: TailoredResume, master: dict) -> str:
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["html", "j2"]))
    return env.get_template("resume.html.j2").render(**view_model(tr, master))


def write_pdf(html: str, path: Path) -> None:
    try:
        import weasyprint
    except OSError as e:  # GTK/Pango system libraries missing (common on Windows)
        raise RenderError("WeasyPrint is installed but its GTK/Pango system libraries are missing; "
                          "PDF not written. HTML was saved instead.") from e
    weasyprint.HTML(string=html, base_url=str(ROOT)).write_pdf(path)


def verification_report(job, result, cost: float) -> str:
    v = result.verification
    lines = [f"# Verification report — {job['company']}: {job['title']}", "",
             f"- **Status:** {result.status}",
             f"- **Attempts:** {result.attempts}  |  **API calls:** {result.calls}  |  **Cost:** ${cost:.4f}",
             f"- **Generated:** {date.today().isoformat()}", ""]
    for n, issues in enumerate(result.attempt_issues, 1):
        lines.append(f"**Attempt {n}:** " + ("passed all checks" if not issues else f"{len(issues)} problem(s)"))
        lines += [f"  - {i}" for i in issues]
    lines += ["", "## Every claim, checked", "",
              "| Where | Text | Cites | Numbers | Tools | LLM judge |", "|---|---|---|---|---|---|"]
    for c in v.checked:
        judge = "—" if c.llm_supported is None else ("supported" if c.llm_supported else f"NOT supported: {c.llm_problem}")
        text = c.text.replace("|", "/")
        lines.append(f"| {c.where} | {text} | {', '.join(c.source_ids)} | {'ok' if c.numbers_ok else 'FAIL'} | "
                     f"{'ok' if c.terms_ok else 'FAIL'} | {judge} |")
    if result.removed:
        lines += ["", "## Removed (could not be sourced after all attempts)", ""] + [f"- {r}" for r in result.removed]
    if result.resume is not None and not result.resume.summary.text:
        lines += ["", "**Summary:** the model's summary was removed; the resume uses the fixed fallback summary "
                  "built only from master facts (see render.fallback_summary)."]
    if v.issues:
        lines += ["", "## Remaining problems", ""] + [f"- {i}" for i in v.issues]
    lines += ["", "Titles, companies, dates, education, GPA, skill names, and course names are filled in from",
              "master_resume.yaml by id and are not model-written."]
    return "\n".join(lines) + "\n"
