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


def view_model(tr: TailoredResume, master: dict, summary: str, wording: dict[str, str] | None = None) -> dict:
    """wording: skill id -> display text in the posting's wording (src/keywords.py; only approved names/aliases/
    covers of that same skill), used in the headline and skills section."""
    skills = {s["id"]: s["name"] for s in master["skills"]}
    skills.update({k: v for k, v in (wording or {}).items() if k in skills})
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
        "headline": " | ".join(head), "summary": summary,
        "experience": [{"title": _v(exp[s.id]["title"]), "company": exp[s.id]["company"],
                        "dates": _dates(exp[s.id].get("dates")), "bullets": [b.text for b in s.bullets]}
                       for s in tr.experience if s.id in exp and s.bullets],
        "projects": [{"name": _v(proj[s.id]["name"]), "tech": ", ".join(t["name"] for t in proj[s.id].get("tech", [])),
                      "bullets": [b.text for b in s.bullets]} for s in tr.projects if s.id in proj and s.bullets],
        "skill_groups": [{"label": g.label, "skills": [skills[i] for i in g.skill_ids if i in skills]}
                         for g in tr.skill_groups],
        "education": education,
    }


def render_html(tr: TailoredResume, master: dict, summary: str, wording: dict[str, str] | None = None) -> str:
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["html", "j2"]))
    return env.get_template("resume.html.j2").render(**view_model(tr, master, summary, wording))


_PAGE = re.compile(rb"/Type\s*/Page(?!s)")


def write_pdf(html: str, path: Path) -> int:
    """Print HTML to PDF with headless Chromium (Playwright). Returns the page count.
    (WeasyPrint was dropped: on Windows it needs GTK/Pango system libraries.)"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RenderError("Playwright is not installed: pip install playwright && playwright install chromium") from e
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(path=str(path), prefer_css_page_size=True, print_background=True)
        browser.close()
    return len(_PAGE.findall(Path(path).read_bytes()))


def fit_one_page(tr: TailoredResume, master: dict, summary: str, path: Path,
                 optional: set[str], required: set[str],
                 wording: dict[str, str] | None = None) -> tuple[str, int, list[str]]:
    """Write a PDF; while it runs past one page, cut content (never add or reword):
    1) drop optional experience entries (last first), then
    2) drop the last bullet of the section with the most bullets, skipping required bullets and sole bullets.
    Returns (final html, pages, list of cuts)."""
    cuts = []
    while True:
        html = render_html(tr, master, summary, wording)
        pages = write_pdf(html, path)
        if pages <= 1:
            return html, pages, cuts
        opt = [e for e in tr.experience if e.id in optional]
        if opt:
            tr.experience.remove(opt[-1])
            cuts.append(f"dropped optional entry {opt[-1].id} to fit one page")
            continue
        candidates = [s for s in tr.experience + tr.projects if len(s.bullets) > 1
                      and not set(s.bullets[-1].source_ids) & required]
        if not candidates:
            cuts.append(f"still {pages} pages; nothing left that can be cut automatically")
            return html, pages, cuts
        sec = max(candidates, key=lambda s: len(s.bullets))
        b = sec.bullets.pop()
        cuts.append(f"cut {sec.id} bullet to fit one page: {b.text}")


def verification_report(job, result, cost: float, summary_id: str = "") -> str:
    v = result.verification
    lines = [f"# Verification report — {job['company']}: {job['title']}", "",
             f"- **Status:** {result.status}",
             f"- **Attempts:** {result.attempts}  |  **API calls:** {result.calls}  |  **Cost:** ${cost:.4f}",
             f"- **Summary:** pre-approved `{summary_id}` from master_resume.yaml (not model-written)",
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
    if v.issues:
        lines += ["", "## Remaining problems", ""] + [f"- {i}" for i in v.issues]
    lines += ["", "Titles, companies, dates, education, GPA, skill names, and course names are filled in from",
              "master_resume.yaml by id and are not model-written."]
    return "\n".join(lines) + "\n"
