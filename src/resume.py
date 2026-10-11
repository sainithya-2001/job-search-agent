"""Load data/master_resume.yaml and render the candidate profile that LLM prompts see.

What the model sees is deliberately narrower than the file:
- no contact details (name, email, phone) — not needed to judge fit, so never sent to an API
- no old summaries/headlines — they contain superseded claims ("5+ years")
- nothing marked `outdated`
If any conflict is still unresolved, loading fails: the agent must never guess between versions.
"""
from pathlib import Path

import yaml


class ResumeError(RuntimeError):
    pass


def load_master(path: str | Path = "data/master_resume.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise ResumeError(f"{p} not found — the master resume is required for scoring.")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    open_items = [c["id"] for c in data.get("conflicts", []) if c.get("status") != "resolved"]
    open_items += [q["id"] for q in data.get("missing", []) if q.get("status") != "resolved"]
    if open_items:
        raise ResumeError(f"master_resume.yaml has unresolved items: {open_items}")
    return data


def _v(x):
    """Unwrap {value: ..., sources: ...} wrappers; refuse anything that still holds competing options."""
    if isinstance(x, dict):
        if "options" in x:
            raise ResumeError(f"unresolved field: {x}")
        if "value" in x:
            return x["value"]
    return x


def _dates(d) -> str:
    if not d or d == "none":
        return ""
    start, end = _v(d.get("start")), _v(d.get("end"))
    return f" ({start} to {end})"


def profile_text(data: dict) -> str:
    lines = [f"Years of experience: {_v(data['years_experience'])}", "", "EXPERIENCE"]
    for e in data["experience"]:
        lines.append(f"- {_v(e['title'])}, {e['company']}{_dates(e.get('dates'))}")
        lines += [f"    * {b['text']}" for b in e["bullets"] if "outdated" not in b]

    lines += ["", "PROJECTS"]
    for p in data["projects"]:
        tech = ", ".join(t["name"] for t in p.get("tech", []))
        lines.append(f"- {_v(p['name'])} [{tech}]")
        lines += [f"    * {b['text']}" for b in p["bullets"] if "outdated" not in b]
    for m in data["metrics"]:
        if m["id"] == "m_readmit_dataset":  # the corrected wording replaces the outdated bullet's phrase
            lines.append(f"    * Dataset: {_v(m)}")

    lines += ["", "SKILLS", ", ".join(s["name"] for s in data["skills"])]

    lines += ["", "EDUCATION"]
    for ed in data["education"]:
        status = _v(ed.get("status")) or ""
        lines.append(f"- {_v(ed['degree'])}, {ed['institution']}{_dates(ed.get('dates'))} {status}".rstrip())
        cw = ed.get("coursework")
        if cw:
            courses = [c["name"] for c in cw.get("core", []) + (cw.get("electives") or [])]
            lines.append(f"    Coursework: {'; '.join(courses)}")
    return "\n".join(lines)
