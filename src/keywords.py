"""ATS keyword matching for tailoring.

1. extract: one fast-tier LLM call pulls the 15-20 most important keywords from the posting's requirements, in the
   posting's exact wording. Cached per job in SQLite (job_keywords), so it is paid for once.
2. match: CODE (never the model) decides which keywords Nithya has, using only master_resume.yaml:
     skill name or alias  -> same skill, other wording         ("ELT" for ETL/ELT)
     skill covers         -> broader term the skill evidences  ("data warehouse" for Snowflake)
     bullet / project tech text containing the phrase
   Everything else is "missing" and is never claimed. Near-misses (the keyword contains one of her skill names, e.g.
   "Python scripting") are listed as possible synonyms for her to approve by adding an alias in master_resume.yaml.
3. wording: for matched skills, the resume shows the posting's wording (alias -> the posting's term;
   covers -> "Snowflake (data warehouse)").
4. resume_coverage: read-only check of an already-rendered resume against the job's keywords.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from src import tracing

EXTRACT_SYSTEM = """You extract ATS keywords from ONE job posting.
Return the 15-20 most important keywords and phrases a resume-screening system would look for: skills, tools,
technologies, methods, and domain terms. Rules:
- Copy each keyword exactly as the posting writes it (same words, same casing), as short as possible (1-4 words).
- Most important first: required qualifications before nice-to-haves; repeated terms rank higher.
- Exclude soft skills and filler ("team player", "fast-paced", "passionate"), years of experience, degrees,
  job titles, the company's own name and product names, benefits, and locations.
- No duplicates or near-duplicates (keep the posting's most common form)."""

ESTIMATE_OUTPUT_TOKENS = 500


class Keyword(BaseModel):
    text: str = Field(description="exact wording from the posting")
    kind: Literal["skill", "tool", "method", "domain"]
    importance: int = Field(ge=1, le=3, description="3 = required/core, 2 = important, 1 = nice to have")


class KeywordList(BaseModel):
    keywords: list[Keyword]


def norm(text: str) -> str:
    """Case/punctuation-insensitive form; keeps + and # (C++, C#)."""
    t = re.sub(r"[^a-z0-9+#]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _contains(haystack_norm: str, needle_norm: str) -> bool:
    return bool(needle_norm) and re.search(r"(?<![a-z0-9+#])" + re.escape(needle_norm) + r"(?![a-z0-9+#])",
                                           haystack_norm) is not None


# ---- extraction (cached) ----------------------------------------------------------------------------------------
def posting_text(job) -> str:
    from src.skills_gap import requirements_text
    req = requirements_text(job["description"] or "", job["company"])
    return req if len(req) > 400 else (job["description"] or "")


def extract_prompt(job) -> str:
    return f"JOB: {job['title']} at {job['company']}\n\nPOSTING\n{posting_text(job)}"


def cached(store, job_id: str) -> list[Keyword] | None:
    row = store.conn.execute("SELECT keywords FROM job_keywords WHERE job_id=?", (job_id,)).fetchone()
    return [Keyword(**k) for k in json.loads(row["keywords"])] if row else None


def get_keywords(store, job, llm, tier: str = "fast") -> tuple[list[Keyword], object | None]:
    """Cached keywords, or extract them now. Returns (keywords, LLMResponse or None if cached)."""
    hit = cached(store, job["id"])
    if hit is not None:
        return hit, None
    with tracing.get().span("keyword_extraction", job_id=job["id"]) as span:
        resp = llm.generate_json(extract_prompt(job), KeywordList, tier=tier, system=EXTRACT_SYSTEM)
        span.set(keywords=min(len(resp.data.keywords), 20), model=resp.model, ok=True)
    seen, kws = set(), []
    for k in sorted(resp.data.keywords, key=lambda k: -k.importance):
        if norm(k.text) and norm(k.text) not in seen:
            seen.add(norm(k.text))
            kws.append(k)
    kws = kws[:20]
    with store.conn:
        store.conn.execute("INSERT OR REPLACE INTO job_keywords (job_id, keywords, model, extracted_at) VALUES (?,?,?,?)",
                           (job["id"], json.dumps([k.model_dump() for k in kws]), resp.model,
                            datetime.now(timezone.utc).isoformat()))
    return kws, resp


# ---- matching against master facts (code only) ------------------------------------------------------------------
@dataclass
class KeywordMatch:
    keyword: Keyword
    have: bool
    how: str = ""                 # "skill name" | "alias" | "covers" | "bullet text"
    skill_id: str | None = None
    source_id: str | None = None
    display: str | None = None    # how the matched skill is shown on the resume
    hint: str = ""                # for missing keywords: a possible synonym for Nithya to approve


@dataclass
class Coverage:
    matches: list[KeywordMatch] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)      # soft skills / filler dropped before matching

    @property
    def have(self) -> list[KeywordMatch]:
        return [m for m in self.matches if m.have]

    @property
    def missing(self) -> list[KeywordMatch]:
        return [m for m in self.matches if not m.have]

    @property
    def percent(self) -> int:
        return round(100 * len(self.have) / len(self.matches)) if self.matches else 0

    def wording(self) -> dict[str, str]:
        """skill_id -> display text in the posting's wording (most important keyword wins)."""
        out = {}
        for m in sorted(self.have, key=lambda m: -m.keyword.importance):
            if m.skill_id and m.display and m.skill_id not in out:
                out[m.skill_id] = m.display
        return out


def split_soft(keywords: list[Keyword], patterns: list[str]) -> tuple[list[Keyword], list[str]]:
    """-> (keywords to match, soft-skill keywords ignored)."""
    rx = [re.compile(p, re.I) for p in patterns or []]
    keep, soft = [], []
    for k in keywords:
        (soft.append(k.text) if any(r.search(k.text) for r in rx) else keep.append(k))
    return keep, soft


def match(keywords: list[Keyword], master: dict, texts: dict[str, str], tech: dict[str, list[str]],
          soft_patterns: list[str] | None = None) -> Coverage:
    skills = master["skills"]
    keywords, soft = split_soft(keywords, soft_patterns or [])
    cov = Coverage(ignored=soft)
    for kw in keywords:
        k = norm(kw.text)
        m = KeywordMatch(kw, False)
        for s in skills:
            if k == norm(s["name"]):
                # same words, possibly different casing/punctuation: show the posting's form
                m.have, m.how, m.skill_id, m.display = True, "skill name", s["id"], kw.text
                break
            if any(k == norm(a) for a in s.get("aliases") or []):
                m.have, m.how, m.skill_id, m.display = True, f"alias of {s['name']}", s["id"], kw.text
                break
            if any(k == norm(c) for c in s.get("covers") or []):
                m.have, m.how, m.skill_id, m.display = True, f"covered by {s['name']}", s["id"], f"{s['name']} ({kw.text})"
                break
        if not m.have:
            for sid, text in texts.items():
                if _contains(norm(text), k):
                    m.have, m.how, m.source_id = True, "bullet text", sid
                    break
        if not m.have:
            for pid, names in tech.items():
                if any(k == norm(n) for n in names):
                    m.have, m.how, m.source_id = True, "project tech", pid
                    break
        if not m.have:
            near = [s["name"] for s in skills if len(norm(s["name"])) >= 3 and _contains(k, norm(s["name"]))]
            if near:
                m.hint = f"contains your skill {near[0]!r}: approve as an alias in master_resume.yaml if it's the same thing"
        cov.matches.append(m)
    return cov


# ---- equivalent wordings (for verification) ---------------------------------------------------------------------
def equivalents(master: dict) -> list[tuple[set[str], set[str]]]:
    """Approved wordings per skill as (triggers, allowed). If a source names any trigger (the skill's name or an
    alias), any allowed wording may be used. `covers` terms are allowed but never triggers: "Snowflake" supports
    "data warehouse", but a source saying "data warehouse" does not support "Snowflake"."""
    groups = []
    for s in master["skills"]:
        aliases, covers = s.get("aliases") or [], s.get("covers") or []
        if aliases or covers:
            triggers = {s["name"], *aliases}
            groups.append((triggers, triggers | set(covers)))
    return groups


# ---- reports ----------------------------------------------------------------------------------------------------
def report_section(cov: Coverage, title: str = "Keyword coverage (ATS)") -> str:
    lines = [f"## {title}", "", f"**{len(cov.have)} of {len(cov.matches)} keywords covered ({cov.percent}%)**", "",
             "| Keyword (posting's wording) | Importance | You have it? | How |", "|---|---|---|---|"]
    if cov.ignored:
        lines.insert(3, f"Ignored as soft skills / filler: {', '.join(cov.ignored)}\n")
    for m in cov.matches:
        how = m.how + (f" ({m.skill_id or m.source_id})" if (m.skill_id or m.source_id) else "")
        lines.append(f"| {m.keyword.text} | {m.keyword.importance} | {'yes' if m.have else 'no'} | "
                     f"{how if m.have else (m.hint or 'not in master_resume.yaml: never claimed')} |")
    return "\n".join(lines) + "\n"


@dataclass
class ResumeCoverage:
    exact: list[str]               # posting's wording appears on the resume
    synonym: list[str]             # you have it and the resume shows it, but in different words (ATS may miss it)
    not_shown: list[str]           # you have it (master_resume.yaml) but this resume doesn't show it
    missing: list[str]             # not in master_resume.yaml
    hints: list[str]
    ignored: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.exact) + len(self.synonym) + len(self.not_shown) + len(self.missing)

    def pct(self, n: int) -> int:
        return round(100 * n / self.total) if self.total else 0


def resume_coverage(resume_text: str, cov: Coverage, master: dict) -> ResumeCoverage:
    """Read-only: which of the job's keywords this already-rendered resume actually shows."""
    text = norm(resume_text)
    names = {s["id"]: s for s in master["skills"]}
    out = ResumeCoverage([], [], [], [], [], list(cov.ignored))
    for m in cov.matches:
        kw = m.keyword.text
        if _contains(text, norm(kw)):
            out.exact.append(kw)
        elif not m.have:
            out.missing.append(kw)
            if m.hint:
                out.hints.append(f"{kw}: {m.hint}")
        elif m.skill_id and (shown := next((w for w in [names[m.skill_id]["name"], *(names[m.skill_id].get("aliases") or []),
                                                        *(names[m.skill_id].get("covers") or [])]
                                            if _contains(text, norm(w))), None)):
            # the skill is on the resume under another of its approved wordings (e.g. "evals" for "evaluations")
            out.synonym.append(f"{kw} (shown as {shown})")
        else:
            out.not_shown.append(kw)
    return out


def strip_coverage(report: str) -> str:
    """Remove earlier keyword-coverage sections from a verification report, keeping every other section
    (one-page cuts, post-verification edits) intact."""
    parts = re.split(r"(?m)^(?=## )", report)
    kept = [p for p in parts if not p.startswith("## Keyword coverage (ATS)")]
    return "".join(kept).rstrip() + "\n"


def resume_section(rc: ResumeCoverage) -> str:
    lines = ["## Keyword coverage (ATS) — read-only check of this resume", "",
             f"- **On the resume in the posting's wording:** {len(rc.exact)} of {rc.total} ({rc.pct(len(rc.exact))}%)",
             f"- **On the resume in other words** (ATS may miss): {len(rc.synonym)}",
             f"- **You have it, but this resume doesn't show it:** {len(rc.not_shown)}",
             f"- **Not in master_resume.yaml (never claimed):** {len(rc.missing)}", "",
             "Covered: " + (", ".join(rc.exact) or "none"),
             "", "In other words: " + (", ".join(rc.synonym) or "none"),
             "", "Not shown: " + (", ".join(rc.not_shown) or "none"),
             "", "Don't have: " + (", ".join(rc.missing) or "none"),
             "", "Ignored as soft skills / filler: " + (", ".join(rc.ignored) or "none")]
    if rc.hints:
        lines += ["", "Possible synonyms to approve:"] + [f"- {h}" for h in rc.hints]
    return "\n".join(lines) + "\n"
