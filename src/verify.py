"""Phase 3 guardrail: every claim in a tailored resume must trace to master_resume.yaml.

Two layers:
1. Deterministic checks (free, exact). Structure and ids; the headline rule; and, for every bullet and the summary,
   that each NUMBER and each known TOOL/SKILL term in the text also appears in the facts it cites.
2. An LLM judge (strong tier) that reads each bullet next to its cited sources and flags anything not supported:
   added outcomes, scope, seniority, or tools the string checks can't see.

Problems from both layers go back to the tailoring step for a retry.
"""
import re
from dataclasses import dataclass, field

from src.llm import LLMProvider, LLMResponse
from src.models import TailoredResume, Verification
from src.skills_gap import VOCAB

# Numbers like 40+, 30%, 6.5%, 200K+, 3.2K, 1,500. Compared by their core value ("100k", "4.5").
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)(\s?[kK]\b)?")


# Summary rules. "4+ years of experience" may stand alone or span roles; it must not be attached to specific work
# ("4+ years building GenAI apps") because the years are total experience, mostly business analysis.
_YEARS = re.compile(r"\b4\+\s*years(\s+of\s+(professional\s+|work\s+)?experience)?", re.I)
_YEARS_OK_AFTER = re.compile(r"\s*($|[,.;)]|across\b|spanning\b|in business analysis\b|as an? (business )?analyst\b)", re.I)
# Self-descriptions implying a title she hasn't held.
_SELF_TITLE = re.compile(r"\b((ai|ml|software|applied ai|genai|llm|solutions?|forward[- ]deployed|analytics|data|"
                         r"machine learning)\s+(engineer|architect|scientist)s?|data scientists?|engineer)\b", re.I)


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUM.finditer(text or ""):
        core = m.group(1).replace(",", "")
        out.add(core + ("k" if m.group(2) else ""))
    return out


@dataclass
class Issue:
    where: str          # e.g. "exp_emp_a bullet 2", "summary", "headline"
    problem: str

    def __str__(self) -> str:
        return f"{self.where}: {self.problem}"


@dataclass
class CheckedText:
    where: str
    text: str
    source_ids: list[str]
    numbers_ok: bool = True
    terms_ok: bool = True
    llm_supported: bool | None = None
    llm_problem: str | None = None


@dataclass
class VerificationResult:
    issues: list[Issue] = field(default_factory=list)
    checked: list[CheckedText] = field(default_factory=list)
    llm_response: LLMResponse | None = None

    @property
    def passed(self) -> bool:
        return not self.issues


class Verifier:
    def __init__(self, master: dict, texts: dict[str, str], tech: dict[str, list[str]], cfg: dict, role_family: str):
        self.master, self.texts, self.tech = master, texts, tech
        t = cfg["tailoring"]
        self.allowed_labels = t["headline_role_labels"].get(role_family, [])
        self.forbidden = [w.lower() for w in t["headline_forbidden_words"]]
        self.exp_ids = [e["id"] for e in master["experience"]]
        self.proj_ids = [p["id"] for p in master["projects"]]
        self.skill_ids = {s["id"] for s in master["skills"]}
        univ = next(ed for ed in master["education"] if ed.get("coursework"))
        cw = univ["coursework"]
        self.course_ids = {c["id"] for c in cw.get("core", []) + (cw.get("electives") or [])}
        # Tool/skill terms: the skills-gap vocabulary plus every master skill name and project tech name.
        names = {s["name"] for s in master["skills"]} | {n for v in tech.values() for n in v}
        self.terms = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in VOCAB]
        self.terms += [(n, [re.compile(r"(?<![\w])" + re.escape(n) + r"(?![\w])", re.I)]) for n in sorted(names)]

    # --- deterministic -------------------------------------------------------------------------------------------
    def _source_text(self, ids: list[str], section: str | None) -> str:
        parts = [self.texts.get(i, "") for i in ids]
        if section in self.tech:          # a project's tech list counts as a fact of that project
            parts.append(", ".join(self.tech[section]))
        return " \n ".join(parts)

    def _check_text(self, where: str, text: str, ids: list[str], section: str | None, res: VerificationResult,
                    allowed_prefixes: tuple[str, ...] | None = None) -> CheckedText:
        ct = CheckedText(where, text, ids)
        if not ids:
            res.issues.append(Issue(where, "cites no source ids"))
        for i in ids:
            if i not in self.texts:
                res.issues.append(Issue(where, f"cites unknown or unusable id '{i}'"))
            elif allowed_prefixes and not i.startswith(allowed_prefixes):
                res.issues.append(Issue(where, f"cites '{i}', which belongs to a different entry"))
        src = self._source_text(ids, section)
        extra_nums = numbers_in(text) - numbers_in(src)
        if extra_nums:
            ct.numbers_ok = False
            res.issues.append(Issue(where, f"number(s) not in cited sources: {sorted(extra_nums)}"))
        extra_terms = [name for name, pats in self.terms
                       if any(p.search(text) for p in pats) and not any(p.search(src) for p in pats)]
        if extra_terms:
            ct.terms_ok = False
            res.issues.append(Issue(where, f"tool/skill term(s) not in cited sources: {sorted(set(extra_terms))}"))
        if re.search(r"\b5\+\s*years\b", text, re.I):
            res.issues.append(Issue(where, "says '5+ years' (must be '4+ years')"))
        res.checked.append(ct)
        return ct

    def check_structure(self, tr: TailoredResume) -> VerificationResult:
        res = VerificationResult()
        # headline
        label = tr.headline.role_label
        if label and label not in self.allowed_labels:
            res.issues.append(Issue("headline", f"role label '{label}' not allowed; use one of {self.allowed_labels} or null"))
        for sid in tr.headline.skill_ids:
            if sid not in self.skill_ids:
                res.issues.append(Issue("headline", f"unknown skill id '{sid}'"))
        if not 3 <= len(tr.headline.skill_ids) <= 5:
            res.issues.append(Issue("headline", f"needs 3-5 skills, got {len(tr.headline.skill_ids)}"))
        if label and any(re.search(rf"\b{w}\b", label.lower()) for w in self.forbidden):
            res.issues.append(Issue("headline", "contains a forbidden seniority word"))
        # summary
        self._check_text("summary", tr.summary.text, tr.summary.source_ids, None, res)
        if len(tr.summary.text.split()) > 50:
            res.issues.append(Issue("summary", "longer than 50 words"))
        for m in _YEARS.finditer(tr.summary.text):
            if not _YEARS_OK_AFTER.match(tr.summary.text[m.end():]):
                res.issues.append(Issue("summary", "ties '4+ years' to specific work or skills; the years are total "
                                        "experience (mostly business analysis). Write '4+ years of experience' standalone "
                                        "or '4+ years of experience across business analysis, data, and AI roles'"))
        title = _SELF_TITLE.search(tr.summary.text)
        if title:
            res.issues.append(Issue("summary", f"describes the candidate as '{title.group(0)}', a title she has not held; "
                                    "use 'Analyst' or 'data and AI professional' (role labels go in the headline only)"))
        # experience: every entry, in master order
        got = [s.id for s in tr.experience]
        if got != self.exp_ids:
            res.issues.append(Issue("experience", f"must list every entry in order {self.exp_ids}; got {got}"))
        for sec in tr.experience + tr.projects:
            is_proj = sec.id in self.proj_ids
            if not is_proj and sec.id not in self.exp_ids:
                res.issues.append(Issue(sec.id, "unknown section id"))
                continue
            if not sec.bullets:
                res.issues.append(Issue(sec.id, "has no bullets"))
            for n, b in enumerate(sec.bullets, 1):
                self._check_text(f"{sec.id} bullet {n}", b.text, b.source_ids, sec.id, res, (sec.id + "_b",))
        if not 1 <= len(tr.projects) <= 3:
            res.issues.append(Issue("projects", f"choose 1-3 projects, got {len(tr.projects)}"))
        total = sum(len(s.bullets) for s in tr.experience + tr.projects)
        if total > 16:
            res.issues.append(Issue("length", f"{total} bullets; keep to 16 or fewer for one page"))
        # skills and coursework
        for g in tr.skill_groups:
            for sid in g.skill_ids:
                if sid not in self.skill_ids:
                    res.issues.append(Issue(f"skills '{g.label}'", f"unknown skill id '{sid}'"))
        bad_cw = [c for c in tr.coursework_ids if c not in self.course_ids]
        if bad_cw:
            res.issues.append(Issue("coursework", f"unknown course ids {bad_cw}"))
        if not 4 <= len(tr.coursework_ids) <= 6:
            res.issues.append(Issue("coursework", f"pick 4-6 courses, got {len(tr.coursework_ids)}"))
        return res

    # --- LLM judge ------------------------------------------------------------------------------------------------
    def check_with_llm(self, res: VerificationResult, llm: LLMProvider, tier: str) -> None:
        items = []
        for i, ct in enumerate(res.checked):
            section = ct.where.split(" bullet")[0] if " bullet" in ct.where else None
            items.append(f"#{i}\nTEXT: {ct.text}\nSOURCES:\n{self._source_text(ct.source_ids, section)}")
        prompt = ("For each numbered item, decide whether TEXT is fully supported by its SOURCES.\n"
                  "Rewording, shortening, and merging are fine. Mark supported=false if TEXT adds anything the sources "
                  "do not state: a tool, number, outcome, scale, customer, responsibility, seniority or leadership "
                  "claim, or an intensifier that changes the claim (e.g. 'production', 'enterprise', 'led', 'owned', "
                  "'architected'). For the summary: the candidate's '4+ years' is total experience, mostly business "
                  "analysis; mark it unsupported if it implies years of a specific skill or kind of work, or describes "
                  "the candidate with a job title the sources don't show. Return one verdict per item, using the item "
                  "number as index.\n\n"
                  + "\n\n".join(items))
        resp = llm.generate_json(prompt, Verification, tier=tier,
                                 system="You are a strict fact-checker for resumes. Never invent; judge only support.")
        res.llm_response = resp
        by_index = {v.index: v for v in resp.data.verdicts}
        for i, ct in enumerate(res.checked):
            v = by_index.get(i)
            if v is None:
                res.issues.append(Issue(ct.where, "verifier returned no verdict"))
                continue
            ct.llm_supported, ct.llm_problem = v.supported, v.problem
            if not v.supported:
                res.issues.append(Issue(ct.where, f"not supported by sources: {v.problem}"))
