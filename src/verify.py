"""Phase 3 guardrail: every claim in a tailored resume must trace to master_resume.yaml.

Two layers for model-written bullets:
1. Deterministic checks (free, exact). Structure and ids; the headline rule; the tailoring rules from config.yaml
   (required bullets/projects, preferences, optional entries, one bullet per same-work group); and, for every
   bullet, that each NUMBER and each known TOOL/SKILL term in the text also appears in the facts it cites.
2. An LLM judge (strong tier) that reads each bullet next to its cited sources and flags anything not supported:
   added outcomes, scope, seniority, or tools the string checks can't see.

The summary is not model-written: it is one of the pre-approved summaries in master_resume.yaml, picked by code
(`select_summary`) and checked once (`check_approved_summaries`).
"""
import re
from dataclasses import dataclass, field

from src.llm import LLMProvider, LLMResponse
from src.models import TailoredResume, Verification
from src.skills_gap import VOCAB

# Numbers like 40+, 30%, 6.5%, 200K+, 3.2K, 1,500. Compared by their core value ("100k", "4.5").
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)(\s?[kK]\b)?")
# Summary wording checks (used when checking the approved summaries).
_YEARS = re.compile(r"\b4\+\s*years(\s+of\s+(professional\s+|work\s+)?experience)?", re.I)
_YEARS_OK_AFTER = re.compile(r"\s*($|[,.;)]|across\b|spanning\b|in business analysis\b|as an? (business )?analyst\b)", re.I)
_SELF_TITLE = re.compile(r"\b((ai|ml|software|applied ai|genai|llm|solutions?|forward[- ]deployed|analytics|data|"
                         r"machine learning)\s+(engineer|architect|scientist)s?|data scientists?|engineer)\b", re.I)


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUM.finditer(text or ""):
        core = m.group(1).replace(",", "")
        out.add(core + ("k" if m.group(2) else ""))
    return out


def _compile_terms(master: dict, tech: dict[str, list[str]]):
    """Tool/skill terms: the skills-gap vocabulary plus every master skill name and project tech name."""
    names = {s["name"] for s in master["skills"]} | {n for v in tech.values() for n in v}
    terms = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in VOCAB]
    terms += [(n, [re.compile(r"(?<![\w])" + re.escape(n) + r"(?![\w])", re.I)]) for n in sorted(names)]
    return terms


@dataclass
class Issue:
    where: str          # e.g. "exp_emp_a bullet 2", "headline", "rules"
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


def _applies(rule: dict, family: str, title: str) -> bool:
    if family not in rule.get("families", []):
        return False
    return not (rule.get("unless_title") and re.search(rule["unless_title"], title or "", re.I))


def select_summary(master: dict, family: str, title: str) -> dict:
    """Pick the pre-approved summary for this job: role family first, then title conditions."""
    for s in master.get("approved_summaries", []):
        use = s["use_for"]
        if family not in use.get("role_families", []):
            continue
        if use.get("title") and not re.search(use["title"], title or "", re.I):
            continue
        if use.get("title_not") and re.search(use["title_not"], title or "", re.I):
            continue
        return s
    raise ValueError(f"no approved summary for role family '{family}' / title '{title}'")


def check_approved_summaries(master: dict, texts: dict[str, str], tech: dict[str, list[str]]) -> list[str]:
    """One-time check of the pre-approved summaries against ALL facts in the master resume.
    Returns findings (empty = every number and tool term appears somewhere in the master facts)."""
    corpus = " \n ".join(texts.values()) + " \n " + " \n ".join(n for v in tech.values() for n in v)
    terms, findings = _compile_terms(master, tech), []
    for s in master.get("approved_summaries", []):
        t = s["text"]
        nums = numbers_in(t) - numbers_in(corpus)
        missing = [n for n, pats in terms if any(p.search(t) for p in pats) and not any(p.search(corpus) for p in pats)]
        if nums:
            findings.append(f"{s['id']}: numbers not in master facts {sorted(nums)}")
        if missing:
            findings.append(f"{s['id']}: terms not in master facts {sorted(set(missing))}")
        for m in _YEARS.finditer(t):
            if not _YEARS_OK_AFTER.match(t[m.end():]):
                findings.append(f"{s['id']}: ties '4+ years' to specific work ('{t[m.start():m.end() + 45]}…')")
        if _SELF_TITLE.search(t):
            findings.append(f"{s['id']}: self-describes as '{_SELF_TITLE.search(t).group(0)}'")
    return findings


class Verifier:
    def __init__(self, master: dict, texts: dict[str, str], tech: dict[str, list[str]], cfg: dict,
                 role_family: str, job_title: str = ""):
        self.master, self.texts, self.tech = master, texts, tech
        self.t = cfg["tailoring"]
        self.family, self.title = role_family, job_title
        self.allowed_labels = self.t["headline_role_labels"].get(role_family, [])
        self.forbidden = [w.lower() for w in self.t["headline_forbidden_words"]]
        self.max_head = self.t.get("max_headline_skills", 4)
        self.optional = set(self.t.get("optional_experience", []))
        self.exp_ids = [e["id"] for e in master["experience"]]
        self.proj_ids = [p["id"] for p in master["projects"]]
        self.skill_ids = {s["id"] for s in master["skills"]}
        self.groups = [set(g) for g in master.get("same_work_groups", [])]
        univ = next(ed for ed in master["education"] if ed.get("coursework"))
        cw = univ["coursework"]
        self.course_ids = {c["id"] for c in cw.get("core", []) + (cw.get("electives") or [])}
        self.terms = _compile_terms(master, tech)

    def required_bullets(self) -> list[str]:
        return [i for r in self.t.get("required_bullets", []) if _applies(r, self.family, self.title) for i in r["ids"]]

    def required_projects(self) -> list[str]:
        return [r["id"] for r in self.t.get("required_projects", []) if _applies(r, self.family, self.title)]

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
        if not 3 <= len(tr.headline.skill_ids) <= self.max_head:
            res.issues.append(Issue("headline", f"needs 3-{self.max_head} skills, got {len(tr.headline.skill_ids)}"))
        if label and any(re.search(rf"\b{w}\b", label.lower()) for w in self.forbidden):
            res.issues.append(Issue("headline", "contains a forbidden seniority word"))
        # experience: every non-optional entry, in master order
        got = [s.id for s in tr.experience]
        expected = [e for e in self.exp_ids if e in got or e not in self.optional]
        if got != expected:
            res.issues.append(Issue("experience", f"must list {[e for e in self.exp_ids if e not in self.optional]} "
                                    f"(optional: {sorted(self.optional)}) in this order; got {got}"))
        for sec in tr.experience + tr.projects:
            if sec.id not in self.proj_ids and sec.id not in self.exp_ids:
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
        self._check_rules(tr, res)
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

    def _check_rules(self, tr: TailoredResume, res: VerificationResult) -> None:
        """Tailoring rules from config.yaml + same-work groups from master_resume.yaml."""
        bullets = [(f"{s.id} bullet {n}", set(b.source_ids))
                   for s in tr.experience + tr.projects for n, b in enumerate(s.bullets, 1)]
        cited = set().union(*(ids for _, ids in bullets)) if bullets else set()
        # No two bullets about the same work.
        for i, (w1, ids1) in enumerate(bullets):
            for w2, ids2 in bullets[i + 1:]:
                if ids1 & ids2:
                    res.issues.append(Issue("rules", f"{w1} and {w2} cite the same source {sorted(ids1 & ids2)}"))
                    continue
                for g in self.groups:
                    if ids1 & g and ids2 & g:
                        res.issues.append(Issue("rules", f"{w1} and {w2} describe the same work (group {sorted(g)}); keep one"))
                        break
        for need in self.required_bullets():
            if need not in cited:
                res.issues.append(Issue("rules", f"must include a bullet citing {need} for this role"))
        projects = {p.id for p in tr.projects}
        for need in self.required_projects():
            if need not in projects:
                res.issues.append(Issue("rules", f"must include project {need} for this role"))
        for p in self.t.get("preferences", []):
            if p["over"] in cited and p["prefer"] not in cited:
                res.issues.append(Issue("rules", f"uses {p['over']} without {p['prefer']} ({p.get('why', '')})"))

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
                  "'architected'). Return one verdict per item, using the item number as index.\n\n"
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
