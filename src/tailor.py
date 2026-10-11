"""Phase 3: tailor the master resume to one job, with the verify.py guardrail in the loop.

Loop: draft (strong tier) -> verify (string checks + LLM judge) -> if problems, redraft with the problem list,
up to `tailoring.max_attempts`. If problems remain, unsupported bullets are REMOVED (never kept) and the removal
is recorded in the report. Structural problems that removal can't fix (bad ids, wrong order) fail the job.
"""
import json
from dataclasses import dataclass, field

from src.llm import LLMProvider, cost_usd
from src.models import TailoredResume
from src.verify import Issue, VerificationResult, Verifier

SYSTEM = """You tailor a candidate's resume to one job. You may only SELECT, REORDER, and REWORD facts from the
SOURCE list. Never add a tool, number, metric, outcome, employer, title, date, customer, scale, or responsibility
that the cited sources do not state. Rewording must keep meaning: no intensifiers that raise the claim
("production", "enterprise-grade", "led", "owned", "architected", "spearheaded") unless the cited source says so.

Output rules:
- experience: include EVERY experience entry, in the SOURCE order. 1-4 bullets each (1-2 for short or older roles).
- projects: the 1-3 most relevant projects, 1-3 bullets each.
- Every bullet cites the ids of the SOURCE bullets it is built from, and only bullets from its OWN entry
  (e.g. a bullet under exp_emp_b cites exp_emp_b_bNN ids only). You may merge bullets from the same entry.
- Keep every number exactly as written in its source (40+, 30% → 6%, 200K+, ~3%). Do not round or combine.
- Total bullets across experience and projects: 16 or fewer (one page).
- summary: 1-2 sentences, at most 45 words, aimed at this job. Cite the ids it relies on (bullet ids, skill ids,
  years_experience, edu_univ). "4+ years" is TOTAL experience, mostly business analysis: write
  "4+ years of experience" on its own or "4+ years of experience across business analysis, data, and AI roles" —
  never attach the years to a skill or kind of work ("4+ years building AI apps" is false). Do not call the
  candidate an engineer, architect, or scientist; "Analyst" or "data and AI professional" are fine.
- headline: role_label must be exactly one of the ALLOWED ROLE LABELS below, or null. Pick 3-5 skill ids most
  relevant to the job. Never use Senior, Lead, Principal, or Staff.
- skill_groups: 3-4 groups of 4-8 skill ids each, only skills relevant to this job. Labels are short category names.
- coursework_ids: the 4-6 courses most relevant to this job.
- Emphasize what the job asks for (see MATCHED and GAPS from the scorer), but never claim a gap you can't source.
"""


@dataclass
class TailorResult:
    resume: TailoredResume | None
    verification: VerificationResult
    attempts: int
    removed: list[str] = field(default_factory=list)
    attempt_issues: list[list[str]] = field(default_factory=list)
    cost: float = 0.0
    calls: int = 0
    status: str = "FAIL"


def build_prompt(job, score_row, source: str, allowed_labels: list[str], feedback: list[str] | None) -> str:
    matched = json.loads(score_row["matched_skills"] or "[]") if score_row else []
    gaps = json.loads(score_row["gaps"] or "[]") if score_row else []
    parts = [f"JOB: {job['title']} at {job['company']} ({job['location']})",
             f"ALLOWED ROLE LABELS for this job: {allowed_labels or 'none — use null'}",
             f"MATCHED (from scorer): {', '.join(matched) or 'n/a'}",
             f"GAPS (from scorer; do NOT claim these): {', '.join(gaps) or 'n/a'}",
             "", "POSTING", job["description"], "", "SOURCE (master resume facts, cite by id)", source]
    if feedback:
        parts += ["", "YOUR PREVIOUS DRAFT WAS REJECTED. Fix every problem below; if a claim can't be sourced, drop it:"]
        parts += [f"- {f}" for f in feedback]
    return "\n".join(parts)


def _remove_failing(tr: TailoredResume, res: VerificationResult) -> list[str]:
    """Drop bullets that failed any check; return what was removed."""
    failing = {i.where for i in res.issues}
    removed = []
    for sec in tr.experience + tr.projects:
        keep = []
        for n, b in enumerate(sec.bullets, 1):
            if f"{sec.id} bullet {n}" in failing:
                removed.append(f"{sec.id}: {b.text}")
            else:
                keep.append(b)
        sec.bullets = keep
    if "summary" in failing:
        removed.append(f"summary: {tr.summary.text}")
        tr.summary.text, tr.summary.source_ids = "", []
    tr.projects = [p for p in tr.projects if p.bullets]
    return removed


def tailor_job(job, score_row, master: dict, source: str, verifier: Verifier, llm: LLMProvider, cfg: dict,
               budget_left: float) -> TailorResult:
    tcfg, pricing = cfg["tailoring"], cfg["llm"].get("pricing", {})
    tier = tcfg.get("tier", "strong")
    out = TailorResult(None, VerificationResult(), 0)
    feedback = None

    def spend(resp):
        out.cost += cost_usd(resp.model, resp.input_tokens, resp.output_tokens, pricing) or 0.0
        out.calls += 1

    for attempt in range(1, tcfg.get("max_attempts", 3) + 1):
        if out.cost >= budget_left:
            out.attempt_issues.append(["stopped: tailoring budget reached"])
            break
        out.attempts = attempt
        draft = llm.generate_json(build_prompt(job, score_row, source, verifier.allowed_labels, feedback),
                                  TailoredResume, tier=tier, system=SYSTEM)
        spend(draft)
        tr = draft.data
        res = verifier.check_structure(tr)
        if res.passed:                           # only spend an LLM-judge call on drafts that pass the string checks
            verifier.check_with_llm(res, llm, tier)
            spend(res.llm_response)
        out.resume, out.verification = tr, res
        out.attempt_issues.append([str(i) for i in res.issues])
        if res.passed:
            out.status = "PASS"
            return out
        feedback = [str(i) for i in res.issues]

    # Out of attempts: remove what still fails. Whatever survives must still pass the string checks AND the LLM
    # judge (which may not have run on a draft that failed the string checks); anything the judge flags is removed too.
    if out.resume is not None:
        out.removed = _remove_failing(out.resume, out.verification)
        recheck = _recheck(verifier, out.resume)
        if recheck.passed and out.cost < budget_left:
            verifier.check_with_llm(recheck, llm, tier)
            spend(recheck.llm_response)
            if not recheck.passed:
                out.removed += _remove_failing(out.resume, recheck)
                judged = recheck
                recheck = _recheck(verifier, out.resume)
                _carry_verdicts(judged, recheck)
        out.verification = recheck
        # PASS only if every remaining item passed the string checks AND was judged supported by the LLM.
        judged_ok = all(c.llm_supported is True for c in recheck.checked)
        out.status = "PASS (bullets removed)" if recheck.passed and judged_ok else "FAIL"
    return out


def _recheck(verifier: Verifier, tr: TailoredResume) -> VerificationResult:
    res = verifier.check_structure(tr)
    # A summary blanked on purpose isn't a new problem; drop its "cites no source ids" issue and its row.
    if not tr.summary.text:
        res.issues = [i for i in res.issues if i.where != "summary"]
        res.checked = [c for c in res.checked if c.where != "summary"]
    return res


def _carry_verdicts(src: VerificationResult, dst: VerificationResult) -> None:
    by_text = {c.text: c for c in src.checked}
    for c in dst.checked:
        if c.text in by_text:
            c.llm_supported, c.llm_problem = by_text[c.text].llm_supported, by_text[c.text].llm_problem
