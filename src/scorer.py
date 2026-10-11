"""Phase 2: LLM fit scoring.

Flow per job: fast tier scores it -> if fit_score >= escalate_threshold, the strong tier re-scores it and its answer
wins -> USCIS cross-check -> saved to the `scores` table.

Cost control: the run is estimated before any call, refused if the expected cost exceeds max_run_cost_usd,
and stopped before any call that would push actual spend past it. Unscored jobs simply wait for the next run.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.llm import LLMProvider, cost_usd, estimate_tokens
from src.models import ScoredJob
from src.sponsorship import SponsorshipDB

# Structured output adds the JSON schema (and some instructions) to every prompt; this covers it in estimates.
SCHEMA_OVERHEAD_TOKENS = 600

SYSTEM_TEMPLATE = """You evaluate how well ONE job posting fits ONE candidate. Be strict and evidence-based.

Judge only against what the candidate profile below actually shows. Never assume a skill, tool, or experience
that is not in the profile. Score the candidate's fit for the job; do not reward or penalize the employer.

fit_score rubric (0-100):
- 85-100: the posting's core requirements are clearly shown in the profile and seniority fits.
- 70-84: most core requirements shown; one or two notable gaps.
- 55-69: real overlap, but several core gaps or a stretch on seniority.
- 35-54: weak overlap; mostly a different skill set.
- 0-34: wrong kind of role, or clearly ineligible.
If seniority_ok is false, fit_score must be at most 50.

seniority_ok: the candidate has 4+ years of experience (see profile). True if the posting's required minimum is
5 years or less (e.g. "3+ years", "5+ years"). False if it requires more than 5 years, or the role is
Staff/Principal/Director level. If no years are stated, judge from the level and responsibilities.

role_family — pick exactly one:
- ai_engineer: builds LLM / generative-AI applications (AI Engineer, Applied AI Engineer, GenAI Engineer)
- ai_solutions: customer-facing technical role selling or designing AI solutions (Solutions Engineer/Architect)
- forward_deployed: embedded with customers to build and deploy (Forward Deployed Engineer, Deployment Strategist)
- ai_consultant: AI strategy / consulting / transformation
- ai_eval: AI evaluation, model quality, evals
- analyst_backup: Data Analyst, Business Analyst, Analytics Engineer
- other: none of the above (e.g. general software engineering, research, sales)

sponsorship_signal (from the posting text only):
- "yes" only if the posting explicitly says it sponsors visas / H-1B or offers immigration support.
- "no" if it says it will not sponsor, requires US citizenship, a security clearance, or work authorization
  without sponsorship.
- otherwise "unclear".
quote: copy the exact sentence from the posting that supports "yes" or "no"; null when "unclear".

is_staffing_firm: true only if the poster is a staffing agency or recruiter placing people with a client.
matched_skills: short phrases, only skills that appear in the candidate profile AND the posting asks for (max 8).
gaps: short phrases for requirements in the posting that the profile does not show (max 8).
one_line_reason: one sentence, at most 25 words.

CANDIDATE PROFILE
{profile}
"""


def job_prompt(job) -> str:
    return (f"Company: {job['company']}\nTitle: {job['title']}\nLocation: {job['location']}"
            f"{' (remote)' if job['remote'] else ''}\n\nPOSTING\n{job['description']}")


def location_rank(location: str, tiers: list[dict]) -> int:
    """Index of the first location_priority tier that matches (0 = most preferred)."""
    for i, tier in enumerate(tiers):
        if any(re.search(p, location or "", re.I) for p in tier["patterns"]):
            return i
    return len(tiers)


def select_jobs(rows, cfg: dict, limit: int) -> list:
    """Most relevant first: role-family priority, then location priority, then newest; capped per company."""
    families = {f["name"]: i for i, f in enumerate(cfg["filters"]["role_families"])}
    tiers = cfg["preferences"]["location_priority"]
    cap = cfg["scoring"].get("max_per_company", 3)
    ordered = sorted(rows, key=lambda r: (families.get(r["role_hint"], 99),
                                          location_rank(r["location"], tiers),
                                          -(datetime.fromisoformat(r["posted_at"]).timestamp() if r["posted_at"] else 0)))
    picked, per_company = [], {}
    for r in ordered:
        if per_company.get(r["company"], 0) < cap:
            picked.append(r)
            per_company[r["company"]] = per_company.get(r["company"], 0) + 1
        if len(picked) == limit:
            break
    return picked


@dataclass
class Estimate:
    calls: int
    fast_cost: float
    strong_cost_each: list[float]
    expected: float      # assumes half the jobs escalate
    worst: float         # assumes every job escalates


def estimate(jobs, system: str, llm: LLMProvider, cfg: dict) -> Estimate:
    pricing, out = cfg["llm"].get("pricing", {}), cfg["scoring"]["est_output_tokens"]
    fast_m, strong_m = llm.models["fast"], llm.models["strong"]
    fast, strong = 0.0, []
    for j in jobs:
        tokens_in = estimate_tokens(system + job_prompt(j)) + SCHEMA_OVERHEAD_TOKENS
        fast += cost_usd(fast_m, tokens_in, out["fast"], pricing) or 0.0
        strong.append(cost_usd(strong_m, tokens_in, out["strong"], pricing) or 0.0)
    return Estimate(len(jobs), fast, strong, fast + sum(strong) / 2, fast + sum(strong))


@dataclass
class RunResult:
    scored: list[dict] = field(default_factory=list)
    spent: float = 0.0
    calls: int = 0
    stopped_early: str | None = None


def score_jobs(jobs, system: str, llm: LLMProvider, cfg: dict, store, sponsor_db: SponsorshipDB,
               est: Estimate) -> RunResult:
    pricing = cfg["llm"].get("pricing", {})
    threshold, cap = cfg["scoring"]["escalate_threshold"], cfg["scoring"]["max_run_cost_usd"]
    aliases = {c["name"]: c.get("uscis_names", []) for c in cfg["companies"]}
    res = RunResult()
    fast_each = est.fast_cost / max(est.calls, 1)

    for i, job in enumerate(jobs):
        if res.spent + fast_each > cap:
            res.stopped_early = f"budget cap ${cap:.2f} reached before job {i + 1}"
            break
        prompt = job_prompt(job)
        fast = llm.generate_json(prompt, ScoredJob, tier="fast", system=system)
        calls = [fast]
        final = fast
        if fast.data.fit_score >= threshold:
            if res.spent + (cost_usd(fast.model, fast.input_tokens, fast.output_tokens, pricing) or 0) \
                    + est.strong_cost_each[i] > cap:
                res.stopped_early = f"budget cap ${cap:.2f}: kept fast-tier score for job {i + 1}, stopped after it"
            else:
                final = llm.generate_json(prompt, ScoredJob, tier="strong", system=system)
                calls.append(final)
        job_cost = sum(cost_usd(c.model, c.input_tokens, c.output_tokens, pricing) or 0.0 for c in calls)
        res.spent += job_cost
        res.calls += len(calls)

        s: ScoredJob = final.data
        uscis = sponsor_db.lookup(job["company"], aliases.get(job["company"]))
        row = {
            "job_id": job["id"], "fit_score": s.fit_score, "fast_score": fast.data.fit_score,
            "escalated": int(final is not fast), "model": final.model, "role_family": s.role_family,
            "matched_skills": json.dumps(s.matched_skills), "gaps": json.dumps(s.gaps),
            "sponsorship_signal": s.sponsorship_signal.signal, "sponsorship_quote": s.sponsorship_signal.quote,
            "uscis_status": uscis["status"], "uscis_approvals": uscis.get("approvals"),
            "is_staffing_firm": int(s.is_staffing_firm), "seniority_ok": int(s.seniority_ok),
            "one_line_reason": s.one_line_reason,
            "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
            "cost_usd": job_cost, "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        store.save_score(row)
        res.scored.append({**row, "company": job["company"], "title": job["title"], "location": job["location"]})
        tag = f"{fast.data.fit_score}->{s.fit_score} (strong)" if final is not fast else f"{s.fit_score}"
        print(f"  [{i + 1:2d}/{len(jobs)}] {job['company'][:14]:14s} {job['title'][:52]:52s} {tag:>16s}  ${job_cost:.4f}")
        if res.stopped_early:
            break
    return res
