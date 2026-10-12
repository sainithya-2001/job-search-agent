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
- 0-34: wrong kind of role, or clearly ineligible (requires US citizenship, a security clearance, or US-person /
  permanent-resident status).
Statements about
sponsorship ("we do not sponsor", "must be authorized to work without sponsorship") do not affect fit_score.
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
- program_product_manager: Program Manager, Technical Program Manager, Product Manager, Analytics Manager,
  AI / Data / Solutions Manager
- other: none of the above (e.g. general software engineering, research, sales)

sponsorship_signal (report what the posting says; it never affects fit_score):
- "yes" only if the posting explicitly says it sponsors visas / H-1B or offers immigration support.
- "no" if it says it will not sponsor, or requires work authorization without sponsorship.
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


TIER_ORDER = {"startup": 0, "growth": 1, "big_tech": 2}


def location_rank(location: str, tiers: list[dict], remote: bool = False) -> int:
    """Rank of the first location_priority tier that matches (lower = preferred; equal ranks tie)."""
    text = f"{location or ''}{' remote' if remote else ''}"
    for i, tier in enumerate(tiers):
        if any(re.search(p, text, re.I) for p in tier["patterns"]):
            return tier.get("rank", i)
    return max((t.get("rank", i) for i, t in enumerate(tiers)), default=0) + 1


_FILTERS: dict[int, object] = {}


def job_location_rank(row, cfg: dict) -> int:
    """location_rank, except remote roles restricted to non-preferred states (filters.remote_preferred_states) are
    pushed back to preferences.restricted_remote_rank, even when the location string reads like US-remote."""
    prefs = cfg["preferences"]
    remote = bool(row["remote"] if "remote" in row.keys() else False)
    rank = location_rank(row["location"], prefs["location_priority"], remote)
    if "filters" in cfg:
        from src.filters import JobFilter
        jf = _FILTERS.setdefault(id(cfg["filters"]), JobFilter(cfg["filters"]))
        desc = row["description"] if "description" in row.keys() else ""
        if jf.remote_restricted_elsewhere(row["location"], remote, desc or ""):
            rank = max(rank, prefs.get("restricted_remote_rank", 3))
    return rank


def company_tiers(cfg: dict) -> dict[str, str]:
    return {c["name"]: c.get("tier", "unknown") for c in cfg["companies"]}


def tier_rank(company: str, tiers_by_company: dict[str, str]) -> int:
    return TIER_ORDER.get(tiers_by_company.get(company, "unknown"), len(TIER_ORDER))


def _posted_ts(r) -> float:
    return datetime.fromisoformat(r["posted_at"]).timestamp() if r["posted_at"] else 0.0


def _cap_per_company(ordered, limit: int, cap: int, per_company: dict | None = None) -> list:
    per_company = {} if per_company is None else per_company
    picked = []
    for r in ordered:
        if len(picked) == limit:
            break
        if per_company.get(r["company"], 0) < cap:
            picked.append(r)
            per_company[r["company"]] = per_company.get(r["company"], 0) + 1
    return picked


def select_jobs(rows, cfg: dict, limit: int, max_per_company: int | None = None) -> list:
    """Most relevant first: role-family priority, company tier, location priority, newest; capped per company
    (scoring.max_per_company unless max_per_company is given)."""
    families = {f["name"]: i for i, f in enumerate(cfg["filters"]["role_families"])}
    tiers = company_tiers(cfg)
    ordered = sorted(rows, key=lambda r: (families.get(r["role_hint"], 99), tier_rank(r["company"], tiers),
                                          job_location_rank(r, cfg), -_posted_ts(r)))
    cap = cfg["scoring"].get("max_per_company", 3) if max_per_company is None else max_per_company
    return _cap_per_company(ordered, limit, cap)


def select_mix(rows, cfg: dict) -> tuple[list, list[str]]:
    """Pick jobs per scoring.mix bucket (e.g. 5 Solutions/FDE, 5 Consultant/Eval, ...), only from the allowed
    company tiers. Within a bucket: tier, location priority, newest. Returns (jobs, shortfall notes).
    A bucket with too few jobs is reported, not topped up from other buckets."""
    mix = cfg["scoring"]["mix"]
    tiers = company_tiers(cfg)
    allowed = set(mix["tiers"])
    cap, per_company = cfg["scoring"].get("max_per_company", 3), {}
    picked, notes = [], []
    for bucket in mix["buckets"]:
        pool = [r for r in rows if r["role_hint"] in bucket["families"] and tiers.get(r["company"]) in allowed]
        pool.sort(key=lambda r: (tier_rank(r["company"], tiers), job_location_rank(r, cfg),
                                 -_posted_ts(r)))
        got = _cap_per_company(pool, bucket["n"], cap, per_company)
        if len(got) < bucket["n"]:
            notes.append(f"{bucket['name']}: only {len(got)} of {bucket['n']} available")
        picked += [dict(r) | {"bucket": bucket["name"]} for r in got]
    return picked, notes


SPONSOR_ORDER = {"yes": 0, "unclear": 1, "no": 2}


def _get(row, key):
    return row[key] if key in row.keys() else None


def sponsor_rank(row) -> int:
    """yes > unclear > no. Missing/unknown counts as unclear. Used for ordering only, never for fit or filtering."""
    return SPONSOR_ORDER.get(_get(row, "sponsorship_signal") or "unclear", 1)


def is_fresh(row, cfg: dict, now: datetime | None = None) -> bool:
    """Posted within ranking.fresh_days (default 3). No reliable date = not fresh (treated as 4-14 days old)."""
    posted = _get(row, "posted_at")
    if not posted:
        return False
    posted = datetime.fromisoformat(posted) if isinstance(posted, str) else posted
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)
    age = (now or datetime.now(timezone.utc)) - posted
    return age.total_seconds() <= cfg.get("ranking", {}).get("fresh_days", 3) * 86400


def tailor_threshold(row, cfg: dict, now: datetime | None = None) -> int:
    """Minimum fit_score to tailor: agent.tailor_min_score (65), or agent.tailor_min_score_fresh (58) for a job
    posted within ranking.fresh_days (0-3 days)."""
    a = cfg["agent"]
    if is_fresh(row, cfg, now):
        return a.get("tailor_min_score_fresh", a["tailor_min_score"])
    return a["tailor_min_score"]


def result_sort_key(row: dict, cfg: dict, now: datetime | None = None):
    """Base order: fit_score, then freshness (posted within fresh_days first), then sponsorship signal
    (yes > unclear > no), then company tier (startup > growth > big_tech), then location priority."""
    return (-row["fit_score"], 0 if is_fresh(row, cfg, now) else 1, sponsor_rank(row),
            tier_rank(row["company"], company_tiers(cfg)),
            job_location_rank(row, cfg))


def _preferences(cfg: dict, names: list[str], now: datetime | None) -> list[tuple]:
    r = cfg.get("ranking", {})
    table = {"fresh": (lambda row: is_fresh(row, cfg, now), r.get("freshness_preference_points", 5)),
             "sponsorship": (lambda row: sponsor_rank(row) == 0, r.get("sponsorship_preference_points", 3))}
    return [table[n] for n in names]


def preference_order(rows: list, cfg: dict, prefs: list[str], now: datetime | None = None) -> list:
    """The base order (result_sort_key), except a job with a preferred trait goes ahead of a job without it that
    scores at most that preference's margin higher. Preferences are in priority order: a lower-priority one
    (sponsorship) never moves a job ahead of one that wins on a higher-priority one (freshness).

    Greedy, one pick at a time: start from the top remaining job and move down to any later job that beats it;
    repeat from there. Picks only move down the list, so this always terminates. (A plain sort key can't express
    "within N points": that rule isn't transitive.)"""
    checks = _preferences(cfg, prefs, now)

    def beats(r, pick) -> bool:
        for i, (has, margin) in enumerate(checks):
            within = pick["fit_score"] - r["fit_score"] <= margin
            not_worse_higher = all(h(r) or not h(pick) for h, _ in checks[:i])
            if has(r) and not has(pick) and within and not_worse_higher:
                return True
        return False

    remaining = sorted(rows, key=lambda r: result_sort_key(r, cfg, now))
    ordered = []
    while remaining:
        i = 0
        moved = True
        while moved:
            moved = False
            for j in range(i + 1, len(remaining)):
                if beats(remaining[j], remaining[i]):
                    i, moved = j, True
                    break
        ordered.append(remaining.pop(i))
    return ordered


def rank_results(rows: list, cfg: dict, now: datetime | None = None) -> list:
    """Ranking shown to Nithya and used for the review queue: base order + freshness preference."""
    return preference_order(rows, cfg, ["fresh"], now)


def tailor_order(rows: list, cfg: dict, now: datetime | None = None) -> list:
    """Order in which jobs are tailored: base order + freshness preference, then sponsorship preference."""
    return preference_order(rows, cfg, ["fresh", "sponsorship"], now)


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
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: str | None = None


def score_jobs(jobs, system: str, llm: LLMProvider, cfg: dict, store, sponsor_db: SponsorshipDB,
               est: Estimate, cap: float | None = None, quiet: bool = False) -> RunResult:
    """`cap` overrides scoring.max_run_cost_usd (the agent passes whatever is left of its own run budget)."""
    pricing = cfg["llm"].get("pricing", {})
    threshold = cfg["scoring"]["escalate_threshold"]
    cap = cfg["scoring"]["max_run_cost_usd"] if cap is None else cap
    companies = {c["name"]: c for c in cfg["companies"]}
    res = RunResult()
    # Budget guard: before each call, project its cost as the larger of 2x the estimate and the most
    # expensive call of that tier seen so far, so an underestimate can't push spend past the cap.
    fast_est = est.fast_cost / max(est.calls, 1)
    worst_seen = {"fast": 0.0, "strong": 0.0}

    def call_cost(r) -> float:
        return cost_usd(r.model, r.input_tokens, r.output_tokens, pricing) or 0.0

    for i, job in enumerate(jobs):
        if res.spent + max(2 * fast_est, worst_seen["fast"]) > cap:
            res.stopped_early = f"budget cap ${cap:.2f} reached before job {i + 1}"
            break
        prompt = job_prompt(job)
        fast = llm.generate_json(prompt, ScoredJob, tier="fast", system=system)
        worst_seen["fast"] = max(worst_seen["fast"], call_cost(fast))
        calls, final = [fast], fast
        if fast.data.fit_score >= threshold:
            projected = max(2 * est.strong_cost_each[i], worst_seen["strong"])
            if res.spent + call_cost(fast) + projected > cap:
                res.stopped_early = f"budget cap ${cap:.2f}: kept fast-tier score for job {i + 1}, stopped after it"
            else:
                final = llm.generate_json(prompt, ScoredJob, tier="strong", system=system)
                worst_seen["strong"] = max(worst_seen["strong"], call_cost(final))
                calls.append(final)
        job_cost = sum(call_cost(c) for c in calls)
        res.spent += job_cost
        res.calls += len(calls)
        res.input_tokens += sum(c.input_tokens for c in calls)
        res.output_tokens += sum(c.output_tokens for c in calls)

        s: ScoredJob = final.data
        company = companies.get(job["company"], {})
        uscis = sponsor_db.lookup(job["company"], company.get("uscis_names"), company.get("e_verify"))
        row = {
            "job_id": job["id"], "fit_score": s.fit_score, "fast_score": fast.data.fit_score,
            "escalated": int(final is not fast), "model": final.model, "role_family": s.role_family,
            "matched_skills": json.dumps(s.matched_skills), "gaps": json.dumps(s.gaps),
            "sponsorship_signal": s.sponsorship_signal.signal, "sponsorship_quote": s.sponsorship_signal.quote,
            "uscis_status": uscis["status"], "uscis_approvals": uscis.get("approvals"), "e_verify": uscis["e_verify"],
            "is_staffing_firm": int(s.is_staffing_firm), "seniority_ok": int(s.seniority_ok),
            "one_line_reason": s.one_line_reason,
            "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
            "cost_usd": job_cost, "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        store.save_score(row)
        res.scored.append({**row, "company": job["company"], "title": job["title"], "location": job["location"],
                           "remote": job["remote"], "posted_at": job["posted_at"], "description": job["description"], "tier": company.get("tier", "unknown"),
                           "bucket": job["bucket"] if "bucket" in job.keys() else None})
        tag = f"{fast.data.fit_score}->{s.fit_score} (strong)" if final is not fast else f"{s.fit_score}"
        if not quiet:
            print(f"  [{i + 1:2d}/{len(jobs)}] {job['company'][:14]:14s} {job['title'][:52]:52s} {tag:>16s}  ${job_cost:.4f}")
        if res.stopped_early:
            break
    return res
