"""CLI entry point. Phase 1: `python main.py search`. Later phases add score | tailor | apply | run."""
import argparse
import asyncio
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import httpx

from src.filters import JobFilter
from src.models import Job
from src.sources import fetch_all
from src.store import Store

ROOT = Path(__file__).parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_EXPORT_WORDS = ("export", "itar", "ear", "person", "permanent", "green")


def _reason_bucket(reason: str) -> str:
    """Group phrase hits so the summary stays short."""
    if not reason.startswith("phrase"):
        return reason
    text = reason.lower()
    if any(w in text for w in _EXPORT_WORDS):
        return "phrase: export control / US person / residency"
    return "phrase: sponsorship / citizenship / clearance / contract"


def _job_from_row(r) -> Job:
    return Job(id=r["id"], company=r["company"], title=r["title"], location=r["location"] or "",
               remote=bool(r["remote"]), url=r["url"] or "", description=r["description"] or "",
               source=r["source"], employment_type=r["employment_type"])


def cmd_search(args) -> None:
    from src.llm import load_env
    from src.scorer import company_tiers, location_rank, tier_rank
    from src.sources import jsearch

    cfg = load_config()
    load_env(ROOT / ".env")
    print(f"Fetching {len(cfg['companies'])} boards...")
    fetched = asyncio.run(fetch_all(cfg["companies"]))
    for board, err in sorted(fetched.errors.items()):
        print(f"  ! {board}: {err}")

    js_cfg, js_key = cfg.get("jsearch", {}), jsearch.api_key()
    if js_cfg.get("enabled") and js_key:
        async def _js():
            async with httpx.AsyncClient(timeout=30) as client:
                return await jsearch.fetch(client, js_cfg, js_key)
        js_jobs, js_dropped, js_errors = asyncio.run(_js())
        fetched.jobs.extend(js_jobs)
        print(f"JSearch: {len(js_jobs)} jobs from {len(js_cfg['queries'])} queries "
              f"({js_dropped} dropped as staffing/excluded employers)")
        for e in js_errors:
            print(f"  ! {e}")
    else:
        print("JSearch: skipped (" + ("disabled in config" if not js_cfg.get("enabled") else "no JSEARCH_API_KEY in .env") + ")")

    jf = JobFilter(cfg["filters"])
    kept, hints, reasons = [], {}, Counter()
    for job in fetched.jobs:
        r = jf.check(job)
        reasons[_reason_bucket(r.reason)] += 1
        if r.keep:
            kept.append(job)
            hints[job.id] = r.role_hint

    store = Store(ROOT / "data" / "jobs.db")
    new = store.save(kept, hints)

    # Re-check unscored jobs already in the DB against the CURRENT filters and company list, so rule changes
    # apply retroactively. Rows are re-labelled, never deleted; scored jobs are left alone.
    configured = {c["name"] for c in cfg["companies"]}
    relabel = Counter()
    for row in store.jobs_by_status("new"):
        if row["source"] != "jsearch" and row["company"] not in configured:
            store.set_status(row["id"], "dropped_company"); relabel["dropped_company"] += 1
        elif not jf.check(_job_from_row(row)).keep:
            store.set_status(row["id"], "excluded"); relabel["excluded"] += 1

    print(f"\nFetched {len(fetched.jobs)} jobs from {len(fetched.per_board)} boards "
          f"({len(fetched.errors)} errors)")
    print("Filter results:")
    for reason, n in reasons.most_common():
        print(f"  {n:6d}  {reason}")
    print(f"Passed filters: {len(kept)}  |  New (after dedupe): {len(new)}  |  Total in DB: {store.count()}")
    if relabel:
        print("Existing unscored jobs re-checked: " + ", ".join(f"{n} -> {k}" for k, n in relabel.items()))
    print("Ready to score (status new): " + str(store.count("new")))

    # "Top" before scoring = role family priority, company tier, location priority, newest.
    priority = {fam["name"]: i for i, fam in enumerate(cfg["filters"]["role_families"])}
    tiers, locs = company_tiers(cfg), cfg["preferences"]["location_priority"]
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    new.sort(key=lambda j: (priority.get(hints[j.id], 99), tier_rank(j.company, tiers),
                            location_rank(j.location, locs, j.remote), -(j.posted_at or epoch).timestamp()))

    print(f"\nTop {min(args.top, len(new))} new jobs:")
    for i, j in enumerate(new[: args.top], 1):
        posted = j.posted_at.date().isoformat() if j.posted_at else "?"
        remote = " [remote]" if j.remote else ""
        print(f"{i:2d}. {j.company} ({tiers.get(j.company, 'unknown')}) — {j.title}")
        print(f"    {hints[j.id]} | {j.location[:70]}{remote} | posted {posted}")
        print(f"    {j.url}")
    store.close()


def cmd_score(args) -> None:
    from src.llm import get_llm
    from src.resume import load_master, profile_text
    from src.scorer import SYSTEM_TEMPLATE, estimate, result_sort_key, score_jobs, select_jobs, select_mix
    from src.sponsorship import SponsorshipDB

    cfg = load_config()
    limit = args.limit or cfg["scoring"]["default_limit"]
    cap = cfg["scoring"]["max_run_cost_usd"]
    store = Store(ROOT / "data" / "jobs.db")
    if args.mix:
        jobs, notes = select_mix(store.unscored_jobs(), cfg)
        mix = cfg["scoring"]["mix"]
        print(f"Mix selection from tiers {mix['tiers']}: "
              + ", ".join(f"{b['n']} {b['name']}" for b in mix["buckets"]))
        for n in notes:
            print(f"  ! {n}")
    else:
        jobs = select_jobs(store.unscored_jobs(), cfg, limit)
    if not jobs:
        print("No unscored jobs. Run `python main.py search` first.")
        return

    system = SYSTEM_TEMPLATE.format(profile=profile_text(load_master(ROOT / "data" / "master_resume.yaml")))
    llm = get_llm(cfg)
    est = estimate(jobs, system, llm, cfg)
    print(f"Scoring {len(jobs)} jobs  |  fast: {llm.models['fast']}  strong (>= {cfg['scoring']['escalate_threshold']}): "
          f"{llm.models['strong']}")
    print(f"Estimated cost: ~${est.expected:.3f} expected (half escalate), ${est.worst:.3f} if all escalate  "
          f"|  run cap ${cap:.2f}")
    if est.expected > cap:
        print(f"Refusing to run: expected cost exceeds the cap. Use a smaller --limit or raise scoring.max_run_cost_usd.")
        return

    sponsor_db = SponsorshipDB(ROOT / "data" / "uscis")
    if not sponsor_db.loaded:
        print("USCIS: no files in data/uscis/ — cross-check will report no_data.")
    print()
    res = score_jobs(jobs, system, llm, cfg, store, sponsor_db, est)

    rows = sorted(res.scored, key=lambda r: result_sort_key(r, cfg))
    print(f"\nActual cost: ${res.spent:.4f} over {res.calls} API calls "
          f"({sum(r['escalated'] for r in res.scored)} escalated to {llm.models['strong']})")
    if res.stopped_early:
        print(f"Stopped early: {res.stopped_early}")
    print(f"\n{'#':>2}  {'Company':16s} {'Tier':8s} {'Title':44s} {'Location':24s} {'Rem':3s} {'Fit':>3}  "
          f"{'Sponsor':7s}  Reason")
    for i, r in enumerate(rows, 1):
        print(f"{i:2d}  {r['company'][:16]:16s} {r['tier'][:8]:8s} {r['title'][:44]:44s} {r['location'][:24]:24s} "
              f"{'yes' if r['remote'] else 'no':3s} {r['fit_score']:3d}  {r['sponsorship_signal']:7s}  "
              f"{r['one_line_reason']}")
    store.close()


def cmd_skills_gap(args) -> None:
    from src.resume import load_master, profile_text
    from src.skills_gap import build_report

    store = Store(ROOT / "data" / "jobs.db")
    rows = store.scored_rows()
    if not rows:
        print("No scored jobs yet. Run `python main.py score` first.")
        return
    report, _ = build_report(rows, profile_text(load_master(ROOT / "data" / "master_resume.yaml")), args.top)
    out = ROOT / "output" / "skills_gap.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {out.relative_to(ROOT)}")
    store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="job-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="Fetch boards, filter, save new jobs")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("score", help="LLM-score the most relevant unscored jobs")
    p.add_argument("--limit", type=int, default=None, help="jobs to score (default: scoring.default_limit)")
    p.add_argument("--mix", action="store_true", help="pick jobs per scoring.mix buckets instead of top-N")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("skills-gap", help="Top skills asked for across scored jobs vs. the master resume")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_skills_gap)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
