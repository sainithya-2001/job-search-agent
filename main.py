"""CLI entry point. Phase 1: `python main.py search`. Later phases add score | tailor | apply | run."""
import argparse
import asyncio
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.filters import JobFilter
from src.sources import fetch_all
from src.store import Store

ROOT = Path(__file__).parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_search(args) -> None:
    cfg = load_config()
    print(f"Fetching {len(cfg['companies'])} boards...")
    fetched = asyncio.run(fetch_all(cfg["companies"]))
    for board, err in sorted(fetched.errors.items()):
        print(f"  ! {board}: {err}")

    jf = JobFilter(cfg["filters"])
    kept, hints, reasons = [], {}, Counter()
    for job in fetched.jobs:
        r = jf.check(job)
        # Bucket phrase hits by their pattern family so the summary stays short.
        reasons[r.reason if not r.reason.startswith("phrase") else "phrase: sponsorship/clearance/contract"] += 1
        if r.keep:
            kept.append(job)
            hints[job.id] = r.role_hint

    store = Store(ROOT / "data" / "jobs.db")
    new = store.save(kept, hints)

    print(f"\nFetched {len(fetched.jobs)} jobs from {len(fetched.per_board)} boards "
          f"({len(fetched.errors)} errors)")
    print("Filter results:")
    for reason, n in reasons.most_common():
        print(f"  {n:6d}  {reason}")
    print(f"Passed filters: {len(kept)}  |  New (after dedupe): {len(new)}  |  Total in DB: {store.count()}")

    # "Top" before scoring = highest-priority role family first, then most recent.
    priority = {fam["name"]: i for i, fam in enumerate(cfg["filters"]["role_families"])}
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    new.sort(key=lambda j: (priority.get(hints[j.id], 99), -(j.posted_at or epoch).timestamp()))

    print(f"\nTop {min(args.top, len(new))} new jobs:")
    for i, j in enumerate(new[: args.top], 1):
        posted = j.posted_at.date().isoformat() if j.posted_at else "?"
        remote = " [remote]" if j.remote else ""
        print(f"{i:2d}. {j.company} — {j.title}")
        print(f"    {hints[j.id]} | {j.location[:70]}{remote} | posted {posted}")
        print(f"    {j.url}")
    store.close()


def cmd_score(args) -> None:
    from src.llm import get_llm
    from src.resume import load_master, profile_text
    from src.scorer import SYSTEM_TEMPLATE, estimate, location_rank, score_jobs, select_jobs
    from src.sponsorship import SponsorshipDB

    cfg = load_config()
    limit = args.limit or cfg["scoring"]["default_limit"]
    cap = cfg["scoring"]["max_run_cost_usd"]
    store = Store(ROOT / "data" / "jobs.db")
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

    tiers = cfg["preferences"]["location_priority"]
    rows = sorted(res.scored, key=lambda r: (-r["fit_score"], location_rank(r["location"], tiers)))
    print(f"\nActual cost: ${res.spent:.4f} over {res.calls} API calls "
          f"({sum(r['escalated'] for r in res.scored)} escalated to {llm.models['strong']})")
    if res.stopped_early:
        print(f"Stopped early: {res.stopped_early}")
    print(f"\n{'#':>2}  {'Company':14s} {'Title':46s} {'Location':24s} {'Fit':>3}  {'Sponsor':7s}  Reason")
    for i, r in enumerate(rows, 1):
        print(f"{i:2d}  {r['company'][:14]:14s} {r['title'][:46]:46s} {r['location'][:24]:24s} "
              f"{r['fit_score']:3d}  {r['sponsorship_signal']:7s}  {r['one_line_reason']}")
    store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="job-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="Fetch boards, filter, save new jobs")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("score", help="LLM-score the most relevant unscored jobs")
    p.add_argument("--limit", type=int, default=None, help="jobs to score (default: scoring.default_limit)")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
