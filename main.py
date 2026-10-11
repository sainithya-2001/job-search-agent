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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="job-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="Fetch boards, filter, save new jobs")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
