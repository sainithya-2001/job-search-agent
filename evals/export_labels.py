"""Export 30 already-scored jobs for Nithya to label by hand: evals/label_me.csv.

    python evals/export_labels.py            # estimate only (no API calls)
    python evals/export_labels.py --run      # write the CSV (one fast-tier summary call per job, cached)

Blind by design: the CSV has company, title, link, and a neutral 2-line summary of the POSTING, and nothing the
model concluded (no fit score, role family, reason, matched skills or gaps). Rows are shuffled, not sorted by score.

Selection mixes fit ranges and role families: it targets an even split across four fit bands (<40, 40-54, 55-64,
65+) and, inside each band, takes role families round-robin so no family dominates. Deterministic (seeded).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

EVALS = ROOT / "evals"
OUT = EVALS / "label_me.csv"
CACHE = EVALS / ".summaries.json"          # job_id -> summary lines (so re-exports are free)
BANDS = [("<40", 0, 39), ("40-54", 40, 54), ("55-64", 55, 64), ("65+", 65, 100)]
N, SEED = 30, 20261012
MAX_COST = 0.10

SUMMARY_SYSTEM = """Summarize ONE job posting in exactly two short lines for a job seeker skimming a list.
Line 1: what the role does day to day (team/product, main responsibilities).
Line 2: the key requirements: years of experience, must-have skills or background, and location / work mode.
Use only facts stated in the posting. Do not judge fit, do not mention any candidate, do not rate the job.
Each line at most 160 characters."""


class PostingSummary(BaseModel):
    line1: str = Field(description="What the role does")
    line2: str = Field(description="Key requirements, location / work mode")


def band_of(score: int) -> str:
    return next(name for name, lo, hi in BANDS if lo <= score <= hi)


def select(rows: list[dict], n: int = N, seed: int = SEED) -> list[dict]:
    """Even split across fit bands (spare slots go to bands with more jobs), role families round-robin."""
    rng = random.Random(seed)
    by_band: dict[str, dict[str, list]] = {b: defaultdict(list) for b, _, _ in BANDS}
    for r in rows:
        by_band[band_of(r["fit_score"])][r["role_family"] or "other"].append(r)
    for fams in by_band.values():
        for lst in fams.values():
            rng.shuffle(lst)
    sizes = {b: sum(len(v) for v in fams.values()) for b, fams in by_band.items()}
    quota, left = {b: 0 for b in by_band}, n
    while left and any(quota[b] < sizes[b] for b in by_band):     # one slot per band per round
        for b in sorted(by_band, key=lambda b: quota[b]):
            if left and quota[b] < sizes[b]:
                quota[b] += 1
                left -= 1
    picked = []
    for b, fams in by_band.items():
        order = sorted(fams, key=lambda f: (-len(fams[f]), f))
        take, i = [], 0
        while len(take) < quota[b]:
            fam = order[i % len(order)]
            if fams[fam]:
                take.append(fams[fam].pop())
            i += 1
        picked += take
    rng.shuffle(picked)
    return picked


def candidates(store) -> list[dict]:
    """Scored jobs that are still real postings (not excluded / closed)."""
    return [dict(r) for r in store.conn.execute(
        "SELECT j.*, s.fit_score, s.role_family FROM scores s JOIN jobs j ON j.id = s.job_id "
        "WHERE j.status NOT IN ('excluded', 'closed', 'dropped_company')")]


def main() -> None:
    from main import load_config
    from src.llm import cost_usd, estimate_tokens, get_llm, load_env
    from src.scorer import job_prompt
    from src.store import Store

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="make the summary calls and write the CSV")
    args = ap.parse_args()

    cfg = load_config()
    load_env(ROOT / ".env")
    store = Store(ROOT / "data" / "jobs.db")
    jobs = select(candidates(store))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [j for j in jobs if j["id"] not in cache]

    llm = get_llm(cfg)
    model, pricing = llm.models["fast"], cfg["llm"].get("pricing", {})
    est = sum(cost_usd(model, estimate_tokens(SUMMARY_SYSTEM + job_prompt(j)) + 200, 120, pricing) or 0 for j in todo)
    bands = defaultdict(int)
    fams = defaultdict(int)
    for j in jobs:
        bands[band_of(j["fit_score"])] += 1
        fams[j["role_family"] or "other"] += 1
    print(f"Selected {len(jobs)} jobs. Fit bands: {dict(bands)}")
    print(f"Role families: {dict(fams)}")
    print(f"Summaries needed: {len(todo)} ({len(jobs) - len(todo)} cached) with {model}: "
          f"estimated ${est:.4f} (cap ${MAX_COST:.2f})")
    if not args.run:
        print("Estimate only. Re-run with --run to write evals/label_me.csv.")
        return
    if est > MAX_COST:
        sys.exit("Refusing: estimate exceeds the cap.")

    spent = 0.0
    for j in todo:
        if spent > MAX_COST:
            sys.exit(f"Stopped: spent ${spent:.4f}, cap ${MAX_COST:.2f}. Cached summaries are kept; re-run to resume.")
        r = llm.generate_json(job_prompt(j), PostingSummary, tier="fast", system=SUMMARY_SYSTEM)
        spent += cost_usd(r.model, r.input_tokens, r.output_tokens, pricing) or 0
        cache[j["id"]] = [r.data.line1.strip(), r.data.line2.strip()]
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:      # BOM so Excel opens it as UTF-8
        w = csv.writer(f)
        w.writerow(["n", "job_id", "company", "title", "link", "summary", "label (apply/maybe/skip)", "notes"])
        for i, j in enumerate(jobs, 1):
            w.writerow([i, j["id"], j["company"], j["title"], j["url"], "\n".join(cache[j["id"]]), "", ""])
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(jobs)} rows). Actual cost: ${spent:.4f}")


if __name__ == "__main__":
    main()
