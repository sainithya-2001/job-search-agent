"""Daily digest: output/digest_YYYY-MM-DD.md, written after each `run` (and on demand with `main.py digest`).

Built from the database plus that day's run logs (logs/run_YYYYMMDD_*.jsonl), so it covers every run of the day
and can be regenerated at any time. "Today" is the local date, matching the log file names.

  Tier A  up to 5 jobs tailored today (ranked), with their review-queue status. More than 5 = an overflow line.
  Tier B  the next 10 ranked jobs scoring >= 50 that were not tailored, with the approved summary that fits each.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.scorer import company_tiers, is_fresh, rank_results, tailor_threshold
from src.store import Store
from src.verify import select_summary

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "output"
CLOSED = ("applied", "skipped")


def _local_day(iso: str | None) -> date | None:
    if not iso:
        return None
    t = datetime.fromisoformat(iso)
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone().date()


def run_stats(day: date, log_dir: Path = LOG_DIR) -> list[dict]:
    """One row per agent run that day, from its JSONL log."""
    runs = []
    for path in sorted(log_dir.glob(f"run_{day:%Y%m%d}_*.jsonl")):
        r = {"run": path.stem.removeprefix("run_"), "searched": 0, "passed": 0, "new": 0, "scored": 0,
             "tailored": 0, "jsearch_requests": 0, "cost": 0.0}
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            r["cost"] += e.get("cost_usd") or 0.0
            res = e.get("result") or {}
            if e.get("kind") != "tool_call" or not e.get("ok"):
                continue
            if e["name"] == "search_jobs":
                r["searched"] += res.get("fetched", 0)
                r["passed"] += res.get("passed_filters", 0)
                r["new"] += res.get("new_jobs", 0)
                r["jsearch_requests"] += res.get("jsearch_requests", 0)
            elif e["name"] == "score_jobs":
                r["scored"] += res.get("scored_count", len(res.get("scored", [])))
            elif e["name"] == "tailor_resume" and str(res.get("status", "")).startswith("PASS"):
                r["tailored"] += 1
        runs.append(r)
    return runs


def _freshness(row, cfg: dict, now: datetime) -> str:
    if not row["posted_at"]:
        return "no date"
    p = datetime.fromisoformat(row["posted_at"])
    days = (now - (p if p.tzinfo else p.replace(tzinfo=timezone.utc))).days
    return f"{days}d" + (" (fresh)" if is_fresh(row, cfg, now) else "")


def _cell(s) -> str:
    return str(s or "").replace("|", "/").replace("\n", " ").strip()


def tier_a(store: Store, cfg: dict, day: date) -> tuple[list[dict], int]:
    """Jobs whose current tailored resume was made on `day`, ranked; (top max_a, total)."""
    rows = [dict(r) for r in store.scored_rows()
            if r["status"] == "tailored" and _local_day(r["tailored_at"]) == day]
    ranked = rank_results(rows, cfg)
    return ranked[: cfg["digest"]["tier_a_max"]], len(ranked)


def tier_b(store: Store, cfg: dict, master: dict) -> list[dict]:
    """Next ranked untailored jobs at or above digest.tier_b_min_score, not applied/skipped."""
    closed = {r["job_id"] for r in store.applications() if r["status"] in CLOSED}
    d = cfg["digest"]
    rows = [dict(r) for r in store.scored_rows()
            if r["status"] == "scored" and r["fit_score"] >= d["tier_b_min_score"] and r["id"] not in closed]
    out = []
    for r in rank_results(rows, cfg)[: d["tier_b_max"]]:
        family = r["role_family"] or r["role_hint"]
        try:
            r["summary"] = f"{select_summary(master, family, r['title'])['id']} ({family})"
            r["held_by"] = None
        except ValueError:
            r["summary"], r["held_by"] = f"none approved for {family}", "no approved summary"
        if r["held_by"] is None:
            need = tailor_threshold(r, cfg)
            r["held_by"] = f"below {need}" if r["fit_score"] < need else "eligible: next run"
        out.append(r)
    return out


def build(store: Store, cfg: dict, master: dict, day: date | None = None, now: datetime | None = None,
          log_dir: Path = LOG_DIR) -> str:
    day = day or date.today()
    now = now or datetime.now(timezone.utc)
    tiers = company_tiers(cfg)
    runs = run_stats(day, log_dir)
    a_rows, a_total = tier_a(store, cfg, day)
    b_rows = tier_b(store, cfg, master)
    queue = {r["job_id"]: r for r in store.conn.execute("SELECT job_id, answered, total, queued_at FROM apply_queue")}

    L = [f"# Job digest {day.isoformat()}", ""]
    if runs:
        t = {k: sum(r[k] for r in runs) for k in ("searched", "passed", "new", "scored", "tailored",
                                                  "jsearch_requests", "cost")}
        L += [f"**{len(runs)} run(s):** searched {t['searched']} postings, filtered out {t['searched'] - t['passed']}, "
              f"{t['passed']} passed ({t['new']} new), scored {t['scored']}, tailored {t['tailored']}. "
              f"JSearch requests: {t['jsearch_requests']}. **Run cost: ${t['cost']:.4f}**", ""]
        if len(runs) > 1:
            L += ["| Run | Searched | Filtered out | Passed | New | Scored | Tailored | Cost |",
                  "|---|---|---|---|---|---|---|---|"]
            L += [f"| {r['run']} | {r['searched']} | {r['searched'] - r['passed']} | {r['passed']} | {r['new']} | "
                  f"{r['scored']} | {r['tailored']} | ${r['cost']:.4f} |" for r in runs] + [""]
    else:
        L += ["No agent run logged today.", ""]

    L += [f"## Tier A: tailored today ({min(a_total, len(a_rows))} of max {cfg['digest']['tier_a_max']})", ""]
    if a_rows:
        L += ["| # | Fit | Freshness | Company (tier) | Role | Why | Queue | Link |", "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(a_rows, 1):
            q = queue.get(r["id"])
            qs = (f"queued, {q['answered']}/{q['total']} answered" if q and _local_day(q["queued_at"]) == day
                  else "not queued yet")
            L.append(f"| {i} | {r['fit_score']} | {_freshness(r, cfg, now)} | {_cell(r['company'])} "
                     f"({tiers.get(r['company'], 'unknown')}) | {_cell(r['title'])} | {_cell(r['one_line_reason'])} | "
                     f"{qs} | [posting]({r['url']}) |")
        if a_total > len(a_rows):
            L.append(f"\n+ {a_total - len(a_rows)} more tailored today (in the review queue; not listed, Tier A "
                     f"is capped at {cfg['digest']['tier_a_max']}).")
    else:
        L.append("None tailored today.")
    L += ["", f"## Tier B: next {cfg['digest']['tier_b_max']} ranked, fit >= {cfg['digest']['tier_b_min_score']}, "
              "not tailored", ""]
    if b_rows:
        L += ["| # | Fit | Freshness | Company | Role | Why | Best approved summary | Not tailored because | Link |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(b_rows, 1):
            L.append(f"| {i} | {r['fit_score']} | {_freshness(r, cfg, now)} | {_cell(r['company'])} | "
                     f"{_cell(r['title'])} | {_cell(r['one_line_reason'])} | {r['summary']} | {r['held_by']} | "
                     f"[posting]({r['url']}) |")
    else:
        L.append("None.")
    return "\n".join(L) + "\n"


def write(store: Store, cfg: dict, master: dict, day: date | None = None, out_dir: Path = OUT_DIR) -> Path:
    day = day or date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"digest_{day.isoformat()}.md"
    path.write_text(build(store, cfg, master, day), encoding="utf-8")
    return path
