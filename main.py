"""CLI entry point. Phase 1: `python main.py search`. Later phases add score | tailor | apply | run."""
import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.store import Store

ROOT = Path(__file__).parent


LOCAL_CONFIG = ROOT / "config.local.yaml"


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into `base`: nested dicts merge, anything else (lists included) is replaced."""
    for k, v in over.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    """config.yaml, overlaid with config.local.yaml (gitignored: personal values such as the tailoring rules that
    name master-resume bullet ids). Tests set JOB_AGENT_NO_LOCAL_CONFIG so they see only the committed config."""
    import os

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if LOCAL_CONFIG.exists() and not os.environ.get("JOB_AGENT_NO_LOCAL_CONFIG"):
        cfg = _merge(cfg, yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8")) or {})
    return cfg


_EXPORT_WORDS = ("export", "itar", "ear", "person", "permanent", "green")


def _reason_bucket(reason: str) -> str:
    """Group phrase hits so the summary stays short."""
    if not reason.startswith("phrase"):
        return reason
    text = reason.lower()
    if any(w in text for w in _EXPORT_WORDS):
        return "phrase: export control / US person / residency"
    return "phrase: citizenship / clearance / contract"


def cmd_search(args) -> None:
    from src.llm import load_env
    from src.pipeline import run_search
    from src.scorer import company_tiers, location_rank, tier_rank

    cfg = load_config()
    load_env(ROOT / ".env")
    print(f"Fetching {len(cfg['companies'])} boards...")
    store = Store(ROOT / "data" / "jobs.db")
    res = run_search(cfg, store)
    for board, err in sorted(res.errors.items()):
        print(f"  ! {board}: {err}")
    for note in res.notes:
        print(note)
    reasons = Counter()
    for reason, n in res.reasons.items():
        reasons[_reason_bucket(reason)] += n
    new, hints = res.new, res.hints

    print(f"\nFetched {res.fetched} jobs from {res.boards} boards ({len(res.errors)} errors)")
    print("Filter results:")
    for reason, n in reasons.most_common():
        print(f"  {n:6d}  {reason}")
    print(f"Passed filters: {res.kept}  |  New (after dedupe): {len(new)}  |  Total in DB: {store.count()}")
    if res.relabel:
        print("Existing unscored jobs re-checked: " + ", ".join(f"{n} -> {k}" for k, n in res.relabel.items()))
    print("Ready to score (status new): " + str(store.count("new")))
    js_new = Counter(j.company for j in new if j.source == "jsearch")
    if js_new:
        print(f"New from JSearch: {sum(js_new.values())} jobs from {len(js_new)} companies: "
              + ", ".join(f"{c} ({n})" for c, n in js_new.most_common()))

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
    from src.scorer import system_prompt, estimate, hide_stale, is_fresh, rank_results, score_jobs, select_jobs, \
        select_mix
    from src.sponsorship import SponsorshipDB

    cfg = load_config()
    limit = args.limit or cfg["scoring"]["default_limit"]
    cap = args.cap if args.cap is not None else cfg["scoring"]["max_run_cost_usd"]
    store = Store(ROOT / "data" / "jobs.db")
    previous = {}
    if args.job_ids:
        # Re-score specific jobs (e.g. after a prompt change). The new score replaces the old one; the old one is
        # printed alongside for comparison. Tailored jobs keep their status.
        jobs = []
        for jid in args.job_ids:
            row = store.conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            if row is None:
                print(f"Unknown job id: {jid}")
                return
            old = store.conn.execute("SELECT fit_score FROM scores WHERE job_id=?", (jid,)).fetchone()
            previous[jid] = old["fit_score"] if old else None
            jobs.append(row)
    elif args.mix:
        jobs, notes = select_mix(store.unscored_jobs(), cfg)
        mix = cfg["scoring"]["mix"]
        print(f"Mix selection from tiers {mix['tiers']}: "
              + ", ".join(f"{b['n']} {b['name']}" for b in mix["buckets"]))
        for n in notes:
            print(f"  ! {n}")
    elif args.all:
        # Every unscored job, no per-company cap, still in priority order (AI families first, PM family last).
        unscored = store.unscored_jobs()
        jobs = select_jobs(unscored, cfg, len(unscored), max_per_company=len(unscored))
    else:
        jobs = select_jobs(store.unscored_jobs(), cfg, limit)
    if not jobs:
        print("No unscored jobs. Run `python main.py search` first.")
        return

    system = system_prompt(cfg, load_master(ROOT / "data" / "master_resume.yaml"))
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
    res = score_jobs(jobs, system, llm, cfg, store, sponsor_db, est, cap=cap)

    rows, stale = hide_stale(res.scored, cfg)
    rows = rank_results(rows, cfg)
    print(f"\nActual cost: ${res.spent:.4f} over {res.calls} API calls "
          f"({sum(r['escalated'] for r in res.scored)} escalated to {llm.models['strong']})")
    if res.stopped_early:
        print(f"Stopped early: {res.stopped_early}")
    now = datetime.now(timezone.utc)

    def posted(r) -> str:
        if not r.get("posted_at"):
            return "no date"
        p = datetime.fromisoformat(r["posted_at"])
        days = (now - (p if p.tzinfo else p.replace(tzinfo=timezone.utc))).days
        return f"{days}d" + (" *" if is_fresh(r, cfg, now) else "")

    if stale:
        print(f"{stale} scored job(s) hidden: posted more than {cfg['filters']['max_posting_age_days']} days ago.")
    print("Ranked by fit, then freshness (* = posted 0-3 days ago; a fresh job goes ahead of an older one up to "
          f"{cfg.get('ranking', {}).get('freshness_preference_points', 5)} points higher), sponsorship, tier, location.")
    print(f"\n{'#':>2}  {'Fit':>3}  {'Posted':7s} {'Family':23s} {'Company':16s} {'Title':44s} {'Location':22s} "
          f"{'Sponsor':7s}  Reason")
    for i, r in enumerate(rows, 1):
        was = f"(was {previous[r['job_id']]}) " if r["job_id"] in previous else ""
        print(f"{i:2d}  {r['fit_score']:3d}  {posted(r):7s} {r['role_family'][:23]:23s} {r['company'][:16]:16s} "
              f"{r['title'][:44]:44s} {r['location'][:22]:22s} {r['sponsorship_signal']:7s}  "
              f"{was}{r['one_line_reason']}")
    store.close()


def cmd_tailor(args) -> None:
    from src.llm import get_llm
    from src.pipeline import OUT_DIR, tailor_and_publish, tailor_context, tailor_estimate
    from src.verify import check_approved_summaries

    cfg = load_config()
    tcfg = cfg["tailoring"]
    store = Store(ROOT / "data" / "jobs.db")
    ctx = tailor_context()
    llm = get_llm(cfg)
    model = llm.models[tcfg.get("tier", "strong")]

    findings = check_approved_summaries(ctx.master, ctx.texts, ctx.tech)
    print("Approved summaries checked against master facts: " + ("all numbers and tools found" if not findings else ""))
    for f in findings:
        print(f"  note: {f}")

    jobs = []
    for jid in args.job_ids:
        job = store.conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not job:
            print(f"Unknown job id: {jid}")
            return
        jobs.append(job)

    # Estimate: one draft + one LLM fact-check per job, assuming ~1.5 attempts on average (worst case: max_attempts).
    per_job = [tailor_estimate(job, store, ctx, llm, cfg) for job in jobs]
    expected, worst = sum(per_job) * 1.5, sum(per_job) * tcfg["max_attempts"]
    cap = tcfg["max_run_cost_usd"]
    print(f"Tailoring {len(jobs)} job(s) with {model}")
    print(f"Estimated cost: ~${expected:.3f} expected, ${worst:.3f} if every job needs {tcfg['max_attempts']} attempts"
          f"  |  run cap ${cap:.2f}")
    if expected > cap:
        print("Refusing to run: expected cost exceeds tailoring.max_run_cost_usd.")
        return

    from src.linkcheck import check_before_tailoring
    spent = 0.0
    for job in jobs:
        check = check_before_tailoring(job, cfg)          # closed posting / unverified JSearch link: skip, no spend
        if check is not None:
            if check.closed:
                store.mark_closed(job["id"], check.note)
            else:
                store.save_link_check(job["id"], False, check.note)
            print(f"  {'SKIPPED':22s} {job['company'][:16]:16s} {job['title'][:40]:40s} {check.note}")
            continue
        out = tailor_and_publish(job, store, ctx, llm, cfg, budget_left=cap - spent, pdf=args.pdf)
        res = out.result
        spent += res.cost
        print(f"  {res.status:22s} {job['company'][:16]:16s} {job['title'][:40]:40s} attempts={res.attempts} "
              f"removed={len(res.removed)} ${res.cost:.4f}  -> {', '.join(out.files)}")
    print(f"\nActual cost: ${spent:.4f}  (output in {OUT_DIR.relative_to(ROOT)})")
    store.close()


def cmd_render(args) -> None:
    """Re-render already-verified resumes from their saved JSON (no API calls)."""
    from src.models import TailoredResume
    from src.pipeline import family_and_title, job_coverage, output_base, publish, tailor_context
    from src.verify import Verifier

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    ctx = tailor_context()
    for jid in args.job_ids:
        job = store.conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        base = output_base(job) if job else None
        if not job or not base.with_suffix(".json").exists():
            print(f"  ! no verified resume for {jid} — run `tailor` first")
            continue
        report = (base.parent / (base.name + "_verification.md")).read_text(encoding="utf-8")
        if "**Status:** PASS" not in report:
            print(f"  ! {job['company']}: verification did not pass; not rendering")
            continue
        tr = TailoredResume.model_validate_json(base.with_suffix(".json").read_text(encoding="utf-8"))
        fam, title = family_and_title(store, job)
        verifier = Verifier(ctx.master, ctx.texts, ctx.tech, cfg, fam, title)
        recheck = verifier.check_structure(tr)          # rules may have changed since it was tailored
        if not recheck.passed:
            print(f"  ! {job['company']}: saved resume no longer passes current rules ({len(recheck.issues)} issue(s)); "
                  f"re-run `tailor`")
            continue
        cov, _ = job_coverage(store, job, ctx, None, cfg)  # cached keywords only: re-rendering never calls the API
        try:
            files, _, _ = publish(tr, ctx.master, verifier, base, report, args.pdf, cov.wording() if cov else None)
        except PermissionError as e:
            # Windows locks a PDF that's open in a viewer. The PDF is written first, so nothing was changed.
            print(f"  ! {job['company']}: can't write {Path(e.filename).name} (open in another program?); "
                  f"skipped, files unchanged")
            continue
        print(f"  {job['company'][:16]:16s} -> {', '.join(files)}")
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


def cmd_run(args) -> None:
    """Phase 4: the daily cycle (search -> score -> tailor 65+ -> log), driven by the agent loop. Never applies."""
    from src.agent import RunLog, Toolbox, plan, run_agent, summary_markdown
    from src.llm import get_llm, load_env

    cfg = load_config()
    load_env(ROOT / ".env")
    store = Store(ROOT / "data" / "jobs.db")
    llm = get_llm(cfg)
    if args.dry_run:
        print(plan(cfg, store, llm))
        store.close()
        return
    _backup(cfg)
    from src.notion import try_sync
    print(try_sync(store))              # pull Status changes made in Notion before the run uses them
    log = RunLog()
    a = cfg["agent"]
    print(f"Agent run (model {llm.models[a.get('tier', 'fast')]}) | cap ${a['max_run_cost_usd']:.2f} | "
          f"max {a['max_tool_calls']} tool calls | tailor >= {a['tailor_min_score']}, max {a['max_tailor_per_run']}")
    state = run_agent(cfg, llm, Toolbox(cfg, store, llm), log)
    report = summary_markdown(state, log)
    summary_path = log.path.with_name(log.path.stem + "_summary.md")
    summary_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Logs: {log.path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}")
    if cfg["digest"].get("queue_after_run"):
        _queue_today(cfg, store)
    _write_digest(cfg, store)
    print(try_sync(store))
    store.close()


def _backup(cfg: dict) -> None:
    from src.backup import backup_db

    path, deleted = backup_db(ROOT / "data" / "jobs.db", ROOT / "data" / "backups", cfg["backup"]["keep_days"])
    print(f"Backup: {path.relative_to(ROOT)}" + (f" ({len(deleted)} older than {cfg['backup']['keep_days']} days "
                                                  f"deleted)" if deleted else ""))


def _queue_today(cfg: dict, store: Store) -> None:
    """Read the forms of jobs tailored today that aren't queued yet (headless; no LLM; nothing filled/submitted)."""
    from datetime import date

    from src.apply import queue_jobs
    from src.digest import _local_day

    queued = {r["job_id"] for r in store.conn.execute("SELECT job_id FROM apply_queue")}
    from src.digest import apply_route
    ids = [r["id"] for r in store.jobs_by_status("tailored")       # only forms review mode can pre-fill
           if _local_day(r["tailored_at"]) == date.today() and r["id"] not in queued and apply_route(r) is None]
    if not ids:
        return
    print(f"\nQueueing {len(ids)} job(s) tailored today (reading forms headlessly)...")
    try:
        queue_jobs(cfg, store, ids)
    except Exception as e:  # noqa: BLE001 — the digest still gets written; `queue` can be rerun by hand
        print(f"  ! queue failed: {type(e).__name__}: {e}")


def _write_digest(cfg: dict, store: Store, day=None) -> None:
    from src.digest import write
    from src.resume import load_master

    path = write(store, cfg, load_master(ROOT / "data" / "master_resume.yaml"), day)
    print(f"Digest: {path.relative_to(ROOT)}")


def cmd_digest(args) -> None:
    from datetime import date

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    _write_digest(cfg, store, date.fromisoformat(args.date) if args.date else None)
    store.close()


def cmd_keywords(args) -> None:
    """Read-only ATS keyword coverage for tailored resumes (no re-tailoring). Keyword extraction is the only paid
    step (fast tier, cached per job); without --run this only prints its estimated cost."""
    from src.keywords import EXTRACT_SYSTEM, ESTIMATE_OUTPUT_TOKENS, cached, extract_prompt, resume_coverage, \
        report_section, resume_section, strip_coverage
    from src.llm import cost_usd, estimate_tokens, get_llm
    from src.pipeline import job_coverage, output_base, resume_text, tailor_context

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    if args.job_ids:
        jobs = [store.conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone() for j in args.job_ids]
    else:
        jobs = store.jobs_by_status("tailored")
    jobs = [j for j in jobs if j is not None and output_base(j).with_suffix(".html").exists()]
    todo = [j for j in jobs if cached(store, j["id"]) is None]
    llm = get_llm(cfg)
    model, pricing = llm.models["fast"], cfg["llm"].get("pricing", {})
    est = sum(cost_usd(model, estimate_tokens(EXTRACT_SYSTEM + extract_prompt(j)) + 300, ESTIMATE_OUTPUT_TOKENS,
                       pricing) or 0 for j in todo)
    print(f"{len(jobs)} tailored resume(s); keywords cached for {len(jobs) - len(todo)}, "
          f"{len(todo)} need extraction with {model}: estimated ${est:.4f}")
    if todo and not args.run:
        print("Nothing spent. Re-run with --run to extract keywords and check coverage.")
        store.close()
        return
    ctx = tailor_context()
    spent = 0.0
    print(f"\n{'Company':14s} {'Title':40s} {'Exact':>6s} {'Other words':>11s} {'Not shown':>9s} {'Missing':>7s}")
    for j in jobs:
        cov, resp = job_coverage(store, j, ctx, llm, cfg)
        if resp:
            spent += cost_usd(resp.model, resp.input_tokens, resp.output_tokens, pricing) or 0
        base = output_base(j)
        rc = resume_coverage(resume_text(base.with_suffix(".html").read_text(encoding="utf-8")), cov, ctx.master)
        rep = base.parent / (base.name + "_verification.md")
        old = rep.read_text(encoding="utf-8") if rep.exists() else ""
        old = strip_coverage(old)          # replace earlier coverage sections; keep every other section
        rep.write_text(old + "\n" + report_section(cov) + "\n" + resume_section(rc), encoding="utf-8")
        print(f"{j['company'][:14]:14s} {j['title'][:40]:40s} {f'{len(rc.exact)}/{rc.total}':>6s} "
              f"{len(rc.synonym):>11d} {len(rc.not_shown):>9d} {len(rc.missing):>7d}   ({rc.pct(len(rc.exact))}%)")
        print(f"{'':14s}   not shown: {', '.join(rc.not_shown) or '-'}")
        print(f"{'':14s}   don't have: {', '.join(rc.missing) or '-'}")
    print(f"\nActual extraction cost: ${spent:.4f}. Coverage sections written to each *_verification.md.")
    store.close()


def cmd_answers_check(args) -> None:
    from src.answers import BankError, load_bank

    try:
        bank = load_bank()
    except BankError as e:
        print(e)
        return
    todo = bank.unfilled()
    print(f"Answer bank: {len(bank.entries)} entries, {len(bank.by_question)} question wordings")
    print(("Not answered yet (these questions will always be left for you): " + ", ".join(todo)) if todo
          else "Every entry has an answer or skip.")


def cmd_queue(args) -> None:
    from src.apply import queue_jobs, queue_report
    from src.notion import try_sync

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    print("Reading application forms (headless; nothing is filled or submitted)...")
    results = queue_jobs(cfg, store, args.job_ids or None)
    print(queue_report(results))
    print(f"\nQueued {sum('matches' in r for r in results)} application(s). "
          f"Shadow auto-submit: {sum(bool(r.get('shadow_qualified')) for r in results)} would qualify "
          f"(there is no submit code; all go to review).")
    print(try_sync(store))
    store.close()


def cmd_review(args) -> None:
    from src.apply import review
    from src.notion import try_sync

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    res = review(cfg, store, args.job_id)
    if res:
        print(try_sync(store))
    store.close()


def cmd_confirm(args) -> None:
    from src.apply import confirm
    from src.notion import try_sync

    store = Store(ROOT / "data" / "jobs.db")
    status = confirm(store, args.job_id, applied=args.applied, note=args.note or "")
    print(f"{args.job_id}: {status}")
    print(try_sync(store))
    store.close()


def cmd_close(args) -> None:
    from src.apply import close
    from src.notion import try_sync

    store = Store(ROOT / "data" / "jobs.db")
    close(store, args.job_id, args.note or "")
    print(f"{args.job_id}: closed")
    print(try_sync(store))
    store.close()


def cmd_status(args) -> None:
    """Set an application's status by hand, e.g. interviewing / offer / rejected after you hear back."""
    from src.apply import change_status
    from src.notion import try_sync

    store = Store(ROOT / "data" / "jobs.db")
    print(f"{args.job_id}: {change_status(store, args.job_id, args.status, args.note or '')}")
    print(try_sync(store))
    store.close()


def cmd_notion_sync(args) -> None:
    from src.notion import try_sync

    store = Store(ROOT / "data" / "jobs.db")
    print(try_sync(store))
    store.close()


def main(argv: list[str] | None = None) -> None:
    from src.apply import STATUSES

    parser = argparse.ArgumentParser(prog="job-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="Fetch boards, filter, save new jobs")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("score", help="LLM-score the most relevant unscored jobs")
    p.add_argument("--limit", type=int, default=None, help="jobs to score (default: scoring.default_limit)")
    p.add_argument("--mix", action="store_true", help="pick jobs per scoring.mix buckets instead of top-N")
    p.add_argument("--job-ids", nargs="+", help="re-score these jobs (replaces their current scores)")
    p.add_argument("--all", action="store_true", help="score every unscored job (no per-company cap)")
    p.add_argument("--cap", type=float, help="cost cap for this run in USD (default scoring.max_run_cost_usd)")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("tailor", help="Tailor + verify resumes for job ids (HTML; --pdf for PDF)")
    p.add_argument("job_ids", nargs="+")
    p.add_argument("--pdf", action="store_true", help="also write PDFs (needs a working PDF engine)")
    p.set_defaults(func=cmd_tailor)
    p = sub.add_parser("render", help="Re-render verified resumes from saved JSON (no API calls)")
    p.add_argument("job_ids", nargs="+")
    p.add_argument("--pdf", action="store_true")
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("skills-gap", help="Top skills asked for across scored jobs vs. the master resume")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_skills_gap)
    p = sub.add_parser("run", help="Daily agent cycle: search, score, tailor 50+ (45+ if fresh; never applies)")
    p.add_argument("--dry-run", action="store_true", help="plan and estimate cost; no LLM calls, no DB writes")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("digest", help="Write output/digest_YYYY-MM-DD.md (Tier A tailored today, Tier B next 10)")
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.set_defaults(func=cmd_digest)
    p = sub.add_parser("keywords", help="ATS keyword coverage of tailored resumes (read-only; extraction is cached)")
    p.add_argument("job_ids", nargs="*")
    p.add_argument("--run", action="store_true", help="spend the (estimated) extraction cost and check coverage")
    p.set_defaults(func=cmd_keywords)
    p = sub.add_parser("answers-check", help="Show which answer-bank entries you still need to fill in")
    p.set_defaults(func=cmd_answers_check)
    p = sub.add_parser("queue", help="Read forms for tailored jobs, match the answer bank, run shadow gates")
    p.add_argument("job_ids", nargs="*")
    p.set_defaults(func=cmd_queue)
    p = sub.add_parser("review", help="Open the next queued form pre-filled; you review and submit")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("apply", help="Same as review, for one job id")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("confirm", help="Mark a job applied (after the employer's confirmation) or skipped")
    p.add_argument("job_id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--applied", action="store_true")
    g.add_argument("--skipped", action="store_true")
    p.add_argument("--note")
    p.set_defaults(func=cmd_confirm)
    p = sub.add_parser("close", help="Mark a posting closed (no longer accepting applications)")
    p.add_argument("job_id")
    p.add_argument("--note")
    p.set_defaults(func=cmd_close)
    p = sub.add_parser("status", help="Set an application's status (e.g. interviewing, offer, rejected)")
    p.add_argument("job_id")
    p.add_argument("status", choices=STATUSES)
    p.add_argument("--note")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("notion-sync", help="Sync the applications table with Notion (pulls Status changes back)")
    p.set_defaults(func=cmd_notion_sync)

    args = parser.parse_args(argv)
    if _spends_llm(args):
        _traced(args)
    else:
        args.func(args)


def _spends_llm(args) -> bool:
    """Commands that make LLM calls get one Langfuse trace each (dry runs and read-only modes make none)."""
    if args.command == "run":
        return not args.dry_run
    if args.command == "keywords":
        return args.run
    return args.command in ("score", "tailor")


def _traced(args) -> None:
    """Run the command inside one trace. Tracing is optional (off without the three LANGFUSE_* settings) and never
    fails the command: src/tracing.py swallows every Langfuse error. Only ids and metadata are sent."""
    from src import tracing
    from src.llm import load_env
    load_env(ROOT / ".env")
    tracer = tracing.configure()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        with tracer.run(f"job-agent {args.command}", command=args.command, run_id=run_id) as root:
            args.func(args)
            root.set(ok=True)
    finally:
        tracer.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
