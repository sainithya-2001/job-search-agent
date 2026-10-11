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
               source=r["source"], employment_type=r["employment_type"],
               posted_at=datetime.fromisoformat(r["posted_at"]) if r["posted_at"] else None)


def job_filter(cfg: dict) -> JobFilter:
    return JobFilter(cfg["filters"], us_hq_companies={c["name"] for c in cfg["companies"] if c.get("hq") == "US"})


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

    jf = job_filter(cfg)
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


def _family_and_title(store, job):
    sc = store.conn.execute("SELECT role_family FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    return (sc["role_family"] if sc else None) or job["role_hint"], job["title"]


def _publish(job, tr, master, verifier, base, report_text: str, pdf: bool) -> list[str]:
    """Write JSON + HTML (+ one-page PDF) for a verified resume; append any one-page cuts to the report."""
    import json as _json
    from src.render import RenderError, fit_one_page, render_html
    from src.verify import select_summary

    summary = select_summary(master, verifier.family, verifier.title)["text"]
    files = []
    if pdf:
        try:
            html, pages, cuts = fit_one_page(tr, master, summary, base.with_suffix(".pdf"),
                                             verifier.optional, set(verifier.required_bullets()))
            files.append(f"{base.name}.pdf ({pages} page{'s' if pages != 1 else ''})")
            if cuts:
                report_text += "\n## One-page fitting (content cut, nothing added)\n\n" + "\n".join(f"- {c}" for c in cuts) + "\n"
        except RenderError as e:
            print(f"  ! {e}")
            html = render_html(tr, master, summary)
    else:
        html = render_html(tr, master, summary)
    base.with_suffix(".json").write_text(_json.dumps(tr.model_dump(), indent=2), encoding="utf-8")
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    (base.parent / (base.name + "_verification.md")).write_text(report_text, encoding="utf-8")
    return [base.name + ".html", *files]


def cmd_tailor(args) -> None:
    import json as _json
    from src.llm import cost_usd, estimate_tokens, get_llm
    from src.render import slug, verification_report
    from src.resume import citable_texts, load_master, project_tech, tailoring_source
    from src.tailor import SYSTEM, build_prompt, tailor_job
    from src.verify import Verifier, check_approved_summaries, select_summary

    cfg = load_config()
    tcfg, pricing = cfg["tailoring"], cfg["llm"].get("pricing", {})
    store = Store(ROOT / "data" / "jobs.db")
    master = load_master(ROOT / "data" / "master_resume.yaml")
    texts, tech, source = citable_texts(master), project_tech(master), tailoring_source(master)
    llm = get_llm(cfg)
    model = llm.models[tcfg.get("tier", "strong")]

    findings = check_approved_summaries(master, texts, tech)
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
    est_out, per_job = tcfg["est_output_tokens"], []
    for job in jobs:
        fam, title = _family_and_title(store, job)
        v = Verifier(master, texts, tech, cfg, fam, title)
        sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job["id"],)).fetchone()
        d_in = estimate_tokens(SYSTEM + build_prompt(job, sc, source, v, None)) + 800
        per_job.append((cost_usd(model, d_in, est_out["draft"], pricing) or 0)
                       + (cost_usd(model, 3000, est_out["verify"], pricing) or 0))
    expected, worst = sum(per_job) * 1.5, sum(per_job) * tcfg["max_attempts"]
    cap = tcfg["max_run_cost_usd"]
    print(f"Tailoring {len(jobs)} job(s) with {model}")
    print(f"Estimated cost: ~${expected:.3f} expected, ${worst:.3f} if every job needs {tcfg['max_attempts']} attempts"
          f"  |  run cap ${cap:.2f}")
    if expected > cap:
        print("Refusing to run: expected cost exceeds tailoring.max_run_cost_usd.")
        return

    out_dir = ROOT / "output" / "resumes"
    out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for job in jobs:
        fam, title = _family_and_title(store, job)
        verifier = Verifier(master, texts, tech, cfg, fam, title)
        sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job["id"],)).fetchone()
        res = tailor_job(job, sc, master, source, verifier, llm, cfg, budget_left=cap - spent)
        spent += res.cost
        base = out_dir / f"{slug(job['company'])}_{slug(job['title'])}"
        report = verification_report(job, res, res.cost, select_summary(master, fam, title)["id"])
        if res.status.startswith("PASS"):
            files = _publish(job, res.resume, master, verifier, base, report, args.pdf)
            store.set_status(job["id"], "tailored")
        else:
            (out_dir / (base.name + "_verification.md")).write_text(report, encoding="utf-8")
            if res.resume is not None:
                base.with_suffix(".failed.json").write_text(_json.dumps(res.resume.model_dump(), indent=2), encoding="utf-8")
            files = [base.name + "_verification.md"]
        print(f"  {res.status:22s} {job['company'][:16]:16s} {job['title'][:40]:40s} attempts={res.attempts} "
              f"removed={len(res.removed)} ${res.cost:.4f}  -> {', '.join(files)}")
    print(f"\nActual cost: ${spent:.4f}  (output in {out_dir.relative_to(ROOT)})")
    store.close()


def cmd_render(args) -> None:
    """Re-render already-verified resumes from their saved JSON (no API calls)."""
    from src.models import TailoredResume
    from src.render import slug
    from src.resume import citable_texts, load_master, project_tech
    from src.verify import Verifier

    cfg = load_config()
    store = Store(ROOT / "data" / "jobs.db")
    master = load_master(ROOT / "data" / "master_resume.yaml")
    texts, tech = citable_texts(master), project_tech(master)
    out_dir = ROOT / "output" / "resumes"
    for jid in args.job_ids:
        job = store.conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        base = out_dir / f"{slug(job['company'])}_{slug(job['title'])}" if job else None
        if not job or not base.with_suffix(".json").exists():
            print(f"  ! no verified resume for {jid} — run `tailor` first")
            continue
        report = (out_dir / (base.name + "_verification.md")).read_text(encoding="utf-8")
        if "**Status:** PASS" not in report:
            print(f"  ! {job['company']}: verification did not pass; not rendering")
            continue
        tr = TailoredResume.model_validate_json(base.with_suffix(".json").read_text(encoding="utf-8"))
        fam, title = _family_and_title(store, job)
        verifier = Verifier(master, texts, tech, cfg, fam, title)
        recheck = verifier.check_structure(tr)          # rules may have changed since it was tailored
        if not recheck.passed:
            print(f"  ! {job['company']}: saved resume no longer passes current rules ({len(recheck.issues)} issue(s)); "
                  f"re-run `tailor`")
            continue
        files = _publish(job, tr, master, verifier, base, report, args.pdf)
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    main()
