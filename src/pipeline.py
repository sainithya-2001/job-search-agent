"""Pipeline steps shared by the CLI commands (main.py) and the agent's tools (src/agent.py).

Each step returns plain data and does no printing, so the agent can turn the result into a compact tool result
and the CLI can print it however it likes.
"""
import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from src.filters import JobFilter
from src.models import Job
from src.sources import fetch_all
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "resumes"


def job_filter(cfg: dict) -> JobFilter:
    return JobFilter(cfg["filters"], us_hq_companies={c["name"] for c in cfg["companies"] if c.get("hq") == "US"})


def job_from_row(r) -> Job:
    return Job(id=r["id"], company=r["company"], title=r["title"], location=r["location"] or "",
               remote=bool(r["remote"]), url=r["url"] or "", description=r["description"] or "",
               source=r["source"], employment_type=r["employment_type"],
               posted_at=datetime.fromisoformat(r["posted_at"]) if r["posted_at"] else None)


# ---- search -----------------------------------------------------------------------------------------------------
@dataclass
class SearchResult:
    fetched: int = 0
    boards: int = 0
    errors: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)   # raw filter reasons (main.py buckets them for display)
    kept: int = 0
    new: list[Job] = field(default_factory=list)
    hints: dict = field(default_factory=dict)            # job id -> role family hint from the title filter
    relabel: Counter = field(default_factory=Counter)
    saved: bool = True


def run_search(cfg: dict, store: Store, *, save: bool = True) -> SearchResult:
    """Fetch every board (+ JSearch if configured), apply filters, and save new jobs.
    save=False (dry run) fetches and filters the same way but only reports which jobs WOULD be new."""
    from src.sources import jsearch

    res = SearchResult(saved=save)
    fetched = asyncio.run(fetch_all(cfg["companies"]))
    res.errors = dict(fetched.errors)

    js_cfg, js_key = cfg.get("jsearch", {}), jsearch.api_key()
    if js_cfg.get("enabled") and js_key:
        async def _js():
            async with httpx.AsyncClient(timeout=30) as client:
                return await jsearch.fetch(client, js_cfg, js_key)
        js_jobs, js_dropped, js_errors = asyncio.run(_js())
        fetched.jobs.extend(js_jobs)
        res.notes.append(f"JSearch: {len(js_jobs)} jobs from {len(js_cfg['queries'])} queries "
                         f"({js_dropped} dropped as staffing/excluded employers)")
        res.notes += [f"! {e}" for e in js_errors]
    else:
        res.notes.append("JSearch: skipped (" + ("disabled in config" if not js_cfg.get("enabled")
                                                 else "no JSEARCH_API_KEY in .env") + ")")

    jf = job_filter(cfg)
    kept = []
    for job in fetched.jobs:
        r = jf.check(job)
        res.reasons[r.reason] += 1
        if r.keep:
            kept.append(job)
            res.hints[job.id] = r.role_hint
    res.fetched, res.boards, res.kept = len(fetched.jobs), len(fetched.per_board), len(kept)

    if not save:
        res.new = store.preview_new(kept)
        return res
    res.new = store.save(kept, res.hints)

    res.relabel = recheck_existing(cfg, store, jf)
    return res


def recheck_existing(cfg: dict, store: Store, jf: JobFilter | None = None) -> Counter:
    # Re-check unscored jobs already in the DB against the CURRENT filters and company list, so rule changes
    # apply retroactively in both directions: newly failing jobs are excluded, and previously excluded jobs that
    # now pass come back as new. Rows are re-labelled, never deleted; scored jobs are left alone.
    jf = jf or job_filter(cfg)
    configured = {c["name"] for c in cfg["companies"]}
    relabel = Counter()
    for row in store.jobs_by_status("new"):
        if row["source"] != "jsearch" and row["company"] not in configured:
            store.set_status(row["id"], "dropped_company"); relabel["dropped_company"] += 1
        elif not jf.check(job_from_row(row)).keep:
            store.set_status(row["id"], "excluded"); relabel["excluded"] += 1
    for row in store.jobs_by_status("excluded"):
        if (row["source"] == "jsearch" or row["company"] in configured) and jf.check(job_from_row(row)).keep:
            store.set_status(row["id"], "new"); relabel["restored (now passes filters)"] += 1
    return relabel


# ---- tailor -----------------------------------------------------------------------------------------------------
@dataclass
class TailorContext:
    """Everything tailoring needs that doesn't change between jobs (loaded once per run)."""
    master: dict
    texts: dict
    tech: dict
    source: str


def tailor_context() -> TailorContext:
    from src.resume import citable_texts, load_master, project_tech, tailoring_source

    master = load_master(ROOT / "data" / "master_resume.yaml")
    return TailorContext(master, citable_texts(master), project_tech(master), tailoring_source(master))


def family_and_title(store: Store, job) -> tuple[str, str]:
    sc = store.conn.execute("SELECT role_family FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    return (sc["role_family"] if sc else None) or job["role_hint"], job["title"]


def output_base(job) -> Path:
    from src.render import slug
    return OUT_DIR / f"{slug(job['company'])}_{slug(job['title'])}"


def tailor_estimate(job, store: Store, ctx: TailorContext, llm, cfg: dict) -> float:
    """Expected cost of ONE attempt (draft + LLM judge) for this job."""
    from src.llm import cost_usd, estimate_tokens
    from src.tailor import SYSTEM, build_prompt
    from src.verify import Verifier

    tcfg, pricing = cfg["tailoring"], cfg["llm"].get("pricing", {})
    model = llm.models[tcfg.get("tier", "strong")]
    fam, title = family_and_title(store, job)
    v = Verifier(ctx.master, ctx.texts, ctx.tech, cfg, fam, title)
    sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    d_in = estimate_tokens(SYSTEM + build_prompt(job, sc, ctx.source, v, None)) + 800
    out = tcfg["est_output_tokens"]
    return (cost_usd(model, d_in, out["draft"], pricing) or 0) + (cost_usd(model, 3000, out["verify"], pricing) or 0)


def publish(tr, master, verifier, base: Path, report_text: str, pdf: bool) -> tuple[list[str], list[str], int | None]:
    """Write JSON + HTML (+ one-page PDF) for a verified resume; append any one-page cuts to the report.
    Returns (files written, one-page cuts, PDF page count or None)."""
    from src.render import RenderError, fit_one_page, render_html
    from src.verify import select_summary

    summary = select_summary(master, verifier.family, verifier.title)["text"]
    files, cuts, pages = [], [], None
    if pdf:
        try:
            html, pages, cuts = fit_one_page(tr, master, summary, base.with_suffix(".pdf"),
                                             verifier.optional, set(verifier.required_bullets()))
            files.append(f"{base.name}.pdf ({pages} page{'s' if pages != 1 else ''})")
            if cuts:
                report_text += ("\n## One-page fitting (content cut, nothing added)\n\n"
                                + "\n".join(f"- {c}" for c in cuts) + "\n")
        except RenderError as e:
            files.append(f"PDF failed: {e}")
            html = render_html(tr, master, summary)
    else:
        html = render_html(tr, master, summary)
    base.with_suffix(".json").write_text(json.dumps(tr.model_dump(), indent=2), encoding="utf-8")
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    (base.parent / (base.name + "_verification.md")).write_text(report_text, encoding="utf-8")
    return [base.name + ".html", *files], cuts, pages


@dataclass
class TailorOutcome:
    result: object          # tailor.TailorResult
    files: list[str]
    pages: int | None
    cuts: list[str]


def tailor_and_publish(job, store: Store, ctx: TailorContext, llm, cfg: dict, budget_left: float,
                       pdf: bool = True) -> TailorOutcome:
    """Tailor + verify one job; on PASS write the resume files and mark the job 'tailored'.
    On FAIL write only the verification report (and the failed draft as .failed.json)."""
    from src.render import verification_report
    from src.tailor import tailor_job
    from src.verify import Verifier, select_summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fam, title = family_and_title(store, job)
    verifier = Verifier(ctx.master, ctx.texts, ctx.tech, cfg, fam, title)
    sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    res = tailor_job(job, sc, ctx.master, ctx.source, verifier, llm, cfg, budget_left=budget_left)
    base = output_base(job)
    report = verification_report(job, res, res.cost, select_summary(ctx.master, fam, title)["id"])
    if res.status.startswith("PASS"):
        files, cuts, pages = publish(res.resume, ctx.master, verifier, base, report, pdf)
        store.set_status(job["id"], "tailored")
        return TailorOutcome(res, files, pages, cuts)
    (OUT_DIR / (base.name + "_verification.md")).write_text(report, encoding="utf-8")
    if res.resume is not None:
        base.with_suffix(".failed.json").write_text(json.dumps(res.resume.model_dump(), indent=2), encoding="utf-8")
    return TailorOutcome(res, [base.name + "_verification.md"], None, [])
