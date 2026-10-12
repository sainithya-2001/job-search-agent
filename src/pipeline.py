"""Pipeline steps shared by the CLI commands (main.py) and the agent's tools (src/agent.py).

Each step returns plain data and does no printing, so the agent can turn the result into a compact tool result
and the CLI can print it however it likes.
"""
import asyncio
import re
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
    js_requests: int = 0                                  # JSearch API requests sent by this search
    js_jobs: int = 0                                      # JSearch jobs kept (before the shared filters)
    js_direct: int = 0                                    # ... of which have the employer's own apply link


def run_search(cfg: dict, store: Store, *, save: bool = True) -> SearchResult:
    """Fetch every board (+ JSearch if configured), apply filters, and save new jobs.
    save=False (dry run) fetches and filters the same way but only reports which jobs WOULD be new."""
    from src.sources import jsearch

    res = SearchResult(saved=save)
    fetched = asyncio.run(fetch_all(cfg["companies"]))
    res.errors = dict(fetched.errors)

    js_cfg, js_key = cfg.get("jsearch", {}), jsearch.api_key()
    if js_cfg.get("enabled") and js_key and not save:
        # Dry runs never call JSearch: every request counts against a hard monthly quota.
        b = jsearch.budget(js_cfg, store)
        res.notes.append(f"JSearch: not called in a dry run (a real run may send up to {b.allowed} request(s); "
                         f"{b.used_today} used today, {b.used_month}/{js_cfg['monthly_soft_limit']} this month)")
    elif js_cfg.get("enabled") and js_key:
        async def _js():
            async with httpx.AsyncClient(timeout=30) as client:
                return await jsearch.fetch(client, js_cfg, js_key, store)
        js_jobs, js_dropped, js_errors, st = asyncio.run(_js())
        fetched.jobs.extend(js_jobs)
        b = st["budget"]
        pages = ", ".join(f"page {p}: {n}" for p, n in sorted(st["pages"].items())) or "none"
        direct = sum(bool(j.apply_direct) for j in js_jobs)
        dropped = ", ".join(f"{n} {k}" for k, n in js_dropped.items()) or "none"
        res.notes.append(f"JSearch: {len(js_jobs)} jobs from {st['sent']} request(s) ({pages}); "
                         f"{direct} with a direct employer link; dropped: {dropped}")
        res.js_jobs, res.js_direct = len(js_jobs), direct
        res.notes.append(f"JSearch requests: {b.used_today + st['sent']}/{js_cfg['max_requests_per_day']} today, "
                         f"{b.used_month + st['sent']}/{js_cfg['monthly_soft_limit']} this month "
                         f"(hard limit {js_cfg['monthly_hard_limit']})"
                         + (f"; stopped: {st['stopped']}" if st["stopped"] else ""))
        res.js_requests = st["sent"]
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
    # Strict source rule for JSearch jobs already saved: no direct employer link -> excluded (not shown anywhere).
    # Tailored jobs and jobs in the review queue / applied are never touched.
    from src.sources.jsearch import classify_stored_url
    for row in store.conn.execute("SELECT id, url, company, status FROM jobs WHERE source='jsearch' AND status IN "
                                  "('new', 'scored') AND (apply_direct IS NULL OR apply_direct=1)").fetchall():
        if store.application(row["id"]):
            continue
        direct, how = classify_stored_url(row["url"], row["company"], cfg.get("jsearch", {}))
        with store.conn:
            store.conn.execute("UPDATE jobs SET apply_direct=?, apply_via=? WHERE id=?", (int(direct), how, row["id"]))
        relabel["jsearch link re-checked"] += 1
    for row in store.conn.execute("SELECT id FROM jobs WHERE source='jsearch' AND status IN ('new', 'scored') "
                                  "AND apply_direct=0").fetchall():
        if not store.application(row["id"]):
            store.set_status(row["id"], "excluded"); relabel["jsearch excluded: no direct employer link"] += 1
    for row in store.jobs_by_status("new"):
        if row["source"] != "jsearch" and row["company"] not in configured:
            store.set_status(row["id"], "dropped_company"); relabel["dropped_company"] += 1
        elif not jf.check(job_from_row(row)).keep:
            store.set_status(row["id"], "excluded"); relabel["excluded"] += 1
    for row in store.jobs_by_status("excluded"):
        if row["source"] == "jsearch" and not row["apply_direct"]:
            continue                  # excluded by the strict source rule, not by the filters
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


def publish(tr, master, verifier, base: Path, report_text: str, pdf: bool,
            wording: dict[str, str] | None = None) -> tuple[list[str], list[str], int | None]:
    """Write JSON + HTML (+ one-page PDF) for a verified resume; append any one-page cuts to the report.
    Returns (files written, one-page cuts, PDF page count or None)."""
    from src.render import RenderError, fit_one_page, render_html
    from src.verify import select_summary

    summary = select_summary(master, verifier.family, verifier.title)["text"]
    files, cuts, pages = [], [], None
    if pdf:
        try:
            html, pages, cuts = fit_one_page(tr, master, summary, base.with_suffix(".pdf"),
                                             verifier.optional, set(verifier.required_bullets()), wording)
            files.append(f"{base.name}.pdf ({pages} page{'s' if pages != 1 else ''})")
            if cuts:
                report_text += ("\n## One-page fitting (content cut, nothing added)\n\n"
                                + "\n".join(f"- {c}" for c in cuts) + "\n")
        except RenderError as e:
            files.append(f"PDF failed: {e}")
            html = render_html(tr, master, summary, wording)
    else:
        html = render_html(tr, master, summary, wording)
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


def job_coverage(store: Store, job, ctx: TailorContext, llm=None, cfg: dict | None = None):
    """Keyword coverage for a job: cached keywords, or (if llm is given) extract them now.
    Returns (Coverage or None, extraction LLMResponse or None)."""
    from src.keywords import cached, get_keywords, match

    kws = cached(store, job["id"])
    resp = None
    if kws is None:
        if llm is None:
            return None, None
        kws, resp = get_keywords(store, job, llm, (cfg or {}).get("tailoring", {}).get("keyword_tier", "fast"))
    soft = (cfg or {}).get("tailoring", {}).get("keyword_soft_skill_patterns", [])
    return match(kws, ctx.master, ctx.texts, ctx.tech, soft), resp


def resume_text(html: str) -> str:
    import html as _html
    return _html.unescape(re.sub(r"<[^>]+>", " ", html))


def tailor_and_publish(job, store: Store, ctx: TailorContext, llm, cfg: dict, budget_left: float,
                       pdf: bool = True) -> TailorOutcome:
    """Tailor + verify one job; on PASS write the resume files and mark the job 'tailored'.
    On FAIL write only the verification report (and the failed draft as .failed.json).
    Before drafting, the posting's ATS keywords are extracted (cached) and matched to master_resume.yaml; matched
    keywords steer the draft and the skills wording, and coverage is added to the verification report."""
    from src.keywords import report_section, resume_coverage, resume_section
    from src.llm import cost_usd
    from src.render import verification_report
    from src.tailor import tailor_job
    from src.verify import Verifier, select_summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fam, title = family_and_title(store, job)
    # Checked before any LLM call: summaries are never generated, so a family without one can't be tailored.
    select_summary(ctx.master, fam, title)
    verifier = Verifier(ctx.master, ctx.texts, ctx.tech, cfg, fam, title)
    sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    cov, kw_resp = job_coverage(store, job, ctx, llm, cfg)
    kw_cost = 0.0
    if kw_resp:
        kw_cost = cost_usd(kw_resp.model, kw_resp.input_tokens, kw_resp.output_tokens,
                           cfg["llm"].get("pricing", {})) or 0.0
    res = tailor_job(job, sc, ctx.master, ctx.source, verifier, llm, cfg, budget_left=budget_left - kw_cost,
                     coverage=cov)
    if kw_resp:
        res.cost += kw_cost
        res.calls += 1
        res.input_tokens += kw_resp.input_tokens
        res.output_tokens += kw_resp.output_tokens
    base = output_base(job)
    report = verification_report(job, res, res.cost, select_summary(ctx.master, fam, title)["id"])
    if cov:
        report += "\n" + report_section(cov)
    if res.status.startswith("PASS"):
        wording = cov.wording() if cov else None
        files, cuts, pages = publish(res.resume, ctx.master, verifier, base, report, pdf, wording)
        if cov:
            html = base.with_suffix(".html").read_text(encoding="utf-8")
            rep = base.parent / (base.name + "_verification.md")
            rep.write_text(rep.read_text(encoding="utf-8") + "\n" +
                           resume_section(resume_coverage(resume_text(html), cov, ctx.master)), encoding="utf-8")
        store.mark_tailored(job["id"])
        return TailorOutcome(res, files, pages, cuts)
    (OUT_DIR / (base.name + "_verification.md")).write_text(report, encoding="utf-8")
    if res.resume is not None:
        base.with_suffix(".failed.json").write_text(json.dumps(res.resume.model_dump(), indent=2), encoding="utf-8")
    return TailorOutcome(res, [base.name + "_verification.md"], None, [])