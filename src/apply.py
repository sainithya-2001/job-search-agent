"""Phase 5: review-mode applications. Pre-fill and queue; Nithya reviews and clicks Submit herself.

  queue   read each tailored job's form (headless), match every question against the answer bank, run the
          shadow auto-submit gates, and store the result. Free: no LLM calls.
  review  open one queued form in a visible browser, pre-fill what the bank answers exactly, attach the tailored
          PDF, outline what's left, then hand over. The form's values are snapshotted after pre-fill and again
          when Nithya clicks Submit, so the log records whether she approved unchanged or edited (and what).
  confirm mark a job applied (after she sees the employer's confirmation) or skipped. Also used for
          applications she submitted entirely by hand, so the 90-day company rule counts them.
There is no submit code anywhere: see src/forms.py for the two safeguards.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from src.answers import ANSWERED, SKIPPED, UNVERIFIED, AnswerBank, Match, load_bank, match_form, profile_tokens
from src.forms import (ats_of, block_submits, detect_captcha, fill, form_url, hand_over, highlight,
                       read_questions, snapshot)
from src.gates import blockers, check_config, evaluate, qualifies
from src.pipeline import output_base
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "apply.jsonl"
DONE = ("applied", "skipped")


def log_event(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **event}, default=str) + "\n")


def application_fields(store: Store, job_id: str) -> dict | None:
    """The Sheets columns that come from the job + its score + its resume (shared with the agent's log tool)."""
    row = store.conn.execute(
        "SELECT j.*, s.fit_score, s.sponsorship_signal, s.uscis_status, s.matched_skills, s.gaps "
        "FROM jobs j LEFT JOIN scores s ON s.job_id = j.id WHERE j.id=?", (job_id,)).fetchone()
    if row is None:
        return None
    pdf = verified_pdf(row)
    return {"job_id": job_id, "date_found": (row["first_seen_at"] or "")[:10], "company": row["company"],
            "role": row["title"], "location": row["location"], "url": row["url"], "fit_score": row["fit_score"],
            "sponsorship": f"{row['sponsorship_signal']} (USCIS: {row['uscis_status']})",
            "matched_skills": row["matched_skills"], "gaps": row["gaps"],
            "resume_file": pdf.name if pdf else None}


def verified_pdf(job) -> Path | None:
    """The job's tailored PDF, only if its verification report says PASS."""
    base = output_base(job)
    pdf, report = base.with_suffix(".pdf"), base.parent / (base.name + "_verification.md")
    if pdf.exists() and report.exists() and "**Status:** PASS" in report.read_text(encoding="utf-8"):
        return pdf
    return None


def _pdf_pages(pdf: Path) -> int:
    import re
    return len(re.findall(rb"/Type\s*/Page(?!s)", pdf.read_bytes()))


def _history(store: Store) -> list[dict]:
    """Every logged application with its role family (from the score, falling back to the title filter's hint)."""
    rows = store.conn.execute(
        "SELECT a.*, COALESCE(s.role_family, j.role_hint) AS role_family FROM applications a "
        "LEFT JOIN scores s ON s.job_id = a.job_id LEFT JOIN jobs j ON j.id = a.job_id").fetchall()
    return [dict(r) for r in rows]


def _qualifying_today(store: Store, exclude: str) -> int:
    today = date.today().isoformat()
    return store.conn.execute("SELECT COUNT(*) FROM apply_queue WHERE shadow_qualified=1 AND substr(queued_at,1,10)=? "
                              "AND job_id != ?", (today, exclude)).fetchone()[0]


def _bank(master: dict) -> AnswerBank:
    return load_bank(tokens=profile_tokens(master))


# ---- queue ------------------------------------------------------------------------------------------------------
def queue_jobs(cfg: dict, store: Store, job_ids: list[str] | None = None, echo=print) -> list[dict]:
    from playwright.sync_api import sync_playwright
    from src.resume import load_master

    acfg = cfg["apply"]
    check_config(acfg)
    bank = _bank(load_master(ROOT / "data" / "master_resume.yaml"))
    if job_ids:
        jobs = [store.conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone() for j in job_ids]
        jobs = [j for j in jobs if j is not None]
    else:
        jobs = [j for j in store.jobs_by_status("tailored")
                if not (store.application(j["id"]) and store.application(j["id"])["status"] in DONE)]
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for job in jobs:
            res = _queue_one(browser, job, store, bank, acfg)
            results.append(res)
            echo(_queue_line(res))
        browser.close()
    return results


def _queue_one(browser, job, store: Store, bank: AnswerBank, acfg: dict) -> dict:
    url, ats = form_url(job["url"]), ats_of(job["url"])
    pdf = verified_pdf(job)
    res = {"job_id": job["id"], "company": job["company"], "title": job["title"], "ats": ats, "form_url": url}
    if ats is None:
        res["error"] = "not a Greenhouse/Lever/Ashby form"
        return res
    page, requested = browser.new_page(), []
    page.on("request", lambda r: requested.append(r.url))
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
        block_submits(page)
        questions = read_questions(page, ats)
        captcha = detect_captcha(page, requested)
    except Exception as e:      # noqa: BLE001 — a page that won't load is reported, not fatal
        questions, captcha, res["error"] = [], [], f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    finally:
        page.close()
    bank.tokens["@resume"] = str(pdf) if pdf else None
    matches = match_form(questions, bank)
    fit = store.conn.execute("SELECT fit_score, role_family FROM scores WHERE job_id=?", (job["id"],)).fetchone()
    gates = evaluate(fit_score=fit["fit_score"] if fit else None, ats=ats, captcha=captcha, matches=matches,
                     company=job["company"], resume_ok=bool(pdf) and _pdf_pages(pdf) == 1,
                     form_loaded=bool(questions), history=_history(store),
                     qualifying_today=_qualifying_today(store, job["id"]), cfg=acfg, job_id=job["id"],
                     role_family=(fit["role_family"] if fit else None) or job["role_hint"])
    ok = qualifies(gates)
    answered = sum(m.status in (ANSWERED, SKIPPED) for m in matches)
    store.save_queue({"job_id": job["id"], "ats": ats, "form_url": url,
                      "questions": json.dumps([q.to_dict() for q in questions]),
                      "matches": json.dumps([asdict(m) for m in matches], default=str),
                      "captcha": json.dumps(captcha), "gates": json.dumps([asdict(g) for g in gates]),
                      "shadow_qualified": int(ok), "answered": answered, "total": len(matches),
                      "queued_at": datetime.now(timezone.utc).isoformat()})
    prev = store.application(job["id"])
    store.upsert_application({**application_fields(store, job["id"]),
                              "status": prev["status"] if prev and prev["status"] in DONE else "queued",
                              "shadow_qualified": int(ok), "shadow_blockers": json.dumps(blockers(gates))})
    res.update({"questions": len(matches), "answered": answered, "matches": matches, "gates": gates,
                "shadow_qualified": ok, "captcha": captcha})
    log_event({"event": "queued", "job_id": job["id"], "company": job["company"], "ats": ats,
               "answered": answered, "total": len(matches), "shadow_qualified": ok, "blockers": blockers(gates),
               "unanswered": [m.label for m in matches if m.status not in (ANSWERED, SKIPPED)]})
    return res


def _queue_line(r: dict) -> str:
    if "error" in r and "questions" not in r:
        return f"  ! {r['company']}: {r['error']}"
    head = f"  {r['company'][:14]:14s} {r['title'][:36]:36s} {r['ats']:10s} {r['answered']:2d}/{r['questions']:<2d} answered"
    shadow = "would qualify" if r["shadow_qualified"] else "review only"
    return f"{head}  shadow: {shadow}"


def queue_report(results: list[dict]) -> str:
    """Per-job detail: which questions need Nithya, and which gates blocked auto-submit."""
    out = []
    for r in results:
        if "matches" not in r:
            continue
        out.append(f"\n{r['company']} — {r['title']}  ({r['form_url']})")
        for m in r["matches"]:
            if m.status in (ANSWERED, SKIPPED):
                continue
            req = "required" if m.required else "optional"
            out.append(f"    needs you: {m.label[:70]!r} [{m.kind}, {req}] — {m.status}: {m.detail}")
        out.append("    shadow gates: " + ("ALL PASS" if r["shadow_qualified"] else
                                            " | ".join(f"{g.name} ✗ ({g.detail})" for g in r["gates"] if not g.passed)))
    return "\n".join(out)


# ---- review -----------------------------------------------------------------------------------------------------
def next_job_id(store: Store, cfg: dict) -> str | None:
    """Next queued application, in the same order as the score ranking (scorer.rank_results)."""
    from src.scorer import rank_results

    rows = store.conn.execute(
        "SELECT a.job_id, j.company, j.location, j.remote, j.posted_at, s.fit_score, s.sponsorship_signal "
        "FROM applications a JOIN apply_queue q ON q.job_id = a.job_id JOIN jobs j ON j.id = a.job_id "
        "JOIN scores s ON s.job_id = a.job_id WHERE a.status = 'queued'").fetchall()
    rows = rank_results([dict(r) for r in rows], cfg)
    return rows[0]["job_id"] if rows else None


def classify(prefilled: dict, final: dict, filled_keys: set[str], labels: dict[str, str]) -> dict:
    """Compare the form right after pre-fill with the form at Submit time."""
    def norm(v):
        return (str(v).strip() if v not in (None, "") else None)
    edited = [k for k in filled_keys if norm(final.get(k)) != norm(prefilled.get(k))]
    by_you = [k for k in labels if k not in filled_keys and norm(final.get(k)) and
              norm(final.get(k)) != norm(prefilled.get(k))]
    return {"outcome": "approved_edited" if edited else "approved_unchanged",
            "edited": [labels[k] for k in edited], "edited_keys": edited,
            "filled_by_you": [labels[k] for k in by_you]}


def review(cfg: dict, store: Store, job_id: str | None = None, echo=print, ask=input,
           interactive: bool | None = None) -> dict | None:
    from playwright.sync_api import sync_playwright
    from src.resume import load_master

    check_config(cfg["apply"])
    interactive = sys.stdin.isatty() if interactive is None else interactive
    job_id = job_id or next_job_id(store, cfg)
    if not job_id:
        echo("Review queue is empty. Run `python main.py queue` first.")
        return None
    job = store.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if job is None:
        echo(f"Unknown job id: {job_id}")
        return None
    pdf = verified_pdf(job)
    bank = _bank(load_master(ROOT / "data" / "master_resume.yaml"))
    bank.tokens["@resume"] = str(pdf) if pdf else None
    url, ats = form_url(job["url"]), ats_of(job["url"])
    echo(f"Opening {job['company']} — {job['title']}\n  {url}")
    if not pdf:
        echo("  ! no verified PDF for this job; the resume field will be left for you")

    captured: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.expose_binding("__jaOnSubmitPy", lambda source, values: captured.update(values=values, at=_now()))
        page.goto(url, wait_until="networkidle", timeout=60000)
        block_submits(page)
        questions = read_questions(page, ats)
        matches = match_form(questions, bank)
        by_key = {q.key: q for q in questions}

        filled, problems = set(), {}
        for m in matches:
            if m.fillable and m.status in (ANSWERED, UNVERIFIED):
                ok, note = fill(page, by_key[m.key], m.value)
                if ok:
                    filled.add(m.key)
                else:
                    problems[m.key] = note
        prefilled = snapshot(page, questions)
        todo = {m.key for m in matches if m.key not in filled and m.status != SKIPPED}
        highlight(page, questions, todo,
                  f"Job agent pre-filled {len(filled)} of {len(questions)} questions. Outlined fields need you. "
                  f"Review everything, then click Submit yourself — or close this window to skip.")
        page.evaluate("(qs) => { window.__jaQs = qs; window.__jaSnapshot = " + _snapshot_fn() + "; "
                      "window.__jaOnSubmit = () => window.__jaOnSubmitPy(window.__jaSnapshot(window.__jaQs)); }",
                      [q.to_dict() for q in questions])
        hand_over(page)
        log_event({"event": "prefilled", "job_id": job_id, "company": job["company"], "filled": len(filled),
                   "total": len(questions), "left_for_you": [by_key[k].label for k in sorted(todo)],
                   "fill_problems": {by_key[k].label: v for k, v in problems.items()}})
        echo(f"  pre-filled {len(filled)}/{len(questions)}; {len(todo)} outlined for you. "
             f"Close the browser window when you're done.")

        last = prefilled
        while True:
            try:
                page.wait_for_timeout(1000)
                if page.url.split("?")[0].rstrip("/") == url.rstrip("/"):
                    snap = snapshot(page, questions)
                    if any(v is not None for v in snap.values()):
                        last = snap
            except Exception:  # noqa: BLE001 — window closed, or the page navigated mid-snapshot
                if page.is_closed() or not browser.is_connected():
                    break
        try:
            browser.close()
        except Exception:      # noqa: BLE001
            pass

    labels = {q.key: q.label for q in questions}
    submitted = bool(captured)
    final = captured.get("values") or last
    result = classify(prefilled, final, filled, labels)
    bank_edits = sorted({m.entry_id for m in matches if m.key in result["edited_keys"] and m.entry_id})
    fields = application_fields(store, job_id)
    record = {"job_id": job_id, "company": job["company"], "submit_clicked": submitted, **result,
              "changed_bank_entries": bank_edits}
    if not submitted:
        record["outcome"] = "abandoned"
    log_event({"event": "review_closed", **record})

    status = "queued"
    if submitted:
        echo(f"  Submit clicked. Outcome: {result['outcome']}"
             + (f" — you changed: {', '.join(result['edited'])}" if result["edited"] else ""))
        if bank_edits:
            echo(f"  You changed answers that came from the bank ({', '.join(bank_edits)}); "
                 f"consider updating data/answer_bank.yaml.")
        if interactive and ask("  Did you see the employer's confirmation that it was submitted? [y/N] ").strip().lower() == "y":
            status = "applied"
        elif not interactive:
            echo(f"  Once you see the confirmation, run: python main.py confirm {job_id} --applied")
        else:
            record["outcome"] = "submit_failed"
    else:
        echo("  Closed without submitting.")
        if interactive and ask("  Skip this job (remove it from the queue)? [y/N] ").strip().lower() == "y":
            status = "skipped"
    store.upsert_application({**fields, "status": status, "review_outcome": record["outcome"],
                              "edited_fields": json.dumps(result["edited"]),
                              "date_applied": date.today().isoformat() if status == "applied" else None,
                              "notes": _notes(result, bank_edits)})
    if status != "queued":
        log_event({"event": status, "job_id": job_id, "company": job["company"]})
    return {**record, "status": status}


def _notes(result: dict, bank_edits: list[str]) -> str:
    parts = []
    if result["filled_by_you"]:
        parts.append("you filled: " + "; ".join(result["filled_by_you"])[:300])
    if bank_edits:
        parts.append("changed bank answers: " + ", ".join(bank_edits))
    return " | ".join(parts)


def _snapshot_fn() -> str:
    from src.forms import _SNAPSHOT_JS
    return _SNAPSHOT_JS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def confirm(store: Store, job_id: str, applied: bool, note: str = "") -> str:
    """Mark a job applied (or skipped). Works for jobs reviewed here and for ones applied to entirely by hand."""
    fields = application_fields(store, job_id)
    if fields is None:
        raise ValueError(f"unknown job id {job_id}")
    prev = store.application(job_id)
    outcome = prev["review_outcome"] if prev and prev["review_outcome"] else "manual"
    if applied and outcome in ("abandoned", "submit_failed"):
        outcome = "manual"
    status = "applied" if applied else "skipped"
    row = {**fields, "status": status, "review_outcome": outcome if applied else (prev["review_outcome"] if prev else None),
           "date_applied": date.today().isoformat() if applied else None}
    if note:
        row["notes"] = note
    store.upsert_application(row)
    log_event({"event": status, "job_id": job_id, "company": fields["company"], "outcome": row["review_outcome"]})
    return status
