"""MCP server: lets an MCP client (Claude Desktop, Claude Code) ask about the pipeline's state.

Read-mostly by construction, not by convention:
- jobs.db is opened read-only (Store.read_only, SQLite mode=ro): any write raises.
- No tool calls the LLM, tailoring, form-filling or apply code (tests/test_mcp_server.py makes each of them raise),
  and .env is never loaded, so there is no API key in this process to spend and no tool path that fills a form.
- run_search_dry_run fetches the public job boards (network, no keys) and filters them, but runs with save=False:
  no database writes and never JSearch (its quota is spent only by real runs).

Run:  .venv/Scripts/python.exe src/mcp_server.py      (stdio transport; see README "MCP server")
"""
import sys
from datetime import date as Date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as a script by an MCP client, from any working directory
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from src.store import Store  # noqa: E402

DB_PATH = ROOT / "data" / "jobs.db"
OUT_DIR = ROOT / "output"
MAX_RANKED = 50
DESCRIPTION_CHARS = 1500

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
# The dry-run search reads the public job boards, so it is "open world" (results change), but still read-only.
READ_ONLY_WEB = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True)


def _config() -> dict:
    from main import load_config
    return load_config()


def _store(db_path: Path) -> Store:
    return Store.read_only(db_path)


def _ranked(store: Store, cfg: dict, now: datetime | None = None) -> list[dict]:
    """Same ranking as the digest: scored/unverified/tailored jobs, not closed or done, stale ones hidden."""
    from src.digest import CLOSED
    from src.scorer import hide_stale, is_fresh, rank_results
    apps = {r["job_id"]: r["status"] for r in store.applications()}
    rows = [dict(r) for r in store.scored_rows()
            if r["status"] in ("scored", "unverified", "tailored") and apps.get(r["id"]) not in CLOSED]
    rows, _ = hide_stale(rows, cfg, now, apps)
    out = []
    for i, r in enumerate(rank_results(rows, cfg, now), 1):
        out.append({"rank": i, "job_id": r["id"], "company": r["company"], "title": r["title"],
                    "fit_score": r["fit_score"], "role_family": r["role_family"] or r["role_hint"],
                    "fresh": is_fresh(r, cfg, now), "posted_at": r["posted_at"], "location": r["location"],
                    "job_status": r["status"], "application_status": apps.get(r["id"]),
                    "reason": r["one_line_reason"], "url": r["url"]})
    return out


def build_server(db_path: Path = DB_PATH, out_dir: Path = OUT_DIR, config=_config) -> MCPServer:
    """Factory so tests can point the server at a temporary database and output folder."""
    server = MCPServer("job-search-agent", instructions=(
        "Read-only view of a job-search pipeline: the daily digest, ranked jobs, job details, application "
        "statuses, and a dry-run search. These tools never tailor resumes, call an LLM, spend API credits, "
        "write to the database, or submit applications."))

    @server.tool(annotations=READ_ONLY)
    def get_digest(date: str | None = None) -> str:
        """The daily digest (Markdown) for a date (YYYY-MM-DD); default: the most recent one.
        Reads output/digest_<date>.md written by the daily run; if that day has no file, builds it in memory
        from the database (nothing is written)."""
        if date is None:
            files = sorted(out_dir.glob("digest_*.md"))
            if files:
                return files[-1].read_text(encoding="utf-8")
            day = Date.today()
        else:
            try:
                day = Date.fromisoformat(date)
            except ValueError:
                raise ValueError(f"date must be YYYY-MM-DD, got {date!r}") from None
            f = out_dir / f"digest_{day.isoformat()}.md"
            if f.exists():
                return f.read_text(encoding="utf-8")
        from src.digest import build
        from src.resume import load_master
        store = _store(db_path)
        try:
            return build(store, config(), load_master(ROOT / "data" / "master_resume.yaml"), day)
        finally:
            store.close()

    @server.tool(annotations=READ_ONLY)
    def list_ranked_jobs(limit: int = 10) -> list[dict]:
        """Scored jobs in the pipeline's ranking order (fit, freshness, sponsorship signal, company tier,
        location), excluding closed, applied, skipped and stale postings. limit: 1-50."""
        limit = max(1, min(int(limit), MAX_RANKED))
        store = _store(db_path)
        try:
            return _ranked(store, config())[:limit]
        finally:
            store.close()

    @server.tool(annotations=READ_ONLY)
    def get_job(job_id: str) -> dict:
        """Everything stored about one job: posting, score and reasoning, application, review-queue summary."""
        import json
        store = _store(db_path)
        try:
            job = store.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise ValueError(f"unknown job_id {job_id!r}")
            out = {k: job[k] for k in ("id", "company", "title", "location", "remote", "url", "source",
                                        "posted_at", "status", "role_hint", "first_seen_at", "tailored_at")}
            desc = job["description"] or ""
            out["description"] = desc[:DESCRIPTION_CHARS] + ("…" if len(desc) > DESCRIPTION_CHARS else "")
            sc = store.conn.execute("SELECT * FROM scores WHERE job_id=?", (job_id,)).fetchone()
            out["score"] = None if sc is None else {
                "fit_score": sc["fit_score"], "fast_score": sc["fast_score"], "escalated": bool(sc["escalated"]),
                "model": sc["model"], "role_family": sc["role_family"], "reason": sc["one_line_reason"],
                "matched_skills": json.loads(sc["matched_skills"] or "[]"), "gaps": json.loads(sc["gaps"] or "[]"),
                "sponsorship_signal": sc["sponsorship_signal"], "sponsorship_quote": sc["sponsorship_quote"],
                "seniority_ok": bool(sc["seniority_ok"]), "cost_usd": sc["cost_usd"]}
            app = store.application(job_id)
            out["application"] = None if app is None else {
                k: app[k] for k in ("status", "date_found", "date_applied", "resume_file", "notes",
                                    "review_outcome", "shadow_qualified")}
            q = store.queue_row(job_id)
            # Question counts and gate results only: the pre-filled answer values stay out of the response.
            out["review_queue"] = None if q is None else {
                "ats": q["ats"], "form_url": q["form_url"], "answered": q["answered"], "total": q["total"],
                "captcha": json.loads(q["captcha"] or "[]"), "queued_at": q["queued_at"],
                "gates": [{"name": g["name"], "passed": g["passed"], "detail": g["detail"]}
                          for g in json.loads(q["gates"] or "[]")]}
            return out
        finally:
            store.close()

    @server.tool(annotations=READ_ONLY)
    def get_application_status() -> dict:
        """Every application with its status (queued, applied, interviewing, offer, rejected, skipped, closed),
        newest activity first, plus counts per status."""
        store = _store(db_path)
        try:
            rows = sorted(store.applications(), key=lambda r: r["updated_at"] or "", reverse=True)
            apps = [{"job_id": r["job_id"], "company": r["company"], "role": r["role"], "status": r["status"],
                     "fit_score": r["fit_score"], "date_found": r["date_found"], "date_applied": r["date_applied"],
                     "updated_at": r["updated_at"], "url": r["url"]} for r in rows]
            counts: dict[str, int] = {}
            for a in apps:
                counts[a["status"]] = counts.get(a["status"], 0) + 1
            return {"counts": counts, "applications": apps}
        finally:
            store.close()

    @server.tool(annotations=READ_ONLY_WEB)
    def run_search_dry_run() -> dict:
        """Fetch every configured job board now and apply the filters, without saving anything: how many
        postings were fetched, how many pass, why the rest were filtered out, and which would be new.
        No database writes, no LLM calls, never JSearch. Takes about 20 seconds."""
        from src.pipeline import run_search
        store = _store(db_path)
        try:
            res = run_search(config(), store, save=False)
        finally:
            store.close()
        return {"fetched": res.fetched, "boards": res.boards, "passed_filters": res.kept,
                "would_be_new": len(res.new), "board_errors": len(res.errors),
                "filtered_out_by_reason": dict(res.reasons.most_common()),
                "new_jobs": [{"job_id": j.id, "company": j.company, "title": j.title, "location": j.location,
                              "posted_at": j.posted_at.isoformat() if j.posted_at else None, "url": j.url}
                             for j in res.new[:25]],
                "notes": res.notes, "checked_at": datetime.now(timezone.utc).isoformat()}

    return server


if __name__ == "__main__":
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per board fetch is noise in client logs
    build_server().run()      # stdio: stdout carries the protocol, so nothing here may print
