"""Phase 4: the daily agent loop — a plain tool-use loop, no framework.

The model (fast tier) decides the order of tool calls; CODE enforces every limit, so a confused model can waste a
turn but can never overspend, loop forever, tailor a weak match, or apply anywhere:
  - hard cost cap (agent.max_run_cost_usd): model turns + scoring + tailoring all draw from one budget, and each
    paid step is pre-checked against a projected cost before it starts
  - at most agent.max_tool_calls tool calls per run
  - a tool that fails agent.max_consecutive_failures times in a row stops the run
  - tailor_resume refuses jobs below agent.tailor_min_score, already-tailored jobs, and more than
    agent.max_tailor_per_run per run
  - there is no apply tool; log_application cannot write status "applied"
Every model turn and tool call is logged (name, args, latency, tokens, cost) to logs/run_<timestamp>.jsonl.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.llm import LLMProvider, ToolSpec, cost_usd, estimate_tokens
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

SYSTEM = """You run Nithya's daily job-search cycle by calling tools. Do exactly this, in order:
1. Call search_jobs once.
2. Call score_jobs once.
3. For each job in score_jobs' tailor_candidates (already filtered to fit_score >= the minimum and in the
   required order), call tailor_resume in exactly that order, at most {max_tailor} times in total.
4. After each tailor_resume, call log_application for that job: status "to_review" if its status starts with PASS,
   otherwise "skipped" with a short note on why.
5. Then reply with a 2-4 sentence summary and no tool call. That ends the run.
Never call search_jobs or score_jobs twice. You cannot apply to jobs; Nithya reviews and applies herself.
If a tool returns an error, read it: retry once only if the error suggests a retry could work, otherwise move on.
If a tool says a limit or budget was reached, stop calling that tool."""

TOOLS = [
    ToolSpec("search_jobs", "Fetch postings from every configured job board, apply the hard filters, and save new "
             "jobs to the database. Free (no LLM calls). Returns counts.",
             {"type": "object", "properties": {}, "additionalProperties": False}),
    ToolSpec("score_jobs", "LLM-score the most relevant unscored jobs (fast model, strong model for promising ones). "
             "Returns the scored jobs and tailor_candidates: untailored jobs at or above the minimum fit score.",
             {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1,
                                                         "description": "max jobs to score this run"}},
              "additionalProperties": False}),
    ToolSpec("tailor_resume", "Tailor and verify a one-page resume PDF for one scored job. Only jobs from "
             "tailor_candidates are allowed. Returns PASS/FAIL, the PDF file, and the cost.",
             {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"],
              "additionalProperties": False}),
    ToolSpec("log_application", "Record a job in the application log for Nithya's review (synced to Google Sheets "
             "in Phase 5). Does not apply.",
             {"type": "object", "properties": {
                 "job_id": {"type": "string"},
                 "status": {"type": "string", "enum": ["to_review", "skipped"]},
                 "notes": {"type": "string"}},
              "required": ["job_id", "status"], "additionalProperties": False}),
]


class ToolError(Exception):
    """A tool could not do what was asked (bad id, guard refused, budget too low...). Reported to the model."""


# ---- run state + logging ----------------------------------------------------------------------------------------
@dataclass
class Limits:
    max_cost: float
    max_tool_calls: int
    max_consecutive_failures: int
    tailor_min_score: int
    max_tailor: int
    score_limit: int
    max_score_cost: float
    max_turns: int
    agent_reserve: float            # kept back from tools so the model can always finish its turns

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Limits":
        a = cfg["agent"]
        return cls(a["max_run_cost_usd"], a["max_tool_calls"], a["max_consecutive_failures"], a["tailor_min_score"],
                   a["max_tailor_per_run"], a["score_limit"], a["max_score_cost_usd"], a["max_turns"],
                   a["agent_reserve_usd"])


@dataclass
class RunState:
    limits: Limits
    spent: float = 0.0
    tool_calls: int = 0
    turns: int = 0
    tailored: list[dict] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)      # consecutive failures per tool
    events: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    final_text: str = ""

    @property
    def remaining(self) -> float:
        return self.limits.max_cost - self.spent


class RunLog:
    """One JSON line per model turn / tool call."""

    def __init__(self, log_dir: Path = LOG_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"run_{self.stamp}.jsonl"

    def write(self, record: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ---- tools ------------------------------------------------------------------------------------------------------
@dataclass
class ToolOutput:
    result: dict
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class Toolbox:
    """The real tools. Each one takes the run state (for budget/limit checks) and returns a ToolOutput."""

    def __init__(self, cfg: dict, store: Store, llm: LLMProvider):
        self.cfg, self.store, self.llm = cfg, store, llm
        self._ctx = None

    @property
    def ctx(self):
        if self._ctx is None:
            from src.pipeline import tailor_context
            self._ctx = tailor_context()
        return self._ctx

    def search_jobs(self, state: RunState) -> ToolOutput:
        from src.pipeline import run_search

        res = run_search(self.cfg, self.store)
        return ToolOutput({"fetched": res.fetched, "passed_filters": res.kept, "new_jobs": len(res.new),
                           "unscored_in_db": len(self.store.unscored_jobs()), "board_errors": len(res.errors)})

    def score_jobs(self, state: RunState, limit: int | None = None) -> ToolOutput:
        from src.resume import load_master, profile_text
        from src.scorer import SYSTEM_TEMPLATE, estimate, score_jobs, select_jobs
        from src.sponsorship import SponsorshipDB

        lim = state.limits
        limit = min(limit or lim.score_limit, lim.score_limit)
        cap = min(lim.max_score_cost, state.remaining - lim.agent_reserve)
        jobs = select_jobs(self.store.unscored_jobs(), self.cfg, limit)
        out = ToolOutput({"scored": [], "tailor_candidates": self.tailor_candidates()})
        if not jobs:
            out.result["note"] = "no unscored jobs"
            return out
        system = SYSTEM_TEMPLATE.format(profile=profile_text(load_master(ROOT / "data" / "master_resume.yaml")))
        est = estimate(jobs, system, self.llm, self.cfg)
        while jobs and est.expected > cap:          # trim to what the budget can carry, rather than refusing
            jobs = jobs[:-1]
            est = estimate(jobs, system, self.llm, self.cfg)
        if not jobs:
            raise ToolError(f"budget too low to score even one job (scoring budget ${cap:.3f})")
        res = score_jobs(jobs, system, self.llm, self.cfg, self.store, SponsorshipDB(ROOT / "data" / "uscis"), est,
                         cap=cap, quiet=True)
        out.input_tokens, out.output_tokens, out.cost = res.input_tokens, res.output_tokens, res.spent
        out.result = {
            "scored": [{"job_id": r["job_id"], "company": r["company"], "title": r["title"],
                        "fit_score": r["fit_score"]} for r in sorted(res.scored, key=lambda r: -r["fit_score"])],
            # Stated explicitly so the model's summary can't confuse "asked for" with "scored".
            "requested": limit, "scored_count": len(res.scored),
            "note": (f"scored {len(res.scored)} of {limit} requested: trimmed to fit the ${cap:.2f} scoring budget"
                     if len(res.scored) < limit else f"scored {len(res.scored)} as requested"),
            "stopped_early": res.stopped_early,
            "tailor_candidates": self.tailor_candidates(),
        }
        return out

    def tailor_candidates(self, exclude: set[str] = frozenset()) -> list[dict]:
        """Untailored jobs at or above the minimum fit, in tailoring order (scorer.tailor_order)."""
        from src.scorer import tailor_order

        min_score = self.cfg["agent"]["tailor_min_score"]
        rows = [dict(r) for r in self.store.scored_rows()
                if r["status"] == "scored" and r["fit_score"] >= min_score and r["id"] not in exclude]
        return [{"job_id": r["id"], "company": r["company"], "title": r["title"], "fit_score": r["fit_score"],
                 "sponsorship": r["sponsorship_signal"] or "unclear"} for r in tailor_order(rows, self.cfg)]

    def tailor_resume(self, state: RunState, job_id: str) -> ToolOutput:
        from src.pipeline import tailor_and_publish, tailor_estimate

        lim = state.limits
        if len(state.tailored) >= lim.max_tailor:
            raise ToolError(f"limit reached: already tailored {lim.max_tailor} resumes this run")
        job = self.store.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise ToolError(f"unknown job_id {job_id!r}")
        sc = self.store.conn.execute("SELECT fit_score FROM scores WHERE job_id=?", (job_id,)).fetchone()
        if sc is None or sc["fit_score"] < lim.tailor_min_score:
            raise ToolError(f"refused: fit_score {sc['fit_score'] if sc else 'none'} is below {lim.tailor_min_score}")
        if job["status"] == "tailored":
            raise ToolError("refused: this job already has a tailored resume")
        # Tailoring order is enforced here, not left to the model: only the next candidate may be tailored.
        attempted = {t["job_id"] for t in state.tailored}
        nxt = self.tailor_candidates(exclude=attempted)
        if nxt and nxt[0]["job_id"] != job_id:
            raise ToolError(f"refused: tailor candidates in order; next is {nxt[0]['job_id']} "
                            f"({nxt[0]['company']}, fit {nxt[0]['fit_score']}, sponsorship {nxt[0]['sponsorship']})")
        # Budget: tailor_job checks its budget before each attempt, so one attempt can run past it. Hold back a
        # projected attempt (2x the estimate) so even that last attempt stays under the run cap.
        attempt = 2 * tailor_estimate(job, self.store, self.ctx, self.llm, self.cfg)
        budget = state.remaining - lim.agent_reserve - attempt
        if budget < attempt / 2:
            raise ToolError(f"budget too low to tailor (${state.remaining:.3f} left of the run cap)")
        o = tailor_and_publish(job, self.store, self.ctx, self.llm, self.cfg, budget_left=budget, pdf=True)
        r = o.result
        pdf = next((f for f in o.files if ".pdf" in f), None)
        state.tailored.append({"job_id": job_id, "company": job["company"], "title": job["title"],
                               "status": r.status, "pdf": pdf, "pages": o.pages, "cost": r.cost})
        return ToolOutput({"status": r.status, "attempts": r.attempts, "removed_bullets": len(r.removed),
                           "pdf": pdf, "pages": o.pages, "one_page_cuts": len(o.cuts),
                           "remaining_problems": [str(i) for i in r.verification.issues][:5]},
                          r.input_tokens, r.output_tokens, r.cost)

    def log_application(self, state: RunState, job_id: str, status: str, notes: str = "") -> ToolOutput:
        from src.apply import application_fields

        if status not in ("to_review", "skipped"):        # 'applied' is only ever set by Nithya (Phase 5)
            raise ToolError(f"status must be to_review or skipped, not {status!r}")
        fields = application_fields(self.store, job_id)
        if fields is None:
            raise ToolError(f"unknown job_id {job_id!r}")
        prev = self.store.application(job_id)
        if prev and prev["status"] in ("queued", "applied", "skipped"):   # never move a reviewed job backwards
            return ToolOutput({"logged": job_id, "status": prev["status"], "note": "already further along"})
        self.store.upsert_application({**fields, "status": status, "notes": notes})
        return ToolOutput({"logged": job_id, "status": status})


# ---- the loop ---------------------------------------------------------------------------------------------------
def run_agent(cfg: dict, llm: LLMProvider, toolbox, log: RunLog, echo=print) -> RunState:
    lim = Limits.from_cfg(cfg)
    state = RunState(lim)
    acfg, pricing = cfg["agent"], cfg["llm"].get("pricing", {})
    tier = acfg.get("tier", "fast")
    system = SYSTEM.format(max_tailor=lim.max_tailor)
    tools_json = json.dumps([t.__dict__ for t in TOOLS])
    messages: list[dict] = [{"role": "user", "content":
                             f"Run today's cycle. Score up to {lim.score_limit} jobs; tailor jobs with fit_score >= "
                             f"{lim.tailor_min_score}, at most {lim.max_tailor}."}]
    worst_turn = 0.0

    def stop(reason: str) -> None:
        state.stop_reason = reason
        echo(f"  STOP: {reason}")

    while state.stop_reason is None:
        if state.turns >= lim.max_turns:
            stop(f"model turn limit ({lim.max_turns}) reached"); break
        # Project this turn's cost from the conversation so far; never start a turn the budget can't cover.
        est_in = estimate_tokens(system + tools_json + json.dumps(messages, default=str)) + 500
        projected = max(2 * (cost_usd(llm.models[tier], est_in, acfg["est_output_tokens"], pricing) or 0), worst_turn)
        if state.spent + projected > lim.max_cost:
            stop(f"cost cap ${lim.max_cost:.2f} reached (${state.spent:.4f} spent)"); break
        resp = llm.chat(messages, TOOLS, tier=tier, system=system)
        cost = cost_usd(resp.model, resp.input_tokens, resp.output_tokens, pricing) or 0.0
        state.spent += cost
        state.turns += 1
        worst_turn = max(worst_turn, cost)
        log.write({"kind": "model_turn", "name": resp.model, "args": {"turn": state.turns},
                   "latency_s": round(resp.latency_s, 3), "input_tokens": resp.input_tokens,
                   "output_tokens": resp.output_tokens, "cost_usd": round(cost, 6), "ok": True,
                   "stop_reason": resp.stop_reason, "tool_calls": [c.name for c in resp.tool_calls],
                   "text": resp.text[:500]})
        messages.append(resp.as_message())
        if not resp.tool_calls:
            state.final_text = resp.text
            state.stop_reason = "done"
            break

        results = []
        for call in resp.tool_calls:
            if state.tool_calls >= lim.max_tool_calls:
                stop(f"tool call limit ({lim.max_tool_calls}) reached"); break
            # Paid tools pre-check their own projected cost against state.remaining (score_jobs trims, tailor_resume
            # refuses); this is the backstop: once only the reserve is left, no tool starts at all.
            if state.remaining <= lim.agent_reserve:
                stop(f"cost cap ${lim.max_cost:.2f} reached (${state.spent:.4f} spent)"); break
            state.tool_calls += 1
            content, is_error, out, err = _execute(toolbox, state, call)
            state.spent += out.cost if out else 0.0
            ok = not is_error
            state.failures[call.name] = 0 if ok else state.failures.get(call.name, 0) + 1
            event = {"kind": "tool_call", "name": call.name, "args": call.args, "latency_s": content["latency_s"],
                     "input_tokens": out.input_tokens if out else 0, "output_tokens": out.output_tokens if out else 0,
                     "cost_usd": round(out.cost, 6) if out else 0.0, "ok": ok, "error": err,
                     "result": content["result"]}
            log.write(event)
            state.events.append(event)
            echo(f"  [{state.tool_calls:2d}] {call.name}({_short(call.args)}) -> "
                 f"{'ERROR ' + err if err else _short(content['result'])}  ${event['cost_usd']:.4f}  "
                 f"(run total ${state.spent:.4f})")
            results.append({"id": call.id, "name": call.name, "content": json.dumps(content["result"], default=str),
                            "is_error": is_error})
            if state.failures[call.name] >= lim.max_consecutive_failures:
                stop(f"{call.name} failed {state.failures[call.name]} times in a row: {err}"); break
            if state.spent >= lim.max_cost:
                stop(f"cost cap ${lim.max_cost:.2f} reached (${state.spent:.4f} spent)"); break
        messages.append({"role": "tool", "results": results})
    return state


def _execute(toolbox, state: RunState, call) -> tuple[dict, bool, ToolOutput | None, str | None]:
    """Run one tool call. Tool exceptions become an error result for the model, never a crash of the loop."""
    fn = getattr(toolbox, call.name, None) if call.name in {t.name for t in TOOLS} else None
    start = time.monotonic()
    try:
        if fn is None:
            raise ToolError(f"unknown tool {call.name!r}")
        out = fn(state, **call.args)
        return {"result": out.result, "latency_s": round(time.monotonic() - start, 3)}, False, out, None
    except TypeError as e:            # bad arguments from the model
        err = f"bad arguments: {e}"
    except ToolError as e:
        err = str(e)
    except Exception as e:            # noqa: BLE001 — any other tool crash is a failure, reported and counted
        err = f"{type(e).__name__}: {e}"
    return {"result": {"error": err}, "latency_s": round(time.monotonic() - start, 3)}, True, None, err


def _short(obj, n: int = 110) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def summary_markdown(state: RunState, log: RunLog) -> str:
    lim = state.limits
    lines = [f"# Agent run {log.stamp}", "",
             f"- **Ended:** {state.stop_reason}",
             f"- **Cost:** ${state.spent:.4f} of ${lim.max_cost:.2f} cap",
             f"- **Tool calls:** {state.tool_calls} of {lim.max_tool_calls}  |  **Model turns:** {state.turns}",
             f"- **Log:** {log.path.name}", "", "## Tool calls", "",
             "| # | Tool | Args | OK | Latency | Tokens in/out | Cost |", "|---|---|---|---|---|---|---|"]
    for i, e in enumerate(state.events, 1):
        lines.append(f"| {i} | {e['name']} | `{_short(e['args'], 60)}` | {'yes' if e['ok'] else 'NO: ' + str(e['error'])} "
                     f"| {e['latency_s']:.1f}s | {e['input_tokens']}/{e['output_tokens']} | ${e['cost_usd']:.4f} |")
    lines += ["", "## Resumes tailored", ""]
    lines += [f"- {t['status']}: {t['company']} — {t['title']} -> {t['pdf'] or 'no PDF'} (${t['cost']:.4f})"
              for t in state.tailored] or ["- none"]
    if state.final_text:
        lines += ["", "## Agent summary", "", state.final_text]
    return "\n".join(lines) + "\n"


# ---- dry run ----------------------------------------------------------------------------------------------------
def plan(cfg: dict, store: Store, llm: LLMProvider) -> str:
    """What `run` would do, with cost estimates. Fetches the boards (free) but writes nothing and makes no LLM calls."""
    from src.pipeline import run_search, tailor_context, tailor_estimate
    from src.resume import load_master, profile_text
    from src.scorer import SYSTEM_TEMPLATE, estimate, select_jobs

    lim, acfg = Limits.from_cfg(cfg), cfg["agent"]
    pricing = cfg["llm"].get("pricing", {})
    search = run_search(cfg, store, save=False)

    # Jobs that would be eligible for scoring = unscored rows already in the DB that survive search's re-check
    # against the current filters/company list, + the new jobs search would save.
    from src.pipeline import job_filter, job_from_row
    jf, configured = job_filter(cfg), {c["name"] for c in cfg["companies"]}
    pool = [dict(r) for r in store.unscored_jobs()
            if (r["source"] == "jsearch" or r["company"] in configured) and jf.check(job_from_row(r)).keep]
    in_db = len(pool)
    pool += [{"id": j.id, "company": j.company, "title": j.title, "location": j.location, "remote": int(j.remote),
              "description": j.description, "role_hint": search.hints[j.id],
              "posted_at": j.posted_at.isoformat() if j.posted_at else None} for j in search.new]
    to_score = select_jobs(pool, cfg, lim.score_limit)
    system = SYSTEM_TEMPLATE.format(profile=profile_text(load_master(ROOT / "data" / "master_resume.yaml")))
    score_cap = min(lim.max_score_cost, lim.max_cost - lim.agent_reserve)
    est = estimate(to_score, system, llm, cfg)
    while to_score and est.expected > score_cap:
        to_score = to_score[:-1]
        est = estimate(to_score, system, llm, cfg)

    # Tailoring: backlog of untailored 65+ jobs, plus the expected number of new 65+ scores at the historical rate.
    backlog = [r for r in store.scored_rows() if r["status"] == "scored" and r["fit_score"] >= lim.tailor_min_score]
    hist = store.conn.execute("SELECT COUNT(*), SUM(fit_score >= ?) FROM scores", (lim.tailor_min_score,)).fetchone()
    rate = (hist[1] or 0) / hist[0] if hist[0] else 0.1
    expected_new = math.ceil(len(to_score) * rate)
    n_tailor = min(lim.max_tailor, len(backlog) + expected_new)
    ctx = tailor_context()
    typical = [tailor_estimate(r, store, ctx, llm, cfg) for r in store.jobs_by_status("tailored")]
    one_attempt = sum(typical) / len(typical) if typical else 0.06
    tailor_expected, tailor_worst = n_tailor * one_attempt * 1.5, lim.max_tailor * one_attempt * cfg["tailoring"]["max_attempts"]

    # Model turns: ~search, score, then tailor + log per job, plus the final reply; context grows each turn.
    turns = 3 + 2 * n_tailor
    tier = acfg.get("tier", "fast")
    per_turn = cost_usd(llm.models[tier], acfg["est_turn_input_tokens"], acfg["est_output_tokens"], pricing) or 0
    agent_cost = turns * per_turn
    tool_calls = 2 + 2 * n_tailor
    total_expected = est.expected + tailor_expected + agent_cost

    L = ["DRY RUN — no LLM calls made, nothing written to the database.", "",
         "Limits (enforced in code):",
         f"  cost cap ${lim.max_cost:.2f} per run | max {lim.max_tool_calls} tool calls | stop if a tool fails "
         f"{lim.max_consecutive_failures}x in a row",
         f"  tailor only fit_score >= {lim.tailor_min_score}, max {lim.max_tailor} per run | no apply tool | "
         f"agent model: {llm.models[tier]}", "",
         "Plan:",
         f"  1. search_jobs: {search.fetched} fetched from {search.boards} boards ({len(search.errors)} errors), "
         f"{search.kept} pass filters, {len(search.new)} would be new",
         f"  2. score_jobs: {len(to_score)} jobs ({in_db} unscored already in DB + "
         f"{len(search.new)} new, top {lim.score_limit} by relevance; scoring budget ${score_cap:.2f})"]
    for i, j in enumerate(to_score, 1):
        L.append(f"       {i:2d}. {j['company'][:18]:18s} {j['title'][:55]:55s} [{j['role_hint']}]")
    L += [f"  3. tailor_resume: {len(backlog)} untailored job(s) already at {lim.tailor_min_score}+; at the historical "
          f"rate ({rate:.0%} of scored jobs reach {lim.tailor_min_score}+) expect ~{expected_new} new -> plan for "
          f"{n_tailor} (max {lim.max_tailor})",
          f"  4. log_application: one row per tailored job (status to_review, or skipped if verification fails)",
          f"  5. agent replies with a summary and stops (~{turns} model turns, ~{tool_calls} tool calls)", "",
          "Estimated cost:",
          f"  scoring        ~${est.expected:.3f}  (worst ${est.worst:.3f} if every job escalates)",
          f"  tailoring      ~${tailor_expected:.3f}  for {n_tailor} job(s) at ~1.5 attempts "
          f"(worst ${tailor_worst:.3f}: {lim.max_tailor} jobs x {cfg['tailoring']['max_attempts']} attempts)",
          f"  agent turns    ~${agent_cost:.3f}",
          f"  TOTAL          ~${total_expected:.3f} expected  |  hard cap ${lim.max_cost:.2f} "
          f"(worst cases are cut off by the cap, never exceeded)"]
    return "\n".join(L)
