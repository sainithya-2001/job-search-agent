"""Phase 4: the agent loop's code-enforced limits, the real tools' guards, and provider-neutral tool calling."""
import json
from types import SimpleNamespace

import pytest

from src.agent import TOOLS, Limits, RunLog, RunState, ToolError, Toolbox, ToolOutput, run_agent
from src.llm import AnthropicProvider, ChatResponse, ToolCall
from src.models import Job
from src.store import Store

PRICING = {"m": {"input": 1.0, "output": 5.0}}


def agent_cfg(**over):
    a = {"tier": "fast", "max_run_cost_usd": 0.50, "max_tool_calls": 30, "max_consecutive_failures": 2,
         "max_turns": 25, "score_limit": 15, "max_score_cost_usd": 0.15, "tailor_min_score": 65,
         "max_tailor_per_run": 5, "agent_reserve_usd": 0.02, "est_turn_input_tokens": 4000,
         "est_output_tokens": 400}
    return {"agent": {**a, **over}, "llm": {"pricing": PRICING}}


class ScriptedLLM:
    """Replays a list of turns; each turn is a list of (tool name, args) or a final text string."""

    def __init__(self, turns, tokens=(1000, 100)):
        self.models = {"fast": "m", "strong": "m"}
        self.turns, self.tokens, self.seen = list(turns), tokens, []

    def chat(self, messages, tools, *, tier, system):
        self.seen.append([dict(m) for m in messages])
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            return ChatResponse(turn, [], "m", *self.tokens, "end")
        calls = [ToolCall(f"c{len(self.seen)}_{i}", n, a) for i, (n, a) in enumerate(turn)]
        return ChatResponse("", calls, "m", *self.tokens, "tool_use")


class FakeTools:
    def __init__(self, fail=(), cost=0.0):
        self.fail, self.cost, self.calls = dict.fromkeys(fail, 0), cost, []

    def _do(self, name, **args):
        self.calls.append((name, args))
        if name in self.fail:
            raise RuntimeError(f"{name} broke")
        return ToolOutput({"ok": name}, 10, 5, self.cost)

    def search_jobs(self, state):
        return self._do("search_jobs")

    def score_jobs(self, state, limit=None):
        return self._do("score_jobs", limit=limit)

    def tailor_resume(self, state, job_id):
        return self._do("tailor_resume", job_id=job_id)

    def log_application(self, state, job_id, status, notes=""):
        return self._do("log_application", job_id=job_id, status=status)


def run(tmp_path, llm, tools, **over):
    log = RunLog(tmp_path)
    return run_agent(agent_cfg(**over), llm, tools, log, echo=lambda *a: None), log


def test_full_cycle_logs_every_call(tmp_path):
    llm = ScriptedLLM([[("search_jobs", {})], [("score_jobs", {"limit": 10})],
                       [("tailor_resume", {"job_id": "j1"})],
                       [("log_application", {"job_id": "j1", "status": "to_review"})], "All done."])
    state, log = run(tmp_path, llm, FakeTools(cost=0.01))
    assert state.stop_reason == "done" and state.final_text == "All done."
    assert state.tool_calls == 4 and state.turns == 5
    records = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in records].count("tool_call") == 4
    for r in records:
        assert {"name", "args", "latency_s", "input_tokens", "output_tokens", "cost_usd"} <= r.keys()
    # 5 model turns at 1000 in / 100 out on $1/$5 per M, plus 4 tools at $0.01
    assert state.spent == pytest.approx(5 * 0.0015 + 0.04)
    # tool results go back to the model as one provider-neutral "tool" message per turn
    assert llm.seen[1][-1]["role"] == "tool" and llm.seen[1][-1]["results"][0]["name"] == "search_jobs"


def test_tool_call_limit_stops_the_run(tmp_path):
    tools = FakeTools()
    state, _ = run(tmp_path, ScriptedLLM([[("search_jobs", {})]] * 50), tools, max_tool_calls=5)
    assert state.tool_calls == 5 and len(tools.calls) == 5
    assert "tool call limit" in state.stop_reason


def test_parallel_calls_in_one_turn_also_count_toward_limit(tmp_path):
    tools = FakeTools()
    state, _ = run(tmp_path, ScriptedLLM([[("search_jobs", {})] * 4] * 3), tools, max_tool_calls=6)
    assert len(tools.calls) == 6 and "tool call limit" in state.stop_reason


def test_two_failures_in_a_row_stop_the_run(tmp_path):
    tools = FakeTools(fail=["score_jobs"])
    state, _ = run(tmp_path, ScriptedLLM([[("search_jobs", {})], [("score_jobs", {})], [("score_jobs", {})],
                                          [("search_jobs", {})]]), tools)
    assert "score_jobs failed 2 times in a row" in state.stop_reason
    assert len(tools.calls) == 3           # never reached the 4th turn


def test_failure_counter_resets_after_success(tmp_path):
    class Flaky(FakeTools):
        n = 0

        def search_jobs(self, state):
            Flaky.n += 1
            if Flaky.n % 2:
                raise RuntimeError("flaky")
            return ToolOutput({"ok": 1})

    state, _ = run(tmp_path, ScriptedLLM([[("search_jobs", {})]] * 4 + ["end"]), Flaky())
    assert state.stop_reason == "done"     # fail, ok, fail, ok: never two in a row


def test_unknown_tool_and_bad_args_are_errors_not_crashes(tmp_path):
    llm = ScriptedLLM([[("apply_to_job", {"job_id": "j1"})], [("search_jobs", {"x": 1})], "stopping"])
    state, _ = run(tmp_path, llm, FakeTools())
    results = [m for m in llm.seen[-1] if m["role"] == "tool"]
    assert "unknown tool" in results[0]["results"][0]["content"] and results[0]["results"][0]["is_error"]
    assert "bad arguments" in results[1]["results"][0]["content"]
    assert state.stop_reason == "done"


class BudgetedTools(FakeTools):
    """Like the real paid tools: refuse (ToolError) when the projected cost doesn't fit what's left."""

    def search_jobs(self, state):
        if self.cost > state.remaining - state.limits.agent_reserve:
            raise ToolError("budget too low")
        return self._do("search_jobs")


def test_cost_cap_is_never_exceeded(tmp_path):
    # Each call costs $0.20 against a $0.50 cap: two run, the third refuses itself, and the run never goes over.
    llm = ScriptedLLM([[("search_jobs", {})]] * 10)
    state, _ = run(tmp_path, llm, BudgetedTools(cost=0.20))
    assert state.spent <= 0.50
    assert state.tool_calls >= 3 and len([e for e in state.events if e["ok"]]) == 2


def test_loop_backstop_stops_tools_once_only_reserve_is_left(tmp_path):
    # A tool that doesn't pre-check (cost 0.49) leaves only the reserve: the loop starts no further tool.
    tools = FakeTools(cost=0.49)
    state, _ = run(tmp_path, ScriptedLLM([[("search_jobs", {})]] * 5, tokens=(10, 1)), tools)
    assert len(tools.calls) == 1 and "cost cap" in state.stop_reason


def test_model_turn_refused_when_projection_exceeds_cap(tmp_path):
    state, _ = run(tmp_path, ScriptedLLM(["hi"], tokens=(100_000, 1000)), FakeTools(), max_run_cost_usd=0.001)
    assert state.turns == 0 and "cost cap" in state.stop_reason


def test_no_apply_tool_exists():
    assert not any("apply" in t.name and t.name != "log_application" for t in TOOLS)
    status = next(t for t in TOOLS if t.name == "log_application").parameters["properties"]["status"]
    assert "applied" not in status["enum"]


# ---- real Toolbox guards (no LLM calls: every refusal happens before any paid step) ------------------------------
@pytest.fixture
def db(tmp_path):
    store = Store(tmp_path / "jobs.db")
    jobs = [Job(id=f"j{i}", company="Acme", title=f"AI Engineer {i}", location="Atlanta, GA", remote=False,
                url="https://x", description="d", source="greenhouse") for i in range(3)]
    store.save(jobs, {j.id: "ai_engineer" for j in jobs})
    for jid, fit in (("j0", 80), ("j1", 50)):
        store.save_score({"job_id": jid, "fit_score": fit, "fast_score": fit, "escalated": 0, "model": "m",
                          "role_family": "ai_engineer", "matched_skills": "[]", "gaps": "[]",
                          "sponsorship_signal": "unclear", "uscis_status": "no_data", "scored_at": "2026-10-11"})
    yield store
    store.close()


MASTER = {"approved_summaries": [{"id": "sum_ai", "text": "x", "use_for": {"role_families": ["ai_engineer"]}}]}


def toolbox(store):
    cfg = {**agent_cfg(), "companies": [], "preferences": {"location_priority": []}}
    tb = Toolbox(cfg, store, llm=None)
    tb._master = MASTER                       # tiny stand-in for data/master_resume.yaml (not in git)
    return tb, RunState(Limits.from_cfg(cfg))


def test_tailor_refuses_low_scores_unscored_and_unknown(db):
    tb, state = toolbox(db)
    with pytest.raises(ToolError, match="below 65"):
        tb.tailor_resume(state, "j1")
    with pytest.raises(ToolError, match="below 65"):
        tb.tailor_resume(state, "j2")          # never scored
    with pytest.raises(ToolError, match="unknown job_id"):
        tb.tailor_resume(state, "nope")


def test_tailor_refuses_after_max_per_run_and_already_tailored(db):
    tb, state = toolbox(db)
    db.set_status("j0", "tailored")
    with pytest.raises(ToolError, match="already has a tailored resume"):
        tb.tailor_resume(state, "j0")
    state.tailored = [{}] * 5
    with pytest.raises(ToolError, match="limit reached"):
        tb.tailor_resume(state, "j0")


def test_tailor_order_is_enforced_with_sponsorship_preference(db):
    tb, state = toolbox(db)
    db.save_score({"job_id": "j2", "fit_score": 78, "fast_score": 78, "escalated": 0, "model": "m",
                   "role_family": "ai_engineer", "matched_skills": "[]", "gaps": "[]",
                   "sponsorship_signal": "yes", "uscis_status": "no_data", "scored_at": "2026-10-11"})
    # j0 (80, unclear) vs j2 (78, yes): within 3 points, so j2 goes first and j0 is refused for now
    assert [c["job_id"] for c in tb.tailor_candidates()] == ["j2", "j0"]
    with pytest.raises(ToolError, match="next is j2"):
        tb.tailor_resume(state, "j0")
    state.tailored.append({"job_id": "j2"})                      # once j2 has been attempted, j0 is next
    assert [c["job_id"] for c in tb.tailor_candidates(exclude={"j2"})] == ["j0"]


def test_tailor_candidates_are_65_plus_and_untailored(db):
    tb, _ = toolbox(db)
    assert [c["job_id"] for c in tb.tailor_candidates()] == ["j0"]
    db.set_status("j0", "tailored")
    assert tb.tailor_candidates() == []


def test_log_application_never_writes_applied(db):
    tb, state = toolbox(db)
    with pytest.raises(ToolError):
        tb.log_application(state, "j0", "applied")
    tb.log_application(state, "j0", "to_review", "ok")
    row = db.conn.execute("SELECT * FROM applications WHERE job_id='j0'").fetchone()
    assert row["status"] == "to_review" and row["fit_score"] == 80 and row["company"] == "Acme"
    tb.log_application(state, "j0", "skipped", "changed my mind")          # upsert, not a second row
    assert db.conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 1


def test_preview_new_writes_nothing_and_dedupes(db):
    before = db.count()
    fresh = Job(id="n1", company="Acme", title="Applied AI Engineer", location="", remote=True, url="",
                description="", source="lever")
    twin = fresh.model_copy(update={"id": "n2", "title": "Applied AI Engineer (Remote)"})
    dup_of_db = fresh.model_copy(update={"id": "n3", "title": "AI Engineer 0"})
    assert [j.id for j in db.preview_new([fresh, twin, dup_of_db])] == ["n1"]
    assert db.count() == before


# ---- Anthropic tool-calling translation (fake client, no network) ------------------------------------------------
def test_anthropic_chat_wire_format_and_parsing():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Scoring now."),
                     SimpleNamespace(type="tool_use", id="tu_2", name="score_jobs", input={"limit": 5})],
            stop_reason="tool_use", usage=SimpleNamespace(input_tokens=120, output_tokens=30))

    p = object.__new__(AnthropicProvider)
    p.models, p.client = {"fast": "claude-haiku-5-5", "effort": {"fast": "low"}}, SimpleNamespace(
        messages=SimpleNamespace(create=create))
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [ToolCall("tu_1", "search_jobs", {})], "raw": None},
                {"role": "tool", "results": [{"id": "tu_1", "name": "search_jobs", "content": "{}", "is_error": True}]}]
    resp = p._chat("claude-haiku-5-5", "fast", messages, TOOLS, "sys")
    assert sent["messages"][1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "tu_1", "name": "search_jobs", "input": {}}]}
    assert sent["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "{}",
                                                 "is_error": True}
    assert sent["tools"][0]["input_schema"]["type"] == "object" and sent["output_config"] == {"effort": "low"}
    assert resp.text == "Scoring now." and resp.stop_reason == "tool_use"
    assert resp.tool_calls == [ToolCall("tu_2", "score_jobs", {"limit": 5})]
    assert resp.as_message()["raw"] is not None          # replayed verbatim next turn


def test_jobs_without_an_approved_summary_are_held_back(db):
    tb, state = toolbox(db)
    db.save_score({"job_id": "j2", "fit_score": 90, "fast_score": 90, "escalated": 0, "model": "m",
                   "role_family": "program_product_manager", "matched_skills": "[]", "gaps": "[]",
                   "sponsorship_signal": "yes", "uscis_status": "no_data", "scored_at": "2026-10-11"})
    assert [c["job_id"] for c in tb.tailor_candidates()] == ["j0"]          # j2 is higher but has no summary
    assert [c["job_id"] for c in tb.needs_summary()] == ["j2"]
