"""Langfuse tracing (src/tracing.py): one trace per run, spans for scoring / keyword extraction / tailoring /
verification, a generation per LLM call. Only ids and metadata are sent; tracing is optional and never fails a run.
A fake client records every argument the code passes, so the tests can prove no content reaches it."""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from main import _spends_llm, load_config
from src import tracing
from src.keywords import Keyword, KeywordList, get_keywords
from src.llm import LLMError, LLMProvider
from src.models import BulletVerdict, ScoredJob, SponsorshipSignal, TailoredResume, Verification
from src.resume import citable_texts, project_tech, tailoring_source
from src.scorer import estimate, score_jobs
from src.store import Store
from src.tailor import tailor_job
from src.verify import Verifier
from tests.test_phase3 import MASTER, good

KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_HOST": "http://langfuse.test"}
SECRET = "SENTINEL-RESUME-TEXT-7f3a"          # planted in descriptions, prompts and model output


class FakeObs:
    def __init__(self, client, name, as_type, kwargs, parent=None):
        self.client, self.name, self.as_type, self.parent = client, name, as_type, parent
        self.start_kwargs, self.updates, self.ended, self.children = kwargs, [], False, []

    def start_observation(self, *, name, as_type="span", **kwargs):
        return self.client._new(name, as_type, kwargs, self)

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self, **kwargs):
        self.ended = True

    @property
    def meta(self) -> dict:
        out = dict(self.start_kwargs.get("metadata") or {})
        for u in self.updates:
            out.update(u.get("metadata") or {})
        return out

    def field(self, key):
        return next((u[key] for u in reversed(self.updates) if key in u), self.start_kwargs.get(key))


class FakeClient:
    def __init__(self):
        self.roots, self.all, self.flushed = [], [], 0

    def _new(self, name, as_type, kwargs, parent):
        obs = FakeObs(self, name, as_type, kwargs, parent)
        (parent.children if parent else self.roots).append(obs)
        self.all.append(obs)
        return obs

    def start_observation(self, *, name, as_type="span", **kwargs):
        return self._new(name, as_type, kwargs, None)

    def flush(self):
        self.flushed += 1

    def everything_sent(self) -> str:
        return json.dumps([{"name": o.name, "start": o.start_kwargs, "updates": o.updates} for o in self.all],
                          default=str)


@pytest.fixture
def fake():
    client = FakeClient()
    tracer = tracing.configure(env=KEYS, client_factory=lambda pk, sk, host: client)
    assert tracer.enabled
    yield client
    tracing.reset()


class Provider(LLMProvider):
    """A real LLMProvider (generate_json/chat and their tracing hooks run); only the raw API call is faked."""
    name = "fake"

    def __init__(self, replies):
        super().__init__({"fast": "claude-haiku-5-5", "strong": "claude-sonnet-5-5"}, {"max_attempts": 1})
        self.replies, self.prompts = replies, []
        self.pricing = load_config()["llm"]["pricing"]

    def _call(self, model, tier, prompt, schema, system):
        self.prompts.append(prompt)
        return self.replies[schema.__name__](prompt, tier), 1000, 200

    def _retry_hint(self, exc):
        return None


def scored(fit):
    return ScoredJob(fit_score=fit, role_family="ai_engineer", matched_skills=[SECRET], gaps=[SECRET],
                     sponsorship_signal=SponsorshipSignal(signal="unclear", quote=SECRET), is_staffing_firm=False,
                     seniority_ok=True, one_line_reason=SECRET)


def no_input_output(client: FakeClient) -> bool:
    return all("input" not in o.start_kwargs and "output" not in o.start_kwargs
               and all("input" not in u and "output" not in u for u in o.updates) for o in client.all)


# ---- configuration -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("missing", list(KEYS))
def test_off_when_any_key_is_missing(missing):
    tracer = tracing.configure(env={k: v for k, v in KEYS.items() if k != missing},
                               client_factory=lambda *a: pytest.fail("client must not be built"))
    assert not tracer.enabled
    with tracer.run("x") as h, tracer.span("y") as s, tracer.generation("z") as g:
        h.set(job_id="j"), s.set(ok=True), g.set(model="m")
    tracer.flush()


def test_tests_never_reach_real_langfuse(monkeypatch):
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    assert not tracing.configure().enabled          # JOB_AGENT_NO_TRACING is set by tests/conftest.py


def test_broken_sdk_or_client_means_no_tracing(capsys):
    def boom(*a):
        raise RuntimeError("bad host")
    assert not tracing.configure(env=KEYS, client_factory=boom).enabled
    assert "running without tracing" in capsys.readouterr().err


def test_only_llm_commands_are_traced():
    ns = SimpleNamespace
    assert _spends_llm(ns(command="score")) and _spends_llm(ns(command="tailor"))
    assert _spends_llm(ns(command="run", dry_run=False)) and not _spends_llm(ns(command="run", dry_run=True))
    assert _spends_llm(ns(command="keywords", run=True)) and not _spends_llm(ns(command="keywords", run=False))
    assert not any(_spends_llm(ns(command=c)) for c in ("search", "digest", "queue", "review", "notion-sync"))


# ---- what is sent ----------------------------------------------------------------------------------------------

def test_metadata_allowlist_and_truncation():
    out = tracing._clean({"job_id": "greenhouse:acme:1", "fit_score": 66, "prompt": SECRET, "description": SECRET,
                          "text": SECRET, "nested": {"job_id": "x"}, "status": "x" * 500,
                          "issue_kinds": ["number", "tool"]})
    assert out == {"job_id": "greenhouse:acme:1", "fit_score": 66, "status": "x" * tracing.MAX_STR,
                   "issue_kinds": ["number", "tool"]}
    assert tracing._mask(data=SECRET) is None


@pytest.mark.parametrize("problem,kind", [   # real messages from src/verify.py
    ("number(s) not in cited sources: ['90']", "number"),
    ("tool/skill term(s) not in cited sources: ['Go']", "tool"),
    ("cites unknown or unusable id 'exp_zz'", "citation"), ("cites no source ids", "citation"),
    ("exp_a bullet 1 and exp_a bullet 2 describe the same work (group ['a', 'b']); keep one", "same_work"),
    ("must include a bullet citing exp_a_b01 for this role", "required"),
    ("must include project proj_x for this role", "required"),
    ("uses exp_b_b02 without exp_b_b01 (stronger first)", "preference"),
    ("needs 3-4 skills, got 2", "headline"), ("contains a forbidden seniority word", "seniority"),
    ("says '5+ years' (must be '4+ years')", "years"), ("pick 4-6 courses, got 2", "coursework"),
    ("not supported by sources: added a claim", "llm_unsupported"), ("something new", "other"),
])
def test_issue_text_becomes_a_fixed_category(problem, kind):
    assert tracing.issue_kind(problem) == kind


def test_scoring_trace_has_spans_generations_and_no_content(fake):
    cfg = load_config()
    cfg["scoring"]["escalate_threshold"] = 60
    llm = Provider({"ScoredJob": lambda p, tier: scored(70 if tier == "fast" else 75)})
    jobs = [{"id": "greenhouse:acme:1", "company": "Acme", "title": "AI Engineer", "location": "Remote", "remote": 1,
             "description": f"Build LLM apps. {SECRET}", "role_hint": "ai_engineer", "posted_at": None}]
    store = SimpleNamespace(save_score=lambda row: None)
    uscis = SimpleNamespace(lookup=lambda *a, **k: {"status": "no_data", "e_verify": "unknown"})
    with tracing.get().run("job-agent score", command="score", run_id="r1"):
        score_jobs(jobs, f"profile {SECRET}", llm, cfg, store, uscis, estimate(jobs, "system", llm, cfg))
    tracing.get().flush()

    assert len(fake.roots) == 1 and fake.flushed == 1                     # one trace per run
    (root,) = fake.roots
    (span,) = root.children
    assert span.name == "scoring" and span.ended
    assert span.meta["job_id"] == "greenhouse:acme:1"
    assert span.meta["fit_score"] == 75 and span.meta["fast_score"] == 70 and span.meta["escalated"] is True
    gens = span.children
    assert [g.as_type for g in gens] == ["generation", "generation"]
    assert [g.meta["tier"] for g in gens] == ["fast", "strong"]
    assert gens[1].field("model") == "claude-sonnet-5-5"
    assert gens[0].field("usage_details") == {"input": 1000, "output": 200}
    assert gens[0].field("cost_details")["total"] > 0 and gens[0].meta["latency_s"] >= 0
    assert SECRET in llm.prompts[0]                                        # the model saw it ...
    assert SECRET not in fake.everything_sent() and no_input_output(fake)  # ... Langfuse never did


def test_tailoring_and_verification_spans_carry_outcomes_not_text(fake):
    cfg = copy.deepcopy(load_config())
    cfg["tailoring"].update({"optional_experience": ["exp_c"], "required_bullets": [], "preferences": [],
                             "required_projects": []})
    verifier = Verifier(MASTER, citable_texts(MASTER), project_tech(MASTER), cfg, "analyst_backup", "Data Analyst")
    bad = good()
    bad.experience[0].bullets[0].text = "Built SQL pipelines across 90+ regions, error 30% to 6%."
    drafts = [bad, good()]
    judge = lambda p, tier: Verification(verdicts=[BulletVerdict(index=i, supported=True)
                                                   for i in range(p.count("\nTEXT: "))])
    llm = Provider({"TailoredResume": lambda p, tier: drafts.pop(0), "Verification": judge})
    job = {"id": "greenhouse:acme:2", "company": "Acme", "title": "Data Analyst", "location": "Remote",
           "description": f"SQL role {SECRET}"}
    with tracing.get().run("job-agent tailor", command="tailor"):
        res = tailor_job(job, None, MASTER, tailoring_source(MASTER), verifier, llm, cfg, budget_left=1.0)
    assert res.status == "PASS" and res.attempts == 2

    (tailoring,) = fake.roots[0].children
    assert tailoring.name == "tailoring" and tailoring.meta["status"] == "PASS" and tailoring.meta["attempts"] == 2
    kinds = [(c.name, c.as_type) for c in tailoring.children]
    assert kinds == [("llm.TailoredResume", "generation"), ("verification", "span"),
                     ("llm.TailoredResume", "generation"), ("verification", "span")]
    v1, v2 = tailoring.children[1], tailoring.children[3]
    assert v1.meta["passed"] is False and v1.meta["issue_kinds"] == ["number"] and v1.meta["llm_checked"] is False
    assert v2.meta["passed"] is True and v2.meta["llm_checked"] is True and v2.meta["llm_unsupported"] == 0
    assert [c.name for c in v2.children] == ["llm.Verification"]           # the judge call nests under its check
    sent = fake.everything_sent()
    master_texts = [t for t in citable_texts(MASTER).values() if len(t) > 12]
    assert SECRET not in sent and "90+ regions" not in sent and not any(t in sent for t in master_texts)
    assert "Test Person" not in sent and "t@example.com" not in sent and no_input_output(fake)


def test_keyword_extraction_span_only_when_not_cached(fake, tmp_path):
    from src.models import Job
    store = Store(tmp_path / "jobs.db")
    store.save([Job(id="j1", company="Acme", title="AI Engineer", location="NYC", remote=False, url="u",
                    description=f"Python, SQL. {SECRET}", source="greenhouse")])
    row = store.conn.execute("SELECT * FROM jobs WHERE id='j1'").fetchone()
    llm = Provider({"KeywordList": lambda p, tier: KeywordList(keywords=[
        Keyword(text="Python", kind="skill", importance=3), Keyword(text=SECRET, kind="skill", importance=1)])})
    with tracing.get().run("job-agent keywords"):
        get_keywords(store, row, llm)
        get_keywords(store, row, llm)                                      # cached: no span, no call
    (span,) = fake.roots[0].children
    assert span.name == "keyword_extraction" and span.meta == {"job_id": "j1", "keywords": 2, "ok": True,
                                                                "model": "claude-haiku-5-5"}
    assert len(span.children) == 1 and SECRET not in fake.everything_sent()


# ---- never fails a run ---------------------------------------------------------------------------------------

def test_llm_error_is_recorded_and_still_raised(fake):
    class Failing(Provider):
        def _call(self, *a):
            raise ValueError("bad request")
    with pytest.raises(ValueError):
        with tracing.get().run("job-agent score"):
            Failing({}).generate_json("p", ScoredJob)
    (gen,) = fake.roots[0].children
    assert gen.meta["ok"] is False and gen.meta["error_type"] == "ValueError" and gen.field("level") == "ERROR"


def test_langfuse_errors_disable_tracing_but_the_run_continues(capsys):
    class Exploding(FakeClient):
        def start_observation(self, **kw):
            raise ConnectionError("unreachable")

        def flush(self):
            raise ConnectionError("unreachable")
    tracing.configure(env=KEYS, client_factory=lambda *a: Exploding())
    try:
        llm = Provider({"ScoredJob": lambda p, tier: scored(40)})
        with tracing.get().run("job-agent score") as root:
            resp = llm.generate_json("p", ScoredJob)
            root.set(ok=True)
        tracing.get().flush()
        assert resp.data.fit_score == 40 and not tracing.get().enabled
        assert capsys.readouterr().err.count("Langfuse disabled") == 1     # one warning, not one per call
    finally:
        tracing.reset()


def test_real_sdk_with_unreachable_host_does_not_fail_or_hang():
    pytest.importorskip("langfuse")
    env = {**KEYS, "LANGFUSE_HOST": "http://127.0.0.1:9"}                  # nothing listens on port 9
    tracer = tracing.configure(env=env)
    try:
        assert tracer.enabled
        llm = Provider({"ScoredJob": lambda p, tier: scored(55)})
        start = time.monotonic()
        with tracer.run("job-agent score", command="score"):
            with tracer.span("scoring", job_id="j1"):
                assert llm.generate_json("p", ScoredJob).data.fit_score == 55
        tracer.flush()
        assert time.monotonic() - start < 30
    finally:
        try:
            tracer.client.shutdown()
        except Exception:  # noqa: BLE001
            pass
        tracing.reset()
