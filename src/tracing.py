"""Optional Langfuse tracing: one trace per CLI run, a span per scoring / keyword-extraction / tailoring /
verification step, and a generation for every LLM call made through src/llm.py.

Privacy: only ids and metadata ever leave the machine. No prompt, completion, job description, resume text or
contact detail is passed to Langfuse:
- observations are created with metadata only (never `input` / `output`);
- metadata goes through `_clean`: an allowlist of keys, scalar values or short lists of short strings;
- the client is built with a `mask` that blanks any input/output that might still reach it.
Verifier issues are sent as fixed categories (`issue_kind`), never as their message text.

Optional and never fatal: tracing is on only when LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are
all set and the SDK imports. Every call into Langfuse is wrapped; the first error turns tracing off for the rest of
the run with one warning on stderr. Export happens in the SDK's background thread, so an unreachable host costs
at most the flush timeout at the end of the run.
"""
from __future__ import annotations

import contextvars
import os
import re
import sys
from contextlib import contextmanager

ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
TIMEOUT_S = 3                  # per export request; the SDK's default is 5

# The only metadata keys that are ever sent. Anything else is dropped.
ALLOWED_KEYS = {
    "command", "run_id", "job_id", "source", "role_family", "tier", "model", "schema", "attempt", "attempts",
    "latency_s", "input_tokens", "output_tokens", "cost_usd", "ok", "error_type", "fit_score", "fast_score",
    "escalated", "cached", "keywords", "passed", "status", "issue_count", "issue_kinds", "items", "llm_checked",
    "llm_unsupported", "removed", "calls", "tools", "tool_calls", "stop_reason", "jobs", "dry_run",
}
MAX_STR = 80
MAX_LIST = 20

# Matched in order against the messages src/verify.py writes (Issue.problem), first match wins.
_ISSUE_KINDS = [
    ("llm_unsupported", r"not supported by sources|no verdict"), ("number", r"^number\(s\) not in"),
    ("tool", r"^tool/skill term"), ("years", r"years"), ("seniority", r"seniority word"),
    ("headline", r"role label|needs \d+-\d+ skills"), ("same_work", r"same source|same work"),
    ("required", r"^must include"), ("preference", r"^uses .+ without"), ("citation", r"^cites"),
    ("coursework", r"course"), ("length", r"bullets; keep"), ("structure", r"must list|section id|no bullets|projects"),
    ("skill_id", r"unknown skill id"),
]


def issue_kind(problem: str) -> str:
    """Map a verifier issue message to a fixed category, so the message text (which can quote resume content)
    never leaves the machine."""
    for kind, pat in _ISSUE_KINDS:
        if re.search(pat, problem, re.I):
            return kind
    return "other"


def _value(v):
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:MAX_STR]
    if isinstance(v, (list, tuple, set)):
        return [str(x)[:MAX_STR] for x in list(v)[:MAX_LIST] if isinstance(x, (str, int, float, bool))]
    return None                 # dicts, objects: never sent


def _clean(meta: dict) -> dict:
    return {k: _value(v) for k, v in meta.items() if k in ALLOWED_KEYS and _value(v) is not None}


def _mask(*, data, **kwargs):
    """Installed as the client's mask: input/output must never carry content; blank whatever arrives."""
    return None


class Handle:
    """What `span` / `generation` yield: add metadata as it becomes known."""

    def __init__(self, tracer: "Tracer", obs=None):
        self._tracer, self._obs, self.meta = tracer, obs, {}

    def set(self, **meta) -> None:
        self.meta.update(meta)


class Tracer:
    """No-op tracer, used when Langfuse isn't configured. LangfuseTracer overrides the hooks."""
    enabled = False

    @contextmanager
    def run(self, name: str, **meta):
        yield Handle(self)

    @contextmanager
    def span(self, name: str, **meta):
        yield Handle(self)

    @contextmanager
    def generation(self, name: str, **meta):
        yield Handle(self)

    def flush(self) -> None:
        pass


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client):
        self.client = client
        self._parent: contextvars.ContextVar = contextvars.ContextVar("langfuse_parent", default=None)

    def _fail(self, exc: Exception) -> None:
        if self.enabled:
            self.enabled = False
            print(f"  [tracing] Langfuse disabled for this run ({type(exc).__name__}); the run continues",
                  file=sys.stderr)

    def _start(self, name: str, as_type: str, meta: dict, **extra):
        if not self.enabled:
            return None
        try:
            parent = self._parent.get()
            starter = parent if parent is not None else self.client
            return starter.start_observation(name=name, as_type=as_type, metadata=_clean(meta), **extra)
        except Exception as exc:  # noqa: BLE001 — tracing must never fail a run
            self._fail(exc)
            return None

    def _end(self, obs, meta: dict, **extra) -> None:
        if obs is None or not self.enabled:
            return
        try:
            obs.update(metadata=_clean(meta), **extra)
            obs.end()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    @contextmanager
    def _observe(self, name: str, as_type: str, meta: dict, **extra):
        obs = self._start(name, as_type, meta, **extra)
        h = Handle(self, obs)
        h.meta.update(meta)
        token = self._parent.set(obs) if obs is not None else None
        try:
            yield h
        except BaseException as exc:
            h.meta.setdefault("ok", False)
            h.meta.setdefault("error_type", type(exc).__name__)
            raise
        finally:
            if token is not None:
                self._parent.reset(token)
            self._end(obs, h.meta, **self._generation_fields(as_type, h.meta),
                      **({"level": "ERROR"} if h.meta.get("ok") is False else {}))

    @staticmethod
    def _generation_fields(as_type: str, meta: dict) -> dict:
        if as_type != "generation":
            return {}
        out = {}
        if meta.get("model"):
            out["model"] = str(meta["model"])
        if meta.get("input_tokens") is not None:
            out["usage_details"] = {"input": int(meta["input_tokens"]), "output": int(meta.get("output_tokens") or 0)}
        if meta.get("cost_usd") is not None:
            out["cost_details"] = {"total": float(meta["cost_usd"])}
        return out

    @contextmanager
    def run(self, name: str, **meta):
        with self._observe(name, "span", meta) as h:
            yield h

    @contextmanager
    def span(self, name: str, **meta):
        with self._observe(name, "span", meta) as h:
            yield h

    @contextmanager
    def generation(self, name: str, **meta):
        with self._observe(name, "generation", meta) as h:
            yield h

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self.client.flush()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)


_tracer: Tracer = Tracer()


def get() -> Tracer:
    return _tracer


def configure(env=None, client_factory=None) -> Tracer:
    """Turn tracing on if all three Langfuse settings are present; otherwise (or on any error) stay a no-op.
    client_factory(public_key, secret_key, host) -> client is for tests."""
    global _tracer
    if env is None:
        env = os.environ
        if env.get("JOB_AGENT_NO_TRACING"):     # set by tests/conftest.py: tests never reach a real Langfuse
            _tracer = Tracer()
            return _tracer
    if not all(env.get(k) for k in ENV_KEYS):
        _tracer = Tracer()
        return _tracer
    try:
        if client_factory is None:
            from langfuse import Langfuse

            def client_factory(public_key, secret_key, host):
                return Langfuse(public_key=public_key, secret_key=secret_key, host=host, timeout=TIMEOUT_S,
                                mask=_mask)
        client = client_factory(env["LANGFUSE_PUBLIC_KEY"], env["LANGFUSE_SECRET_KEY"], env["LANGFUSE_HOST"])
        _tracer = LangfuseTracer(client)
    except Exception as exc:  # noqa: BLE001 — a broken SDK install or bad host must not stop a run
        print(f"  [tracing] Langfuse not started ({type(exc).__name__}); running without tracing", file=sys.stderr)
        _tracer = Tracer()
    return _tracer


def reset() -> None:
    """Back to the no-op tracer (tests)."""
    global _tracer
    _tracer = Tracer()
