import pytest
from pydantic import BaseModel

import src.llm as llm
from main import load_config
from src.llm import AnthropicProvider, GeminiProvider, LLMError, LLMProvider, get_llm


class Out(BaseModel):
    score: int


class Transient(Exception):
    pass


class FakeProvider(LLMProvider):
    """Fails `fail_times` with a retryable error, then succeeds."""
    name = "fake"

    def __init__(self, fail_times, retryable=True):
        super().__init__({"fast": "f-model", "strong": "s-model"}, {"max_attempts": 3, "base_delay": 0, "max_delay": 0})
        self.fail_times, self.retryable, self.calls = fail_times, retryable, []

    def _call(self, model, prompt, schema, system):
        self.calls.append(model)
        if len(self.calls) <= self.fail_times:
            raise Transient("boom")
        return schema(score=7), 10, 2

    def _retry_hint(self, exc):
        return 0.0 if self.retryable else None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_retries_then_succeeds():
    p = FakeProvider(fail_times=2)
    r = p.generate_json("hi", Out)
    assert r.data.score == 7 and r.attempts == 3 and r.input_tokens == 10


def test_gives_up_after_max_attempts():
    with pytest.raises(LLMError, match="gave up after 3"):
        FakeProvider(fail_times=5).generate_json("hi", Out)


def test_non_retryable_raises_immediately():
    p = FakeProvider(fail_times=1, retryable=False)
    with pytest.raises(Transient):
        p.generate_json("hi", Out)
    assert len(p.calls) == 1


def test_tier_selects_model():
    p = FakeProvider(fail_times=0)
    assert p.generate_json("hi", Out, tier="strong").model == "s-model"


def test_gemini_error_classification():
    from google.genai import errors

    hint = GeminiProvider._retry_hint
    rate = errors.APIError(429, {"error": {"code": 429, "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"}]}})
    assert hint(None, rate) == 31.0
    assert hint(None, errors.APIError(503, {"error": {"code": 503}})) == 0.0
    assert hint(None, errors.APIError(400, {"error": {"code": 400}})) is None
    assert hint(None, ValueError("bad json")) is None


def test_anthropic_error_classification():
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    p = AnthropicProvider.__new__(AnthropicProvider)  # skip __init__: no key needed to classify errors
    p._anthropic = anthropic
    rate = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=req, headers={"retry-after": "12"}), body=None)
    bad = anthropic.BadRequestError("nope", response=httpx2.Response(400, request=req), body=None)
    assert p._retry_hint(rate) == 12.0
    assert p._retry_hint(bad) is None


def test_config_switch_and_missing_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config()
    assert cfg["llm"]["provider"] in ("gemini", "anthropic")
    cfg["llm"]["provider"] = "gemini"
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_llm(cfg)
    cfg["llm"]["provider"] = "anthropic"
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm(cfg)


def test_load_env_does_not_override_real_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FOO_KEY=from_file\nBAR_KEY='quoted'\n# comment\n")
    monkeypatch.setenv("FOO_KEY", "from_env")
    monkeypatch.delenv("BAR_KEY", raising=False)
    llm.load_env(tmp_path / ".env")
    import os
    assert os.environ["FOO_KEY"] == "from_env" and os.environ["BAR_KEY"] == "quoted"
