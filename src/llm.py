"""Provider-neutral LLM interface. The rest of the agent only ever calls `get_llm(cfg).generate_json(...)`.

Switching providers = changing `llm.provider` in config.yaml. Each provider maps two tiers
("fast" for first-pass scoring, "strong" for jobs that clear the escalation threshold) to its own models.

Retry policy lives here, once, for every provider: rate limits (429) and transient server errors
(5xx, network) back off exponentially with jitter; anything else (bad request, auth) fails immediately.
SDK-level retries are turned off so the two policies don't stack.
"""
from __future__ import annotations

import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "strong"]


@dataclass
class LLMResponse(Generic[T]):
    data: T                 # validated pydantic instance of the requested schema
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float        # wall-clock including retries — what the Phase 4 logs will record
    attempts: int


class LLMError(RuntimeError):
    """Non-retryable failure, or retries exhausted."""


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines) so we don't need python-dotenv. Real env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise LLMError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


class LLMProvider(ABC):
    name: str

    def __init__(self, models: dict[str, str], retry: dict | None = None):
        self.models = models
        retry = retry or {}
        self.max_attempts = retry.get("max_attempts", 5)
        self.base_delay = retry.get("base_delay", 2.0)
        self.max_delay = retry.get("max_delay", 60.0)

    def generate_json(self, prompt: str, schema: type[T], *, tier: Tier = "fast",
                      system: str | None = None) -> LLMResponse[T]:
        """Ask the model for output matching `schema`; returns it parsed and validated."""
        model = self.models[tier]
        start = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                data, in_tok, out_tok = self._call(model, tier, prompt, schema, system)
                return LLMResponse(data, model, in_tok, out_tok, time.monotonic() - start, attempt)
            except Exception as exc:  # noqa: BLE001 — classified below
                hint = self._retry_hint(exc)
                if hint is None:
                    raise
                if attempt == self.max_attempts:
                    raise LLMError(f"{self.name}/{model}: gave up after {attempt} attempts: {exc}") from exc
                # Exponential backoff with full jitter; honor the server's suggested delay when it gives one.
                backoff = random.uniform(0, min(self.max_delay, self.base_delay * 2 ** (attempt - 1)))
                delay = min(self.max_delay, max(hint, backoff))
                print(f"  [{self.name}] {type(exc).__name__}, retry {attempt}/{self.max_attempts - 1} in {delay:.1f}s")
                time.sleep(delay)
        raise AssertionError("unreachable")

    @abstractmethod
    def _call(self, model: str, tier: Tier, prompt: str, schema: type[T], system: str | None) -> tuple[T, int, int]:
        """One raw API call -> (parsed object, input tokens, output tokens)."""

    @abstractmethod
    def _retry_hint(self, exc: Exception) -> float | None:
        """None = don't retry. Otherwise the minimum seconds to wait (0 if the server gave no hint)."""


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, models, retry=None):
        super().__init__(models, retry)
        from google import genai  # imported lazily so the unused provider's SDK is optional

        self._genai = genai
        self.client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))

    def _call(self, model, tier, prompt, schema, system):
        from google.genai import types

        resp = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,  # Gemini constrains decoding to this schema
            ),
        )
        # Validate ourselves rather than trusting resp.parsed, so a malformed reply raises clearly.
        data = schema.model_validate_json(resp.text or "")
        usage = resp.usage_metadata
        return data, (usage.prompt_token_count or 0), (usage.candidates_token_count or 0)

    def _retry_hint(self, exc):
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            if exc.code == 429 or (exc.code or 0) >= 500:
                return _gemini_retry_delay(exc.details)
            return None
        if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__module__.startswith(("httpx", "httpcore")):
            return 0.0
        return None


def _gemini_retry_delay(details) -> float:
    """Gemini 429s carry a google.rpc.RetryInfo entry like {"retryDelay": "31s"}."""
    try:
        for d in details.get("error", {}).get("details", []):
            if "RetryInfo" in d.get("@type", ""):
                m = re.match(r"([\d.]+)s", d.get("retryDelay", ""))
                if m:
                    return float(m.group(1))
    except AttributeError:
        pass
    return 0.0


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, models, retry=None):
        super().__init__(models, retry)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"), max_retries=0)

    def _call(self, model, tier, prompt, schema, system):
        kwargs = {"system": system} if system else {}
        # Effort controls how much the model thinks; thinking tokens bill as output, so it's the main cost lever.
        effort = (self.models.get("effort") or {}).get(tier)
        if effort:
            kwargs["output_config"] = {"effort": effort}   # the SDK merges output_format into this
        resp = self.client.messages.parse(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,  # structured outputs: response validated against the pydantic model
            **kwargs,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise LLMError(f"anthropic/{model}: no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens

    def _retry_hint(self, exc):
        a = self._anthropic
        if isinstance(exc, a.RateLimitError):
            try:
                return float(exc.response.headers.get("retry-after", 0))
            except (TypeError, ValueError):
                return 0.0
        if isinstance(exc, a.APIStatusError):
            return 0.0 if exc.status_code >= 500 else None
        if isinstance(exc, a.APIConnectionError):  # includes timeouts
            return 0.0
        return None


PROVIDERS: dict[str, type[LLMProvider]] = {"gemini": GeminiProvider, "anthropic": AnthropicProvider}


def cost_usd(model: str, input_tokens: int, output_tokens: int, pricing: dict) -> float | None:
    """Dollar cost from config.yaml llm.pricing (USD per 1M tokens). None if the model has no price listed."""
    p = pricing.get(model)
    if not p:
        return None
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough, deliberately high token estimate (~3.5 chars/token) for pre-run cost estimates."""
    return int(len(text) / 3.5) + 1


def get_llm(cfg: dict) -> LLMProvider:
    """Build the provider named in config.yaml's `llm.provider`."""
    llm_cfg = cfg["llm"]
    name = llm_cfg["provider"]
    if name not in PROVIDERS:
        raise LLMError(f"Unknown llm.provider {name!r}; choose one of {sorted(PROVIDERS)}")
    load_env()
    return PROVIDERS[name](llm_cfg["models"][name], llm_cfg.get("retry"))
