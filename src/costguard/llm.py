"""One wrapper for every LLM call. Backends: anthropic | mock.

Every call returns an LLMResult with token counts and cost, so the UI can show
cost per AI call and the running monthly AI cost.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import config


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    model: str
    backend: str
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    p = config.PRICES_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------- backends
_anthropic_client = None


def _call_anthropic(system: str, user: str) -> tuple[str, int, int, str]:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    resp = _anthropic_client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": config.ANTHROPIC_EFFORT},
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Model declined the request (stop_reason=refusal)")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens, config.ANTHROPIC_MODEL


def _call_mock(system: str, user: str, mock: Callable[[], object]) -> tuple[str, int, int, str]:
    time.sleep(config.MOCK_LATENCY_SECONDS)
    out = mock()
    text = out if isinstance(out, str) else json.dumps(out, indent=2)
    return text, _estimate_tokens(system + user), _estimate_tokens(text), "mock"


def call(system: str, user: str, mock: Optional[Callable[[], object]] = None,
         backend: Optional[str] = None) -> LLMResult:
    """Make one LLM call. `mock` produces the canned response for the mock backend."""
    backend = (backend or config.BACKEND).lower()
    t0 = time.time()
    try:
        if backend == "anthropic":
            text, tin, tout, model = _call_anthropic(system, user)
        elif backend == "mock":
            if mock is None:
                raise ValueError("mock backend needs a mock response")
            text, tin, tout, model = _call_mock(system, user, mock)
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        err = None
    except Exception as e:  # surfaced in the agent trace, never crashes the UI
        text, tin, tout, err = "", 0, 0, f"{type(e).__name__}: {e}"
        model = config.ANTHROPIC_MODEL if backend == "anthropic" else backend
    return LLMResult(text, tin, tout, price(model, tin, tout), time.time() - t0, model, backend, err)


def parse_json(text: str) -> Optional[dict]:
    """Lenient JSON extraction: handles ```json fences and leading/trailing prose."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
