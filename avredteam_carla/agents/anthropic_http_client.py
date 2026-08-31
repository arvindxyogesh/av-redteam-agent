"""Minimal Anthropic Messages API client over raw HTTP.

Exists because this project's Python 3.7 pin (CARLA 0.9.11's client wheel
is cp37-only - `docs/setup.md`) cannot install any `anthropic` SDK version
that supports tool use. Checked directly, not assumed: every SDK release
from 0.27.0 onward depends on `jiter`, and `jiter`'s package metadata
declares `Requires-Python: >=3.8` on every version ever published,
including the very first - a hard, declared floor pip enforces before any
build is attempted, confirmed by installing a real local Rust toolchain
(`rustup`) specifically to rule out "just a missing prebuilt wheel" as the
cause. The only cp37-installable `anthropic` version, 0.26.0, predates the
`tools` parameter on `messages.create()` entirely (checked via
`inspect.signature`).

This talks to the same REST endpoint the SDK itself calls
(`https://api.anthropic.com/v1/messages`) via `httpx` (pure Python, already
an `anthropic` 0.26.0 transitive dependency, cp37-compatible) and exposes
just the `.messages.create(...)` surface `LLMAgentSearch`
(`avredteam_carla/agents/llm_agent_search.py`) actually uses, so nothing
else about that module needed to change.
"""
from __future__ import annotations

import os
import time

import httpx

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 3
# Retryable per REST convention: 429 (rate limit) and 5xx (server-side) -
# never 4xx other than 429, those mean the request itself is wrong and a
# retry would just fail identically.
_RETRYABLE_MIN_STATUS = 429


class AnthropicAPIError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"Anthropic API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _to_jsonable(obj):
    """Recursively unwrap `_Block`s (from a prior response, fed back into
    `messages` by `LLMAgentSearch`) into plain dicts for the outgoing
    request body; plain dicts/lists (e.g. the tool_result messages
    `LLMAgentSearch` builds itself) pass through unchanged."""
    if isinstance(obj, _Block):
        return obj.raw
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


class _Block:
    """One content block from a response, with attribute access
    (`.type`/`.id`/`.name`/`.input`/`.text`) matching how `LLMAgentSearch`
    reads the real SDK's block objects - it never needs anything beyond
    attribute reads, so a full Pydantic-model reimplementation isn't
    needed."""

    def __init__(self, raw: dict):
        self.raw = raw
        for key, value in raw.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"_Block({self.raw!r})"


class _Response:
    def __init__(self, raw: dict):
        self.raw = raw
        self.content = [_Block(b) for b in raw.get("content", [])]
        self.stop_reason = raw.get("stop_reason")
        self.id = raw.get("id")
        self.model = raw.get("model")
        self.usage = raw.get("usage")


class _Messages:
    def __init__(self, api_key: str, base_url: str, timeout_s: float, max_retries: int, client: httpx.Client):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._http = client

    def create(self, *, model, max_tokens, messages, system=None, tools=None, **kwargs) -> _Response:
        body = {"model": model, "max_tokens": max_tokens, "messages": _to_jsonable(messages)}
        if system is not None:
            body["system"] = system
        if tools is not None:
            body["tools"] = tools
        body.update(kwargs)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._http.post(self._base_url, json=body, headers=headers, timeout=self._timeout_s)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(min(2 ** attempt, 30))
                continue

            if resp.status_code >= _RETRYABLE_MIN_STATUS:
                last_error = AnthropicAPIError(resp.status_code, resp.text)
                if attempt < self._max_retries:
                    time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code >= 400:
                raise AnthropicAPIError(resp.status_code, resp.text)
            return _Response(resp.json())

        raise last_error


class AnthropicHTTPClient:
    """Drop-in replacement for `anthropic.Anthropic()` exposing just the
    `.messages.create(...)` surface `LLMAgentSearch` needs - see module
    docstring for why this exists instead of the real SDK. `http_client`
    is injectable so tests never touch the network (pass an
    `httpx.Client(transport=httpx.MockTransport(...))`)."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = ANTHROPIC_API_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client = None,
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set and no api_key passed to AnthropicHTTPClient")
        self.messages = _Messages(api_key, base_url, timeout_s, max_retries, http_client or httpx.Client())
