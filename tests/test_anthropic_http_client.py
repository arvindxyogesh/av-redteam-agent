"""Unit tests for AnthropicHTTPClient (avredteam_carla/agents/anthropic_http_client.py)
- the raw-HTTP stand-in for the anthropic SDK, needed because no SDK version
supporting tool use can be installed under this project's pinned Python 3.7
(see that module's docstring). All requests go through httpx.MockTransport
- no real network call, no API key needed beyond a dummy string."""
import json

import httpx
import pytest

from avredteam_carla.agents.anthropic_http_client import (
    AnthropicAPIError,
    AnthropicHTTPClient,
    _Block,
    _to_jsonable,
)


def _client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return AnthropicHTTPClient(api_key="test-key", http_client=httpx.Client(transport=transport), **kwargs)


def _text_response_body(text="hello"):
    return {
        "id": "msg_1",
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_create_returns_attribute_accessible_blocks():
    def handler(request):
        return httpx.Response(200, json=_text_response_body("hi there"))

    client = _client(handler)
    resp = client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert resp.stop_reason == "end_turn"
    assert len(resp.content) == 1
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "hi there"


def test_create_sends_expected_request_shape():
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_response_body())

    client = _client(handler)
    client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        system="be helpful",
        tools=[{"name": "t", "input_schema": {}}],
        messages=[{"role": "user", "content": "hi"}],
    )

    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"]
    assert captured["body"]["model"] == "claude-sonnet-5"
    assert captured["body"]["max_tokens"] == 256
    assert captured["body"]["system"] == "be helpful"
    assert captured["body"]["tools"] == [{"name": "t", "input_schema": {}}]


def test_tool_use_block_has_attribute_access():
    body = {
        "id": "msg_2",
        "model": "m",
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "run_attack_trial", "input": {"a": 1}}],
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _client(handler)
    resp = client.messages.create(model="m", max_tokens=10, messages=[])

    block = resp.content[0]
    assert block.type == "tool_use"
    assert block.id == "toolu_1"
    assert block.name == "run_attack_trial"
    assert block.input == {"a": 1}


def test_response_blocks_fed_back_into_next_request_serialize_cleanly():
    """LLMAgentSearch does messages.append({"role": "assistant", "content":
    response.content}) and then passes that same `messages` list into the
    next .create() call - the _Block objects in it must round-trip back to
    plain JSON on the outgoing request."""
    first_body = {
        "id": "msg_3",
        "model": "m",
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "toolu_2", "name": "run_attack_trial", "input": {"x": 2}}],
    }
    captured = {}
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=first_body)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_response_body())

    client = _client(handler)
    messages = [{"role": "user", "content": "start"}]

    resp1 = client.messages.create(model="m", max_tokens=10, messages=messages)
    messages.append({"role": "assistant", "content": resp1.content})
    messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "ok"}]})

    client.messages.create(model="m", max_tokens=10, messages=messages)

    sent_assistant_msg = captured["body"]["messages"][1]
    assert sent_assistant_msg["content"] == [
        {"type": "tool_use", "id": "toolu_2", "name": "run_attack_trial", "input": {"x": 2}}
    ]


def test_to_jsonable_passes_through_plain_dicts_and_lists():
    plain = {"a": [1, 2, {"b": "c"}]}
    assert _to_jsonable(plain) == plain


def test_to_jsonable_unwraps_blocks():
    block = _Block({"type": "text", "text": "hi"})
    assert _to_jsonable([block]) == [{"type": "text", "text": "hi"}]


def test_4xx_error_raises_immediately_without_retry():
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        return httpx.Response(400, text="bad request")

    client = _client(handler)
    with pytest.raises(AnthropicAPIError) as exc_info:
        client.messages.create(model="m", max_tokens=10, messages=[])

    assert exc_info.value.status_code == 400
    assert call_count["n"] == 1


def test_429_retries_then_succeeds():
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_text_response_body())

    client = _client(handler, timeout_s=5.0)
    resp = client.messages.create(model="m", max_tokens=10, messages=[])

    assert call_count["n"] == 3
    assert resp.content[0].text == "hello"


def test_5xx_exhausts_retries_and_raises():
    def handler(request):
        return httpx.Response(503, text="server error")

    client = _client(handler)
    with pytest.raises(AnthropicAPIError) as exc_info:
        client.messages.create(model="m", max_tokens=10, messages=[])

    assert exc_info.value.status_code == 503


def test_missing_api_key_raises():
    import os

    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            AnthropicHTTPClient()
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old
