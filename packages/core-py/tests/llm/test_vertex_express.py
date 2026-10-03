"""Vertex AI Express passthrough adapter tests (litellm#21036 workaround).

Covers request building (endpoint, header, generationConfig), multimodal
inline-image translation, response translation (finish_reason / usage /
empty-candidate), the httpx sync+async calls, and the metrics hook.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from hull_core.llm import vertex_express as ve

# ---------------------------------------------------------------------------
# build_express_request
# ---------------------------------------------------------------------------


def test_build_request_endpoint_header_and_config():
    url, headers, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        response_format={"type": "json_object"},
    )
    assert url == (
        "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-3.5-flash:generateContent"
    )
    # api_key in the header, NEVER the query string.
    assert headers["x-goog-api-key"] == "KEY"
    assert "key=" not in url
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    gen = body["generationConfig"]
    assert gen["temperature"] == 0.7
    assert gen["maxOutputTokens"] == 1024
    assert gen["topP"] == 0.9
    assert gen["responseMimeType"] == "application/json"


def test_build_request_strips_express_prefix():
    url, _, _ = ve.build_express_request(
        model="vertex_express/gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
    )
    assert "models/gemini-3.5-flash:generateContent" in url
    assert "vertex_express" not in url


def test_build_request_hoists_system_to_system_instruction():
    _, _, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        api_key="KEY",
    )
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    # System message must NOT leak into contents.
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_build_request_two_system_messages_merge():
    _, _, body = ve.build_express_request(
        model="m",
        messages=[
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "hi"},
        ],
        api_key="KEY",
    )
    assert body["systemInstruction"] == {"parts": [{"text": "a"}, {"text": "b"}]}


def test_build_request_maps_assistant_role_to_model():
    _, _, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        api_key="KEY",
    )
    assert [c["role"] for c in body["contents"]] == ["user", "model"]


def test_build_request_multimodal_inline_image():
    """An [image_url data: part, text part] message must become an inlineData part."""
    b64 = "aGVsbG8="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "extract bubbles"},
            ],
        }
    ]
    _, _, body = ve.build_express_request(model="gemini-3.5-flash", messages=messages, api_key="KEY")
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"inlineData": {"mimeType": "image/png", "data": b64}}
    assert parts[1] == {"text": "extract bubbles"}


def test_build_request_rejects_remote_image_url():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }
    ]
    with pytest.raises(ve.VertexExpressError, match="inline data"):
        ve.build_express_request(model="m", messages=messages, api_key="KEY")


def test_build_request_rejects_unknown_block_type():
    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": {"url": "data:x"}}]},
    ]
    with pytest.raises(ve.VertexExpressError, match="unsupported content block"):
        ve.build_express_request(model="m", messages=messages, api_key="KEY")


def test_build_request_rejects_streaming():
    with pytest.raises(ve.VertexExpressError, match="streaming"):
        ve.build_express_request(model="m", messages=[{"role": "user", "content": "hi"}], api_key="KEY", stream=True)


def test_build_request_stop_sequences():
    _, _, body = ve.build_express_request(
        model="m", messages=[{"role": "user", "content": "hi"}], api_key="KEY", stop="END"
    )
    assert body["generationConfig"]["stopSequences"] == ["END"]


def test_content_parts_none_returns_empty():
    """An assistant/tool message with content=None must not iterate None (TypeError)."""
    assert ve._content_parts(None) == []


def test_build_request_assistant_content_none_no_crash():
    _, _, body = ve.build_express_request(
        model="m",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ],
        api_key="KEY",
    )
    assert body["contents"][1] == {"role": "model", "parts": []}


def test_build_request_omits_generation_config_when_no_params():
    _, _, body = ve.build_express_request(model="m", messages=[{"role": "user", "content": "hi"}], api_key="KEY")
    assert "generationConfig" not in body


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


def test_translate_response_content_finish_usage():
    raw = {
        "candidates": [{"content": {"parts": [{"text": "Xin "}, {"text": "chao"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }
    r = ve.translate_response(raw, model="vertex_express/gemini-3.5-flash")
    assert r["choices"][0]["message"]["content"] == "Xin chao"
    assert r.choices[0].finish_reason == "stop"
    assert r.model == "gemini-3.5-flash"
    usage = dict(r["usage"])
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2


def test_translate_response_max_tokens_raises_instead_of_returning_partial():
    """A truncated completion is not a usable answer, so it must not read as success.

    Returning it as an ordinary 200 is what let 2xx-only billing middleware
    charge a credit for half a chapter.
    """
    raw = {
        "candidates": [{"content": {"parts": [{"text": "half a chapt"}]}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"thoughtsTokenCount": 3932, "candidatesTokenCount": 160},
    }
    with pytest.raises(ve.VertexExpressError, match="MAX_TOKENS") as exc:
        ve.translate_response(raw, model="m")
    # The error has to say what to do about it, not just that it happened.
    assert "3932 thinking" in str(exc.value)
    assert "split it" in str(exc.value)


def test_translate_response_no_candidate_raises_with_block_reason():
    raw = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(ve.VertexExpressError, match="SAFETY"):
        ve.translate_response(raw, model="m")


def test_translate_response_raw_finish_stashed_when_choice_coerced():
    """litellm's Choices coerces an unmapped enum (OTHER/MALFORMED/SPII) to
    'stop'; the RAW reason survives on _hidden_params so the empty-response log
    is not masked."""
    raw = {"candidates": [{"content": {"parts": []}, "finishReason": "OTHER"}]}
    r = ve.translate_response(raw, model="m")
    assert r._hidden_params["vertex_finish_reason"] == "OTHER"


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_from_prefixed_env(monkeypatch):
    monkeypatch.setenv("KLPRISM_VERTEX_EXPRESS_KEY", "envkey")
    assert ve._resolve_api_key(None, "KLPRISM_") == "envkey"


def test_resolve_api_key_falls_back_to_default_env(monkeypatch):
    monkeypatch.delenv("KLPRISM_VERTEX_EXPRESS_KEY", raising=False)
    monkeypatch.setenv("HULL_VERTEX_EXPRESS_KEY", "hullkey")
    assert ve._resolve_api_key(None, "KLPRISM_") == "hullkey"


def test_resolve_api_key_explicit_wins(monkeypatch):
    monkeypatch.setenv("HULL_VERTEX_EXPRESS_KEY", "envkey")
    assert ve._resolve_api_key("explicit", "HULL_") == "explicit"


def test_resolve_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("KLPRISM_VERTEX_EXPRESS_KEY", raising=False)
    monkeypatch.delenv("HULL_VERTEX_EXPRESS_KEY", raising=False)
    with pytest.raises(ve.VertexExpressError, match="requires an API key"):
        ve._resolve_api_key(None, "KLPRISM_")


# ---------------------------------------------------------------------------
# httpx sync + async calls (patched transport)
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    # Pooled (no context manager): the adapter reuses one client per loop.
    last: ClassVar[dict]

    def __init__(self, resp: _FakeResp):
        self._resp = resp

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.last = {"url": url, "headers": headers, "json": json}
        return self._resp


class _FakeSyncClient:
    last: ClassVar[dict]

    def __init__(self, resp: _FakeResp):
        self._resp = resp

    def post(self, url, headers=None, json=None):
        _FakeSyncClient.last = {"url": url, "headers": headers, "json": json}
        return self._resp


_OK_PAYLOAD = {
    "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


async def test_acompletion_express_success(monkeypatch):
    fake = _FakeAsyncClient(_FakeResp(200, _OK_PAYLOAD))
    monkeypatch.setattr(ve, "_get_async_client", lambda: fake)
    out = await ve.acompletion_express(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
    )
    assert out["choices"][0]["message"]["content"] == "hello"
    assert _FakeAsyncClient.last["headers"]["x-goog-api-key"] == "KEY"


async def test_acompletion_express_non_2xx_raises(monkeypatch):
    fake = _FakeAsyncClient(_FakeResp(500, {}, text="boom"))
    monkeypatch.setattr(ve, "_get_async_client", lambda: fake)
    with pytest.raises(ve.VertexExpressError, match="HTTP 500"):
        await ve.acompletion_express(model="m", messages=[{"role": "user", "content": "hi"}], api_key="KEY")


def test_completion_express_success(monkeypatch):
    fake = _FakeSyncClient(_FakeResp(200, _OK_PAYLOAD))
    monkeypatch.setattr(ve, "_get_sync_client", lambda: fake)
    out = ve.completion_express(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
    )
    assert out["choices"][0]["message"]["content"] == "hello"


def test_completion_express_non_2xx_raises(monkeypatch):
    fake = _FakeSyncClient(_FakeResp(403, {}, text="permission denied"))
    monkeypatch.setattr(ve, "_get_sync_client", lambda: fake)
    with pytest.raises(ve.VertexExpressError, match="HTTP 403"):
        ve.completion_express(model="m", messages=[{"role": "user", "content": "hi"}], api_key="KEY")


def test_sync_client_is_pooled():
    """Repeated getter calls reuse one client (connection pooling, no per-call TLS)."""
    a = ve._get_sync_client()
    b = ve._get_sync_client()
    assert a is b and not a.is_closed


async def test_async_client_is_pooled_within_loop():
    a = ve._get_async_client()
    b = ve._get_async_client()
    assert a is b and not a.is_closed


# ---------------------------------------------------------------------------
# thinkingConfig — Gemini bills thoughts against maxOutputTokens, so an
# unbounded thinking budget silently truncates long completions.
# ---------------------------------------------------------------------------


def test_build_request_emits_thinking_config_when_budget_given():
    _, _, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
        thinking_budget=1024,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}


def test_build_request_omits_thinking_config_by_default():
    """A model with no thinking mode must not receive a thinkingConfig it can reject."""
    _, _, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
        max_tokens=1024,
    )
    assert "thinkingConfig" not in body["generationConfig"]


def test_build_request_thinking_budget_zero_is_emitted_not_dropped():
    """0 disables thinking; it must survive the `is not None` guard, unlike a falsy skip."""
    _, _, body = ve.build_express_request(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="KEY",
        thinking_budget=0,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


# ---------------------------------------------------------------------------
# finish_reason_observer hook
# ---------------------------------------------------------------------------


def test_finish_reason_observer_is_called(monkeypatch):
    """The host's metrics hook receives (bare model, raw finishReason)."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(ve, "finish_reason_observer", lambda m, r: seen.append((m, r)))
    ve.translate_response(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
        model="vertex_express/gemini-3.5-flash",
    )
    assert seen == [("gemini-3.5-flash", "STOP")]


def test_finish_reason_observer_failure_does_not_break_call(monkeypatch):
    def boom(_m, _r):
        raise RuntimeError("metrics down")

    monkeypatch.setattr(ve, "finish_reason_observer", boom)
    r = ve.translate_response(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
        model="m",
    )
    assert r["choices"][0]["message"]["content"] == "ok"
