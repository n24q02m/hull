"""Tests for MCP 2026-07-28 extensions: headers, files, tasks, and elicitation fallback."""

from __future__ import annotations

import base64
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from hull_core.server import (
    FILES_EXTENSION_ID,
    MCP_2026_07_28_HEADERS,
    TASKS_EXTENSION_ID,
    ElicitationOutcome,
    TaskConfig,
    build_file_payload,
    build_task_config,
    elicit_with_fallback,
)


def test_extension_constants() -> None:
    assert MCP_2026_07_28_HEADERS == (
        "Mcp-Method",
        "Mcp-Name",
        "Mcp-Protocol-Version",
        "Mcp-Session-Id",
    )
    assert FILES_EXTENSION_ID == "io.modelcontextprotocol/files"
    assert TASKS_EXTENSION_ID == "io.modelcontextprotocol/tasks"


# --- build_file_payload ----------------------------------------------------------------


def test_build_file_payload_json_dict_and_list() -> None:
    payload = build_file_payload({"ok": True, "count": 2}, "report.json", description="Run summary")
    assert payload["extension"] == FILES_EXTENSION_ID
    assert payload["filename"] == "report.json"
    assert payload["mimeType"] == "application/json"
    assert payload["description"] == "Run summary"
    assert payload["resource"]["uri"] == "file:///report.json"
    assert '"count": 2' in payload["resource"]["text"]
    assert payload["_meta"][FILES_EXTENSION_ID]["uri"] == "file:///report.json"

    list_payload = build_file_payload([1, 2, 3], "items.json", uri="https://example.com/items.json")
    assert list_payload["resource"]["uri"] == "https://example.com/items.json"
    assert list_payload["resource"]["text"] == "[1, 2, 3]"


def test_build_file_payload_text_and_binary() -> None:
    text_payload = build_file_payload("hello\nworld", "notes.txt")
    assert text_payload["mimeType"] == "text/plain"
    assert text_payload["sizeBytes"] == 11
    assert text_payload["resource"]["text"] == "hello\nworld"

    raw = b"\x00\x01\x02\xff"
    bin_payload = build_file_payload(bytearray(raw), "blob.bin")
    assert bin_payload["mimeType"] == "application/octet-stream"
    assert bin_payload["sizeBytes"] == 4
    assert base64.b64decode(bin_payload["resource"]["blob"]) == raw


def test_build_file_payload_validation_errors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_file_payload("abc", "   ")
    with pytest.raises(TypeError, match="Unsupported file payload type"):
        build_file_payload(cast(Any, 123), "bad.txt")


# --- build_task_config -----------------------------------------------------------------


def test_build_task_config_modes_and_validation() -> None:
    cfg = build_task_config("required", poll_interval_ms=2500)
    assert isinstance(cfg, TaskConfig)
    assert cfg.mode == "required"
    assert cfg.poll_interval == timedelta(milliseconds=2500)

    default_cfg = build_task_config()
    assert default_cfg.mode == "optional"

    with pytest.raises(ValueError, match="Invalid task mode"):
        build_task_config(cast(Any, "invalid"))
    with pytest.raises(ValueError, match="poll_interval_ms must be positive"):
        build_task_config("optional", poll_interval_ms=0)


# --- elicit_with_fallback --------------------------------------------------------------


async def test_elicit_with_fallback_headless_and_error_cases() -> None:
    none_outcome = await elicit_with_fallback(None, "Confirm?", bool, default=True)
    assert none_outcome == ElicitationOutcome(
        status="fallback",
        data=True,
        used_fallback=True,
        reason="no_elicitation_context",
    )

    no_method_outcome = await elicit_with_fallback(SimpleNamespace(), "Confirm?", bool, default=False)
    assert no_method_outcome.status == "fallback"
    assert no_method_outcome.data is False
    assert no_method_outcome.used_fallback is True

    class _ErrCtx:
        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        async def elicit(self, *_args: Any, **_kwargs: Any) -> Any:
            raise self._exc

    for exc in (
        ToolError("modern era"),
        MCPError(-32601, "Method not found"),
        RuntimeError("no session"),
        NotImplementedError("unsupported"),
    ):
        outcome = await elicit_with_fallback(_ErrCtx(exc), "Name?", str, default="anon")
        assert outcome.status == "fallback"
        assert outcome.data == "anon"
        assert outcome.used_fallback is True
        assert outcome.reason


async def test_elicit_with_fallback_accept_decline_cancel_and_unexpected() -> None:
    class _FixedCtx:
        def __init__(self, result: Any) -> None:
            self._result = result

        async def elicit(self, *_args: Any, **_kwargs: Any) -> Any:
            return self._result

    acc = await elicit_with_fallback(_FixedCtx(AcceptedElicitation(data="yes")), "Ok?", str, default="no")
    assert acc == ElicitationOutcome(status="accepted", data="yes", used_fallback=False)

    dec = await elicit_with_fallback(_FixedCtx(DeclinedElicitation()), "Ok?", str, default="no")
    assert dec == ElicitationOutcome(status="declined", data=None, used_fallback=False, reason="declined")

    can = await elicit_with_fallback(_FixedCtx(CancelledElicitation()), "Ok?", str, default="no")
    assert can == ElicitationOutcome(status="cancelled", data=None, used_fallback=False, reason="cancelled")

    with pytest.raises(TypeError, match="Unexpected elicitation result"):
        await elicit_with_fallback(_FixedCtx("bogus"), "Ok?", str)


async def test_elicit_with_fallback_real_fastmcp_client() -> None:
    class _ConfirmInput(BaseModel):
        approved: bool

    mcp = FastMCP("elicit-test")

    @mcp.tool
    async def deploy(env: str, ctx: Context) -> dict[str, Any]:
        outcome = await elicit_with_fallback(
            ctx,
            f"Approve deploy to {env}?",
            _ConfirmInput,
            default=_ConfirmInput(approved=False),
        )
        approved = outcome.data.approved if outcome.data is not None else False
        return {"status": outcome.status, "approved": approved, "used_fallback": outcome.used_fallback}

    # Modern client without MRTR or headless client -> falls back cleanly.
    async with Client(mcp) as modern_client:
        res = await modern_client.call_tool("deploy", {"env": "staging"})
        assert res.data == {"status": "fallback", "approved": False, "used_fallback": True}

    # Legacy client without elicitation handler -> falls back cleanly.
    async with Client(mcp, mode="legacy") as headless_legacy_client:
        res = await headless_legacy_client.call_tool("deploy", {"env": "staging"})
        assert res.data == {"status": "fallback", "approved": False, "used_fallback": True}

    # Legacy client with elicitation handler accepting -> returns accepted data.
    async def accept_handler(_message: str, _response_type: Any, _params: Any, _context: Any) -> ElicitResult:
        return ElicitResult(action="accept", content={"approved": True})

    async with Client(mcp, elicitation_handler=accept_handler, mode="legacy") as interactive_client:
        res = await interactive_client.call_tool("deploy", {"env": "prod"})
        assert res.data == {"status": "accepted", "approved": True, "used_fallback": False}
