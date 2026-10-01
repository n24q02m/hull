"""Full-stack server tests: MCP handshake, §4 isolation, 401/403 over HTTP.

These drive the REAL Starlette app (FastMCP streamable HTTP + hull auth
middleware) through TestClient inside a `with` block so the app lifespan
(the MCP session manager) runs, exactly as uvicorn does for `server start`.
"""

from __future__ import annotations

import json
from pathlib import Path

from starlette.testclient import TestClient

from hull_core.auth.tokens import hash_token
from hull_core.config.settings import HullSettings, ServerSettings
from hull_core.server.app import build_app
from hull_core.storage.sqlite import HullDatabase

ALICE_TOKEN = "alice-token"
BOB_TOKEN = "bob-token"

USERS_TOML = f"""\
[users.alice]
token_hash = "{hash_token(ALICE_TOKEN)}"
enabled = true
namespace = "alice"

[users.bob]
token_hash = "{hash_token(BOB_TOKEN)}"
enabled = true
namespace = "bob"

[users.carol]
token_hash = "{hash_token("carol-token")}"
enabled = false
namespace = "carol"
"""

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _multi_settings(tmp_path: Path) -> HullSettings:
    (tmp_path / "users.toml").write_text(USERS_TOML, encoding="utf-8")
    return HullSettings(config_dir=tmp_path, server=ServerSettings(auth="multi", users_file=tmp_path / "users.toml"))


def _mcp_call(client: TestClient, token: str, session: str | None, method: str, params: dict | None = None):
    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    if session:
        headers["Mcp-Session-Id"] = session
    body: dict = {"jsonrpc": "2.0", "method": method}
    if method != "notifications/initialized":
        body["id"] = 1
    if params is not None:
        body["params"] = params
    return client.post("/mcp", content=json.dumps(body), headers=headers)


def _initialize(client: TestClient, token: str) -> str:
    resp = _mcp_call(
        client,
        token,
        None,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    assert resp.status_code == 200, resp.text
    session = resp.headers.get("mcp-session-id")
    assert session, "server must issue a streamable-http session id"
    notified = _mcp_call(client, token, session, "notifications/initialized")
    assert notified.status_code in (200, 202), notified.text
    return session


def _payload_text(resp) -> str:  # noqa: ANN001
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                return line[5:].strip()
    return resp.text


def _call_tool(client: TestClient, token: str, session: str, name: str, arguments: dict) -> dict:
    resp = _mcp_call(
        client,
        token,
        session,
        "tools/call",
        {"name": name, "arguments": arguments},
    )
    assert resp.status_code == 200, resp.text
    payload = json.loads(_payload_text(resp))
    assert not payload["result"].get("isError"), payload
    return json.loads(payload["result"]["content"][0]["text"])


def test_wrong_token_401_before_mcp(tmp_path: Path) -> None:
    with TestClient(build_app(_multi_settings(tmp_path), database=HullDatabase(tmp_path / "a.db"))) as client:
        resp = _mcp_call(client, "wrong-token", None, "initialize", {"protocolVersion": "2025-06-18"})
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"


def test_missing_token_401(tmp_path: Path) -> None:
    with TestClient(build_app(_multi_settings(tmp_path), database=HullDatabase(tmp_path / "b.db"))) as client:
        resp = client.post(
            "/mcp",
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            headers=HEADERS,
        )
        assert resp.status_code == 401


def test_disabled_user_403(tmp_path: Path) -> None:
    with TestClient(build_app(_multi_settings(tmp_path), database=HullDatabase(tmp_path / "c.db"))) as client:
        resp = _mcp_call(client, "carol-token", None, "initialize", {"protocolVersion": "2025-06-18"})
        assert resp.status_code == 403
        assert resp.json()["error"] == "forbidden"
        assert "carol" in resp.json()["detail"]


def test_initialize_handshake_per_token(tmp_path: Path) -> None:
    with TestClient(build_app(_multi_settings(tmp_path), database=HullDatabase(tmp_path / "d.db"))) as client:
        for token in (ALICE_TOKEN, BOB_TOKEN):
            assert _initialize(client, token)


def test_mode3_two_tokens_two_namespaces_one_process(tmp_path: Path) -> None:
    """THE §4 isolation proof: 2 users, 2 tokens, 2 namespaces, 1 server."""
    with TestClient(build_app(_multi_settings(tmp_path), database=HullDatabase(tmp_path / "e.db"))) as client:
        session_a = _initialize(client, ALICE_TOKEN)
        session_b = _initialize(client, BOB_TOKEN)

        who_a = _call_tool(client, ALICE_TOKEN, session_a, "namespace_info", {})
        who_b = _call_tool(client, BOB_TOKEN, session_b, "namespace_info", {})
        assert who_a == {"uid": "alice", "namespace": "alice", "mode": "multi"}
        assert who_b == {"uid": "bob", "namespace": "bob", "mode": "multi"}

        _call_tool(client, ALICE_TOKEN, session_a, "kv_put", {"key": "topic", "value": "alice-data"})
        _call_tool(client, BOB_TOKEN, session_b, "kv_put", {"key": "topic", "value": "bob-data"})

        got_a = _call_tool(client, ALICE_TOKEN, session_a, "kv_get", {"key": "topic"})
        got_b = _call_tool(client, BOB_TOKEN, session_b, "kv_get", {"key": "topic"})
        assert got_a["value"] == "alice-data", "alice must see only her namespace"
        assert got_b["value"] == "bob-data", "bob must see only his namespace"


def test_shared_token_mode_one_namespace(tmp_path: Path) -> None:
    settings = HullSettings(
        config_dir=tmp_path,
        server=ServerSettings(auth="token", token_hash=hash_token("shared-token")),
    )
    with TestClient(build_app(settings, database=HullDatabase(tmp_path / "f.db"))) as client:
        session = _initialize(client, "shared-token")
        who = _call_tool(client, "shared-token", session, "namespace_info", {})
        assert who == {"uid": "shared", "namespace": "default", "mode": "token"}
        _call_tool(client, "shared-token", session, "kv_put", {"key": "k", "value": "v"})
        assert _call_tool(client, "shared-token", session, "kv_get", {"key": "k"})["value"] == "v"
