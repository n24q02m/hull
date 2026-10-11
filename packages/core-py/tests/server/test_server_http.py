"""Consumer HTTP runner: /mcp behind hull auth, public /health, loopback guard.

The app tests drive the real FastMCP streamable-HTTP app through TestClient
(lifespan included); ``run_http_server`` tests replace uvicorn's Server/Config
so nothing binds a socket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from hull_core.auth.context import current_user
from hull_core.auth.tokens import hash_token
from hull_core.config.settings import HullSettings, ServerSettings
from hull_core.server import ServerConfigError, build_http_app, run_http_server
from hull_core.server import http as http_mod

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}


def _mcp() -> FastMCP:
    mcp = FastMCP("consumer-test")

    @mcp.tool
    def whoami() -> dict[str, str]:
        user = current_user()
        return {"uid": user.uid, "namespace": user.namespace}

    return mcp


def _token_settings(tmp_path: Path) -> HullSettings:
    return HullSettings(config_dir=tmp_path, server=ServerSettings(auth="token", token_hash=hash_token("tok")))


def _multi_settings(tmp_path: Path) -> HullSettings:
    users = tmp_path / "users.toml"
    users.write_text(
        f'[users.alice]\ntoken_hash = "{hash_token("alice-token")}"\nenabled = true\nnamespace = "alice"\n',
        encoding="utf-8",
    )
    return HullSettings(config_dir=tmp_path, server=ServerSettings(auth="multi", users_file=users))


def _payload(resp) -> dict[str, Any]:  # noqa: ANN001
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
    return resp.json()


# --- build_http_app -----------------------------------------------------------------


def test_health_is_public_in_token_mode(tmp_path: Path) -> None:
    with TestClient(build_http_app(_mcp(), _token_settings(tmp_path))) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_mcp_requires_token(tmp_path: Path) -> None:
    with TestClient(build_http_app(_mcp(), _token_settings(tmp_path))) as client:
        missing = client.post("/mcp", content=json.dumps(INIT), headers=HEADERS)
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        wrong = client.post("/mcp", content=json.dumps(INIT), headers={**HEADERS, "Authorization": "Bearer nope"})
        assert wrong.status_code == 401


def test_mcp_handshake_and_tool_call_with_identity(tmp_path: Path) -> None:
    auth = {**HEADERS, "Authorization": "Bearer tok"}
    with TestClient(build_http_app(_mcp(), _token_settings(tmp_path))) as client:
        init = client.post("/mcp", content=json.dumps(INIT), headers=auth)
        assert init.status_code == 200, init.text
        session = init.headers["mcp-session-id"]
        auth["Mcp-Session-Id"] = session
        notified = client.post(
            "/mcp", content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), headers=auth
        )
        assert notified.status_code in (200, 202)
        call = client.post(
            "/mcp",
            content=json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "whoami", "arguments": {}}}
            ),
            headers=auth,
        )
        assert call.status_code == 200, call.text
        result = _payload(call)["result"]
        assert not result.get("isError"), result
        assert json.loads(result["content"][0]["text"]) == {"uid": "shared", "namespace": "default"}


def test_health_rejects_non_get(tmp_path: Path) -> None:
    with TestClient(build_http_app(_mcp(), _token_settings(tmp_path))) as client:
        assert client.post("/health").status_code == 405


def test_multi_mode_with_explicit_users(tmp_path: Path) -> None:
    from hull_core.server import load_users_for

    settings = _multi_settings(tmp_path)
    with TestClient(build_http_app(_mcp(), settings, users=load_users_for(settings))) as client:
        ok = client.post("/mcp", content=json.dumps(INIT), headers={**HEADERS, "Authorization": "Bearer alice-token"})
        assert ok.status_code == 200, ok.text
        assert client.get("/health").status_code == 200


# --- run_http_server ------------------------------------------------------------------


class _FakeServer:
    instances: list[_FakeServer] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.served = False
        _FakeServer.instances.append(self)

    async def serve(self) -> None:
        self.served = True


class _FakeConfig:
    def __init__(self, app: Any, **kwargs: Any) -> None:
        self.app = app
        self.kwargs = kwargs


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> type[_FakeServer]:
    _FakeServer.instances = []
    monkeypatch.setattr(http_mod.uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(http_mod.uvicorn, "Config", _FakeConfig)
    return _FakeServer


async def test_run_uses_settings_host_and_port(tmp_path: Path, fake_uvicorn: type[_FakeServer]) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="no-auth", host="127.0.0.1", port=8123))
    await run_http_server(_mcp(), settings)
    (server,) = fake_uvicorn.instances
    assert server.served
    assert server.config.kwargs == {"host": "127.0.0.1", "port": 8123, "log_level": "warning"}


async def test_run_explicit_host_and_port_override(tmp_path: Path, fake_uvicorn: type[_FakeServer]) -> None:
    await run_http_server(_mcp(), _token_settings(tmp_path), host="0.0.0.0", port=9001)
    (server,) = fake_uvicorn.instances
    assert server.config.kwargs["host"] == "0.0.0.0"
    assert server.config.kwargs["port"] == 9001


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
async def test_run_no_auth_refuses_non_loopback(tmp_path: Path, fake_uvicorn: type[_FakeServer], host: str) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="no-auth"))
    with pytest.raises(ServerConfigError, match="only permits loopback"):
        await run_http_server(_mcp(), settings, host=host)
    assert fake_uvicorn.instances == []


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.8.8", "localhost"])
async def test_run_no_auth_allows_loopback(tmp_path: Path, fake_uvicorn: type[_FakeServer], host: str) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="no-auth"))
    await run_http_server(_mcp(), settings, host=host)
    assert fake_uvicorn.instances[0].config.kwargs["host"] == host


async def test_run_multi_loads_users_from_settings(tmp_path: Path, fake_uvicorn: type[_FakeServer]) -> None:
    await run_http_server(_mcp(), _multi_settings(tmp_path), host="0.0.0.0")
    (server,) = fake_uvicorn.instances
    with TestClient(server.config.app) as client:
        ok = client.post("/mcp", content=json.dumps(INIT), headers={**HEADERS, "Authorization": "Bearer alice-token"})
        assert ok.status_code == 200, ok.text


async def test_run_multi_without_users_file_fails(tmp_path: Path, fake_uvicorn: type[_FakeServer]) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="multi"))
    with pytest.raises(RuntimeError, match="users_file"):
        await run_http_server(_mcp(), settings, host="0.0.0.0")
    assert fake_uvicorn.instances == []


async def test_run_multi_with_injected_users_skips_file(tmp_path: Path, fake_uvicorn: type[_FakeServer]) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="multi"))
    await run_http_server(_mcp(), settings, host="0.0.0.0", users={})
    assert fake_uvicorn.instances[0].served
