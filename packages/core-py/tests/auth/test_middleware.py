"""Pure-ASGI middleware: JSON rejections, WWW-Authenticate, contextvar binding."""

from __future__ import annotations

import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from hull_core.auth.asgi import HullAuthMiddleware
from hull_core.auth.context import current_user
from hull_core.auth.middleware import Authenticator
from hull_core.auth.tokens import hash_token
from hull_core.config.settings import HullSettings, ServerSettings


async def _endpoint(request):  # noqa: ANN001
    from starlette.responses import JSONResponse

    user = current_user()
    return JSONResponse({"seen": user.uid, "namespace": user.namespace})


def _client(settings: HullSettings) -> TestClient:
    star = Starlette(routes=[Route("/whoami", _endpoint, methods=["GET"])])
    app = HullAuthMiddleware(star, Authenticator(settings))
    return TestClient(app)


def _token_settings(tmp_path: Path) -> HullSettings:
    return HullSettings(config_dir=tmp_path, server=ServerSettings(auth="token", token_hash=hash_token("real")))


def test_reject_401_json_with_www_authenticate(tmp_path: Path) -> None:
    resp = _client(_token_settings(tmp_path)).get("/whoami", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    payload = resp.json()
    assert payload["error"] == "unauthorized"
    assert "detail" in payload


def test_missing_header_401(tmp_path: Path) -> None:
    resp = _client(_token_settings(tmp_path)).get("/whoami")
    assert resp.status_code == 401


def test_forward_sets_contextvar_identity(tmp_path: Path) -> None:
    resp = _client(_token_settings(tmp_path)).get("/whoami", headers={"Authorization": "Bearer real"})
    assert resp.status_code == 200
    assert resp.json()["seen"] == "shared"
    assert resp.json()["namespace"] == "default"


def test_no_auth_binds_local_identity(tmp_path: Path) -> None:
    settings = HullSettings(config_dir=tmp_path, server=ServerSettings(auth="no-auth"))
    resp = _client(settings).get("/whoami")
    assert resp.status_code == 200
    assert resp.json()["seen"] == "local"
