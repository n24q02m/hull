"""Pure-ASGI middleware: JSON rejections, WWW-Authenticate, contextvar binding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


class _OkWithoutContext:
    """Authenticator that reports success but hands back no identity."""

    def authenticate(self, authz):  # noqa: ANN001, ANN003
        return SimpleNamespace(ok=True, status=None, detail="", context=None)


def test_ok_without_context_fails_closed(tmp_path: Path) -> None:
    """A drifting authenticator must never bind a null identity into the contextvar."""
    before = current_user()
    star = Starlette(routes=[Route("/whoami", _endpoint, methods=["GET"])])
    # ty: the stub is the drift this test simulates - a non-Authenticator implementation.
    middleware = HullAuthMiddleware(star, _OkWithoutContext())  # ty: ignore[invalid-argument-type]
    resp = TestClient(middleware).get("/whoami")

    assert resp.status_code == 500
    assert resp.json()["error"] == "server misconfiguration"
    assert current_user() == before, "no identity may be bound for a rejected request"


async def _public(request):  # noqa: ANN001
    from starlette.responses import JSONResponse

    return JSONResponse({"public": True})


def _client_with_public(settings: HullSettings) -> TestClient:
    star = Starlette(
        routes=[
            Route("/whoami", _endpoint, methods=["GET"]),
            Route("/health", _public, methods=["GET"]),
            Route("/health/deep", _public, methods=["GET"]),
        ]
    )
    return TestClient(HullAuthMiddleware(star, Authenticator(settings), public_paths=("/health",)))


def test_public_path_skips_auth(tmp_path: Path) -> None:
    resp = _client_with_public(_token_settings(tmp_path)).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"public": True}


def test_public_paths_are_exact_match(tmp_path: Path) -> None:
    client = _client_with_public(_token_settings(tmp_path))
    assert client.get("/health/deep").status_code == 401, "a prefix is not an allow-list entry"
    assert client.get("/whoami").status_code == 401
    assert client.get("/whoami", headers={"Authorization": "Bearer real"}).status_code == 200


def test_public_paths_default_empty(tmp_path: Path) -> None:
    star = Starlette(routes=[Route("/health", _public, methods=["GET"])])
    middleware = HullAuthMiddleware(star, Authenticator(_token_settings(tmp_path)))
    assert middleware.public_paths == frozenset()
    assert TestClient(middleware).get("/health").status_code == 401
