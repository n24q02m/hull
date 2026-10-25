"""Bundled client identity: CLI > env > bundled, kill-switch, token/client binding."""

from __future__ import annotations

import pytest

from hull_core.auth import BundledClientSpec, ResolvedClient, resolve_bundled_client, token_client_mismatch

SPEC = BundledClientSpec(
    provider="google",
    env_id="GOOGLE_CLIENT_ID",
    env_secret="GOOGLE_CLIENT_SECRET",
    bundled_id="bundled-id",
    bundled_secret="bundled-secret",
    use_bundled_env="USE_BUNDLED_GOOGLE_CLIENT",
)
NO_SECRET_SPEC = BundledClientSpec(
    provider="microsoft",
    env_id="MS_CLIENT_ID",
    env_secret="MS_CLIENT_SECRET",
    bundled_id="ms-public",
    bundled_secret="",
    use_bundled_env="USE_BUNDLED_MS_CLIENT",
    requires_secret=False,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "USE_BUNDLED_GOOGLE_CLIENT",
        "MS_CLIENT_ID",
        "MS_CLIENT_SECRET",
        "USE_BUNDLED_MS_CLIENT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_bundled_default() -> None:
    assert resolve_bundled_client(SPEC) == ResolvedClient("bundled-id", "bundled-secret", "bundled")


def test_cli_pair_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    assert resolve_bundled_client(SPEC, "cli-id", "cli-secret") == ResolvedClient("cli-id", "cli-secret", "cli")


def test_env_pair_wins_over_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    assert resolve_bundled_client(SPEC) == ResolvedClient("env-id", "env-secret", "env")


def test_env_used_when_cli_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    assert resolve_bundled_client(SPEC, "", None).source == "env"


@pytest.mark.parametrize(("cid", "secret", "match"), [("cli-id", None, "set together"), (None, "s", "without an id")])
def test_cli_half_pair_rejected(cid: str | None, secret: str | None, match: str) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        resolve_bundled_client(SPEC, cid, secret)
    assert "--client-id/--client-secret" in str(excinfo.value)
    assert "google" in str(excinfo.value)


@pytest.mark.parametrize(
    ("env", "match"),
    [({"GOOGLE_CLIENT_ID": "e"}, "set together"), ({"GOOGLE_CLIENT_SECRET": "e"}, "without an id")],
)
def test_env_half_pair_rejected(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], match: str) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value.replace("e", "VALUE-MARKER"))
    with pytest.raises(ValueError, match=match) as excinfo:
        resolve_bundled_client(SPEC)
    assert "GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET" in str(excinfo.value)
    assert "VALUE-MARKER" not in str(excinfo.value), "values are never echoed"


def test_public_client_needs_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_bundled_client(NO_SECRET_SPEC, "cli-id") == ResolvedClient("cli-id", "", "cli")
    monkeypatch.setenv("MS_CLIENT_ID", "env-id")
    assert resolve_bundled_client(NO_SECRET_SPEC) == ResolvedClient("env-id", "", "env")


def test_public_client_secret_without_id_still_rejected() -> None:
    with pytest.raises(ValueError, match="without an id"):
        resolve_bundled_client(NO_SECRET_SPEC, None, "s")


@pytest.mark.parametrize("value", ["0", "false", "No", " OFF "])
def test_kill_switch_fails_loud(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("USE_BUNDLED_GOOGLE_CLIENT", value)
    with pytest.raises(RuntimeError, match="USE_BUNDLED_GOOGLE_CLIENT disables the bundled client"):
        resolve_bundled_client(SPEC)


@pytest.mark.parametrize("value", ["1", "true", "yes", ""])
def test_kill_switch_truthy_keeps_bundled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("USE_BUNDLED_GOOGLE_CLIENT", value)
    assert resolve_bundled_client(SPEC).source == "bundled"


def test_kill_switch_does_not_block_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_BUNDLED_GOOGLE_CLIENT", "0")
    assert resolve_bundled_client(SPEC, "cli-id", "cli-secret").source == "cli"
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    assert resolve_bundled_client(SPEC).source == "env"


@pytest.mark.parametrize(
    ("token", "client_id", "expected"),
    [
        (None, "a", False),
        ({"client_id": "a", "refresh_token": "r"}, "a", False),
        ({"client_id": "b", "refresh_token": "r"}, "a", True),
        ({"refresh_token": "r"}, "a", True),
    ],
)
def test_token_client_mismatch(token: dict | None, client_id: str, expected: bool) -> None:
    assert token_client_mismatch(token, client_id) is expected
