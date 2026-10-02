"""Tests for the seeded identity layer (fingerprint.build_identity)."""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from hull_web import fingerprint
from hull_web.fingerprint import IdentityProfile, build_identity


@dataclass
class _Screen:
    width: int = 1280
    height: int = 720


@dataclass
class _Gpu:
    vendor: str = "Intel"
    renderer: str = "Iris Xe"


@dataclass
class _Profile:
    screen: _Screen
    gpu: _Gpu


def _install_fake_invisible_core(monkeypatch: pytest.MonkeyPatch, seed_echo: dict[str, Any]) -> None:
    """Install a fake invisible_core so no real dependency is needed."""
    calls = seed_echo

    def generate_profile(seed: int) -> _Profile:
        calls["seed"] = seed
        return _Profile(screen=_Screen(width=1600, height=900), gpu=_Gpu())

    def resolve_session_timezone(tz: str, proxy: Any) -> str:
        calls["tz"] = tz
        return "Europe/Berlin"

    def resolve_session_locale(locale: Any, proxy: Any) -> str:
        calls["locale"] = locale
        return "de-DE"

    mod = types.ModuleType("invisible_core")
    mod.generate_profile = generate_profile  # ty: ignore[unresolved-attribute]
    mod.resolve_session_timezone = resolve_session_timezone  # ty: ignore[unresolved-attribute]
    mod.resolve_session_locale = resolve_session_locale  # ty: ignore[unresolved-attribute]
    constants = types.ModuleType("invisible_core.constants")
    firefox_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0"
    constants.USER_AGENT = firefox_ua  # ty: ignore[unresolved-attribute]
    constants.OSCPU_OVERRIDE = "Windows NT 10.0; Win64; x64"  # ty: ignore[unresolved-attribute]
    constants.PLATFORM_OVERRIDE = "Win32"  # ty: ignore[unresolved-attribute]
    mod.constants = constants  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "invisible_core", mod)
    monkeypatch.setitem(sys.modules, "invisible_core.constants", constants)


def test_build_identity_coherent_and_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    echo: dict[str, Any] = {}
    _install_fake_invisible_core(monkeypatch, echo)

    identity = build_identity(seed=42)

    assert isinstance(identity, IdentityProfile)
    assert identity.seed == 42
    assert echo["seed"] == 42
    assert identity.user_agent.startswith("Mozilla/5.0")
    assert identity.timezone_id == "Europe/Berlin"
    assert identity.locale == "de-DE"
    assert identity.viewport_width == 1600
    assert identity.webgl_vendor == "Intel"  # populated, not the headless tell
    assert identity.impersonate.startswith("firefox") or identity.impersonate == "chrome131"
    assert identity.raw is not None


def test_build_identity_missing_extra_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "invisible_core", None)  # forces ImportError
    with pytest.raises(ImportError, match=r"hull-core\[identity\]"):
        build_identity(seed=1)


def test_select_impersonate_prefers_newest_firefox() -> None:
    targets = ["chrome131", "firefox133", "chrome136", "firefox133_1", "safari184"]
    # firefox133_1 does not fullmatch firefoxNNN; newest plain firefox wins.
    assert fingerprint._select_impersonate(targets) == "firefox133"

    targets2 = ["chrome131", "firefox133", "firefox147", "chrome136"]
    assert fingerprint._select_impersonate(targets2) == "firefox147"


def test_select_impersonate_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="hull_web.fingerprint"):
        result = fingerprint._select_impersonate(["chrome131", "chrome136"])
    assert result == "chrome131"
    assert any("firefoxNNN" in rec.message for rec in caplog.records)


def test_cached_identity_reuses_same_object(monkeypatch: pytest.MonkeyPatch) -> None:
    echo: dict[str, Any] = {}
    _install_fake_invisible_core(monkeypatch, echo)
    fingerprint._cached_identity.cache_clear()

    a = fingerprint._cached_identity(7, "auto", "auto")
    b = fingerprint._cached_identity(7, "auto", "auto")
    assert a is b
    fingerprint._cached_identity.cache_clear()


def test_default_seed_env_and_random(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fingerprint.IDENTITY_SEED_ENV, "12345")
    fingerprint._process_seed.cache_clear()
    assert fingerprint.default_seed() == 12345

    monkeypatch.setenv(fingerprint.IDENTITY_SEED_ENV, "not-an-int")
    fingerprint._process_seed.cache_clear()
    random_seed = fingerprint.default_seed()
    assert 0 < random_seed < 2**31
    fingerprint._process_seed.cache_clear()
