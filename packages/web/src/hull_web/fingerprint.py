"""Browser identity, seeded and reproducible — replacing the domain-hash version.

The previous implementation called FingerprintGenerator().generate() with no
seed, picked viewport/locale/timezone independently from three bytes of
sha256(domain), and reported empty WebGL strings when BrowserForge had no GPU
profile. Three problems, all of them things a page can test:

  1. No seed. Two processes on the same domain got two different people.
  2. Independent picks. de-DE with America/New_York and UTC was reachable.
  3. Empty WebGL. That is the headless signature, not a way to avoid it.

invisible_core samples the whole profile from a Bayesian network, so screen,
GPU, fonts, audio and codec agree with each other by construction, and one
integer seed reproduces the same machine everywhere.

invisible_core is an OPTIONAL dependency (``hull-core[identity]``): every
import of it is lazy, inside the functions that need it, so ``hull_web``
imports fine without the extra installed.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, get_args

logger = logging.getLogger(__name__)

#: One identity for the whole process, not one per domain. A scraper that is a
#: different person on every host is the inconsistency this exists to remove:
#: sites sharing a CDN, an analytics vendor or a login domain see the switches.
IDENTITY_SEED_ENV = "WET_IDENTITY_SEED"

#: TLS impersonation fallback when the installed curl_cffi ships no Firefox
#: profile at all. Divergence (Chromium TLS under a Firefox UA) is logged once
#: at build time — see :func:`_select_impersonate`.
_FALLBACK_IMPERSONATE = "chrome131"

_FIREFOX_TARGET_RE = re.compile(r"firefox(\d+)$")


@dataclass(frozen=True, slots=True)
class IdentityProfile:
    """A coherent browser identity, plus the TLS impersonation that matches it."""

    seed: int
    user_agent: str
    oscpu: str
    platform: str
    locale: str
    timezone_id: str
    viewport_width: int
    viewport_height: int
    webgl_vendor: str
    webgl_renderer: str
    impersonate: str
    raw: Any = None  # the invisible_core Profile, for consumers that want depth


def _supported_impersonate_targets() -> list[str]:
    """Return the impersonate target names the installed curl_cffi supports.

    ``BrowserTypeLiteral`` is the authoritative source; ``REAL_TARGET_MAP``
    values are the fallback for versions that dropped the literal. Anything
    unreadable yields an empty list and the caller falls back to Chrome.
    """
    try:
        from curl_cffi.requests.impersonate import BrowserTypeLiteral

        targets = [str(t) for t in get_args(BrowserTypeLiteral)]
    except Exception:  # pragma: no cover - curl_cffi API drift
        targets = []
    if not targets:
        try:
            from curl_cffi.requests.impersonate import REAL_TARGET_MAP

            targets = [str(v) for v in REAL_TARGET_MAP.values()]
        except Exception:  # pragma: no cover - curl_cffi API drift
            targets = []
    return targets


def _select_impersonate(targets: list[str] | None = None) -> str:
    """Pick the newest ``firefoxNNN`` target the installed curl_cffi supports.

    curl_cffi can only impersonate a browser it ships a profile for, so this
    is the seam where a Firefox identity meets the TLS layer: the honest move
    is to take the newest Firefox profile available. With no Firefox target at
    all we fall back to ``_FALLBACK_IMPERSONATE`` and log once — a Chromium
    TLS fingerprint under a Firefox UA is a divergence, but a detectable one
    beats a hard failure on curl_cffi versions that change the target list.
    """
    candidates = list(targets) if targets is not None else _supported_impersonate_targets()
    best: tuple[int, str] | None = None
    for target in candidates:
        match = _FIREFOX_TARGET_RE.fullmatch(str(target))
        if match is not None and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), str(target))
    if best is not None:
        return best[1]
    logger.warning(
        "curl_cffi exposes no firefoxNNN impersonation target; falling back to %r. "
        "The identity user-agent and the TLS fingerprint will diverge.",
        _FALLBACK_IMPERSONATE,
    )
    return _FALLBACK_IMPERSONATE


def default_seed() -> int:
    """The process-wide identity seed.

    ``WET_IDENTITY_SEED`` wins when set (the consumer persists it across
    restarts). Otherwise derive once per process and cache it: every caller in
    this process then presents the same identity. A seed that must outlive the
    process is the consumer's job — wet stores it in ``~/.wet-mcp/config.json``.
    """
    return _process_seed()


@lru_cache(maxsize=1)
def _process_seed() -> int:
    raw = os.environ.get(IDENTITY_SEED_ENV, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("%s=%r is not an integer; deriving a random seed", IDENTITY_SEED_ENV, raw)
    return secrets.randbelow(2**31 - 1) + 1


def build_identity(
    *,
    seed: int,
    timezone: str = "auto",
    locale: str = "auto",
    proxy: dict[str, str] | None = None,
) -> IdentityProfile:
    """Build one identity: sampled profile + geo-resolved locale/timezone.

    ``locale``/``timezone`` default to ``"auto"``: invisible_core derives both
    from the proxy egress country (or the host's public IP without a proxy), so
    the language and the clock follow the exit. An explicit IANA zone wins and
    triggers no network call.

    Raises ``ImportError`` with the install hint when the ``identity`` extra
    (invisible-core) is not installed.
    """
    try:
        from invisible_core import (  # ty: ignore[unresolved-import]  # optional [identity] extra
            generate_profile,
            resolve_session_locale,
            resolve_session_timezone,
        )
        from invisible_core.constants import (  # ty: ignore[unresolved-import]
            OSCPU_OVERRIDE,
            PLATFORM_OVERRIDE,
            USER_AGENT,
        )
    except ImportError as exc:
        raise ImportError(
            "build_identity requires the optional 'identity' extra (invisible-core). "
            'Install it with: pip install "hull-core[identity]"',
        ) from exc

    profile = generate_profile(seed)

    # USER_AGENT is a module constant tied to the sealed Firefox build, not a
    # per-seed field: one engine presents one Firefox. That is the point - the
    # UA in basic_http's headers and the UA the engine sends are the same string.
    tz = resolve_session_timezone(timezone, proxy)
    loc = resolve_session_locale(None, proxy)

    return IdentityProfile(
        seed=seed,
        user_agent=USER_AGENT,
        oscpu=OSCPU_OVERRIDE,
        platform=PLATFORM_OVERRIDE,
        locale=loc,
        timezone_id=tz,
        viewport_width=profile.screen.width,
        viewport_height=profile.screen.height,
        # Populated, not blank. A real desktop has a GPU; an empty string here
        # is the headless tell the old code deliberately mirrored.
        webgl_vendor=profile.gpu.vendor,
        webgl_renderer=profile.gpu.renderer,
        impersonate=_select_impersonate(),
        raw=profile,
    )


@lru_cache(maxsize=4)
def _cached_identity(seed: int, timezone: str, locale: str) -> IdentityProfile:
    """Process-wide, so repeated calls in one scrape reuse one identity.

    Unlike the old lru_cache(maxsize=256) keyed by domain, this is keyed by the
    seed: re-keying on the hot path cannot silently swap the identity mid-run.
    """
    return build_identity(seed=seed, timezone=timezone, locale=locale)
