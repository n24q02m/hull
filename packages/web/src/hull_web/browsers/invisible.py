"""dots' engine, behind the existing BrowserProvider Protocol.

invisible_playwright's async_api.InvisiblePlaywright mirrors Playwright's own
API, so this differs from PatchrightProvider in how the browser is launched and
in nothing else. That is why it fits the Protocol without a shim.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Set to the patched build the caller already has, to skip the download.
#: The engine verifies it against its own seal and refuses a mismatch - the
#: check is the point, so this skips the transfer and not the verification.
BINARY_ENV = "STEALTHFOX_BINARY"


class InvisibleProvider:
    """Stealth Firefox provider; satisfies BrowserProvider structurally."""

    def __init__(
        self,
        *,
        seed: int = 0,
        profile_dir: str | None = None,
        humanize: bool = True,
        headless: bool = True,
        binary_path: str | None = None,
    ) -> None:
        self._seed = seed
        self._profile_dir = profile_dir
        self._humanize = humanize
        # NOTE the default. InvisiblePlaywright defaults headless=False, which
        # is right for a desktop demo and wrong for a server with no display.
        self._headless = headless
        self._binary_path = binary_path
        self._session = None

    @property
    def name(self) -> str:
        return "invisible"

    @property
    def supports_arm64(self) -> bool:
        # The engine ships aarch64 builds; see the release assets. Distinct from
        # the greenlet wheel gap that caps wet's greenlet pin.
        return True

    async def launch(self, config: dict[str, Any] | None = None) -> Any:
        cfg = dict(config or {})
        # Lazy import: this is a heavy optional dep and the slim container has
        # neither the engine nor a display. It must be able to fail here without
        # the chain noticing - see the strategy's except branch.
        from invisible_playwright.async_api import (  # ty: ignore[unresolved-import]  # optional [invisible] extra
            InvisiblePlaywright,
        )

        self._session = InvisiblePlaywright(
            seed=self._seed or cfg.get("seed"),
            headless=cfg.get("headless", self._headless),
            humanize=cfg.get("humanize", self._humanize),
            proxy=cfg.get("proxy"),
            profile_dir=self._profile_dir or cfg.get("profile_dir"),
            binary_path=self._binary_path or cfg.get("binary_path"),
            locale=cfg.get("locale", "auto"),
            timezone=cfg.get("timezone", ""),
        )
        return await self._session.__aenter__()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
