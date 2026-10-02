"""Stealth-Firefox browser strategy: dots' engine behind the Patchright seam.

Subclasses :class:`PatchrightStrategy` and overrides only provider creation
(:meth:`PatchrightStrategy._resolve_provider`). The Cloudflare challenge
machinery stays in exactly one place on purpose: if that logic changes it
must change once, not twice. This is the last local resort before the captcha
solver.
"""

from __future__ import annotations

from typing import Any

from hull_web.browsers.invisible import InvisibleProvider
from hull_web.scraper.strategies.patchright_browser import PatchrightStrategy


class InvisibleStrategy(PatchrightStrategy):
    """Patchright flow, launched through the invisible engine provider."""

    name: str = "invisible"

    def _resolve_provider(self) -> tuple[Any, bool]:
        """Always the invisible engine; injected provider still wins.

        Returns ``(provider, owns_provider=False)`` when a provider was
        injected (caller-owned lifecycle), else a fresh InvisibleProvider the
        fetch teardown closes.
        """
        if self._provider is not None:
            return self._provider, False
        return InvisibleProvider(headless=self.headless), True
