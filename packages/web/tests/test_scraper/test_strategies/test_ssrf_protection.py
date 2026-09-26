from unittest.mock import patch

import pytest

from hull_web.scraper.strategies.captcha import CaptchaStrategy
from hull_web.scraper.strategies.headless import HeadlessStrategy
from hull_web.scraper.strategies.patchright_browser import PatchrightStrategy


@pytest.mark.asyncio
async def test_patchright_strategy_blocks_ssrf():
    strategy = PatchrightStrategy()
    with (
        patch("hull_web.scraper.strategies.patchright_browser.is_safe_url", return_value=False),
        pytest.raises(ValueError, match="SSRF blocked"),
    ):
        await strategy.fetch("http://127.0.0.1")


@pytest.mark.asyncio
async def test_captcha_strategy_blocks_ssrf():
    strategy = CaptchaStrategy(capsolver_api_key="key")
    with (
        patch("hull_web.scraper.strategies.captcha.is_safe_url", return_value=False),
        pytest.raises(ValueError, match="SSRF blocked"),
    ):
        await strategy.fetch("http://127.0.0.1")


@pytest.mark.asyncio
async def test_headless_strategy_blocks_ssrf():
    strategy = HeadlessStrategy()
    with (
        patch("hull_web.scraper.strategies.headless.is_safe_url", return_value=False),
        pytest.raises(ValueError, match="SSRF blocked"),
    ):
        await strategy.fetch("http://127.0.0.1")
