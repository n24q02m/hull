"""Scraping strategies."""

from hull_web.scraper.strategies.api_direct import APIDirectStrategy
from hull_web.scraper.strategies.basic_http import BasicHTTPStrategy
from hull_web.scraper.strategies.captcha import CaptchaStrategy
from hull_web.scraper.strategies.headless import HeadlessStrategy
from hull_web.scraper.strategies.patchright_browser import PatchrightStrategy
from hull_web.scraper.strategies.remote_render import RemoteRenderStrategy, RenderClient
from hull_web.scraper.strategies.tls_spoof import TLSSpoofStrategy

__all__ = [
    "APIDirectStrategy",
    "BasicHTTPStrategy",
    "CaptchaStrategy",
    "HeadlessStrategy",
    "PatchrightStrategy",
    "RemoteRenderStrategy",
    "RenderClient",
    "TLSSpoofStrategy",
]
