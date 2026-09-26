import socket

import hull_web.http.client
from hull_web.http.client import _dns_cache, is_safe_url


def test_ssrf_bypass():
    _dns_cache.clear()

    original_getaddrinfo = hull_web.http.client._original_getaddrinfo

    def mock_getaddrinfo(host, port, *args, **kwargs):
        if host == "vuln.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))]
        return original_getaddrinfo(host, port, *args, **kwargs)

    try:
        hull_web.http.client._original_getaddrinfo = mock_getaddrinfo

        assert is_safe_url("http://vuln.com", allow_private=True) is True
        assert is_safe_url("http://vuln.com", allow_private=False) is False
    finally:
        hull_web.http.client._original_getaddrinfo = original_getaddrinfo
