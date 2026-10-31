from __future__ import annotations

from typing import Any
from urllib.error import HTTPError, URLError


class PooledTransport:
    """Reuse TCP/TLS connections; retries and pacing remain in HttpClient."""

    def __init__(self):
        import requests
        self.requests = requests
        self.session = requests.Session()
        # No hidden adapter retries: every attempt must pass our rate limiter.
        adapter = requests.adapters.HTTPAdapter(pool_connections=3, pool_maxsize=1, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def open(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        try:
            response = self.session.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=False)
        except self.requests.exceptions.Timeout as exc:
            raise TimeoutError(str(exc)) from exc
        except self.requests.exceptions.RequestException as exc:
            raise URLError(str(exc)) from exc
        if response.status_code >= 300:
            # Never hide redirects (which could cause unaccounted network calls).
            # API endpoints are canonical; a moved endpoint should be configured.
            error = HTTPError(url, response.status_code, response.reason, response.headers, None)
            response.close()
            raise error

        class Response:
            headers = response.headers

            def read(self, size: int | None = None) -> bytes:
                try:
                    return response.raw.read(size, decode_content=False)
                except Exception as exc:
                    raise URLError(str(exc)) from exc

            def __enter__(self):
                return self

            def __exit__(self, *args):
                response.close()

        return Response()

    def close(self) -> None:
        self.session.close()
