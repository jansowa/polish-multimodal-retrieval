from __future__ import annotations

import hashlib
import gzip
import json
import logging
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config import HttpConfig
from ..storage.checkpoint import atomic_json
from .rate_limit import RateLimiter

LOG = logging.getLogger(__name__)


class ApiError(RuntimeError):
    def __init__(self, message: str, retry_after: float = 0):
        self.retry_after = retry_after
        super().__init__(message)


class HttpClient:
    """Sequential, rate limited requests. Cached responses form a raw snapshot."""

    def __init__(self, config: HttpConfig, raw_dir: Path):
        self.config = config
        self.raw_dir = raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(config, Path(config.state_path) if config.state_path else raw_dir.parent / "http_state.json")
        self.started = time.monotonic()
        self.metrics = {"started_at": datetime.now(timezone.utc).isoformat(), "requests": 0,
                        "successes": 0, "cache_hits": 0, "bytes_received": 0,
                        "errors": {}, "by_host": {}}
        self.transport = None
        if config.keep_alive:
            from .transport import PooledTransport
            self.transport = PooledTransport()

    def save_metrics(self) -> None:
        elapsed = max(.001, time.monotonic() - self.started)
        atomic_json(self.raw_dir.parent / "http_metrics.json", {
            **self.metrics, "elapsed_seconds": elapsed,
            "requests_per_hour": self.metrics["requests"] * 3600 / elapsed,
            "configured_requests_per_minute": self.config.requests_per_minute,
            "effective_min_interval": self.limiter.interval,
            "cooldowns": self.limiter.state["cooldowns"]})

    def _request(self, url: str) -> Any:
        from urllib.parse import urlparse
        self.limiter.reserve(url)
        self.metrics["requests"] += 1
        host = urlparse(url).hostname
        self.metrics["by_host"][host] = self.metrics["by_host"].get(host, 0) + 1
        self.save_metrics()
        agent = self.config.user_agent + (f" ({self.config.contact})" if self.config.contact else "")
        headers = {"User-Agent": agent, "Accept": "application/json", "Accept-Encoding": "gzip"}
        if self.transport:
            return self.transport.open(url, headers, self.config.timeout)
        return urlopen(Request(url, headers=headers), timeout=self.config.timeout)

    def close(self) -> None:
        self.save_metrics()
        if self.transport:
            self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _retry_after(value: str | None) -> float:
        if not value:
            return 0.0
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                return 0.0

    def _retry(self, operation: Any, endpoint: str | None = None) -> Any:
        for attempt in range(self.config.retries + 1):
            delay = self.config.backoff * 2 ** attempt
            try:
                return operation()
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                delay = max(delay, self._retry_after(exc.headers.get("Retry-After")))
                status = str(exc.code)
                self.metrics["errors"][status] = self.metrics["errors"].get(status, 0) + 1
                if endpoint:
                    delay = max(delay, 5.0) if exc.code == 429 else delay
                    deferred = self.limiter.cooldown(endpoint, delay, status, slow_down=exc.code == 429)
                    self.save_metrics()
                    # Record the server deadline even when the last retry fails.
                    # Return control to the scheduler instead of sleeping minutes.
                    if delay > self.config.max_inline_wait or attempt == self.config.retries:
                        LOG.warning("Deferred %s for %.1fs after HTTP %s", deferred.scope, delay, status)
                        raise deferred from exc
                error: Exception = exc
            except (URLError, TimeoutError, socket.timeout, ConnectionError, ApiError, json.JSONDecodeError) as exc:
                error = exc
                delay = max(delay, getattr(exc, "retry_after", 0))
                self.metrics["errors"][type(exc).__name__] = self.metrics["errors"].get(type(exc).__name__, 0) + 1
                if endpoint and (delay > self.config.max_inline_wait or attempt == self.config.retries):
                    deferred = self.limiter.cooldown(endpoint, max(delay, 5.0), str(exc))
                    self.save_metrics()
                    raise deferred from exc
            if attempt == self.config.retries:
                raise error
            LOG.warning("Request failed (%s); retry %d in %.1fs", error, attempt + 1, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def request_url(endpoint: str, params: dict[str, Any]) -> str:
        params = {k: v for k, v in params.items() if v is not None}
        params.setdefault("format", "json")
        if "action" in params:
            params.setdefault("maxlag", 5)
        return endpoint + "?" + urlencode(sorted(params.items()))

    def cached(self, endpoint: str, **params: Any) -> dict[str, Any] | None:
        if not self.config.cache:
            return None
        url = self.request_url(endpoint, params)
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.raw_dir / f"{key}.json"
        for root in [self.raw_dir, *[Path(v) for v in self.config.cache_dirs]]:
            candidate = root / path.name
            if not candidate.exists():
                continue
            try:
                snapshot = json.loads(candidate.read_text(encoding="utf-8"))
                if snapshot.get("url") != url:
                    continue
            except (OSError, ValueError):
                LOG.warning("Ignoring damaged cache entry %s", candidate)
                continue
            if candidate != path:
                atomic_json(path, snapshot)
            self.metrics["cache_hits"] += 1
            return {**snapshot["response"], "_request_key": key, "_retrieved_at": snapshot["retrieved_at"]}
        return None

    def item_get(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Parsed per-item cache points to original batch responses for provenance."""
        if not self.config.cache:
            return None
        path = self.raw_dir.parent / "items" / namespace / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                self.metrics["cache_hits"] += 1
                return value
            except (OSError, ValueError):
                LOG.warning("Ignoring damaged item cache %s", path)
        return None

    def item_put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        path = self.raw_dir.parent / "items" / namespace / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        atomic_json(path, value)

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        if cached := self.cached(endpoint, **params):
            return cached
        url = self.request_url(endpoint, params)
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.raw_dir / f"{key}.json"
        def request() -> dict[str, Any]:
            with self._request(url) as response:
                retry_after = self._retry_after(getattr(response, "headers", {}).get("Retry-After"))
                data = response.read()
                if getattr(response, "headers", {}).get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                self.metrics["bytes_received"] += len(data)
                payload = json.loads(data)
            if "error" in payload:
                error = payload["error"]
                if error.get("code") in {"maxlag", "ratelimited", "readonly"}:
                    raise ApiError(str(error), retry_after)
                raise ValueError(f"API error: {error}")
            return payload
        payload = self._retry(request, endpoint)
        self.metrics["successes"] += 1
        self.limiter.success()
        snapshot = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "response": payload}
        atomic_json(path, snapshot)
        self.save_metrics()
        return {**snapshot["response"], "_request_key": key, "_retrieved_at": snapshot["retrieved_at"]}

    def download(self, url: str, path: Path, max_bytes: int) -> None:
        tmp = path.with_suffix(path.suffix + ".part")
        def request() -> None:
            try:
                with self._request(url) as response, tmp.open("wb") as output:
                    stream = gzip.GzipFile(fileobj=response) if getattr(response, "headers", {}).get("Content-Encoding") == "gzip" else response
                    total = 0
                    while chunk := stream.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"Download exceeds {max_bytes} bytes")
                        output.write(chunk)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        self._retry(request, url)
