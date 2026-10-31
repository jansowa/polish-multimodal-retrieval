from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from ..config import HttpConfig
from ..storage.checkpoint import atomic_json


class RetryLater(RuntimeError):
    """A durable service cooldown; a scheduler can continue other services."""

    def __init__(self, scope: str, retry_at: float, reason: str):
        self.scope, self.retry_at, self.reason = scope, retry_at, reason
        super().__init__(f"{scope}: retry after {max(0, retry_at-time.time()):.1f}s ({reason})")


def scope_for(url: str) -> str:
    host = urlparse(url).hostname or "unknown"
    if host in {"query.wikidata.org", "query-scholarly.wikidata.org"}:
        return "wdqs:" + host
    if host.endswith(".wikimedia.org") or host.endswith(".wikipedia.org") or host.endswith(".wikidata.org"):
        if host in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
            return "media:" + host
        return "wikimedia-api"
    return host


class RateLimiter:
    """One sequential client, one shared gateway budget, durable server cooldowns."""

    def __init__(self, config: HttpConfig, path: Path):
        self.config, self.path = config, path
        self.state = {"cooldowns": {}, "next_request_at": 0.0, "adaptive_interval": 0.0}
        if path.exists():
            self.state.update(json.loads(path.read_text(encoding="utf-8")))

    @property
    def interval(self) -> float:
        # 5% margin below the configured ceiling; never burst at minute boundaries.
        return max(self.config.interval, 63.0 / self.config.requests_per_minute,
                   self.state.get("adaptive_interval", 0.0))

    def save(self) -> None:
        atomic_json(self.path, self.state)

    @contextmanager
    def shared_state(self):
        """Serialize reservations and reload state across collector/media processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + '.lock').open('a+b') as lock:
            if os.name == 'nt':
                import msvcrt
                # Windows byte locks also forbid reads by another process.
                # Inspect length without reading the byte before acquiring it.
                lock.seek(0, os.SEEK_END)
                if lock.tell() == 0:
                    lock.write(b'0'); lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.path.exists():
                    self.state.update(json.loads(self.path.read_text(encoding='utf-8')))
                yield
            finally:
                if os.name == 'nt':
                    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def reserve(self, url: str) -> None:
        with self.shared_state():
            self._reserve(url)

    def _reserve(self, url: str) -> None:
        now = time.time()
        scope = scope_for(url)
        until = self.state["cooldowns"].get(scope, 0.0)
        if until > now:
            raise RetryLater(scope, until, "server cooldown")
        # API gateway accounting is shared across Wikimedia projects; media has
        # the same conservative global pacing as well, never an extra API budget.
        delay = max(0.0, self.state.get("next_request_at", 0.0) - now)
        if delay > self.config.max_inline_wait and delay > self.interval + 0.1:
            raise RetryLater(scope, now + delay, "global request budget")
        if delay:
            time.sleep(delay)
        self.state["next_request_at"] = time.time() + self.interval
        self.save()

    def cooldown(self, url: str, delay: float, reason: str, slow_down: bool = False) -> RetryLater:
        with self.shared_state():
            return self._cooldown(url, delay, reason, slow_down)

    def _cooldown(self, url: str, delay: float, reason: str, slow_down: bool = False) -> RetryLater:
        scope, until = scope_for(url), time.time() + delay
        self.state["cooldowns"][scope] = max(until, self.state["cooldowns"].get(scope, 0.0))
        if slow_down:
            self.state["adaptive_interval"] = max(self.interval * 2, 1.0)
        self.save()
        return RetryLater(scope, self.state["cooldowns"][scope], reason)

    def success(self) -> None:
        with self.shared_state():
            self._success()

    def _success(self) -> None:
        # Gradual recovery after throttling; the configured ceiling always wins.
        adaptive = self.state.get("adaptive_interval", 0.0)
        if adaptive:
            self.state["adaptive_interval"] = adaptive * .98
            self.save()

    def next_retry(self) -> float | None:
        future = [v for v in self.state["cooldowns"].values() if v > time.time()]
        return min(future) if future else None
