"""API-key authentication and a single-process fixed-window rate limiter."""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException

API_KEY_ENV = "FABRAG_API_KEY"
RATE_LIMIT_ENV = "FABRAG_RATE_LIMIT_REQUESTS"
RATE_WINDOW_ENV = "FABRAG_RATE_LIMIT_WINDOW_SECONDS"


@dataclass
class WindowCounter:
    started_at: float
    count: int


class FixedWindowRateLimiter:
    """Thread-safe per-process limiter; use a shared store for multiple replicas."""

    def __init__(self) -> None:
        self._counters: dict[str, WindowCounter] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear counters; intended for tests and controlled process resets."""
        with self._lock:
            self._counters.clear()

    def check(self, identity: str, limit: int, window_seconds: int) -> int:
        now = time.monotonic()
        with self._lock:
            counter = self._counters.get(identity)
            if counter is None or now - counter.started_at >= window_seconds:
                self._counters[identity] = WindowCounter(started_at=now, count=1)
                return window_seconds

            retry_after = max(1, int(window_seconds - (now - counter.started_at) + 0.999))
            if counter.count >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )
            counter.count += 1
            return retry_after


rate_limiter = FixedWindowRateLimiter()


def _positive_int_setting(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="API security configuration invalid") from exc
    if value <= 0:
        raise HTTPException(status_code=503, detail="API security configuration invalid")
    return value


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    configured_key = os.environ.get(API_KEY_ENV)
    if not configured_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(
            status_code=401,
            detail="invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    identity = hashlib.sha256(x_api_key.encode()).hexdigest()
    limit = _positive_int_setting(RATE_LIMIT_ENV, 30)
    window_seconds = _positive_int_setting(RATE_WINDOW_ENV, 60)
    rate_limiter.check(identity, limit, window_seconds)
    return identity
