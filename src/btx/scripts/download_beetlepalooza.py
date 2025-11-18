# src/btx/scripts/download_beetlepalooza.py
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "requests",
#     "tyro",
# ]
# ///

import datetime as dt
import pathlib
import sys
import time
from email.utils import parsedate_to_datetime

import huggingface_hub as hfhub
import requests
import tyro
import random
import httpx
import httpcore
import threading

# Rate limiter to enforce 1000 requests per 5 minutes across all httpx.Client calls.
# This monkeypatch wraps httpx.Client.send so all synchronous httpx requests are counted.
RATE_LIMIT = 1000
WINDOW_S = 300.0  # 5 minutes
_rl_lock = threading.Lock()
_rl_window_start = time.time()
_rl_count = 0

_original_httpx_client_send = httpx.Client.send


def _rate_limited_send(self, request, *args, **kwargs):
    """Wrap httpx.Client.send to enforce a fixed-window rate limit."""
    global _rl_window_start, _rl_count
    while True:
        with _rl_lock:
            now = time.time()
            # reset window when elapsed
            if now - _rl_window_start >= WINDOW_S:
                _rl_window_start = now
                _rl_count = 0
            if _rl_count < RATE_LIMIT:
                _rl_count += 1
                break
            # compute remaining time in window and wait (release lock while sleeping)
            wait = _rl_window_start + WINDOW_S - now
        sleep = max(wait, 0.0)
        print(f"[RateLimiter] reached {RATE_LIMIT} requests; sleeping {int(sleep)}s...", file=sys.stderr)
        time.sleep(sleep)
    # perform the real send
    return _original_httpx_client_send(self, request, *args, **kwargs)


# Apply the monkeypatch so all future httpx.Client.send calls go through the limiter.
httpx.Client.send = _rate_limited_send


def _retry_delay_from_headers(
    resp: requests.Response | None, default_s: float
) -> float:
    """Pick a sensible sleep based on HTTP headers; fall back to default."""
    if resp is None:
        return default_s

    # Standard header first
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        retry_after = retry_after.strip()
        if retry_after.isdigit():
            return max(default_s, float(retry_after))
        # Could be an HTTP-date
        try:
            when = parsedate_to_datetime(retry_after)
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            return max(
                default_s, (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
            )
        except Exception:
            pass

    # Common rate-limit headers (values vary by provider)
    reset = resp.headers.get("X-RateLimit-Reset") or resp.headers.get("Ratelimit-Reset")
    if reset:
        try:
            val = float(reset)
            # If it's an epoch, convert to delta; if it's already a delta (seconds), use it.
            now = time.time()
            if val > now + 5:
                return max(default_s, val - now)
            return max(default_s, val)
        except Exception:
            pass

    return default_s


def main(
    dump_to: pathlib.Path = pathlib.Path("data/beetlepalooza"),
    sleep_s: float = 60.0,
    max_attempts: int | None = None,
    max_workers: int = 1,
    repo_id: str = "imageomics/2018-NEON-beetles",
    revision: str = "refs/pr/32",
):
    # Safely resolve optional HF error classes (avoid AttributeError if not present)
    hf_errors = getattr(hfhub, "errors", None)
    hf_hub_http_error = getattr(hf_errors, "HfHubHTTPError", None) if hf_errors else None
    local_entry_exc = getattr(hf_errors, "LocalEntryNotFoundError", None) if hf_errors else None
    # build the exception tuple we'll catch
    # include common timeout/transport exceptions from httpx/httpcore as retryable
    _base_excs = (requests.HTTPError, httpx.HTTPStatusError, httpx.ReadTimeout, httpx.TransportError, httpcore.ReadTimeout)
    if hf_hub_http_error is not None:
        _base_excs = _base_excs + (hf_hub_http_error,)

    attempt = 0
    while True:
        attempt += 1
        try:
            hfhub.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dump_to),
                revision=revision,
                max_workers=max_workers,
            )
            print("Download complete.")
            return
        except _base_excs as e:
            # try to get underlying response (works for httpx, requests, and wrapped HF errors)
            resp = getattr(e, "response", None) or getattr(getattr(e, "__cause__", None), "response", None)
            status = getattr(resp, "status_code", None)

            # If this is a wrapped "local entry not found" from HF (when available), treat as retryable
            if local_entry_exc is not None and isinstance(e, local_entry_exc):
                print(
                    f"[Attempt {attempt}] Transient hub error: {e.__class__.__name__}. Sleeping {int(sleep_s)}s...",
                    file=sys.stderr,
                )
                if max_attempts is not None and attempt >= max_attempts:
                    raise
                jitter = random.uniform(0.8, 1.2)
                time.sleep(sleep_s * jitter)
                continue

            # treat 429 and 5xx as transient
            if status == 429 or (status is not None and 500 <= status < 600):
                # exponential backoff with jitter
                backoff_factor = min(2 ** (attempt - 1), 32)
                base_delay = _retry_delay_from_headers(resp, sleep_s)
                jitter = random.uniform(0.8, 1.25)
                delay = min(base_delay * backoff_factor * jitter, 3600.0)
                print(
                    f"[Attempt {attempt}] HTTP {status}. Sleeping {int(delay)}s then retrying... (backoff x{backoff_factor:.0f})",
                    file=sys.stderr,
                )
                # reduce concurrency aggressively on 429 to avoid further rate-limiting
                if status == 429 and max_workers > 1:
                    max_workers = 1
                    print("[Info] Reducing max_workers to 1 to avoid further rate limiting.", file=sys.stderr)
                if max_attempts is not None and attempt >= max_attempts:
                    raise
                time.sleep(delay)
                continue

            # Handle read/transport timeouts (no HTTP status). Treat as transient and back off.
            if isinstance(e, (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.TransportError)):
                backoff_factor = min(2 ** (attempt - 1), 32)
                # use sleep_s as base when there's no response headers
                base_delay = _retry_delay_from_headers(None, sleep_s)
                jitter = random.uniform(0.9, 1.3)
                delay = min(base_delay * backoff_factor * jitter, 3600.0)
                print(
                    f"[Attempt {attempt}] Read/Transport timeout ({e.__class__.__name__}). Sleeping {int(delay)}s then retrying...",
                    file=sys.stderr,
                )
                # reduce concurrency to avoid parallel timeouts
                if max_workers > 1:
                    max_workers = 1
                    print("[Info] Reducing max_workers to 1 after timeouts.", file=sys.stderr)
                if max_attempts is not None and attempt >= max_attempts:
                    raise
                time.sleep(delay)
                continue

            # Non-transient client errors: surface immediately.
            raise


if __name__ == "__main__":
    tyro.cli(main)
