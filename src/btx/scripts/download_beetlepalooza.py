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
    revision: str = "refs/pr/26",
):
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
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 or (status is not None and 500 <= status < 600):
                delay = _retry_delay_from_headers(e.response, sleep_s)
                print(
                    f"[Attempt {attempt}] HTTP {status}. Sleeping {int(delay)}s then retrying...",
                    file=sys.stderr,
                )
                if max_attempts is not None and attempt >= max_attempts:
                    raise
                time.sleep(delay)
                continue

            # Non-transient client errors: surface immediately.
            raise

        except hfhub.LocalEntryNotFoundError as e:
            # HF may wrap transient HTTP errors into this. Treat as retryable.
            print(
                f"[Attempt {attempt}] Transient hub error: {e.__class__.__name__}. Sleeping {int(sleep_s)}s...",
                file=sys.stderr,
            )
            if max_attempts is not None and attempt >= max_attempts:
                raise

            time.sleep(sleep_s)
            continue


if __name__ == "__main__":
    tyro.cli(main)
