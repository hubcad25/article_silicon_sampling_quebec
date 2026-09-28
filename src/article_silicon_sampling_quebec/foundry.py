"""Chat client for a Foundry deployment — retries, never skips.

A draw lost to a 429 is not a missing value we can ignore: dropping it biases
the empirical distribution toward whatever the service answered while it was
not throttling, and shrinks the effective N of the cell silently. So every
throttled or transient failure is retried with capped exponential backoff and
jitter until it succeeds; if the total wait for one call exceeds ``max_wait``
the call **raises** (:class:`CallFailed`) and the run stops, rather than
returning a placeholder that would be counted as an answer or quietly dropped.

Only errors that no retry can fix (400, 401, 403, 404) raise immediately.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["CallFailed", "FoundryChat"]

#: Retried: throttling, timeouts, server-side and gateway errors.
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
#: A 400 whose body names the content filter: the service refused this sampled
#: output, not the request. Resampled a bounded number of times and counted.
FILTER_MARKERS = ("content_filter", "ResponsibleAIPolicyViolation", "content management policy")


class CallFailed(RuntimeError):
    """A call that could not succeed: fatal status, or retries exhausted."""


@dataclass
class FoundryChat:
    deployment: str
    base: str = ""
    key: str = ""
    api_version: str = "2024-10-21"
    timeout: float = 60.0
    #: Total seconds one call may spend waiting between attempts.
    max_wait: float = 900.0
    max_backoff: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)
    #: One kept-alive HTTPS connection per worker thread. Opening a fresh TCP+TLS
    #: connection per call exhausted the container's outbound SNAT ports after
    #: ~500 calls, leaving every worker stuck in SYN_SENT.
    _local: threading.local = field(default_factory=threading.local, repr=False, compare=False)
    #: Counters, for the run log: how often the service pushed back.
    retries: int = 0
    throttled: int = 0
    filtered: int = 0
    #: Content-filter refusals resampled before the filtered draw is returned
    #: as an empty (therefore invalid, but checkpointed) response.
    max_filtered: int = 5

    def __post_init__(self) -> None:
        self.base = (self.base or os.environ["FOUNDRY_CHAT_ENDPOINT"]).rstrip("/").split("/openai")[0]
        self.key = self.key or os.environ["FOUNDRY_CHAT_KEY"]

    @property
    def url(self) -> str:
        return (f"{self.base}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _connection(self) -> http.client.HTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            host = urllib.parse.urlsplit(self.base).netloc
            conn = http.client.HTTPSConnection(host, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def _drop_connection(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _post(self, body: dict) -> dict:
        split = urllib.parse.urlsplit(self.url)
        path = f"{split.path}?{split.query}"
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        try:
            conn = self._connection()
            conn.request("POST", path, body=json.dumps(body).encode(), headers=headers)
            resp = conn.getresponse()
            payload = resp.read()
        except (http.client.HTTPException, OSError) as e:
            self._drop_connection()
            if isinstance(e, TimeoutError):
                raise
            raise urllib.error.URLError(e) from e
        if resp.will_close:
            self._drop_connection()
        if resp.status >= 400:
            raise urllib.error.HTTPError(self.url, resp.status, resp.reason,
                                         resp.headers, io.BytesIO(payload))
        return json.loads(payload)

    def complete(self, messages: list[dict], *, temperature: float = 1.0,
                 max_tokens: int = 8, top_p: float = 1.0) -> str:
        """The assistant text of one completion. Retries until it succeeds or raises."""
        body = {"messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "top_p": top_p}
        waited, attempt, filtered = 0.0, 0, 0
        while True:
            retry_after = None
            try:
                data = self._post(body)
                return (data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS:
                    detail = _error_body(e)
                    if e.code == 400 and any(m in detail for m in FILTER_MARKERS):
                        filtered += 1
                        self.filtered += 1
                        if filtered <= self.max_filtered:
                            continue
                        return ""
                    raise CallFailed(f"HTTP {e.code} on {self.deployment}: {detail[:500]}") from e
                if e.code == 429:
                    self.throttled += 1
                header = e.headers.get("Retry-After") if e.headers else None
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
                error = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                error = e
            delay = retry_after if retry_after is not None else min(2 ** attempt, self.max_backoff)
            delay *= 1 + 0.25 * self.rng.random()  # jitter: threads must not retry in lockstep
            if waited + delay > self.max_wait:
                raise CallFailed(f"gave up after {attempt + 1} attempts, "
                                 f"{waited:.0f}s waited: {error}") from error
            self.retries += 1
            self.sleep(delay)
            waited += delay
            attempt += 1


def _error_body(error: urllib.error.HTTPError) -> str:
    try:
        return (error.read() or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - the body is diagnostic only
        return ""
