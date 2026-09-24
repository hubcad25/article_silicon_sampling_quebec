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

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["CallFailed", "FoundryChat"]

#: Retried: throttling, timeouts, server-side and gateway errors.
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


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
    #: Counters, for the run log: how often the service pushed back.
    retries: int = 0
    throttled: int = 0

    def __post_init__(self) -> None:
        self.base = (self.base or os.environ["FOUNDRY_CHAT_ENDPOINT"]).rstrip("/").split("/openai")[0]
        self.key = self.key or os.environ["FOUNDRY_CHAT_KEY"]

    @property
    def url(self) -> str:
        return (f"{self.base}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"api-key": self.key,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def complete(self, messages: list[dict], *, temperature: float = 1.0,
                 max_tokens: int = 8, top_p: float = 1.0) -> str:
        """The assistant text of one completion. Retries until it succeeds or raises."""
        body = {"messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "top_p": top_p}
        waited, attempt = 0.0, 0
        while True:
            retry_after = None
            try:
                data = self._post(body)
                return (data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS:
                    raise CallFailed(f"HTTP {e.code} on {self.deployment}") from e
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
