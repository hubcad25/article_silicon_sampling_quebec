"""A throttled or failed call is retried, never skipped; a hopeless one raises."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from article_silicon_sampling_quebec.foundry import CallFailed, FoundryChat

OK = {"choices": [{"message": {"content": " 3 "}}]}


def http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(b""))


def client(responses, **kw):
    slept = []
    chat = FoundryChat("dep", base="https://x", key="k", sleep=slept.append, **kw)
    queue = list(responses)

    def post(body):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    chat._post = post
    return chat, slept


def test_429_is_retried_until_success():
    chat, slept = client([http_error(429), http_error(429, retry_after=2), OK])
    assert chat.complete([{"role": "user", "content": "q"}]) == "3"
    assert chat.throttled == 2 and chat.retries == 2
    assert 2 <= slept[1] <= 2.5  # Retry-After honoured, with jitter


def test_transient_server_and_network_errors_are_retried():
    chat, _ = client([http_error(503), urllib.error.URLError("reset"), TimeoutError(), OK])
    assert chat.complete([]) == "3"


def test_fatal_status_raises_at_once():
    chat, slept = client([http_error(400)])
    with pytest.raises(CallFailed):
        chat.complete([])
    assert slept == []


def test_exhausted_retries_raise_instead_of_skipping():
    chat, slept = client([http_error(429)] * 100, max_wait=10)
    with pytest.raises(CallFailed):
        chat.complete([])
    assert sum(slept) <= 10
