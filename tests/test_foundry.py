"""A throttled or failed call is retried, never skipped; a hopeless one raises."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from article_silicon_sampling_quebec.foundry import CallFailed, FoundryChat

OK = {"choices": [{"message": {"content": " 3 "}}]}


def http_error(code, retry_after=None, body=b""):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(body))


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


class FakeResponse:
    def __init__(self, status, payload, will_close=False):
        self.status, self.reason, self.will_close = status, "r", will_close
        self.headers = {"Retry-After": "1"} if status == 429 else {}
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()


class FakeConnection:
    def __init__(self, responses):
        self.responses, self.requests, self.closed = responses, 0, False

    def request(self, *args, **kwargs):
        self.requests += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        self.next = item

    def getresponse(self):
        return self.next

    def close(self):
        self.closed = True


def test_connection_is_kept_alive_across_calls():
    chat = FoundryChat("dep", base="https://x", key="k", sleep=lambda s: None)
    conn = FakeConnection([FakeResponse(200, OK), FakeResponse(429, {}), FakeResponse(200, OK)])
    chat._local.conn = conn
    assert chat.complete([]) == "3" and chat.complete([]) == "3"
    assert conn.requests == 3 and not conn.closed and chat.throttled == 1


def test_broken_connection_is_dropped_and_retried():
    chat = FoundryChat("dep", base="https://x", key="k", sleep=lambda s: None)
    broken = FakeConnection([ConnectionResetError()])
    fresh = FakeConnection([FakeResponse(200, OK)])
    chat._local.conn = broken
    chat._connection = lambda: getattr(chat._local, "conn", None) or setattr(chat._local, "conn", fresh) or fresh
    assert chat.complete([]) == "3"
    assert broken.closed and fresh.requests == 1


FILTERED = b'{"error":{"code":"content_filter","message":"filtered"}}'


def test_content_filter_400_is_resampled_and_counted():
    chat, slept = client([http_error(400, body=FILTERED), OK])
    assert chat.complete([]) == "3"
    assert chat.filtered == 1 and slept == []


def test_content_filter_becomes_a_checkpointed_invalid_draw_after_resampling():
    chat, _ = client([http_error(400, body=FILTERED) for _ in range(3)], max_filtered=2)
    assert chat.complete([]) == ""
    assert chat.filtered == 3


def test_empty_choices_are_resampled_then_checkpointed_as_invalid():
    chat, slept = client([{"choices": []}, {"choices": []}, OK], max_filtered=2)
    assert chat.complete([]) == "3"
    assert chat.filtered == 2 and slept == []

    chat, _ = client([{} for _ in range(3)], max_filtered=2)
    assert chat.complete([]) == ""
    assert chat.filtered == 3


def test_other_400_raises_with_its_body():
    chat, _ = client([http_error(400, body=b'{"error":"max_tokens too large"}')])
    with pytest.raises(CallFailed, match="max_tokens too large"):
        chat.complete([])
