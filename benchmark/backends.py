"""Model backends: one call = chat messages in, assistant text out.

A model of benchmark/models.json names its backend and that backend's settings:

    foundry   a deployment of the Azure Foundry account (our fine-tunes, the base
              Llama). ``ensure`` creates the deployment with the ``high-only``
              content-filter policy and ``teardown`` deletes it: deployments
              bill by the hour.
    openai    any OpenAI-compatible chat endpoint (vLLM, Together, OpenRouter…):
              ``base_url``, ``model``, and the env variable holding the key.

Adding a backend means adding a class with ``complete(messages, temperature,
max_tokens) -> str`` and registering it in BACKENDS.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

from . import REPO

sys.path.insert(0, str(REPO / "src"))
load_dotenv(REPO / ".env")

from article_silicon_sampling_quebec.foundry import FoundryChat  # noqa: E402

SUBSCRIPTION = "a54061e5-b5f7-49c0-94f0-d2e7b95de4a4"
RESOURCE_GROUP = "rg-opubliq-sondages"
ACCOUNT = "info-4552-resource"
ARM_API = "2024-10-01"
#: Blocks only "high" severity, so items like "Canada should admit: … Fewer
#: immigrants" (hate severity 5 of 7) are not refused at inference.
RAI_POLICY = "benchmark-high-only"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _arm(method: str, path: str, body: dict | None = None) -> dict | None:
    url = (f"https://management.azure.com/subscriptions/{SUBSCRIPTION}/resourceGroups/"
           f"{RESOURCE_GROUP}/providers/Microsoft.CognitiveServices/accounts/{ACCOUNT}"
           f"/{path}?api-version={ARM_API}")
    if shutil.which("az"):
        token = subprocess.run(["az", "account", "get-access-token", "--query", "accessToken",
                                "-o", "tsv"], check=True, capture_output=True, text=True).stdout.strip()
    else:  # cloud container: its user-assigned managed identity
        from azure.identity import ManagedIdentityCredential
        token = ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID")).get_token(
            "https://management.azure.com/.default").token
    request = urllib.request.Request(url, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
    for attempt in range(40):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code == 409 and attempt < 39:
                # Another deployment operation on the account: they are serialized.
                time.sleep(30)
                request = urllib.request.Request(request.full_url, method=method,
                                                 data=request.data, headers=dict(request.headers))
                continue
            raise RuntimeError(f"{method} {path}: HTTP {error.code} "
                               f"{error.read().decode(errors='replace')[:500]}") from error


class FoundryBackend:
    def __init__(self, deployment: str, model: str, version: str = "1",
                 sku: str = "GlobalStandard", capacity: int = 1000, **_: Any):
        self.deployment, self.model, self.version = deployment, model, version
        self.sku, self.capacity = sku, capacity
        self.chat = FoundryChat(deployment=deployment)

    def ensure(self, timeout: float = 3600) -> None:
        if _arm("GET", f"deployments/{self.deployment}") is None:
            _log(f"creating deployment {self.deployment} ({self.sku} x {self.capacity})")
            _arm("PUT", f"deployments/{self.deployment}", {
                "sku": {"name": self.sku, "capacity": self.capacity},
                "properties": {"model": {"format": "Meta", "name": self.model,
                                         "version": self.version},
                               "raiPolicyName": RAI_POLICY}})
        deadline = time.monotonic() + timeout
        while True:
            state = ((_arm("GET", f"deployments/{self.deployment}") or {})
                     .get("properties", {}).get("provisioningState"))
            if state == "Succeeded":
                _log(f"deployment {self.deployment} ready")
                return
            if state in ("Failed", "Canceled") or time.monotonic() > deadline:
                raise RuntimeError(f"deployment {self.deployment}: {state}")
            time.sleep(20)

    def teardown(self) -> None:
        _log(f"deleting deployment {self.deployment}")
        _arm("DELETE", f"deployments/{self.deployment}")

    def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        return self.chat.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def counters(self) -> dict:
        return {"retries": self.chat.retries, "throttled": self.chat.throttled,
                "content_filtered": self.chat.filtered}


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, api_key_env: str, **_: Any):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, os.environ[api_key_env]
        self.retries = 0

    def ensure(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
        for attempt in range(10):
            request = urllib.request.Request(self.url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    choices = json.loads(response.read()).get("choices") or [{}]
                    return (choices[0].get("message", {}).get("content") or "").strip()
            except urllib.error.HTTPError as error:
                if error.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {error.code}: {error.read()[:300]!r}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            self.retries += 1
            time.sleep(min(2 ** attempt, 60) * (1 + random.random() / 4))
        raise RuntimeError("retries exhausted")

    def counters(self) -> dict:
        return {"retries": self.retries}


BACKENDS = {"foundry": FoundryBackend, "openai": OpenAIBackend}


def load(name: str) -> tuple[dict, Any]:
    registry = json.loads((REPO / "benchmark" / "models.json").read_text())
    if name not in registry:
        raise SystemExit(f"unknown model {name!r}; known: {', '.join(registry)}")
    config = registry[name]
    return config, BACKENDS[config["backend"]](**config)
