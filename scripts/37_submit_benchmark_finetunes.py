"""Upload the CES 2025 benchmark datasets and submit the six fine-tuning jobs.

Two context arms (sem, fix) x three nested sizes (20k, 50k, 100k), each an
independent job from the base model with the recipe of the earlier runs
(docs/plan_article.md §0): Llama-3.3-70B-Instruct-9, globalStandard, 1 epoch,
batch 64, LR multiplier 1, fixed seed. Each arm has its own 500-pair
validation file.

Job ids are appended to data/datasets_ces2025/jobs.json. Run::

    .venv/bin/python scripts/37_submit_benchmark_finetunes.py            # upload + submit
    .venv/bin/python scripts/37_submit_benchmark_finetunes.py --status   # poll
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

DATA = REPO / "data" / "datasets_ces2025"
JOBS = DATA / "jobs.json"
BASE = os.environ["FOUNDRY_CHAT_ENDPOINT"].rstrip("/").split("/openai")[0]
KEY = os.environ["FOUNDRY_CHAT_KEY"]
FILES_API = "2024-10-21"
JOBS_API = "2025-04-01-preview"
MODEL = "Llama-3.3-70B-Instruct-9"
SEED = 20260930
ARMS = ("sem", "fix")
SIZES = (20_000, 50_000, 100_000)


def call(method: str, path: str, body: dict | None = None, raw: bytes | None = None,
         ctype: str | None = None, api: str = FILES_API) -> tuple[int, dict]:
    url = f"{BASE}{path}{'&' if '?' in path else '?'}api-version={api}"
    headers = {"api-key": KEY}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if raw is not None:
        data, headers["Content-Type"] = raw, ctype
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except Exception:  # noqa: BLE001 - diagnostic only
            return error.code, {}


def upload(path: Path) -> str:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nfine-tune\r\n'.encode(),
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
         "Content-Type: application/jsonl\r\n\r\n").encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    status, data = call("POST", "/openai/files", raw=b"".join(parts),
                        ctype=f"multipart/form-data; boundary={boundary}")
    if status >= 300 or "id" not in data:
        sys.exit(f"upload failed for {path.name}: {status} {data}")
    print(f"uploaded {path.name} -> {data['id']}", flush=True)
    return data["id"]


def wait_processed(ids: list[str]) -> None:
    for _ in range(240):
        states = {i: call("GET", f"/openai/files/{i}")[1].get("status") for i in ids}
        if all(s in ("processed", "error", "failed") for s in states.values()):
            break
        time.sleep(15)
    if not all(s == "processed" for s in states.values()):
        sys.exit(f"file processing failed: {states}")


def submit() -> None:
    validation = {arm: upload(DATA / f"{arm}_validation.jsonl") for arm in ARMS}
    train = {(arm, size): upload(DATA / f"{arm}_train_{size}.jsonl") for arm in ARMS for size in SIZES}
    wait_processed([*validation.values(), *train.values()])
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
    for (arm, size), file_id in train.items():
        suffix = f"ces25-{arm}-{size // 1000}k"
        body = {"model": MODEL, "trainingType": "globalStandard", "training_file": file_id,
                "validation_file": validation[arm], "suffix": suffix, "seed": SEED,
                "method": {"type": "supervised", "supervised": {"hyperparameters": {
                    "n_epochs": 1, "batch_size": 64, "learning_rate_multiplier": 1}}}}
        status, data = call("POST", "/openai/fine_tuning/jobs", body=body, api=JOBS_API)
        print(status, suffix, data.get("id"), data.get("status"), data.get("error") or "", flush=True)
        jobs.append({"suffix": suffix, "arm": arm, "size": size, "job_id": data.get("id"),
                     "training_file": file_id, "validation_file": validation[arm],
                     "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "http_status": status})
        JOBS.write_text(json.dumps(jobs, indent=2) + "\n")


def show_status() -> None:
    for job in json.loads(JOBS.read_text()):
        _, data = call("GET", f"/openai/fine_tuning/jobs/{job['job_id']}", api=JOBS_API)
        print(f"{job['suffix']:<18} {data.get('status', '?'):<12} "
              f"{data.get('trained_tokens') or '':>12} {data.get('fine_tuned_model') or ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    show_status() if parser.parse_args().status else submit()
