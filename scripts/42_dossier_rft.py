"""C by reward: reinforcement fine-tuning of o4-mini on the dossier exercises.

The model reads a dossier (benchmark/dossier.py) and writes the 15-cell
distribution; benchmark/dossier_grader.py scores it 1 - mean TV against the
real distribution of that past question. Data: data/datasets_dossier/
dossier_rft_{train,validation}.jsonl (scripts/41, screened by scripts/38).

    .venv/bin/python scripts/42_dossier_rft.py baseline   # o4-mini untrained on both sets
    .venv/bin/python scripts/42_dossier_rft.py submit [--seeds 2]
    .venv/bin/python scripts/42_dossier_rft.py status

``baseline`` deploys o4-mini, runs and scores the zero-shot dossier on the main
and novel sets, then deletes the deployment. Job ids go to
data/datasets_dossier/rft_jobs.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from benchmark import backends  # noqa: E402
from benchmark.dossier_grader import grader_definition  # noqa: E402

DATA = REPO / "data" / "datasets_dossier"
JOBS = DATA / "rft_jobs.json"
BASE = os.environ["FOUNDRY_CHAT_ENDPOINT"].rstrip("/").split("/openai")[0]
KEY = os.environ["FOUNDRY_CHAT_KEY"]
MODEL, VERSION = "o4-mini", "2025-04-16"
SEED = 20261017
DEPLOYMENT = "o4-mini-dossier"


def call(method: str, path: str, body: dict | None = None, raw: bytes | None = None,
         ctype: str | None = None) -> tuple[int, dict]:
    headers = {"api-key": KEY}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if raw is not None:
        data, headers["Content-Type"] = raw, ctype
    request = urllib.request.Request(f"{BASE}/openai/v1{path}", data=data, headers=headers, method=method)
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
    raw = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nfine-tune\r\n'.encode(),
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
         "Content-Type: application/jsonl\r\n\r\n").encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    status, data = call("POST", "/files", raw=raw, ctype=f"multipart/form-data; boundary={boundary}")
    if status >= 300 or "id" not in data:
        sys.exit(f"upload failed for {path.name}: {status} {data}")
    print(f"uploaded {path.name} -> {data['id']}", flush=True)
    for _ in range(240):
        state = call("GET", f"/files/{data['id']}")[1].get("status")
        if state in ("processed", "error", "failed"):
            break
        time.sleep(15)
    if state != "processed":
        sys.exit(f"{path.name}: file status {state}")
    return data["id"]


def baseline() -> None:
    backend = backends.FoundryBackend(deployment=DEPLOYMENT, model=MODEL, version=VERSION,
                                      sku="GlobalStandard", capacity=100, format="OpenAI")
    try:
        backend.ensure()
        for target_set in ("main", "novel"):
            env = {**os.environ, "BENCH_SET": target_set}
            for cmd in ("run", "score"):
                subprocess.run([sys.executable, "-m", "benchmark.dossier", cmd, "--deployment", DEPLOYMENT],
                               env=env, cwd=REPO, check=False)
    finally:
        backend.teardown()


def submit(n_seeds: int) -> None:
    grader = grader_definition()
    status, data = call("POST", "/fine_tuning/alpha/graders/validate", body={"grader": grader})
    print("grader validate:", status, json.dumps(data)[:300], flush=True)
    if status >= 300:
        sys.exit("grader rejected")
    train = upload(DATA / "dossier_rft_train.jsonl")
    validation = upload(DATA / "dossier_rft_validation.jsonl")
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
    for s in range(n_seeds):
        suffix = f"ces25-dossier-rft-s{s + 1}"
        body = {"model": f"{MODEL}-{VERSION}", "training_file": train, "validation_file": validation,
                "suffix": suffix, "seed": SEED + s,
                "method": {"type": "reinforcement", "reinforcement": {"grader": grader}}}
        status, data = call("POST", "/fine_tuning/jobs", body=body)
        print(status, suffix, data.get("id"), data.get("status"), data.get("error") or "", flush=True)
        jobs.append({"suffix": suffix, "seed": SEED + s, "job_id": data.get("id"),
                     "training_file": train, "validation_file": validation, "http_status": status,
                     "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        JOBS.write_text(json.dumps(jobs, indent=2) + "\n")


def status() -> None:
    for job in json.loads(JOBS.read_text()):
        _, data = call("GET", f"/fine_tuning/jobs/{job['job_id']}")
        print(f"{job['suffix']:<24} {data.get('status', '?'):<12} {data.get('fine_tuned_model') or ''} "
              f"{(data.get('error') or {}).get('message', '') if isinstance(data.get('error'), dict) else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    p = sub.add_parser("submit")
    p.add_argument("--seeds", type=int, default=2)
    sub.add_parser("status")
    args = parser.parse_args()
    {"baseline": baseline, "status": status}.get(args.cmd, lambda: submit(args.seeds))()
