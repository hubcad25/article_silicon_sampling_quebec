"""C by reward, Azure-only: expert iteration (rejection sampling + SFT) on base Llama 3.3 70B.

Azure has no reinforcement fine-tuning for Llama, and its RFT models are out of
reach (o4-mini retiring, GPT-5 by invitation, open models on dedicated GPUs).
Expert iteration approximates it with what works: the model writes K tables per
training dossier, the reward (benchmark/dossier_grader.grade, 1 - mean TV vs the
real distribution) picks the best one, and an ordinary SFT job teaches the model
to write it. Round 2 would sample from the round-1 model the same way.

    .venv/bin/python scripts/43_dossier_expert_iteration.py launch          # sample in ACI
    .venv/bin/python scripts/43_dossier_expert_iteration.py status
    .venv/bin/python scripts/43_dossier_expert_iteration.py fetch
    .venv/bin/python scripts/43_dossier_expert_iteration.py select          # -> data/datasets_dossier_ei/
    .venv/bin/python scripts/43_dossier_expert_iteration.py sample --out DIR  # what the container runs

Samples go round by round (every dossier's draw 0, then draw 1, ...), so
``select`` works on a partial run. Base Llama quota is 250k tokens/min
(GlobalStandard only), about 40 requests/min for ~5k-token dossiers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmark import backends  # noqa: E402
from benchmark.dossier_grader import CELLS, _parse, grade  # noqa: E402

DATA = REPO / "data" / "datasets_dossier"
OUT = REPO / "data" / "datasets_dossier_ei"
LOCAL_SAMPLES = OUT / "samples"
FILES = {"train": "dossier_rft_train.jsonl", "validation": "dossier_rft_validation.jsonl"}
K = 8
TEMPERATURE = 1.0
MAX_TOKENS = 1200  # a 7-option table is ~700 tokens; the TPM limit counts max_tokens
DEPLOYMENT = dict(deployment="bench-llama33-70b-ei", model="Llama-3.3-70B-Instruct", version="9",
                  sku="GlobalStandard", capacity=250)
LANE, REMOTE = "ei", "benchmark/ei"


# -- sample (local or in the container) -------------------------------------

def sample(data_dir: Path, out_dir: Path, k: int, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = backends.FoundryBackend(**DEPLOYMENT)
    try:
        backend.ensure()
        for split, name in FILES.items():
            lines = [json.loads(l) for l in (data_dir / name).read_text(encoding="utf-8").splitlines()]
            path = out_dir / f"samples_{split}.jsonl"
            done = set()
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    done.add((r["i"], r["s"]))
            jobs = [(i, s) for s in range(k) for i in range(len(lines)) if (i, s) not in done]
            print(f"{split}: {len(done)} done, {len(jobs)} to go", flush=True)
            lock, tick = threading.Lock(), [time.monotonic(), 0]
            with path.open("a", encoding="utf-8") as handle:
                def one(job):
                    i, s = job
                    reply = backend.complete(lines[i]["messages"], TEMPERATURE, MAX_TOKENS)
                    with lock:
                        handle.write(json.dumps({"i": i, "s": s, "item": lines[i]["item"], "raw": reply},
                                                ensure_ascii=False) + "\n")
                        handle.flush()
                        tick[1] += 1
                        if time.monotonic() - tick[0] > 120:
                            tick[0] = time.monotonic()
                            print(f"  {split}: +{tick[1]} · {backend.counters()}", flush=True)
                with ThreadPoolExecutor(workers) as pool:
                    list(pool.map(one, jobs))
    finally:
        backend.teardown()


# -- ACI ---------------------------------------------------------------------

def _cloud():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud20", REPO / "scripts" / "20_cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(k: int, workers: int) -> None:
    from benchmark import cloud
    c20 = _cloud()
    name = cloud.container(LANE)
    current = c20.arm_request("GET", c20.container_path(name))
    if (current or {}).get("properties", {}).get("instanceView", {}).get("state") == "Running":
        raise SystemExit(f"{name} already running")
    if current is not None:
        c20.arm_request("DELETE", c20.container_path(name))
    cloud.mkdir(f"{REMOTE}/data")
    cloud.mkdir(f"{REMOTE}/samples")
    for fname in FILES.values():
        cloud.upload(DATA / fname, f"{REMOTE}/data/{fname}")
    payload = ["scripts/43_dossier_expert_iteration.py", "benchmark/__init__.py", "benchmark/backends.py",
               "benchmark/dossier_grader.py", "src/article_silicon_sampling_quebec"]
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as tar:
            for rel in payload:
                tar.add(REPO / rel, arcname=rel, filter=lambda i: None if "__pycache__" in i.name else i)
    cloud.upload(Path(handle.name), f"{REMOTE}/payload.tar.gz")
    Path(handle.name).unlink()
    mount = cloud.MOUNT
    command = (f"mkdir -p /work && tar -xzf {mount}/{REMOTE}/payload.tar.gz -C /work && cd /work "
               f"&& pip install --no-cache-dir -q {cloud.PIP} "
               f"&& python -u scripts/43_dossier_expert_iteration.py sample --data {mount}/{REMOTE}/data "
               f"--out {mount}/{REMOTE}/samples --k {k} --workers {workers}")
    env = [
        {"name": "FOUNDRY_CHAT_ENDPOINT", "value": c20.env_value("FOUNDRY_CHAT_ENDPOINT")},
        {"name": "AZURE_CLIENT_ID", "value": c20.IDENTITY_CLIENT_ID},
        {"name": "PYTHONPATH", "value": "/work/src:/work"},
        {"name": "FOUNDRY_CHAT_KEY", "secureValue": c20.env_value("FOUNDRY_CHAT_KEY")},
    ]
    c20.arm_request("PUT", c20.container_path(name), {
        "location": c20.LOCATION,
        "identity": {"type": "UserAssigned", "userAssignedIdentities": {c20.IDENTITY: {}}},
        "properties": {
            "osType": "Linux", "restartPolicy": "Never",
            "containers": [{"name": name, "properties": {
                "image": "python:3.11-slim", "command": ["/bin/sh", "-c", command],
                "resources": {"requests": {"cpu": 2, "memoryInGB": 8}},
                "environmentVariables": env,
                "volumeMounts": [{"name": "share", "mountPath": mount}]}}],
            "volumes": [{"name": "share", "azureFile": {
                "shareName": cloud.SHARE, "storageAccountName": cloud.STORAGE,
                "storageAccountKey": c20.storage_key()}}],
        },
    })
    print(f"launched {name}: {k} draws x {sum(1 for f in FILES.values() for _ in (DATA / f).open())} dossiers")


def fetch() -> None:
    from benchmark import cloud
    LOCAL_SAMPLES.mkdir(parents=True, exist_ok=True)
    for split in FILES:
        subprocess.run(["az", "storage", "file", "download", "--share-name", cloud.SHARE,
                        "--path", f"{REMOTE}/samples/samples_{split}.jsonl",
                        "--dest", str(LOCAL_SAMPLES / f"samples_{split}.jsonl"),
                        "--account-name", cloud.STORAGE, "--account-key", _cloud().storage_key(),
                        "-o", "none"], check=False, capture_output=True)
        path = LOCAL_SAMPLES / f"samples_{split}.jsonl"
        print(f"{split}: {sum(1 for _ in path.open()) if path.exists() else 0} samples")


def status() -> None:
    from benchmark import cloud
    c20 = _cloud()
    group = c20.arm_request("GET", c20.container_path(cloud.container(LANE)))
    if group:
        view = group["properties"]["containers"][0]["properties"].get("instanceView", {})
        print("container:", view.get("currentState", {}))
    for split in FILES:
        try:
            size = cloud.az("storage", "file", "show", "--share-name", cloud.SHARE,
                            "--path", f"{REMOTE}/samples/samples_{split}.jsonl",
                            "--query", "properties.contentLength", "-o", "tsv").strip()
            print(f"{split}: {int(size) / 1e6:.1f} MB")
        except Exception:  # noqa: BLE001 - not written yet
            print(f"{split}: no samples yet")


# -- select ------------------------------------------------------------------

def clean_reply(raw: str, codes: list[str]) -> str:
    data = _parse(raw)
    cells = {}
    def number(value) -> float:
        try:
            return max(float(value or 0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    for cell in CELLS:
        values = [number(data["cells"][cell].get(c, 0)) for c in codes]
        total = sum(values)
        values, total = (values, total) if total > 0 else ([1.0] * len(codes), float(len(codes)))
        cells[cell] = {c: round(v / total, 3) for c, v in zip(codes, values)}
    national = data.get("national") if isinstance(data.get("national"), dict) else None
    out = {}
    if national:
        try:
            values = [number(national.get(c, 0)) for c in codes]
            if sum(values) > 0:
                out["national"] = {c: round(v / sum(values), 3) for c, v in zip(codes, values)}
        except (TypeError, ValueError):
            pass
    out["cells"] = cells
    return json.dumps(out, separators=(",", ":"))


def select() -> None:
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"method": "expert iteration round 1: best of K base-Llama draws by 1 - mean TV",
                "temperature": TEMPERATURE, "k_target": K, "deployment": DEPLOYMENT}
    for split, name in FILES.items():
        lines = [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines()]
        draws: dict[int, list[tuple[float, str]]] = {}
        for line in (LOCAL_SAMPLES / f"samples_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            draws.setdefault(r["i"], []).append((grade({"output_text": r["raw"]}, lines[r["i"]]), r["raw"]))
        kept, best_scores, all_scores = [], [], []
        for i, ds in sorted(draws.items()):
            scores = [s for s, _ in ds]
            all_scores += scores
            score, raw = max(ds, key=lambda t: t[0])
            if score <= 0 or score <= float(np.mean(scores)) and len(set(scores)) > 1:
                continue
            best_scores.append(score)
            line = lines[i]
            reply = clean_reply(raw, [str(c) for c in line["codes"]])
            kept.append({"messages": line["messages"] + [{"role": "assistant", "content": reply}]})
        if split == "train":
            target = OUT / f"ei_train_{len(kept)}.jsonl"
        else:
            target = OUT / "ei_validation.jsonl"
        target.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")
        manifest[split] = {
            "file": target.name, "dossiers": len(lines), "with_draws": len(draws),
            "draws": len(all_scores), "draws_per_dossier": round(len(all_scores) / max(1, len(draws)), 2),
            "kept": len(kept),
            "base_mean_score": round(float(np.mean(all_scores)), 4),
            "base_parse_rate": round(float(np.mean([s > 0 for s in all_scores])), 4),
            "best_of_k_mean_score": round(float(np.mean(best_scores)), 4) if best_scores else None,
            "perfect": 1.0,
        }
        print(split, manifest[split])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=LOCAL_SAMPLES)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--workers", type=int, default=24)
    p = sub.add_parser("launch")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--workers", type=int, default=24)
    for name in ("status", "fetch", "select"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.cmd == "sample":
        sample(args.data, args.out, args.k, args.workers)
    elif args.cmd == "launch":
        launch(args.k, args.workers)
    else:
        {"status": status, "fetch": fetch, "select": select}[args.cmd]()
