"""Run the benchmark in Azure Container Instances, one container per quota lane.

Same infrastructure as scripts/20_cloud.py: the ``c0-inference-runner`` managed
identity, the ``inference`` Azure Files share. The frozen set and the runs live
on the share under ``benchmark/``; a container runs its lane's models one after
the other (``benchmark.run --deploy --teardown``), so each deployment exists only
while its model is being queried. Runs resume from their checkpoint.

    python -m benchmark.cloud upload-frozen                    # once
    python -m benchmark.cloud push-runs                        # local checkpoints -> share
    python -m benchmark.cloud launch gs ces25-sem-50k ces25-sem-100k
    python -m benchmark.cloud status
    python -m benchmark.cloud fetch                            # share -> benchmark/runs/
    python -m benchmark.cloud stop gs
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tarfile
import tempfile
from pathlib import Path

from . import REPO, ROOT

_spec = importlib.util.spec_from_file_location("cloud20", REPO / "scripts" / "20_cloud.py")
cloud20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cloud20)

SHARE, STORAGE = cloud20.SHARE, cloud20.STORAGE
MOUNT = f"/mnt/{SHARE}"
REMOTE = "benchmark"
PIP = "pandas numpy polars pyarrow python-dotenv azure-identity"
PAYLOAD = ["benchmark/__init__.py", "benchmark/backends.py", "benchmark/run.py",
           "benchmark/models.json", "src/article_silicon_sampling_quebec"]


def az(*args: str) -> str:
    return cloud20.az(*args, "--account-name", STORAGE, "--account-key", cloud20.storage_key())


def container(lane: str) -> str:
    return f"bench-{lane}"


def upload(local: Path, remote: str) -> None:
    az("storage", "file", "upload", "--share-name", SHARE, "--source", str(local),
       "--path", remote, "-o", "none")


def mkdir(remote: str) -> None:
    cloud20.ensure_directory(cloud20.storage_key(), remote)


def upload_frozen() -> None:
    mkdir(f"{REMOTE}/frozen")
    for name in ("requests.jsonl", "manifest.json", "profiles.csv"):
        upload(ROOT / "frozen" / name, f"{REMOTE}/frozen/{name}")
        print(f"uploaded frozen/{name}")


def push_runs() -> None:
    for folder in sorted((ROOT / "runs").iterdir()):
        if folder.name.endswith("-smoke") or not (folder / "responses.jsonl").exists():
            continue
        mkdir(f"{REMOTE}/runs/{folder.name}")
        upload(folder / "responses.jsonl", f"{REMOTE}/runs/{folder.name}/responses.jsonl")
        print(f"pushed runs/{folder.name}")


def fetch() -> None:
    for folder in az_list(f"{REMOTE}/runs"):
        local = ROOT / "runs" / folder
        local.mkdir(parents=True, exist_ok=True)
        for name in ("responses.jsonl", "responses.csv", "run_manifest.json"):
            subprocess.run(["az", "storage", "file", "download", "--share-name", SHARE,
                            "--path", f"{REMOTE}/runs/{folder}/{name}", "--dest", str(local / name),
                            "--account-name", STORAGE, "--account-key", cloud20.storage_key(),
                            "-o", "none"], capture_output=True)
        print(f"fetched runs/{folder}")


def az_list(remote: str) -> list[str]:
    out = az("storage", "file", "list", "--share-name", SHARE, "--path", remote,
             "--query", "[].name", "-o", "tsv")
    return [line for line in out.splitlines() if line]


def launch(lane: str, models: list[str], workers: int) -> None:
    current = cloud20.arm_request("GET", cloud20.container_path(container(lane)))
    state = (current or {}).get("properties", {}).get("instanceView", {}).get("state")
    if state == "Running":
        raise SystemExit(f"{container(lane)} already running")
    if current is not None:
        cloud20.arm_request("DELETE", cloud20.container_path(container(lane)))
    mkdir(f"{REMOTE}/lanes/{lane}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as tar:
            for rel in PAYLOAD:
                tar.add(REPO / rel, arcname=rel,
                        filter=lambda info: None if "__pycache__" in info.name else info)
    upload(Path(handle.name), f"{REMOTE}/lanes/{lane}/payload.tar.gz")
    Path(handle.name).unlink()

    runs = " && ".join(f"python -u -m benchmark.run --model {m} --workers {workers} "
                       f"--deploy --teardown" for m in models)
    command = (f"mkdir -p /work && tar -xzf {MOUNT}/{REMOTE}/lanes/{lane}/payload.tar.gz -C /work "
               f"&& cd /work && pip install --no-cache-dir -q {PIP} && {runs}")
    env = [
        {"name": "BENCH_FROZEN", "value": f"{MOUNT}/{REMOTE}/frozen"},
        {"name": "BENCH_RUNS", "value": f"{MOUNT}/{REMOTE}/runs"},
        {"name": "FOUNDRY_CHAT_ENDPOINT", "value": cloud20.env_value("FOUNDRY_CHAT_ENDPOINT")},
        {"name": "AZURE_CLIENT_ID", "value": cloud20.IDENTITY_CLIENT_ID},
        {"name": "PYTHONPATH", "value": "/work/src:/work"},
        {"name": "FOUNDRY_CHAT_KEY", "secureValue": cloud20.env_value("FOUNDRY_CHAT_KEY")},
    ]
    key = cloud20.storage_key()
    cloud20.arm_request("PUT", cloud20.container_path(container(lane)), {
        "location": cloud20.LOCATION,
        "identity": {"type": "UserAssigned", "userAssignedIdentities": {cloud20.IDENTITY: {}}},
        "properties": {
            "osType": "Linux", "restartPolicy": "Never",
            "containers": [{"name": container(lane), "properties": {
                "image": "python:3.11-slim", "command": ["/bin/sh", "-c", command],
                "resources": {"requests": {"cpu": 2, "memoryInGB": 8}},
                "environmentVariables": env,
                "volumeMounts": [{"name": "share", "mountPath": MOUNT}]}}],
            "volumes": [{"name": "share", "azureFile": {
                "shareName": SHARE, "storageAccountName": STORAGE, "storageAccountKey": key}}],
        },
    })
    print(f"launched {container(lane)}: {', '.join(models)}")


def status() -> None:
    for model in az_list(f"{REMOTE}/runs"):
        props = az("storage", "file", "show", "--share-name", SHARE,
                   "--path", f"{REMOTE}/runs/{model}/responses.jsonl",
                   "--query", "[properties.contentLength, properties.lastModified]", "-o", "tsv")
        size, modified = (props.split() + ["?", "?"])[:2]
        done = az_list(f"{REMOTE}/runs/{model}")
        print(f"{model:<20} {int(size) / 1e6 if size.isdigit() else 0:>7.1f} MB  "
              f"last write {modified}  {'DONE' if 'run_manifest.json' in done else ''}")
    for lane in ("gs", "dzs", "base"):
        group = cloud20.arm_request("GET", cloud20.container_path(container(lane)))
        if group:
            view = group["properties"]["containers"][0]["properties"].get("instanceView", {})
            print(f"container {container(lane)}: {view.get('currentState', {}).get('state')}")


def stop(lane: str) -> None:
    cloud20.arm_request("DELETE", cloud20.container_path(container(lane)))
    print(f"deleted {container(lane)} (its deployment must be deleted separately if mid-run)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload-frozen")
    sub.add_parser("push-runs")
    sub.add_parser("fetch")
    sub.add_parser("status")
    p = sub.add_parser("launch")
    p.add_argument("lane")
    p.add_argument("models", nargs="+")
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("stop")
    p.add_argument("lane")
    args = parser.parse_args()
    if args.cmd == "launch":
        launch(args.lane, args.models, args.workers)
    elif args.cmd == "stop":
        stop(args.lane)
    else:
        {"upload-frozen": upload_frozen, "push-runs": push_runs,
         "fetch": fetch, "status": status}[args.cmd]()


if __name__ == "__main__":
    main()
