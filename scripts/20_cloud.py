"""Launch and follow ADR 0001 inference campaigns in Azure Container Instances.

One campaign = one fine-tuned model = one deployment = one container, running
the campaign's arms side by side (c0-8k: A; c1-8k: B and B0). The container
creates the deployment, runs, and deletes the deployment when it exits —
success or failure. Results live on the Azure File share ``inference``
(``<campaign>/<arm>.*``) and survive the container; relaunching resumes.

    python scripts/20_cloud.py launch c0-8k c1-8k
    python scripts/20_cloud.py status
    python scripts/20_cloud.py logs c1-8k
    python scripts/20_cloud.py fetch c0-8k        # -> data/analysis/inference/c0-8k/
    python scripts/20_cloud.py stop c1-8k         # container + deployment
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.inference import CAMPAIGNS  # noqa: E402

SUBSCRIPTION = "a54061e5-b5f7-49c0-94f0-d2e7b95de4a4"  # sponsored: the credits
RESOURCE_GROUP = "rg-opubliq-sondages"
ACCOUNT = "info-4552-resource"
STORAGE = "opubliqsondagesdata"
SHARE = "inference"
LOCATION = "canadaeast"
IDENTITY = (f"/subscriptions/{SUBSCRIPTION}/resourcegroups/{RESOURCE_GROUP}/providers/"
            "Microsoft.ManagedIdentity/userAssignedIdentities/c0-inference-runner")
IDENTITY_CLIENT_ID = "9f2e33c5-9817-4ccf-b910-d92b69f7b4a3"
PIP = "polars pyarrow duckdb numpy python-dotenv azure-storage-blob azure-identity"

#: What the container needs, relative to the repo. Survey microdata is pulled
#: from blob storage at run time.
PAYLOAD = [
    "pyproject.toml",
    "scripts/19_run_inference.py",
    "src/article_silicon_sampling_quebec",
    "data/items.parquet",
    "data/item_similarity.parquet",
    "data/strata_definition.json",
    "data/analysis/test_blocks.csv",
    "data/split/heldout_respondents.parquet",
    "data/split/heldout_items.json",
    "data/split/split_manifest.json",
    "data/split/heldout_halves.csv",
    "data/crosswalks",
]


def az(*args: str, capture: bool = True, check: bool = True) -> str:
    result = subprocess.run(["az", *args], capture_output=capture, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"az {' '.join(args[:4])} … failed:\n{result.stderr.strip()}")
    return (result.stdout or "").strip()


def az_json(*args: str) -> dict | list | None:
    result = subprocess.run(["az", *args, "-o", "json"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout) if result.stdout.strip() else None


def check_subscription() -> None:
    current = az("account", "show", "--query", "id", "-o", "tsv")
    if current != SUBSCRIPTION:
        raise SystemExit(f"active subscription is {current}; run: az account set -s {SUBSCRIPTION}")


def storage_key() -> str:
    return az("storage", "account", "keys", "list", "-g", RESOURCE_GROUP, "-n", STORAGE,
              "--query", "[0].value", "-o", "tsv")


def container_name(campaign: str) -> str:
    return f"inf-{campaign}"


def env_value(name: str) -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing from .env")


def build_payload() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(fileobj=handle, mode="w:gz") as tar:
        for rel in PAYLOAD:
            path = REPO / rel
            if not path.exists():
                raise SystemExit(f"payload file missing: {rel}")
            tar.add(path, arcname=rel,
                    filter=lambda info: None if "__pycache__" in info.name else info)
    handle.close()
    return Path(handle.name)


def ensure_share() -> None:
    exists = az_json("storage", "share-rm", "exists", "-g", RESOURCE_GROUP,
                     "--storage-account", STORAGE, "-n", SHARE)
    if not (exists or {}).get("exists"):
        az("storage", "share-rm", "create", "-g", RESOURCE_GROUP, "--storage-account", STORAGE,
           "-n", SHARE, "--quota", "100", "-o", "none")


def launch(name: str, key: str, workers: int | None, smoke: bool,
           arms: list[str] | None = None) -> None:
    campaign = CAMPAIGNS[name]
    arms = arms or list(campaign.arms)
    unknown = sorted(set(arms) - set(campaign.arms))
    if unknown:
        raise SystemExit(f"{name} serves arms {campaign.arms}, not {unknown}")
    workers = workers or max(1, 24 // len(arms))
    if campaign.model is None:
        raise SystemExit(f"{name}: model not trained yet — set it in inference.CAMPAIGNS")
    state = az_json("container", "show", "-g", RESOURCE_GROUP, "-n", container_name(name),
                    "--query", "instanceView.state")
    if state == "Running":
        raise SystemExit(f"{name}: container already running (status / stop first)")
    if state is not None:
        az("container", "delete", "-g", RESOURCE_GROUP, "-n", container_name(name),
           "--yes", "-o", "none")

    folder = f"{name}-smoke" if smoke else name
    az("storage", "directory", "create", "--account-name", STORAGE, "--account-key", key,
       "--share-name", SHARE, "--name", folder, "-o", "none")
    payload = build_payload()
    az("storage", "file", "upload", "--account-name", STORAGE, "--account-key", key,
       "--share-name", SHARE, "--source", str(payload),
       "--path", f"{folder}/runner.tar.gz", "-o", "none")
    payload.unlink()

    run_args = (f"--campaign {name} --arms {' '.join(arms)} "
                f"--root /mnt/{SHARE}/{folder} --workers {workers} "
                "--ensure-deployment --delete-deployment-after")
    if smoke:
        run_args += " --limit-item-cell-pairs 2 --draws 2 --temperatures 0.7"
    command = (f"mkdir -p /work && tar -xzf /mnt/{SHARE}/{folder}/runner.tar.gz -C /work "
               f"&& cd /work && pip install --no-cache-dir -q {PIP} "
               f"&& python -u scripts/19_run_inference.py {run_args}")
    az("container", "create", "-g", RESOURCE_GROUP, "-n", container_name(name),
       "--location", LOCATION, "--image", "python:3.11-slim", "--os-type", "Linux",
       "--cpu", "2", "--memory", "8", "--restart-policy", "Never",
       "--assign-identity", IDENTITY,
       "--azure-file-volume-account-name", STORAGE, "--azure-file-volume-account-key", key,
       "--azure-file-volume-share-name", SHARE, "--azure-file-volume-mount-path", f"/mnt/{SHARE}",
       "--environment-variables",
       f"FOUNDRY_CHAT_ENDPOINT={env_value('FOUNDRY_CHAT_ENDPOINT')}",
       f"AZURE_STORAGE_ACCOUNT={STORAGE}",
       f"AZURE_STORAGE_CONTAINER={env_value('AZURE_STORAGE_CONTAINER')}",
       f"AZURE_SUBSCRIPTION_ID={SUBSCRIPTION}", f"AZURE_CLIENT_ID={IDENTITY_CLIENT_ID}",
       "--secure-environment-variables",
       f"FOUNDRY_CHAT_KEY={env_value('FOUNDRY_CHAT_KEY')}", f"AZURE_STORAGE_KEY={key}",
       "--command-line", f"/bin/sh -c '{command}'", "-o", "none")
    print(f"{name}: launched {container_name(name)} → {SHARE}/{folder}/ "
          f"({', '.join(arms)}; {campaign.deployment} "
          f"{campaign.sku} x {campaign.capacity})")


def read_progress(key: str, folder: str, arm: str) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "p.json"
        result = subprocess.run(
            ["az", "storage", "file", "download", "--account-name", STORAGE,
             "--account-key", key, "--share-name", SHARE,
             "--path", f"{folder}/{arm}.progress.json", "--dest", str(dest), "-o", "none"],
            capture_output=True, text=True)
        if result.returncode != 0 or not dest.exists():
            return None
        return json.loads(dest.read_text(encoding="utf-8"))


def _age(stamp: str | None) -> str:
    if not stamp:
        return "?"
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def status(names: list[str], key: str, smoke: bool) -> None:
    deployments = {d["name"]: d for d in az_json(
        "cognitiveservices", "account", "deployment", "list",
        "-g", RESOURCE_GROUP, "-n", ACCOUNT) or []}
    for name in names:
        campaign = CAMPAIGNS[name]
        folder = f"{name}-smoke" if smoke else name
        container = az_json("container", "show", "-g", RESOURCE_GROUP, "-n", container_name(name),
                            "--query", "{state:instanceView.state, "
                            "exit:containers[0].instanceView.currentState.exitCode}")
        dep = deployments.get(campaign.deployment)
        dep_text = ("absent" if dep is None else
                    f"{dep['properties']['provisioningState']} "
                    f"({dep['sku']['name']} x {dep['sku']['capacity']})")
        cont_text = ("absent" if container is None else
                     f"{container['state']}" + (f" exit={container['exit']}"
                                                if container.get("exit") is not None else ""))
        print(f"■ {name:7s} model={'ready' if campaign.model else 'not trained'} · "
              f"container {cont_text} · deployment {campaign.deployment}: {dep_text}")
        for arm in campaign.arms:
            p = read_progress(key, folder, arm)
            if p is None:
                print(f"    {arm:3s} no progress yet")
                continue
            pct = 100 * p["done"] / p["tasks"] if p["tasks"] else 0
            eta = (f"~{p['pending'] / p['per_minute_last_5min'] / 60:.1f} h"
                   if p["per_minute_last_5min"] and p["pending"] else "—")
            line = (f"    {arm:3s} {p['state']:9s} {p['done']:>7,}/{p['tasks']:,} ({pct:5.1f} %) · "
                    f"invalid {p['invalid']:,} · {p['per_minute_last_5min']:,.0f}/min · "
                    f"429 {p['throttled_total']:,} · ETA {eta} · updated {_age(p['updated_at'])} ago")
            print(line)
            if p.get("error"):
                print(f"        error: {p['error']}")


def fetch(name: str, key: str, smoke: bool) -> None:
    folder = f"{name}-smoke" if smoke else name
    dest = REPO / "data" / "analysis" / "inference"
    dest.mkdir(parents=True, exist_ok=True)
    for arm in CAMPAIGNS[name].arms:
        az("storage", "file", "download-batch", "--account-name", STORAGE, "--account-key", key,
           "--source", SHARE, "--destination", str(dest), "--pattern", f"{folder}/{arm}.*",
           "-o", "none")
    files = sorted((dest / folder).glob("*"))
    print(f"{name}: {len(files)} files in {dest / folder}")
    for f in files:
        print(f"    {f.name:28s} {f.stat().st_size:>12,} B")


def stop(name: str, keep_deployment: bool) -> None:
    az("container", "delete", "-g", RESOURCE_GROUP, "-n", container_name(name), "--yes",
       "-o", "none", check=False)
    print(f"{name}: container deleted")
    if not keep_deployment:
        az("cognitiveservices", "account", "deployment", "delete", "-g", RESOURCE_GROUP,
           "-n", ACCOUNT, "--deployment-name", CAMPAIGNS[name].deployment, check=False)
        print(f"{name}: deployment {CAMPAIGNS[name].deployment} deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("launch", "status", "logs", "fetch", "stop"):
        p = sub.add_parser(command)
        p.add_argument("campaigns", nargs="*" if command == "status" else "+",
                       metavar="campaign", help=f"one of {sorted(CAMPAIGNS)}")
        p.add_argument("--smoke", action="store_true",
                       help="2 pairs x 2 draws x T=0.7, into <campaign>-smoke/")
        if command == "launch":
            p.add_argument("--arms", nargs="+",
                           help="Subset of the campaign's arms (default: all).")
            p.add_argument("--workers", type=int,
                           help="Per arm. Default: 24 split across the campaign's arms "
                                "(the deployment caps at ~1 000 calls/min anyway).")
        if command == "stop":
            p.add_argument("--keep-deployment", action="store_true")
    args = parser.parse_args()

    names = args.campaigns or list(CAMPAIGNS)
    unknown = sorted(set(names) - set(CAMPAIGNS))
    if unknown:
        parser.error(f"unknown campaign(s) {unknown}; choose from {sorted(CAMPAIGNS)}")
    check_subscription()
    if args.command == "logs":
        for name in names:
            print(az("container", "logs", "-g", RESOURCE_GROUP, "-n", container_name(name),
                     check=False))
        return
    if args.command == "stop":
        for name in names:
            stop(name, args.keep_deployment)
        return
    key = storage_key()
    if args.command == "launch":
        ensure_share()
        for name in names:
            launch(name, key, args.workers, args.smoke, args.arms)
    elif args.command == "status":
        status(names, key, args.smoke)
    elif args.command == "fetch":
        for name in names:
            fetch(name, key, args.smoke)


if __name__ == "__main__":
    main()
