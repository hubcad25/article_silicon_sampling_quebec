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
from functools import lru_cache
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.inference import CAMPAIGNS, DEFAULT_DRAWS  # noqa: E402

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
ACI_API = "2023-05-01"
COGNITIVE_API = "2024-10-01"

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


@lru_cache(maxsize=1)
def arm_token() -> str:
    return subprocess.run(
        ["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()


def arm_request(method: str, path: str, body: dict | None = None,
                *, api: str = ACI_API) -> dict | None:
    """Call Azure Resource Manager directly; avoids slow ``az container`` discovery."""
    separator = "&" if "?" in path else "?"
    url = f"https://management.azure.com{path}{separator}api-version={api}"
    request = urllib.request.Request(
        url, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {arm_token()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        detail = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"ARM {method} {path}: HTTP {exc.code}: {detail}") from exc


def container_path(name: str) -> str:
    return (f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
            f"/providers/Microsoft.ContainerInstance/containerGroups/{name}")


def deployment_path(name: str) -> str:
    return (f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
            f"/providers/Microsoft.CognitiveServices/accounts/{ACCOUNT}/deployments/{name}")


def check_subscription() -> None:
    current = az("account", "show", "--query", "id", "-o", "tsv")
    if current != SUBSCRIPTION:
        raise SystemExit(f"active subscription is {current}; run: az account set -s {SUBSCRIPTION}")


def storage_key() -> str:
    return env_value("AZURE_STORAGE_KEY")


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


def ensure_share(key: str) -> None:
    exists = az_json("storage", "share", "exists", "--account-name", STORAGE,
                     "--account-key", key, "--name", SHARE)
    if not (exists or {}).get("exists"):
        az("storage", "share", "create", "--account-name", STORAGE,
           "--account-key", key, "--name", SHARE, "--quota", "100", "-o", "none")


def campaign_folder(name: str, output_id: str | None, smoke: bool) -> str:
    """Return an isolated Azure Files folder; reject paths that could escape it."""
    if output_id is None:
        return f"{name}-smoke" if smoke else name
    root = PurePosixPath(output_id)
    if (root.is_absolute() or not root.parts
            or any(part in {"", ".", ".."} for part in root.parts)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", output_id) is None):
        raise ValueError("output_id must be a non-empty relative Azure Files path")
    suffix = f"{output_id}-smoke" if smoke else output_id
    return f"{suffix}/{name}"


def ensure_directory(key: str, folder: str) -> None:
    current = []
    for part in PurePosixPath(folder).parts:
        current.append(part)
        az("storage", "directory", "create", "--account-name", STORAGE,
           "--account-key", key, "--share-name", SHARE,
           "--name", "/".join(current), "-o", "none")


def launch(name: str, key: str, workers: int | None, smoke: bool,
           arms: list[str] | None = None,
           temperatures: list[float] | None = None, draws: int = DEFAULT_DRAWS,
           subset: str = "pilot", output_id: str | None = None,
           wait_for_deployment_absence: str | None = None) -> None:
    campaign = CAMPAIGNS[name]
    arms = arms or list(campaign.arms)
    unknown = sorted(set(arms) - set(campaign.arms))
    if unknown:
        raise SystemExit(f"{name} serves arms {campaign.arms}, not {unknown}")
    deployments = {candidate.deployment for candidate in CAMPAIGNS.values()}
    if wait_for_deployment_absence not in deployments | {None}:
        raise SystemExit(
            f"unknown deployment to wait for: {wait_for_deployment_absence}"
        )
    if wait_for_deployment_absence == campaign.deployment:
        raise SystemExit(f"{name} cannot wait for its own deployment")
    workers = workers or max(1, 24 // len(arms))
    if campaign.model is None:
        raise SystemExit(f"{name}: model not trained yet — set it in inference.CAMPAIGNS")
    current = arm_request("GET", container_path(container_name(name)))
    state = (current or {}).get("properties", {}).get("instanceView", {}).get("state")
    if state == "Running":
        raise SystemExit(f"{name}: container already running (status / stop first)")
    if current is not None:
        arm_request("DELETE", container_path(container_name(name)))
        for _ in range(60):
            if arm_request("GET", container_path(container_name(name))) is None:
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"timed out deleting {container_name(name)}")

    folder = campaign_folder(name, output_id, smoke)
    ensure_directory(key, folder)
    payload = build_payload()
    az("storage", "file", "upload", "--account-name", STORAGE, "--account-key", key,
       "--share-name", SHARE, "--source", str(payload),
       "--path", f"{folder}/runner.tar.gz", "-o", "none")
    payload.unlink()

    run_args = (f"--campaign {name} --arms {' '.join(arms)} "
                 f"--root /mnt/{SHARE}/{folder} --workers {workers} "
                 f"--subset {subset} --draws {2 if smoke else draws} "
                 "--ensure-deployment --delete-deployment-after")
    if smoke:
        values = temperatures or [1.0]
        run_args += " --limit-item-cell-pairs 2 --temperatures " + " ".join(
            str(value) for value in values
        )
    elif temperatures:
        run_args += " --temperatures " + " ".join(str(value) for value in temperatures)
    if wait_for_deployment_absence:
        run_args += f" --wait-for-deployment-absence {wait_for_deployment_absence}"
    command = (f"mkdir -p /work && tar -xzf /mnt/{SHARE}/{folder}/runner.tar.gz -C /work "
               f"&& cd /work && pip install --no-cache-dir -q {PIP} "
               f"&& python -u scripts/19_run_inference.py {run_args}")
    environment = [
        {"name": "FOUNDRY_CHAT_ENDPOINT", "value": env_value("FOUNDRY_CHAT_ENDPOINT")},
        {"name": "AZURE_STORAGE_ACCOUNT", "value": STORAGE},
        {"name": "AZURE_STORAGE_CONTAINER", "value": env_value("AZURE_STORAGE_CONTAINER")},
        {"name": "AZURE_SUBSCRIPTION_ID", "value": SUBSCRIPTION},
        {"name": "AZURE_CLIENT_ID", "value": IDENTITY_CLIENT_ID},
        {"name": "FOUNDRY_CHAT_KEY", "secureValue": env_value("FOUNDRY_CHAT_KEY")},
        {"name": "AZURE_STORAGE_KEY", "secureValue": key},
    ]
    arm_request("PUT", container_path(container_name(name)), {
        "location": LOCATION,
        "identity": {"type": "UserAssigned", "userAssignedIdentities": {IDENTITY: {}}},
        "properties": {
            "osType": "Linux", "restartPolicy": "Never",
            "containers": [{
                "name": container_name(name),
                "properties": {
                    "image": "python:3.11-slim",
                    "command": ["/bin/sh", "-c", command],
                    "resources": {"requests": {"cpu": 2, "memoryInGB": 8}},
                    "environmentVariables": environment,
                    "volumeMounts": [{"name": "inference", "mountPath": f"/mnt/{SHARE}"}],
                },
            }],
            "volumes": [{
                "name": "inference",
                "azureFile": {"shareName": SHARE, "storageAccountName": STORAGE,
                              "storageAccountKey": key},
            }],
        },
    })
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


def status(names: list[str], key: str, smoke: bool, output_id: str | None = None) -> None:
    for name in names:
        campaign = CAMPAIGNS[name]
        folder = campaign_folder(name, output_id, smoke)
        container_raw = arm_request("GET", container_path(container_name(name)))
        properties = (container_raw or {}).get("properties", {})
        instances = properties.get("containers", [])
        current_state = (instances[0].get("properties", {}).get("instanceView", {})
                         .get("currentState", {}) if instances else {})
        container = None if container_raw is None else {
            "state": properties.get("instanceView", {}).get("state"),
            "exit": current_state.get("exitCode"),
        }
        dep = arm_request("GET", deployment_path(campaign.deployment), api=COGNITIVE_API)
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


def fetch(name: str, key: str, smoke: bool, output_id: str | None = None) -> None:
    folder = campaign_folder(name, output_id, smoke)
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
    arm_request("DELETE", container_path(container_name(name)))
    print(f"{name}: container deleted")
    if not keep_deployment:
        arm_request("DELETE", deployment_path(CAMPAIGNS[name].deployment), api=COGNITIVE_API)
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
                        help="2 pairs x 2 draws, T from --temperatures (default 1.0).")
        if command in {"launch", "status", "fetch"}:
            p.add_argument("--output-id", help="Persistent output namespace, e.g. final48-250.")
        if command == "launch":
            p.add_argument("--arms", nargs="+",
                           help="Subset of the campaign's arms (default: all).")
            p.add_argument("--workers", type=int,
                           help="Per arm. Default: 24 split across the campaign's arms "
                                 "(the deployment caps at ~1 000 calls/min anyway).")
            p.add_argument("--temperatures", type=float, nargs="+",
                           help="Temperatures passed to the inference runner.")
            p.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
            p.add_argument("--subset", choices=("pilot", "remaining", "all"), default="pilot")
            p.add_argument(
                "--wait-for-deployment-absence",
                help="Queue this campaign until the named quota-sharing deployment is deleted.",
            )
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
        if not args.output_id:
            parser.error("launch requires --output-id to isolate resumable outputs")
        ensure_share(key)
        for name in names:
            launch(name, key, args.workers, args.smoke, args.arms, args.temperatures,
                   args.draws, args.subset, args.output_id,
                   args.wait_for_deployment_absence)
    elif args.command == "status":
        status(names, key, args.smoke, args.output_id)
    elif args.command == "fetch":
        for name in names:
            fetch(name, key, args.smoke, args.output_id)


if __name__ == "__main__":
    main()
