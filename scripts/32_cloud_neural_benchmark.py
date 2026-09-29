"""Launch and retrieve exploratory benchmark N from Azure Container Instances.

The run reuses benchmark S's complete embedding cache on the existing Azure
File share and writes isolated outputs under ``neural-benchmark/``.

    python scripts/32_cloud_neural_benchmark.py launch
    python scripts/32_cloud_neural_benchmark.py status
    python scripts/32_cloud_neural_benchmark.py logs
    python scripts/32_cloud_neural_benchmark.py fetch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SUBSCRIPTION = "a54061e5-b5f7-49c0-94f0-d2e7b95de4a4"
RESOURCE_GROUP = "rg-opubliq-sondages"
STORAGE = "opubliqsondagesdata"
SHARE = "inference"
LOCATION = "canadaeast"
CONTAINER = "benchmark-n"
FOLDER = "neural-benchmark"
REMOTE_CACHE = "statistical-benchmark/cache/embeddings.parquet"
LOCAL_CACHE = REPO / "data" / ".cache" / "benchmark_s_embedding_cache.parquet"
PIP = "polars pyarrow duckdb numpy scipy scikit-learn python-dotenv azure-storage-blob"
PAYLOAD = (
    "scripts/12_build_similarity_index.py",
    "scripts/27_run_statistical_benchmark.py",
    "scripts/31_run_neural_benchmark.py",
    "src/article_silicon_sampling_quebec",
    "data/datasets/c0_train_8000.jsonl",
    "data/datasets/c0_validation.jsonl",
    "data/items.parquet",
    "data/strata_definition.json",
    "data/analysis/test_blocks.csv",
    "data/analysis/statistical_benchmark/model.npz",
    "data/analysis/statistical_benchmark/manifest.json",
    "data/split/heldout_respondents.parquet",
    "data/split/heldout_items.json",
    "data/crosswalks",
    "docs/adr/0007-benchmark-neuronal-exploratoire.md",
)


def az(*args: str, check: bool = True) -> str:
    result = subprocess.run(["az", *args], capture_output=True, text=True, timeout=900)
    if check and result.returncode != 0:
        raise SystemExit(f"az {' '.join(args[:4])} … failed:\n{result.stderr.strip()}")
    return (result.stdout or "").strip()


def az_json(*args: str) -> dict | list | str | None:
    output = az(*args, "-o", "json", check=False)
    return json.loads(output) if output else None


def env_value(name: str) -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing from .env")


def storage_key() -> str:
    return env_value("AZURE_STORAGE_KEY")


def build_payload() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for relative in PAYLOAD:
            path = REPO / relative
            if not path.exists():
                raise SystemExit(f"payload file missing: {relative}")
            archive.add(
                path,
                arcname=relative,
                filter=lambda info: None if "__pycache__" in info.name else info,
            )
    handle.close()
    return Path(handle.name)


def prepare_storage(key: str) -> None:
    for child in (FOLDER, f"{FOLDER}/output"):
        az(
            "storage", "directory", "create", "--account-name", STORAGE,
            "--account-key", key, "--share-name", SHARE, "--name", child, "-o", "none",
        )
    remote = az_json(
        "storage", "file", "exists", "--account-name", STORAGE, "--account-key", key,
        "--share-name", SHARE, "--path", REMOTE_CACHE,
    )
    if isinstance(remote, dict) and remote.get("exists"):
        return
    if not LOCAL_CACHE.exists():
        raise SystemExit("benchmark S embedding cache is absent locally and remotely")
    print(f"uploading local embedding cache ({LOCAL_CACHE.stat().st_size / 2**20:.0f} MiB)")
    az(
        "storage", "file", "upload", "--account-name", STORAGE, "--account-key", key,
        "--share-name", SHARE, "--source", str(LOCAL_CACHE), "--path", REMOTE_CACHE,
        "-o", "none",
    )


def launch(key: str) -> None:
    current = az("account", "show", "--query", "id", "-o", "tsv")
    if current != SUBSCRIPTION:
        raise SystemExit(f"active subscription is {current}, expected {SUBSCRIPTION}")
    state = az_json(
        "container", "show", "-g", RESOURCE_GROUP, "-n", CONTAINER,
        "--query", "instanceView.state",
    )
    if state == "Running":
        raise SystemExit(f"{CONTAINER} is already running")
    if state is not None:
        az("container", "delete", "-g", RESOURCE_GROUP, "-n", CONTAINER, "--yes", "-o", "none")

    prepare_storage(key)
    payload = build_payload()
    try:
        az(
            "storage", "file", "upload", "--account-name", STORAGE,
            "--account-key", key, "--share-name", SHARE, "--source", str(payload),
            "--path", f"{FOLDER}/runner.tar.gz", "-o", "none",
        )
    finally:
        payload.unlink(missing_ok=True)

    mounted = f"/mnt/{SHARE}"
    command = (
        f"mkdir -p /work {mounted}/{FOLDER}/output && "
        f"tar -xzf {mounted}/{FOLDER}/runner.tar.gz -C /work && cd /work && "
        f"pip install --no-cache-dir -q {PIP} && "
        "python -u scripts/31_run_neural_benchmark.py "
        f"--output-root {mounted}/{FOLDER}/output "
        f"--embedding-cache {mounted}/{REMOTE_CACHE} "
        f"> {mounted}/{FOLDER}/run.log 2>&1"
    )
    az(
        "container", "create", "-g", RESOURCE_GROUP, "-n", CONTAINER,
        "--location", LOCATION, "--image", "python:3.11-slim", "--os-type", "Linux",
        "--cpu", "4", "--memory", "16", "--restart-policy", "Never",
        "--azure-file-volume-account-name", STORAGE,
        "--azure-file-volume-account-key", key,
        "--azure-file-volume-share-name", SHARE,
        "--azure-file-volume-mount-path", mounted,
        "--environment-variables",
        f"AOAI_ENDPOINT={env_value('AOAI_ENDPOINT')}",
        f"AOAI_EMBED_DEPLOYMENT={env_value('AOAI_EMBED_DEPLOYMENT')}",
        f"AZURE_STORAGE_ACCOUNT={env_value('AZURE_STORAGE_ACCOUNT')}",
        f"AZURE_STORAGE_CONTAINER={env_value('AZURE_STORAGE_CONTAINER')}",
        "--secure-environment-variables",
        f"AOAI_KEY={env_value('AOAI_KEY')}",
        f"AZURE_STORAGE_KEY={env_value('AZURE_STORAGE_KEY')}",
        "--command-line", f"/bin/sh -c '{command}'", "-o", "none",
    )
    print(f"launched {CONTAINER} -> {SHARE}/{FOLDER}/ (4 CPU, 16 GiB)")


def status() -> None:
    state = az_json(
        "container", "show", "-g", RESOURCE_GROUP, "-n", CONTAINER,
        "--query", "{state:instanceView.state,detail:containers[0].instanceView.currentState}",
    )
    print(json.dumps(state or {"state": "absent"}, indent=2, ensure_ascii=False))


def logs(key: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "run.log"
        az(
            "storage", "file", "download", "--account-name", STORAGE,
            "--account-key", key, "--share-name", SHARE,
            "--path", f"{FOLDER}/run.log", "--dest", str(destination), "-o", "none",
            check=False,
        )
        if destination.exists():
            print("\n".join(destination.read_text(encoding="utf-8").splitlines()[-100:]))
        else:
            print("no persistent log yet")


def fetch(key: str) -> None:
    destination = REPO / "data" / "analysis" / "neural_benchmark"
    destination.mkdir(parents=True, exist_ok=True)
    az(
        "storage", "file", "download-batch", "--account-name", STORAGE,
        "--account-key", key, "--source", SHARE, "--destination", str(destination),
        "--pattern", f"{FOLDER}/output/*", "-o", "none",
    )
    nested = destination / FOLDER / "output"
    for path in nested.glob("*") if nested.exists() else ():
        path.replace(destination / path.name)
    print(f"downloaded benchmark N outputs to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "status", "logs", "fetch"))
    args = parser.parse_args()
    if args.command == "status":
        status()
        return
    key = storage_key()
    if args.command == "launch":
        launch(key)
    elif args.command == "logs":
        logs(key)
    else:
        fetch(key)


if __name__ == "__main__":
    main()
