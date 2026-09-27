"""Run one ADR 0001 inference campaign: its arms side by side on one deployment.

Meant to run in the cloud container (see scripts/20_cloud.py), but works
locally too. Outputs go to ``<root>/<arm>.{jsonl,csv,manifest.json,
diagnostics.csv,progress.json,context.csv}``; rerunning resumes from the JSONL.

    python scripts/19_run_inference.py --campaign c1-8k --root data/analysis/inference/c1-8k \
        --ensure-deployment --delete-deployment-after
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.inference import (  # noqa: E402
    CAMPAIGNS,
    DEFAULT_DRAWS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURES,
    Campaign,
    RunSettings,
    build_item_cells,
    run,
)

AZURE_RESOURCE_GROUP = "rg-opubliq-sondages"
AZURE_ACCOUNT = "info-4552-resource"
ARM_API = "2024-10-01"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {message}", flush=True)


def _arm_request(method: str, deployment: str, body: dict | None = None) -> dict | None:
    """Management-plane call on one deployment: az CLI locally, managed identity in ACI."""
    path = (f"/subscriptions/{os.environ['AZURE_SUBSCRIPTION_ID']}"
            f"/resourceGroups/{AZURE_RESOURCE_GROUP}"
            f"/providers/Microsoft.CognitiveServices/accounts/{AZURE_ACCOUNT}"
            f"/deployments/{deployment}?api-version={ARM_API}")
    if shutil.which("az"):
        token = subprocess.run(
            ["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
            check=True, capture_output=True, text=True).stdout.strip()
    else:
        from azure.identity import ManagedIdentityCredential
        token = ManagedIdentityCredential(
            client_id=os.environ.get("AZURE_CLIENT_ID")
        ).get_token("https://management.azure.com/.default").token
    request = urllib.request.Request(
        f"https://management.azure.com{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"{method} deployment {deployment}: HTTP {exc.code} "
                           f"{exc.read().decode(errors='replace')[:500]}") from exc


def ensure_deployment(campaign: Campaign, timeout: float = 3600) -> None:
    """Create the deployment if absent, then wait until it serves."""
    current = _arm_request("GET", campaign.deployment)
    if current is None:
        _log(f"creating deployment {campaign.deployment} "
             f"({campaign.sku} x {campaign.capacity})")
        _arm_request("PUT", campaign.deployment, {
            "sku": {"name": campaign.sku, "capacity": campaign.capacity},
            "properties": {"model": {"format": "Meta", "name": campaign.model,
                                     "version": campaign.version}},
        })
    deadline = time.monotonic() + timeout
    while True:
        current = _arm_request("GET", campaign.deployment) or {}
        state = current.get("properties", {}).get("provisioningState")
        if state == "Succeeded":
            _log(f"deployment {campaign.deployment} ready "
                 f"(capacity {current.get('sku', {}).get('capacity')})")
            return
        if state in ("Failed", "Canceled") or time.monotonic() > deadline:
            raise RuntimeError(f"deployment {campaign.deployment} ended in state {state}")
        time.sleep(30)


def delete_deployment(deployment: str) -> None:
    _log(f"deleting deployment {deployment}")
    _arm_request("DELETE", deployment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, choices=sorted(CAMPAIGNS))
    parser.add_argument("--root", type=Path, required=True,
                        help="Output directory; one file set per arm.")
    parser.add_argument("--arms", nargs="+", help="Default: the campaign's arms.")
    parser.add_argument("--workers", type=int, default=20, help="Per arm.")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit-item-cell-pairs", type=int)
    parser.add_argument("--stall-seconds", type=float, default=600)
    parser.add_argument("--ensure-deployment", action="store_true")
    parser.add_argument(
        "--delete-deployment-after", action="store_true",
        help="Delete the deployment when every arm has exited, including after failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO / ".env")
    campaign = CAMPAIGNS[args.campaign]
    if campaign.model is None:
        raise SystemExit(f"{campaign.name}: model not trained yet (CAMPAIGNS has no model)")
    arms = args.arms or list(campaign.arms)
    unknown = set(arms) - set(campaign.arms)
    if unknown:
        raise SystemExit(f"{campaign.name} serves arms {campaign.arms}, not {sorted(unknown)}")

    def one_arm(arm: str) -> dict:
        settings = RunSettings(
            deployment=campaign.deployment, model=campaign.model, arm=arm,
            temperatures=tuple(args.temperatures), draws=args.draws,
            workers=args.workers, max_tokens=args.max_tokens,
            limit_item_cell_pairs=args.limit_item_cell_pairs,
        )
        _log(f"arm {arm}: start")
        summary = run(settings, args.root / f"{arm}.csv", stall_seconds=args.stall_seconds)
        _log(f"arm {arm}: done {json.dumps(summary)}")
        return summary

    failed = False
    try:
        # Download the survey parquets once: concurrent arms would otherwise
        # race on the same cache file.
        build_item_cells()
        if args.ensure_deployment:
            ensure_deployment(campaign)
        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            futures = {arm: pool.submit(one_arm, arm) for arm in arms}
            for arm, future in futures.items():
                try:
                    future.result()
                except Exception:
                    failed = True
                    _log(f"arm {arm}: FAILED\n{traceback.format_exc()}")
    except Exception:
        failed = True
        _log(f"campaign {campaign.name}: FAILED\n{traceback.format_exc()}")
    finally:
        if args.delete_deployment_after:
            try:
                delete_deployment(campaign.deployment)
            except Exception:
                failed = True
                _log(f"could not delete {campaign.deployment}\n{traceback.format_exc()}")
    _log(f"campaign {campaign.name}: {'FAILED' if failed else 'completed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # A stalled arm may leave worker threads blocked in a socket; a normal exit
    # would join them forever. The outputs are already written.
    os._exit(code)
