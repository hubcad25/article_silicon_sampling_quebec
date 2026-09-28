from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_cloud_module():
    spec = importlib.util.spec_from_file_location("cloud_script", REPO / "scripts/20_cloud.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_id_isolates_campaign_and_smoke_paths():
    cloud = load_cloud_module()
    assert cloud.campaign_folder("c0-8k", "final48-250", False) == "final48-250/c0-8k"
    assert cloud.campaign_folder("c0-8k", "final48-250", True) == "final48-250-smoke/c0-8k"
    with pytest.raises(ValueError, match="relative"):
        cloud.campaign_folder("c0-8k", "../pilot", False)


def test_launch_passes_remaining_draws_temperature_and_output_id(monkeypatch, tmp_path):
    cloud = load_cloud_module()
    payload = tmp_path / "runner.tar.gz"
    payload.write_bytes(b"payload")
    requests = []

    def fake_request(method, path, body=None, **kwargs):
        requests.append((method, path, body))
        return None if method == "GET" else {}

    monkeypatch.setattr(cloud, "arm_request", fake_request)
    monkeypatch.setattr(cloud, "ensure_directory", lambda key, folder: None)
    monkeypatch.setattr(cloud, "build_payload", lambda: payload)
    monkeypatch.setattr(cloud, "az", lambda *args, **kwargs: "")
    monkeypatch.setattr(cloud, "env_value", lambda name: f"{name}-value")

    cloud.launch(
        "c0-20k", "storage-key", 7, False, ["A20"], [1.0],
        draws=250, subset="remaining", output_id="final48-250",
        wait_for_deployment_absence="c0-8k-txt",
    )

    put = next(body for method, _, body in requests if method == "PUT")
    command = put["properties"]["containers"][0]["properties"]["command"][2]
    assert "--root /mnt/inference/final48-250/c0-20k" in command
    assert "--subset remaining" in command
    assert "--draws 250" in command
    assert "--temperatures 1.0" in command
    assert "--wait-for-deployment-absence c0-8k-txt" in command
