"""Upload the 20k train files and submit C0 20k and C1 20k in parallel, same recipe as the 8k runs."""
import sys, time, json
sys.path.insert(0, "scratch")
from ft import call, upload

API = "2025-04-01-preview"
RUNS = {
    "c0-20k-txt": ("data/datasets/c0_train_20000.jsonl", "file-84f674166e8648fe9c8f5f34d1ec4055"),
    "c1-20k-txt": ("data/datasets/c1_train_20000.jsonl", "file-e73439026bc84e82a7c35c16aa32ed0c"),
}

train_ids = {}
for suffix, (path, _) in RUNS.items():
    s, d = upload(path)
    train_ids[suffix] = d.get("id")
    print(s, path, train_ids[suffix], flush=True)
for _ in range(120):
    sts = [call("GET", f"/openai/files/{i}")[1].get("status") for i in train_ids.values()]
    if all(x in ("processed", "error", "failed") for x in sts):
        break
    time.sleep(15)
if not all(x == "processed" for x in sts):
    sys.exit(f"file processing failed: {sts}")

for suffix, (_, valid_id) in RUNS.items():
    body = {"model": "Llama-3.3-70B-Instruct-9", "trainingType": "globalStandard",
            "training_file": train_ids[suffix], "validation_file": valid_id,
            "suffix": suffix, "seed": 20260924,
            "method": {"type": "supervised", "supervised": {"hyperparameters": {
                "n_epochs": 1, "batch_size": 64, "learning_rate_multiplier": 1}}}}
    s, d = call("POST", "/openai/fine_tuning/jobs", body=body, api=API)
    print(s, suffix, "job:", d.get("id"), d.get("status"), d.get("error"),
          json.dumps({"train": train_ids[suffix], "valid": valid_id}), flush=True)
