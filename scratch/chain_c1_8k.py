"""Wait for run 1bis (C0 8k) to finish, then upload and submit C1 8k. Sequential by design."""
import sys, time, json
sys.path.insert(0, "scratch")
from ft import call, upload

API = "2025-04-01-preview"
PREV = "ftjob-0413559e867f42a59860f17f6153597b"
VALID_C1 = "data/datasets/c1_validation.jsonl"
TRAIN_C1 = "data/datasets/c1_train_8000.jsonl"

while True:
    s, d = call("GET", f"/openai/fine_tuning/jobs/{PREV}", api=API)
    st = d.get("status")
    if st in ("succeeded", "failed", "cancelled"):
        print(f"[{time.strftime('%H:%M:%S')}] previous job {st}", flush=True)
        break
    time.sleep(60)
if st != "succeeded":
    sys.exit(f"previous job {st}: C1 8k NOT submitted")

ids = {}
for f in (TRAIN_C1, VALID_C1):
    s, d = upload(f)
    ids[f] = d.get("id")
    print(s, f, ids[f], flush=True)
for _ in range(60):
    sts = [call("GET", f"/openai/files/{i}")[1].get("status") for i in ids.values()]
    if all(x in ("processed", "error", "failed") for x in sts):
        break
    time.sleep(15)
if not all(x == "processed" for x in sts):
    sys.exit(f"file processing failed: {sts}")

body = {"model": "Llama-3.3-70B-Instruct-9", "trainingType": "globalStandard",
        "training_file": ids[TRAIN_C1], "validation_file": ids[VALID_C1],
        "suffix": "c1-8k-txt", "seed": 20260924,
        "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 1}}}}
s, d = call("POST", "/openai/fine_tuning/jobs", body=body, api=API)
print(s, "C1 8k job:", d.get("id"), d.get("status"), d.get("error"), json.dumps(ids), flush=True)
