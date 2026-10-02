"""Send the frozen requests to one model and record its answers (plan §5-§6).

    python -m benchmark.run --model ces25-fix-100k --deploy --teardown
    python -m benchmark.run --model ces25-fix-100k --profiles-per-cell 10   # smoke test

Every call is appended to benchmark/runs/<model>/responses.jsonl as soon as it
returns, so an interrupted run resumes where it stopped. Temperature 1.0, one
answer per request, no retry on a completed answer (the backend retries
transport failures only). A reply is mapped to an option by exact text, then by
``prompts.normalise_answer``; anything else is invalid and kept as such.

At the end: responses.csv (one row per request) and run_manifest.json (model,
frozen-set hash, counts, invalid and content-filter rates).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from . import FROZEN, REPO, RUNS, backends

sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.prompts import normalise_answer  # noqa: E402

TEMPERATURE = 1.0
MAX_TOKENS = 32


#: Typographic quotes only: the catalogue mixes "Don't" and "Don’t".
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def parse(reply: str, options: dict[str, str]) -> str | None:
    if reply in options:
        return options[reply]
    key = normalise_answer(reply.translate(_QUOTES))
    matches = {code for text, code in options.items()
               if key and normalise_answer(text.translate(_QUOTES)) == key}
    return matches.pop() if len(matches) == 1 else None


def load_requests(profiles_per_cell: int | None) -> list[dict]:
    requests, kept = [], defaultdict(set)
    with (FROZEN / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            seen = kept[record["cell"]]
            if profiles_per_cell is not None and record["profile_id"] not in seen \
                    and len(seen) >= profiles_per_cell:
                continue
            seen.add(record["profile_id"])
            requests.append(record)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--profiles-per-cell", type=int, default=None,
                        help="subsample for a smoke test (results go to <model>-smoke)")
    parser.add_argument("--deploy", action="store_true", help="create the deployment first")
    parser.add_argument("--teardown", action="store_true", help="delete the deployment at the end")
    args = parser.parse_args()

    config, backend = backends.load(args.model)
    out = RUNS / (args.model + ("-smoke" if args.profiles_per_cell else ""))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "responses.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[record["request_id"]] = record
    requests = [r for r in load_requests(args.profiles_per_cell) if r["request_id"] not in done]
    print(f"{args.model}: {len(done)} done, {len(requests)} to go", flush=True)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock, progress = threading.Lock(), Counter()
    tick = time.monotonic()
    try:
        if args.deploy:
            backend.ensure()
        with checkpoint.open("a", encoding="utf-8") as handle:
            def work(request: dict) -> None:
                nonlocal tick
                reply = backend.complete(request["messages"], TEMPERATURE, MAX_TOKENS)
                code = parse(reply, request["options"])
                record = {k: request[k] for k in ("request_id", "profile_id", "cell", "language", "item")}
                record |= {"raw_response": reply, "code": code or "", "valid": code is not None}
                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    done[request["request_id"]] = record
                    progress["n"] += 1
                    if time.monotonic() - tick > 60:
                        tick = time.monotonic()
                        print(f"  {len(done)} done · {backend.counters()}", flush=True)

            with ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(work, requests))
    finally:
        if args.teardown:
            backend.teardown()

    frame = pd.DataFrame(list(done.values())).sort_values("request_id")
    frame.to_csv(out / "responses.csv", index=False)
    frozen = json.loads((FROZEN / "manifest.json").read_text())
    manifest = {
        "model": args.model, "config": config, "frozen_sha256": frozen["sha256"],
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "profiles_per_cell": args.profiles_per_cell, "n_requests": len(frame),
        "invalid_rate": round(float(1 - frame.valid.mean()), 5),
        "invalid_by_item": frame.groupby("item").valid.apply(lambda v: round(1 - v.mean(), 4)).to_dict(),
        "backend_counters_this_session": backend.counters(),
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_requests", "invalid_rate", "invalid_by_item")},
                     indent=1))


if __name__ == "__main__":
    main()
