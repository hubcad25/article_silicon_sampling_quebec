"""Screen every training item with Azure AI Content Safety (hate category).

Foundry's fine-tuning data check rejects a training file when too many lines are
flagged for hate/fairness, and its thresholds cannot be changed. The flagged
lines are attitude batteries rendered with an agreement answer ("Immigrants
take jobs away from other Canadians. → Strongly agree"). This script scores,
with the Content Safety API of the same Foundry resource, every rendering an
item can take in a training line:

    target   the question block: wording + option list
    context  one line per option: "wording → option"

in each language the item exists in, and keeps the maximum hate severity
(EightSeverityLevels, 0-7). scripts/36 drops items at or above the cut from
targets and contexts alike.

    .venv/bin/python scripts/38_content_safety_screen.py            # score items
    .venv/bin/python scripts/38_content_safety_screen.py --lines FILE [N]  # audit N lines of a JSONL
    .venv/bin/python scripts/38_content_safety_screen.py --filter FILE...  # drop lines >= MAX_SEVERITY

``--filter`` scores whole lines in chunks of at most 9,000 characters (the API
limit; long dossier lines would otherwise be judged on their head only), keeps
each line's maximum, and rewrites every FILE without the flagged lines (the
original is kept as FILE.unscreened).

Scores are cached in data/content_safety/cache.json (text -> severity).
"""

from __future__ import annotations

import http.client
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
load_dotenv(REPO / ".env")

import os  # noqa: E402

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.prompts import build_item_specs  # noqa: E402

OUT = REPO / "data" / "content_safety"
CACHE = OUT / "cache.json"
BASE = os.environ["FOUNDRY_CHAT_ENDPOINT"].rstrip("/").split("/openai")[0]
KEY = os.environ["FOUNDRY_CHAT_KEY"]
URL = f"{BASE}/contentsafety/text:analyze?api-version=2024-09-01"
SURVEYS = ("ces_2019_online", "ces_2021", "dc_2019", "dc_2020", "dc_2021", "dc_2022",
           "dc_2023", "dc_2024")
ITEMS = [REPO / "data" / "items.parquet", REPO / "data" / "items_extra.parquet"]
FRENCH = [ds.FRENCH_WORDING_PATH, REPO / "data" / "extra_french_wording.json"]

OUT.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


def severity(text: str) -> int:
    body = json.dumps({"text": text[:9000], "categories": ["Hate"],
                       "outputType": "EightSeverityLevels"}).encode()
    for attempt in range(8):
        request = urllib.request.Request(URL, data=body, method="POST", headers={
            "Ocp-Apim-Subscription-Key": KEY, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                analysis = json.loads(response.read())["categoriesAnalysis"]
            if analysis:  # an empty list now and then under load: retry
                return analysis[0]["severity"]
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            pass  # incl. RemoteDisconnected, raised raw by urllib under load
        time.sleep(min(2 ** attempt, 30) * (1 + random.random() / 4))
    raise RuntimeError("content safety: retries exhausted")


def score_all(texts: set[str]) -> dict[str, int]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = sorted(t for t in texts if t not in cache)
    print(f"{len(texts)} distinct texts, {len(todo)} to score", flush=True)

    def work(text: str) -> None:
        value = severity(text)
        with _lock:
            cache[text] = value
            if len(cache) % 500 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"  {len(cache)} scored", flush=True)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, todo))
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


def renderings(spec) -> tuple[str, list[str]]:
    target = spec.text + "\n" + "\n".join(f"- {o.text}" for o in spec.options)
    context = [f"{spec.context_text} → {o.text}" for o in spec.options]
    return target, context


def screen_items() -> None:
    items = pl.concat([pl.read_parquet(p) for p in ITEMS], how="diagonal_relaxed")
    items = items.filter(pl.col("survey_id").is_in(SURVEYS)).unique(
        ["survey_id", "variable"], keep="last", maintain_order=True)
    french = [row for path in FRENCH for row in ds.french_item_rows(items, path)]
    variants = {"en": build_item_specs(items.iter_rows(named=True)), "fr": build_item_specs(french)}
    plan = []
    for lang, specs in variants.items():
        for key, spec in specs.items():
            if spec.options and spec.text:
                target, context = renderings(spec)
                plan.append((key, lang, target, context))
    cache = score_all({t for _, _, target, context in plan for t in (target, *context)})
    rows = [{"survey_id": k[0], "variable": k[1], "language": lang,
             "target_severity": cache[target],
             "context_severity": max(cache[c] for c in context),
             "wording": target.split("\n")[0][:200]}
            for k, lang, target, context in plan]
    frame = pl.DataFrame(rows).sort("survey_id", "variable", "language")
    frame.write_csv(OUT / "item_hate.csv")
    worst = frame.with_columns(pl.max_horizontal("target_severity", "context_severity").alias("max"))
    print(worst.group_by("max").len().sort("max"))


MAX_SEVERITY = 4
CHUNK = 9000


def chunks(text: str) -> list[str]:
    """Pieces of at most CHUNK characters, cut at line breaks where possible."""
    out, current = [], ""
    for line in text.split("\n"):
        while len(line) > CHUNK:
            out.append(line[:CHUNK])
            line = line[CHUNK:]
        if len(current) + len(line) + 1 > CHUNK:
            out.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    return out + ([current] if current else [])


def line_text(line: str) -> str:
    return "\n".join(m["content"] for m in json.loads(line)["messages"])


def filter_files(paths: list[Path]) -> None:
    lines = {p: p.read_text(encoding="utf-8").splitlines() for p in paths}
    pieces = {p: {l: chunks(line_text(l)) for l in ls} for p, ls in lines.items()}
    cache = score_all({c for d in pieces.values() for cs in d.values() for c in cs})
    for path, ls in lines.items():
        kept = [l for l in ls if max(cache[c] for c in pieces[path][l]) < MAX_SEVERITY]
        backup = path.with_name(path.name + ".unscreened")
        if not backup.exists():
            path.rename(backup)
        path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        print(f"{path.name}: kept {len(kept)} / {len(ls)}")


def audit_lines(path: Path, n: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    sample = random.Random(0).sample(lines, min(n, len(lines)))
    texts = {"\n".join(m["content"] for m in json.loads(l)["messages"]) for l in sample}
    cache = score_all(texts)
    values = [cache[t] for t in texts]
    for cut in (2, 4, 6):
        print(f"severity >= {cut}: {sum(v >= cut for v in values)} / {len(values)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--filter":
        filter_files([Path(a) for a in sys.argv[2:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "--lines":
        audit_lines(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    else:
        screen_items()
