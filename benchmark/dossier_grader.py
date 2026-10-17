"""Python grader for reinforcement fine-tuning of the dossier task (C by reward).

Azure OpenAI RFT calls ``grade(sample, item)`` for each generated answer:
``sample["output_text"]`` is the model's reply (the JSON of benchmark/dossier.py),
``item`` the dataset line (data/datasets_dossier/dossier_rft_*.jsonl) with
``target`` (cell -> code -> share, the real distribution) and ``codes``.

Score = 1 - mean over the 15 cells of the total variation distance between the
reply and the target: 1 is perfect, 0 for unparseable JSON or a missing cell.
Shares in a cell are clipped at 0 and renormalized, so only the shape counts;
a cell whose shares sum to 0 counts as maximally wrong (TV 1).

Pure Python, no imports beyond json/re: it must run in the grader sandbox.
``grader_definition()`` gives the JSON object for ``method.reinforcement.grader``.

    .venv/bin/python -m benchmark.dossier_grader     # self-test on SFT targets
"""

import json
import re

CELLS = [f"{r}|{a}" for r in ("BC", "Prairies", "ON", "QC", "Atlantic")
         for a in ("18-34", "35-54", "55+")]


def _parse(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None


def grade(sample, item):
    data = sample.get("output_json") if isinstance(sample.get("output_json"), dict) else None
    if data is None:
        data = _parse(sample.get("output_text", ""))
    if not isinstance(data, dict) or not isinstance(data.get("cells"), dict):
        return 0.0
    codes = [str(c) for c in item["codes"]]
    total = 0.0
    for cell in CELLS:
        reply, target = data["cells"].get(cell), item["target"][cell]
        if not isinstance(reply, dict):
            return 0.0
        p = []
        for c in codes:
            try:
                p.append(max(float(reply.get(c, 0) or 0), 0.0))
            except (TypeError, ValueError):
                p.append(0.0)
        s = sum(p)
        if s <= 0:
            total += 1.0
            continue
        total += 0.5 * sum(abs(x / s - float(target[c])) for x, c in zip(p, codes))
    return max(0.0, 1.0 - total / len(CELLS))


def grader_definition() -> dict:
    """The ``method.reinforcement.grader`` object: this file's source, run as is."""
    with open(__file__, encoding="utf-8") as handle:
        source = handle.read()
    source = source.split("\n\ndef grader_definition")[0] + "\n"
    return {"type": "python", "name": "dossier_tv", "source": source}


if __name__ == "__main__":
    import random
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "datasets_dossier" / "dossier_rft_validation.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:20]]
    perfect = [grade({"output_text": json.dumps({"cells": it["target"]})}, it) for it in items]
    uniform = [grade({"output_text": json.dumps({"cells": {c: {k: 1 for k in it["codes"]} for c in CELLS}})}, it)
               for it in items]
    rng = random.Random(0)
    shuffled = [grade({"output_text": json.dumps({"cells": {c: dict(zip(it["codes"], rng.sample(
        list(it["target"][c].values()), len(it["codes"])))) for c in CELLS}})}, it) for it in items]
    broken = [grade({"output_text": "not json"}, items[0]), grade({"output_text": "{\"cells\": {}}"}, items[0])]
    print(f"perfect  min {min(perfect):.4f}  mean {sum(perfect) / len(perfect):.4f}")
    print(f"uniform  mean {sum(uniform) / len(uniform):.4f}")
    print(f"shuffled mean {sum(shuffled) / len(shuffled):.4f}")
    print(f"broken   {broken}")
    exec(compile(grader_definition()["source"], "grader", "exec"), ns := {})
    print("sandbox source ok:", ns["grade"]({"output_text": json.dumps({"cells": items[0]["target"]})}, items[0]))
