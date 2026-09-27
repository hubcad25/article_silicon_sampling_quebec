"""Quick sanity check of a fine-tuned deployment on validation prompts (not the test split).

1. repeated draws: 6 prompts x 20 draws at T=1.0 — valid-code rate, spread;
2. one draw on 100 distinct validation examples — hit rate vs the respondent's
   answer, against a per-item modal baseline built from the training file.

usage: python scratch/quick_eval.py <deployment> [train_jsonl]
"""
import collections, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from article_silicon_sampling_quebec.foundry import FoundryChat

load_dotenv(".env")
DEP = sys.argv[1]
TRAIN = sys.argv[2] if len(sys.argv) > 2 else "data/datasets/c0_train_8000.jsonl"
CHAT = FoundryChat(DEP)


def ask(messages, temp=1.0):
    return CHAT.complete(messages, temperature=temp)


def question(msgs):
    return re.search(r"^Question : (.*)$", msgs[1]["content"], re.M).group(1)


def codes(msgs):
    return re.findall(r"^(-?\d+)\) ", msgs[1]["content"], re.M)


val = [json.loads(l)["messages"] for l in open("data/datasets/c0_validation.jsonl")]
modal = collections.defaultdict(collections.Counter)
for line in open(TRAIN):
    m = json.loads(line)["messages"]
    modal[question(m)][m[2]["content"]] += 1

rng = random.Random(0)
t0 = time.time()
with ThreadPoolExecutor(3) as pool:
    print("== 1. repeated draws, T=1.0")
    for m in rng.sample(val, 6):
        outs = list(pool.map(lambda _: ask(m[:2]), range(20)))
        valid = [o for o in outs if o in codes(m)]
        dist = " ".join(f"{k}:{v}" for k, v in collections.Counter(outs).most_common(6))
        print(f"  [{question(m)[:70]}]")
        print(f"     valid {len(valid)}/20 | truth={m[2]['content']} | {dist}")

    print("\n== 2. one draw on 100 validation examples, T=1.0")
    sample = rng.sample(val, 100)
    outs = list(pool.map(lambda m: ask(m[:2]), sample))
    greedy = list(pool.map(lambda m: ask(m[:2], 0.0), sample))
json.dump([{"q": question(m), "truth": m[2]["content"], "t1": o, "t0": g}
           for m, o, g in zip(sample, outs, greedy)],
          open(f"logs/quick_eval_{DEP}.json", "w"), ensure_ascii=False, indent=0)
failed = lambda o: False  # FoundryChat retries or raises: every draw is an answer
print(f"  service pushback: {CHAT.throttled} x 429, {CHAT.retries} retries (all recovered)")
for name, answers in (("T=1.0", outs), ("T=0.0", greedy)):
    rows = [(m, o) for m, o in zip(sample, answers) if not failed(o) and modal[question(m)]]
    n = len(rows)
    if not n:
        continue
    valid = sum(o in codes(m) for m, o in rows)
    hit = sum(o == m[2]["content"] for m, o in rows)
    marg = sum(modal[question(m)][m[2]["content"]] / sum(modal[question(m)].values()) for m, _ in rows)
    top = sum(modal[question(m)].most_common(1)[0][0] == m[2]["content"] for m, _ in rows)
    unif = sum(1 / max(len(codes(m)), 1) for m, _ in rows)
    print(f"  {name}  n={n}  valid {100 * valid / n:.0f}%  hit {100 * hit / n:.0f}%  | "
          f"marginal draw {100 * marg / n:.0f}%  modal {100 * top / n:.0f}%  uniform {100 * unif / n:.0f}%")
print(f"\n{time.time() - t0:.0f}s")
