"""Step 1.5 — build the item-to-item similarity index.

    uv run python scripts/12_build_similarity_index.py            # build + diagnose
    uv run python scripts/12_build_similarity_index.py --diagnose # diagnose only
    uv run python scripts/12_build_similarity_index.py --dry-run  # cost estimate

Item source, in order of preference:
  1. data/items.parquet                 (perimeter table, if another step built it)
  2. Azure AI Search `survey-questions`  (the catalogue rail, §3.1)
  3. ../mvp_moteur_recherche_sondages/ingestion/normalized/*.json

Embeddings: the index stores `content_vector` (text-embedding-3-large, 3072d)
but the field is `hidden=True` and therefore NOT retrievable, so we recompute
with the same Azure OpenAI deployment and the same embed-text recipe as the
production ingestion. Results are cached on disk keyed by (deployment, text)
so nothing is ever paid for twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from article_silicon_sampling_quebec.corpus.perimeter import (  # noqa: E402
    EXCLUDED_SURVEYS,
    NORMALIZED_DIR,
    TARGET_VAR_TYPES,
)
from article_silicon_sampling_quebec.corpus.similarity import (  # noqa: E402
    DEFAULT_K,
    build_embed_text,
    build_similarity_frame,
    load_embeddings,
    load_index,
)

ITEMS_PATH = REPO_ROOT / "data" / "items.parquet"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "item_embeddings.parquet"
SIMILARITY_PATH = REPO_ROOT / "data" / "item_similarity.parquet"
CACHE_PATH = REPO_ROOT / "data" / ".cache" / "embedding_cache.parquet"

INDEX_NAME = "survey-questions"
AOAI_API_VERSION = "2024-02-01"
BATCH_SIZE = 100
# text-embedding-3-large list price, USD per 1M input tokens.
PRICE_PER_MTOK = 0.13


# --------------------------------------------------------------------------
# Item sources
# --------------------------------------------------------------------------


def _json_list(value) -> list:
    """`options`/`themes`/`concepts` may arrive as JSON strings or real lists."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return list(value)


def _from_items_parquet() -> pl.DataFrame | None:
    if not ITEMS_PATH.exists():
        return None
    frame = pl.read_parquet(ITEMS_PATH)
    if not {"survey_id", "variable", "question_text"} <= set(frame.columns):
        print(f"[items] {ITEMS_PATH.name} lacks required columns, skipping")
        return None
    print(f"[items] source: {ITEMS_PATH} ({frame.height} rows)")

    rows = []
    for row in frame.iter_rows(named=True):
        options = _json_list(row.get("options")) or _json_list(row.get("response_options"))
        rows.append(
            {
                "survey_id": row["survey_id"],
                "variable": row["variable"],
                "question_text": row.get("question_text") or "",
                "display_label": row.get("display_label"),
                "option_labels": [
                    o.get("label")
                    for o in options
                    if isinstance(o, dict) and o.get("label")
                ],
                "concepts": [c for c in _json_list(row.get("concepts")) if isinstance(c, str)],
                "themes": [t for t in _json_list(row.get("themes")) if isinstance(t, str)],
                "var_type": row.get("var_type"),
                "survey_year": row.get("year") or row.get("survey_year"),
                "language": row.get("language"),
            }
        )
    return _normalize_items(pl.DataFrame(rows))


def _from_search_index() -> pl.DataFrame | None:
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ImportError:
        return None
    endpoint = os.environ.get("SEARCH_ENDPOINT")
    key = os.environ.get("SEARCH_QUERY_KEY") or os.environ.get("SEARCH_ADMIN_KEY")
    if not endpoint or not key:
        return None

    client = SearchClient(endpoint, INDEX_NAME, AzureKeyCredential(key))
    var_filter = " or ".join(f"var_type eq '{t}'" for t in sorted(TARGET_VAR_TYPES))
    results = client.search(
        search_text="*",
        filter=f"doc_type eq 'question' and is_sociodemo eq false and ({var_filter})",
        select=[
            "survey_id",
            "variable",
            "question_text",
            "display_label",
            "response_options",
            "concepts",
            "themes",
            "var_type",
            "survey_year",
            "language",
        ],
        top=100_000,
    )
    rows = [
        {
            "survey_id": d["survey_id"],
            "variable": d["variable"],
            "question_text": d.get("question_text") or "",
            "display_label": d.get("display_label"),
            "option_labels": [
                o.get("label") for o in (d.get("response_options") or []) if o.get("label")
            ],
            "concepts": list(d.get("concepts") or []),
            "themes": list(d.get("themes") or []),
            "var_type": d.get("var_type"),
            "survey_year": d.get("survey_year"),
            "language": d.get("language"),
        }
        for d in results
        if d["survey_id"] not in EXCLUDED_SURVEYS
    ]
    if not rows:
        return None
    print(f"[items] source: AI Search index `{INDEX_NAME}` ({len(rows)} rows)")
    return _normalize_items(pl.DataFrame(rows))


def _from_normalized_json() -> pl.DataFrame:
    directory = (REPO_ROOT / NORMALIZED_DIR).resolve()
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        survey_id = payload["survey"]["survey_id"]
        if survey_id in EXCLUDED_SURVEYS:
            continue
        for q in payload.get("questions", []):
            if q.get("var_type") not in TARGET_VAR_TYPES or q.get("is_sociodemo"):
                continue
            rows.append(
                {
                    "survey_id": survey_id,
                    "variable": q["variable"],
                    "question_text": q.get("question_text") or "",
                    "display_label": q.get("display_label"),
                    "option_labels": [
                        o.get("label")
                        for o in (q.get("response_options") or [])
                        if o.get("label")
                    ],
                    "concepts": list(q.get("concepts") or []),
                    "themes": list(q.get("themes") or []),
                    "var_type": q.get("var_type"),
                    "survey_year": payload["survey"].get("year"),
                    "language": payload["survey"].get("language"),
                }
            )
    print(f"[items] source: normalized JSON ({len(rows)} rows)")
    return _normalize_items(pl.DataFrame(rows))


def _normalize_items(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the perimeter, fill optional columns, dedupe, sort."""
    for name, dtype in (
        ("display_label", pl.String),
        ("var_type", pl.String),
        ("language", pl.String),
    ):
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    for name in ("option_labels", "concepts", "themes"):
        if name not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None).cast(pl.List(pl.String)).alias(name)
            )
    if "survey_year" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int32).alias("survey_year"))

    frame = frame.filter(~pl.col("survey_id").is_in(list(EXCLUDED_SURVEYS)))
    if "var_type" in frame.columns and frame["var_type"].null_count() < frame.height:
        frame = frame.filter(
            pl.col("var_type").is_in(list(TARGET_VAR_TYPES)) | pl.col("var_type").is_null()
        )
    if "is_sociodemo" in frame.columns:
        frame = frame.filter(~pl.col("is_sociodemo").fill_null(False))

    return (
        frame.filter(pl.col("question_text").fill_null("").str.strip_chars() != "")
        .unique(subset=["survey_id", "variable"], keep="first")
        .sort(["survey_id", "variable"])
    )


def _catalogue_enrichment() -> dict[tuple[str, str], dict] | None:
    """`display_label` / `concepts` per item, straight from the search index.

    These two fields are authored at ingestion (LLM enrichment) and live only in
    the index — the normalized JSON does not have them. They matter more than
    they look: `question_text` is capped at 80 characters for the Stata-sourced
    surveys, so the sub-items of a matrix battery share a byte-identical stem and
    `display_label` is the ONLY thing telling them apart. Production embeds it;
    so do we.
    """
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ImportError:
        return None
    endpoint = os.environ.get("SEARCH_ENDPOINT")
    key = os.environ.get("SEARCH_QUERY_KEY") or os.environ.get("SEARCH_ADMIN_KEY")
    if not endpoint or not key:
        return None
    try:
        client = SearchClient(endpoint, INDEX_NAME, AzureKeyCredential(key))
        results = client.search(
            search_text="*",
            filter="doc_type eq 'question'",
            select=["survey_id", "variable", "display_label", "concepts"],
            top=100_000,
        )
        return {
            (d["survey_id"], d["variable"]): {
                "display_label": d.get("display_label"),
                "concepts": list(d.get("concepts") or []),
            }
            for d in results
        }
    except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
        print(f"[items] catalogue enrichment unavailable: {type(exc).__name__}")
        return None


def _enrich(items: pl.DataFrame) -> pl.DataFrame:
    """Fill missing display_label / concepts from the catalogue index."""
    if items["display_label"].null_count() == 0:
        return items
    catalogue = _catalogue_enrichment()
    if not catalogue:
        print("[items] WARNING: no display_label — battery sub-items will collide")
        return items

    labels, concepts, hits = [], [], 0
    for row in items.iter_rows(named=True):
        extra = catalogue.get((row["survey_id"], row["variable"]), {})
        label = row["display_label"] or extra.get("display_label")
        labels.append(label)
        concepts.append(row["concepts"] or extra.get("concepts") or [])
        hits += bool(label)
    print(f"[items] display_label resolved for {hits}/{items.height} items")
    return items.with_columns(
        pl.Series("display_label", labels, dtype=pl.String),
        pl.Series("concepts", concepts, dtype=pl.List(pl.String)),
    )


def load_items() -> pl.DataFrame:
    for source in (_from_items_parquet, _from_search_index):
        frame = source()
        if frame is not None and frame.height > 0:
            return _enrich(frame)
    return _enrich(_from_normalized_json())


# --------------------------------------------------------------------------
# Embeddings with on-disk cache
# --------------------------------------------------------------------------


def _text_hash(deployment: str, text: str) -> str:
    return hashlib.sha256(f"{deployment}\x00{text}".encode()).hexdigest()


def _read_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        return {}
    frame = pl.read_parquet(CACHE_PATH)
    return dict(zip(frame["text_hash"], frame["embedding"].to_list(), strict=True))


def _write_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"text_hash": list(cache), "embedding": [cache[h] for h in cache]}
    ).write_parquet(CACHE_PATH)


def embed_texts(texts: Sequence[str], dry_run: bool = False) -> np.ndarray:
    """Embed every text, hitting Azure only for cache misses."""
    deployment = os.environ["AOAI_EMBED_DEPLOYMENT"]
    cache = _read_cache()
    hashes = [_text_hash(deployment, t) for t in texts]
    missing = sorted({h for h in hashes if h not in cache})

    # Rough token count: text-embedding-3-large is ~4 chars/token on this corpus.
    missing_texts = {h: t for h, t in zip(hashes, texts, strict=True) if h in set(missing)}
    approx_tokens = sum(len(t) for t in missing_texts.values()) / 4
    print(
        f"[embed] {len(texts)} items | {len(set(hashes))} distinct texts | "
        f"to compute {len(missing)} (~{approx_tokens:,.0f} tokens, "
        f"~${approx_tokens / 1e6 * PRICE_PER_MTOK:.4f})"
    )
    if dry_run:
        raise SystemExit(0)

    if missing:
        endpoint = os.environ["AOAI_ENDPOINT"].rstrip("/")
        url = (
            f"{endpoint}/openai/deployments/{deployment}"
            f"/embeddings?api-version={AOAI_API_VERSION}"
        )
        api_key = os.environ["AOAI_KEY"]
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            payload = [missing_texts[h] for h in batch]
            vectors = _embed_batch(url, api_key, payload)
            for h, vec in zip(batch, vectors, strict=True):
                cache[h] = vec
            # Checkpoint every batch: a mid-run failure never costs a redo.
            _write_cache(cache)
            print(f"[embed]   {min(start + BATCH_SIZE, len(missing))}/{len(missing)}")

    return np.asarray([cache[h] for h in hashes], dtype=np.float32)


def _embed_batch(url: str, api_key: str, texts: list[str]) -> list[list[float]]:
    """One REST call, with exponential backoff on throttling / transient errors.

    Uses urllib rather than the `openai` SDK to avoid touching the shared
    pyproject dependency list. The key only ever travels in a request header.
    """
    import urllib.error
    import urllib.request

    body = json.dumps({"input": texts}).encode("utf-8")
    delay = 2.0
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read())
            rows = sorted(data["data"], key=lambda d: d["index"])
            return [r["embedding"] for r in rows]
        except urllib.error.HTTPError as exc:
            # 429 = throttling, 5xx = transient. Anything else is a real bug.
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(
                    f"Azure OpenAI embeddings failed: HTTP {exc.code} {exc.reason}"
                ) from None
            wait = float(exc.headers.get("Retry-After") or delay)
            print(f"[embed]   HTTP {exc.code}, retry in {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 5:
                raise RuntimeError(f"Azure OpenAI embeddings unreachable: {exc}") from None
            print(f"[embed]   connection error, retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build(k: int, dry_run: bool) -> None:
    items = load_items()
    print(f"[items] perimeter: {items.height} items, {items['survey_id'].n_unique()} surveys")

    texts = [
        build_embed_text(row["question_text"], row["display_label"], row["option_labels"], row["concepts"])
        for row in items.iter_rows(named=True)
    ]
    vectors = embed_texts(texts, dry_run=dry_run)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # display_label is carried along: for the 80-char-truncated items it is the
    # only human-readable way to tell two sub-items of a battery apart.
    items.select("survey_id", "variable", "question_text", "display_label").with_columns(
        pl.Series("embedding", vectors.tolist(), dtype=pl.List(pl.Float32))
    ).write_parquet(EMBEDDINGS_PATH)
    print(f"[write] {EMBEDDINGS_PATH} ({items.height} x {vectors.shape[1]})")

    frame = build_similarity_frame(items, vectors, k=k)
    frame.write_parquet(SIMILARITY_PATH)
    print(f"[write] {SIMILARITY_PATH} ({frame.height} pairs, k={k})")


# --------------------------------------------------------------------------
# Diagnostics (§2.3 — the intellectual deliverable of this step)
# --------------------------------------------------------------------------

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _describe(values: np.ndarray, label: str) -> None:
    print(f"\n{label}  (n={values.size})")
    print(f"  mean {values.mean():.4f}   sd {values.std():.4f}")
    qs = np.quantile(values, QUANTILES)
    print("  " + "  ".join(f"p{int(q * 100):02d}={v:.4f}" for q, v in zip(QUANTILES, qs, strict=True)))


def diagnose(sample_size: int = 20, seed: int = 20260921) -> None:
    emb = load_embeddings(EMBEDDINGS_PATH)
    index = load_index(SIMILARITY_PATH)
    frame = index.frame

    raw = pl.read_parquet(EMBEDDINGS_PATH).drop("embedding")
    text_of = {}
    for row in raw.iter_rows(named=True):
        text = row["question_text"]
        if row.get("display_label"):
            text = f"{text}  ⟨{row['display_label']}⟩"
        text_of[(row["survey_id"], row["variable"])] = text

    print("=" * 78)
    print(f"DIAGNOSTIC — {len(emb)} items, k={index.k}")
    print("=" * 78)

    nn_all = frame.filter(pl.col("rank") == 1)
    all_vals = nn_all["cosine"].to_numpy()
    _describe(all_vals, "1-NN cosine, all neighbours")

    cross = (
        frame.filter(~pl.col("same_survey"))
        .sort("cosine", descending=True)
        .group_by(["survey_id", "variable"], maintain_order=True)
        .first()
    )
    cross_vals = cross["cosine"].to_numpy()
    _describe(cross_vals, "1-NN cosine, cross-survey only")

    print("\nShare of items above a cosine threshold")
    print(f"  {'thr':>6} {'all':>10} {'cross-survey':>16}")
    for thr in (0.99, 0.98, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60):
        a = float((all_vals >= thr).mean())
        c = float((cross_vals >= thr).mean()) if cross_vals.size else float("nan")
        print(f"  {thr:>6.2f} {a:>9.1%} {c:>15.1%}")

    # Isolation, the other end of the axis: an item whose best match anywhere is
    # weak has, by construction, no context to retrieve in C2.
    print("\nIsolated items (low 1-NN cosine)")
    print(f"  {'thr':>6} {'all':>10} {'cross-survey':>16}")
    for thr in (0.60, 0.65, 0.70, 0.75, 0.80):
        a = int((all_vals < thr).sum())
        c = int((cross_vals < thr).sum())
        print(f"  <{thr:>5.2f} {a:>6} ({a / all_vals.size:>5.1%}) "
              f"{c:>8} ({c / max(cross_vals.size, 1):>5.1%})")

    print("\nNeighbour composition of the top-1")
    print(f"  same survey : {nn_all['same_survey'].mean():.1%}")
    print(f"  items with no cross-survey neighbour in the index: "
          f"{len(emb) - cross.height}")

    by_survey = (
        nn_all.group_by("survey_id")
        .agg(
            pl.len().alias("n_items"),
            pl.col("cosine").mean().alias("mean_1nn"),
            pl.col("same_survey").mean().alias("share_same_survey"),
        )
        .sort("survey_id")
    )
    print("\nPer survey")
    print(f"  {'survey_id':<26} {'n':>5} {'mean 1-NN':>10} {'same-survey 1-NN':>18}")
    for r in by_survey.iter_rows(named=True):
        print(
            f"  {r['survey_id']:<26} {r['n_items']:>5} "
            f"{r['mean_1nn']:>10.4f} {r['share_same_survey']:>17.1%}"
        )

    _sample_pairs(frame, text_of, sample_size, seed)


def _sample_pairs(
    frame: pl.DataFrame,
    text_of: dict,
    sample_size: int,
    seed: int,
) -> None:
    """Pairs drawn across the whole similarity range, for human adjudication."""
    rng = np.random.default_rng(seed)
    pairs = frame.filter(pl.col("rank") <= 5)
    cosines = pairs["cosine"].to_numpy()

    bands = [(0.98, 1.01), (0.93, 0.98), (0.88, 0.93), (0.82, 0.88), (0.75, 0.82), (0.0, 0.75)]
    per_band = max(1, sample_size // len(bands))

    print("\n" + "=" * 78)
    print("SAMPLE PAIRS — human check of the metric")
    print("=" * 78)
    for lo, hi in bands:
        mask = np.flatnonzero((cosines >= lo) & (cosines < hi))
        if mask.size == 0:
            print(f"\n--- cosine [{lo:.2f}, {hi:.2f}) : empty")
            continue
        take = rng.choice(mask, size=min(per_band, mask.size), replace=False)
        print(f"\n--- cosine [{lo:.2f}, {hi:.2f})  ({mask.size} candidate pairs)")
        for i in take:
            row = pairs.row(int(i), named=True)
            a = (row["survey_id"], row["variable"])
            b = (row["neighbor_survey_id"], row["neighbor_variable"])
            tag = "SAME SURVEY" if row["same_survey"] else "cross"
            print(f"\n  cos={row['cosine']:.4f}  [{tag}]")
            print(f"    A {a[0]}/{a[1]}: {_clip(text_of.get(a, ''))}")
            print(f"    B {b[0]}/{b[1]}: {_clip(text_of.get(b, ''))}")


def _clip(text: str, width: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--diagnose", action="store_true", help="skip the build")
    parser.add_argument("--dry-run", action="store_true", help="cost estimate only")
    parser.add_argument("--sample-size", type=int, default=24)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not args.diagnose:
        build(k=args.k, dry_run=args.dry_run)
    diagnose(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
