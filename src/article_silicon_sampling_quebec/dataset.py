"""Step 3.2 — the training-dataset generator (docs/plan_article.md §2.2, §2.5, §4).

Turns the frozen split plus the shared prompt template into the chat-format
JSONL files Azure AI Foundry ingests:

    {"messages": [{"role": "system", …}, {"role": "user", …},
                  {"role": "assistant", …}]}

Two conditions, C2 having been dropped (§2.2 box):

    C0  persona                                   -> this respondent's answer
    C1  persona + his OWN answers to the k=6      -> this respondent's answer
        nearest items of the same survey

One example is **one real respondent**, and the target is always **his** answer.
Never a cell mean: training on means would collapse the dispersion the product
sells (§2.2).

Why the sampling is stratified and not uniform
----------------------------------------------
Drawing (item, respondent) pairs uniformly would let ``ces_2025`` (20 180
respondents, 290 training targets) and ``ces_2019_online`` swamp
``cecd_charte_2013_10`` (1 000 respondents, 3 targets): the model would mostly
learn the vocabulary of the three big CES waves. Allocation is therefore
equal-share with water-filling down a fixed hierarchy —

    language -> survey -> theme -> item -> respondent

— each level handing back to its siblings whatever it cannot absorb
(:func:`equal_allocation`). Capacity of an item is
``min(eligible respondents, MAX_PAIRS_PER_ITEM)``: without that cap a survey
with three items would answer an equal-share quota by putting the same three
questions in front of six hundred respondents.

`themes` is empty for every row of ``items.parquet`` (the production catalogue
never filled it), so the theme axis is a **measured proxy**: spherical k-means
over the item embeddings that already define the distance axis of §2.3. The
clustering is part of the manifest, deterministic, and only ever used to
balance the sample.

**Then everything is shuffled.** Training the file in thematic order would be
a textbook recipe for catastrophic forgetting: the last theme seen would own
the checkpoint.

Nesting. The 8 000-pair file is the first 8 000 lines of the 20 000-pair file,
and C0 and C1 are rendered from the **same** pairs with the same SES-dropout
draw. So "8 k vs 20 k" measures duration and "C0 vs C1" measures the context
block, neither of them confounded by which respondents were drawn.

Exclusions, all of them checkable on the produced files
------------------------------------------------------
* targets drawn from ``Split.training_items`` only — never a test item, never
  a context-only item;
* respondents drawn outside ``Split.respondents`` — the 30 144 held out;
* a respondent with no valid answer to the target produces no example;
* C1 context never contains the target item, and never an item at cosine
  >= ``CONTEXT_MAX_COSINE`` of it (``prompts.nearest_context_items``);
* C1 context never contains a test item (``Split.context_pool``: a held-out
  item reached as a neighbour would be training data through the back door);
* fixed seed -> byte-identical files.

The sidecar ``pairs.parquet`` carries, per example, the context item keys and
their cosines, so :func:`audit_examples` can verify all of the above **on the
generated data** rather than trusting the code that generated it.

No network: microdata comes from the local ``data/cache/`` parquets.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .corpus.blob import read_survey
from .corpus.ses import MISSING, SesCrosswalk
from .prompts import (
    CONTEXT_MAX_COSINE,
    ContextAnswer,
    ItemSpec,
    Persona,
    PromptTemplate,
    example_rng,
    guess_language,
    has_mojibake,
    nearest_context_items,
    strip_variable_prefix,
)
from .split import Split

ItemKey = tuple[str, str]

_REPO = Path(__file__).resolve().parents[2]
DATASET_DIR = _REPO / "data" / "datasets"

# --------------------------------------------------------------------------
# Frozen generation parameters
# --------------------------------------------------------------------------

#: Seed of the whole generation. Changing it changes every file.
SEED = 20260923

#: Context items per C1 example (§0 decision).
K_CONTEXT = 6

#: The two durations of §2.5. 8 000 brackets the optimum Justin observes
#: (~8 000 examples on 28 items); 20 000 asks whether a corpus two orders of
#: magnitude more diverse pushes the turning point further out. The budget
#: holds to ~13 000 pairs per condition in the nominal regime, so the 20 000
#: file is the one to check the token bill of before launching.
TRAIN_SIZES: tuple[int, ...] = (8_000, 20_000)

#: Validation pairs, shared by both durations of a condition so the two runs
#: are scored on the same set. Drawn from the same stratified pool and
#: disjoint from every training pair.
VALIDATION_SIZE = 500

#: Cap on how many respondents may answer the same target item. 20 500 pairs
#: over 1 573 targets is 13 on average; 40 lets a three-item survey pull its
#: weight without turning its quota into 600 repetitions of one question.
MAX_PAIRS_PER_ITEM = 40

#: Theme proxy — see the module docstring.
N_THEME_CLUSTERS = 12
KMEANS_ITERATIONS = 40

CONDITIONS: tuple[str, ...] = ("C0", "C1")
LANGUAGES: tuple[str, ...] = ("en", "fr")

#: Language each CES respondent actually answered in, and the raw values of
#: it that mean French. The catalogue stores the CES in English only, yet
#: 72-89 % of their Quebec respondents answered in French: a prompt in the
#: language the respondent never read would make the wording — the
#: independent variable — wrong for most of the Quebec cells.
RESPONSE_LANGUAGE: dict[str, tuple[str, frozenset[str]]] = {
    "ces_2019_online": ("cps19_Q_Language", frozenset({"FR-CA"})),
    "ces_2019_phone": ("language_CES", frozenset({"2"})),
    "ces_2021": ("UserLanguage", frozenset({"FR-CA"})),
    "ces_2025": ("cps25_UserLanguage", frozenset({"FR-CA"})),
}

#: French wording of the CES items (scripts/17_extract_ces_french_wording.py).
FRENCH_WORDING_PATH = _REPO / "data" / "ces_french_wording.json"


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def equal_allocation(capacities: Mapping[Any, int], total: int) -> dict[Any, int]:
    """Split `total` as evenly as possible across groups, respecting capacity.

    Water-filling: every group gets the same share until it hits its ceiling,
    and what it cannot absorb is redistributed among the others. Deterministic
    (groups are visited in sorted key order), and it never allocates more than
    a group's capacity nor more than ``sum(capacities)`` in total.
    """
    alloc = {key: 0 for key in capacities}
    remaining = min(int(total), sum(max(0, int(c)) for c in capacities.values()))
    active = sorted(key for key in capacities if capacities[key] > 0)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for key in active:
            if remaining <= 0:
                break
            give = min(share, capacities[key] - alloc[key], remaining)
            if give <= 0:
                continue
            alloc[key] += give
            remaining -= give
            progressed = True
        active = [key for key in active if alloc[key] < capacities[key]]
        if not progressed:
            break
    return alloc


def allocate_pairs(
    items: pl.DataFrame,
    capacity: Mapping[ItemKey, int],
    total: int,
    levels: Sequence[str] = ("language", "survey_id", "theme"),
    key_columns: Sequence[str] = ("survey_id", "variable"),
) -> dict[tuple[str, ...], int]:
    """Quota of pairs per item, balanced down `levels` then across items.

    `items` needs ``survey_id``, ``variable`` and one column per level.
    `capacity` is the number of distinct respondents that item can supply.
    """
    rows = [
        {
            "key": tuple(row[c] for c in key_columns),
            **{lv: str(row[lv]) for lv in levels},
        }
        for row in items.iter_rows(named=True)
    ]
    rows = [r for r in rows if capacity.get(r["key"], 0) > 0]

    def split(group: list[dict[str, Any]], depth: int, budget: int) -> dict[ItemKey, int]:
        if budget <= 0:
            return {}
        if depth == len(levels):
            caps = {r["key"]: capacity[r["key"]] for r in group}
            return {k: v for k, v in equal_allocation(caps, budget).items() if v > 0}
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in group:
            buckets.setdefault(r[levels[depth]], []).append(r)
        caps = {
            name: sum(capacity[r["key"]] for r in members)
            for name, members in buckets.items()
        }
        out: dict[ItemKey, int] = {}
        for name, share in equal_allocation(caps, budget).items():
            out.update(split(buckets[name], depth + 1, share))
        return out

    return split(rows, 0, int(total))


# --------------------------------------------------------------------------
# Theme proxy — spherical k-means over the item embeddings
# --------------------------------------------------------------------------


def spherical_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int,
    iterations: int = KMEANS_ITERATIONS,
) -> np.ndarray:
    """Deterministic k-means++ on the unit sphere; returns one label per row.

    Cosine is the metric the whole design is built on (§2.3), so clusters are
    formed with dot products on already-normalised vectors rather than with
    Euclidean k-means. Pure numpy: no hidden thread-dependent initialisation,
    hence byte-stable labels across runs.
    """
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    n = x.shape[0]
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(seed)

    centres = np.empty((n_clusters, x.shape[1]), dtype=np.float64)
    centres[0] = x[int(rng.integers(n))]
    closest = x @ centres[0]
    for c in range(1, n_clusters):
        weights = np.clip(1.0 - closest, 0.0, None) ** 2
        if weights.sum() <= 0:
            centres[c] = x[int(rng.integers(n))]
        else:
            centres[c] = x[int(rng.choice(n, p=weights / weights.sum()))]
        closest = np.maximum(closest, x @ centres[c])

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        sims = x @ centres.T
        new = sims.argmax(axis=1).astype(np.int32)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(n_clusters):
            members = x[labels == c]
            if members.shape[0] == 0:
                # Empty cluster: reseed on the point worst served so far.
                worst = int(sims.max(axis=1).argmin())
                centres[c] = x[worst]
                continue
            centre = members.mean(axis=0)
            norm = np.linalg.norm(centre)
            centres[c] = centre / norm if norm > 0 else centre
    return labels


def theme_labels(
    keys: Sequence[ItemKey],
    embeddings,
    n_clusters: int = N_THEME_CLUSTERS,
    seed: int = SEED,
) -> dict[ItemKey, str]:
    """Theme cluster of every key, as ``theme_00`` … ``theme_NN``."""
    matrix = np.stack([embeddings.vector(k) for k in keys])
    labels = spherical_kmeans(matrix, n_clusters, seed)
    return {key: f"theme_{int(lab):02d}" for key, lab in zip(keys, labels, strict=True)}


# --------------------------------------------------------------------------
# Items in the respondent's language
# --------------------------------------------------------------------------


#: The write-in line printed after "Autre (spécifier) :" in the French phone
#: questionnaire — layout, not wording.
_FILL_IN_BLANK = re.compile(r"\s*:?\s*_{3,}\s*$")


def french_item_rows(items: pl.DataFrame,
                     path: Path = FRENCH_WORDING_PATH) -> list[dict[str, Any]]:
    """``items.parquet``-shaped rows for the CES items fully available in French.

    Only ``status == "complete"`` entries: the French question **and** a French
    label for every option code of the English item. Anything less and the
    pair is dropped for French respondents rather than shown in English
    (decision of 24 Sept.: the corpus is large enough to be selective). Codes,
    option order, ``code_map`` and year are the English item's — only the words
    change.
    """
    if not path.exists():
        return []
    french = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in items.iter_rows(named=True):
        entry = french.get(row["survey_id"], {}).get(row["variable"])
        if not entry or entry.get("status") != "complete":
            continue
        labels = entry["options_fr"]
        options = json.loads(row["options"]) if isinstance(row["options"], str) else row["options"]
        if any(str(o["code"]) not in labels for o in options):
            continue
        rows.append({
            **row,
            "question_text": entry["question_text_fr"],
            "question_text_source": "questionnaire_fr",
            "display_label": None,
            "language": "fr",
            "options": json.dumps([{"code": o["code"],
                                    "label": _FILL_IN_BLANK.sub("", labels[str(o["code"])])}
                                   for o in options], ensure_ascii=False),
        })
    return rows


class SpecSet(Mapping):
    """Item specs by key, with per-language variants.

    Indexing by key gives the item in its catalogue language, as before;
    :meth:`for_language` gives the version a respondent of that language saw,
    or ``None`` when that version is not available — the caller then drops the
    pair, it never falls back to the other language.
    """

    def __init__(self, base: Mapping[ItemKey, ItemSpec],
                 variants: Mapping[str, Mapping[ItemKey, ItemSpec]] | None = None):
        self.base = dict(base)
        self.variants = {lang: dict(v) for lang, v in (variants or {}).items()}

    def __getitem__(self, key: ItemKey) -> ItemSpec:
        return self.base[key]

    def __iter__(self):
        return iter(self.base)

    def __len__(self) -> int:
        return len(self.base)

    def for_language(self, key: ItemKey, language: str | None) -> ItemSpec | None:
        spec = self.base.get(key)
        if spec is None or language is None or language == spec.language:
            return spec
        return self.variants.get(language, {}).get(key)


def spec_for(specs: Mapping[ItemKey, ItemSpec], key: ItemKey,
             language: str | None) -> ItemSpec | None:
    """Language-aware lookup that also accepts a plain mapping (tests)."""
    if isinstance(specs, SpecSet):
        return specs.for_language(key, language)
    return specs.get(key)


# --------------------------------------------------------------------------
# Microdata access
# --------------------------------------------------------------------------


def normalise_code(value: Any) -> str | None:
    """Raw Parquet values are floats for numeric codes: ``3.0`` -> ``3``.

    Same normalisation as ``scripts/15_prompt_report.py``; the prompt and the
    observed distribution must fold raw codes identically.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _stable_seed(*parts: Any) -> int:
    """Process-independent seed from arbitrary parts.

    ``hash()`` on a str is salted per process (PYTHONHASHSEED), so it cannot
    appear anywhere in a pipeline that must be byte-identical across runs.
    """
    joined = chr(0).join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


class SurveyPanel:
    """One survey's microdata, with the held-out respondents already removed.

    Column-wise on purpose: a CES Parquet is 1 400 columns wide and
    ``DataFrame.row(named=True)`` per lookup would dominate the runtime. Only
    the SES source variables are ever read row-wise, from a narrow projection.
    """

    def __init__(self, survey_id: str, crosswalk: SesCrosswalk, heldout: frozenset[str]):
        self.survey_id = survey_id
        self.crosswalk = crosswalk
        self.raw = read_survey(survey_id)
        self.rids = [str(r) for r in self.raw["__respondent_id"]]
        self.training_rows = np.array(
            [i for i, rid in enumerate(self.rids) if rid not in heldout],
            dtype=np.int64,
        )
        ses_vars: set[str] = set()
        for dim in crosswalk.dimensions_for(survey_id):
            ses_vars |= {
                v for v in crosswalk.source_variables(survey_id, dim)
                if v in self.raw.columns
            }
        self._ses = self.raw.select(sorted(ses_vars)) if ses_vars else None
        self._columns = set(self.raw.columns)
        self._codes: dict[str, list[str | None]] = {}
        self._personas: dict[int, Persona] = {}
        self._languages: list[str | None] | None = None
        spec = RESPONSE_LANGUAGE.get(survey_id)
        if spec and spec[0] in self._columns:
            french = spec[1]
            self._languages = [
                None if (c := normalise_code(v)) is None else ("fr" if c in french else "en")
                for v in self.raw[spec[0]]
            ]

    # -- language --------------------------------------------------------

    def response_language(self, row: int, default: str) -> str:
        """Language this respondent answered in; `default` where the survey
        has a single language (or the value is missing)."""
        if self._languages is None:
            return default
        return self._languages[row] or default

    # -- answers ---------------------------------------------------------

    def codes(self, variable: str) -> list[str | None]:
        cached = self._codes.get(variable)
        if cached is None:
            if variable not in self._columns:
                cached = [None] * len(self.rids)
            else:
                cached = [normalise_code(v) for v in self.raw[variable]]
            self._codes[variable] = cached
        return cached

    def eligible_rows(self, item: ItemSpec,
                      specs: Mapping[ItemKey, ItemSpec] | None = None,
                      language: str | None = None) -> np.ndarray:
        """Training respondents with a valid answer to `item`, as row indices.

        "Valid" means the raw code folds (refusal merge included) onto a
        modality the prompt actually offers. A respondent who did not answer
        the target item never produces an example. With `specs`, a respondent
        whose response language has no complete version of the item is left
        out too (never shown the other language's wording).
        """
        codes = self.codes(item.variable)
        localized: dict[str, ItemSpec | None] = {}

        def version(row: int) -> ItemSpec | None:
            if specs is None:
                return item
            lang = self.response_language(row, item.language)
            if lang not in localized:
                localized[lang] = spec_for(specs, item.key, lang)
            return localized[lang]

        out = []
        for i in self.training_rows:
            if language is not None and self.response_language(int(i), item.language) != language:
                continue
            spec = version(int(i))
            if spec is None:
                continue
            c = spec.canonical_code(codes[i])
            if c is not None and c in {opt.code for opt in spec.options}:
                out.append(i)
        return np.array(out, dtype=np.int64)

    def answer(self, row: int, item: ItemSpec) -> str | None:
        return item.canonical_code(self.codes(item.variable)[row])

    # -- persona ---------------------------------------------------------

    def persona(self, row: int, language: str) -> Persona:
        cached = self._personas.get(row)
        if cached is None:
            lang = "en" if language == "en" else "fr"
            source = self._ses.row(row, named=True) if self._ses is not None else {}
            labels = self.crosswalk.profile_labels(self.survey_id, source, lang=lang)
            labels = {k: v for k, v in labels.items() if v and v != MISSING}
            cached = Persona(fields=labels, survey_id=self.survey_id)
            self._personas[row] = cached
        return cached


# --------------------------------------------------------------------------
# Pairs and examples
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One (target item, respondent) draw — the unit both conditions render."""

    survey_id: str
    variable: str
    respondent_id: str
    row: int
    language: str
    theme: str
    answer_code: str
    context: tuple[tuple[ItemKey, float], ...] = ()

    @property
    def key(self) -> ItemKey:
        return (self.survey_id, self.variable)

    @property
    def example_id(self) -> str:
        return f"{self.survey_id}:{self.variable}:{self.respondent_id}"


@dataclass
class GeneratorConfig:
    seed: int = SEED
    k: int = K_CONTEXT
    train_sizes: tuple[int, ...] = TRAIN_SIZES
    validation_size: int = VALIDATION_SIZE
    max_pairs_per_item: int = MAX_PAIRS_PER_ITEM
    n_theme_clusters: int = N_THEME_CLUSTERS
    conditions: tuple[str, ...] = CONDITIONS
    out_dir: Path = DATASET_DIR
    template_kwargs: dict = field(default_factory=dict)

    @property
    def total_pairs(self) -> int:
        return max(self.train_sizes) + self.validation_size


def sample_pairs(
    quotas: Mapping[ItemKey, int],
    panels: Mapping[str, SurveyPanel],
    specs: Mapping[ItemKey, ItemSpec],
    themes: Mapping[ItemKey, str],
    seed: int,
) -> list[Pair]:
    """Draw the quota of distinct respondents for each item.

    A quota key is ``(survey_id, variable)``, or ``(survey_id, variable,
    language)`` when the allocation unit is the item *in one prompt language*
    (a CES item and its French version are then two units). Deterministic:
    units are visited in sorted order and each unit's draw comes from its own
    seeded generator, so a quota change on one unit does not reshuffle the
    others.
    """
    pairs: list[Pair] = []
    for unit in sorted(quotas):
        quota = quotas[unit]
        if quota <= 0:
            continue
        key, language = tuple(unit[:2]), (unit[2] if len(unit) > 2 else None)
        panel = panels[key[0]]
        item = specs[key]
        pool = panel.eligible_rows(item, specs, language=language)
        if pool.size == 0:
            continue
        rng = np.random.default_rng(_stable_seed(seed, *unit))
        take = min(quota, pool.size)
        chosen = rng.choice(pool, size=take, replace=False)
        for row in sorted(int(r) for r in chosen):
            answer = panel.answer(row, item)
            if answer is None:
                continue
            pairs.append(
                Pair(
                    survey_id=key[0],
                    variable=key[1],
                    respondent_id=panel.rids[row],
                    row=row,
                    language=panel.response_language(row, item.language),
                    theme=themes.get(key, "theme_na"),
                    answer_code=answer,
                )
            )
    return pairs


def context_answers(
    pair: Pair,
    panel: SurveyPanel,
    specs: Mapping[ItemKey, ItemSpec],
    template: PromptTemplate | None = None,
) -> list[ContextAnswer]:
    """This respondent's own answers to the retrieved neighbours, in order.

    Fewer than k is the normal case and is left as is: 163 items of the corpus
    cannot even fill k=6 neighbours below the anti-leak cut, and a respondent
    routinely skipped some of the ones that exist. The spread that produces is
    the §2.3 axis, not a defect to paper over.

    One extra rule, on the *rendering* rather than on the embedding: a context
    item whose rendered wording is byte-identical to the target's, or to a line
    already kept, is dropped. That rule now lives in
    ``PromptTemplate.select_context`` / ``prompts.deduplicate_context``, so the
    inference harness gets it too — it would otherwise come back the day
    stratum context is injected at inference. Applying it here as well only
    keeps ``n_context`` and ``context_used`` describing the block that is
    actually rendered; the operation is idempotent. Measured on the
    20 500-pair draw: 39 pairs over 4 target items hit the target-identical
    case, 1 163 pairs had a duplicated line.
    """
    target = spec_for(specs, pair.key, pair.language)
    out: list[ContextAnswer] = []
    for nkey, _cos in pair.context:
        # In the respondent's language, like the target: a neighbour with no
        # complete version in that language is dropped, not shown translated.
        spec = spec_for(specs, nkey, pair.language)
        if spec is None:
            continue
        code = panel.answer(pair.row, spec)
        label = spec.option_label(code) if code is not None else None
        if label:
            out.append(ContextAnswer(item=spec, code=code, label=label))
    if target is None:
        return out
    template = template or PromptTemplate(condition="C1", k=len(out) or 1)
    return list(template.select_context(target, out))


def render(
    pair: Pair,
    condition: str,
    panel: SurveyPanel,
    specs: Mapping[ItemKey, ItemSpec],
    templates: Mapping[str, PromptTemplate],
    seed: int,
) -> dict[str, Any]:
    """One chat example plus the metadata the audit needs."""
    item = spec_for(specs, pair.key, pair.language)
    if item is None:  # pragma: no cover - sample_pairs never draws such a pair
        raise ValueError(f"no {pair.language} version of {pair.key}")
    persona = panel.persona(pair.row, item.language)
    # One dropout draw per pair, shared by the conditions: the C0/C1 delta is
    # the context block and nothing else (§2.7 dropout, §2.2 contrast).
    dims = templates[condition].keep_dimensions(
        list(persona.fields), example_rng(seed, pair.example_id)
    )
    context = (context_answers(pair, panel, specs, templates[condition])
               if condition != "C0" else [])
    example = templates[condition].build_example(
        persona, item, pair.answer_code, context, dimensions=dims
    )
    return {
        "example": example,
        "n_context": len(context),
        "n_ses_fields": len(dims),
        "context_keys": [list(k) for k, _ in pair.context],
        "context_used": [[c.item.survey_id, c.item.variable] for c in context],
        "context_cosines": [cos for _, cos in pair.context],
    }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_jsonl(path: Path, examples: Iterable[Mapping[str, Any]]) -> int:
    """Write chat examples, one compact JSON object per line. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# Audit — run against the produced files, not against the code above
# --------------------------------------------------------------------------


#: Audit keys that are informational, not violations. A context item whose
#: ``display_label`` summarises the same topic as the target's own label is
#: not a leak: the target is rendered with its verbatim wording, the context
#: line with a generated summary, and a near neighbour below the 0.95 cut is
#: precisely what C1 is for. It is reported because the label vocabulary of
#: ces_2021 is generic enough that the count is worth watching.
#:
#: ``context_line_wrong_language`` is a warning because it rests on the
#: stopword heuristic of :func:`prompts.guess_language`; it is the measure of
#: the display_label defect (30 % of English context lines before the fix)
#: and must stay near zero.
AUDIT_WARNINGS: frozenset[str] = frozenset({
    "context_label_matches_target_label",
    "context_line_wrong_language",
})


#: A target option line in the retired ``N) label`` format.
_CODED_OPTION = re.compile(r"^-?\d+\) ")


def _strip(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def audit_examples(
    examples: Sequence[Mapping[str, Any]],
    meta: pl.DataFrame,
    split: Split,
    specs: Mapping[ItemKey, ItemSpec],
    condition: str,
    max_cosine: float = CONTEXT_MAX_COSINE,
) -> dict[str, Any]:
    """Every exclusion rule of §2.2/§4, checked on the generated data.

    Returns a dict of lists keyed by rule. Every key not in
    :data:`AUDIT_WARNINGS` is a hard violation and must come back empty.
    """
    problems: dict[str, list[Any]] = {
        "target_is_test_item": [],
        "target_is_context_only": [],
        "respondent_held_out": [],
        "context_contains_target": [],
        "context_above_max_cosine": [],
        "context_is_test_item": [],
        "bad_message_shape": [],
        "answer_not_an_option": [],
        "answer_does_not_match_code": [],
        "target_option_has_code": [],
        "context_line_count": [],
        "duplicate_pair": [],
        "context_label_matches_target_label": [],
        "duplicate_context_line": [],
        "variable_prefix_in_wording": [],
        "mojibake": [],
        "context_line_wrong_language": [],
        "prompt_not_in_response_language": [],
    }
    seen: set[tuple[str, str, str]] = set()
    rows = meta.to_dicts()
    if len(rows) != len(examples):
        problems["bad_message_shape"].append("meta/jsonl length mismatch")
        return problems

    for example, row in zip(examples, rows, strict=True):
        key = (row["survey_id"], row["variable"])
        ident = (row["survey_id"], row["variable"], row["respondent_id"])
        if ident in seen:
            problems["duplicate_pair"].append(ident)
        seen.add(ident)

        if split.is_test_item(key):
            problems["target_is_test_item"].append(key)
        if specs[key].key != key:  # pragma: no cover - defensive
            problems["bad_message_shape"].append(key)
        if split.is_test_respondent(row["survey_id"], row["respondent_id"]):
            problems["respondent_held_out"].append(ident)

        messages = example.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [m["role"] for m in messages] != ["system", "user", "assistant"]
            or not all(isinstance(m.get("content"), str) and m["content"] for m in messages)
        ):
            problems["bad_message_shape"].append(ident)
            continue

        language = row.get("language")
        item = spec_for(specs, key, language)
        if item is None:
            problems["prompt_not_in_response_language"].append((ident, language))
            continue
        # The prompt speaks the language the respondent answered in (CES
        # French respondents included), never the catalogue's by default.
        if messages[0]["content"].startswith("Tu es") != (item.language == "fr") or \
                (language is not None and item.language != language):
            problems["prompt_not_in_response_language"].append((ident, language))
        # The assistant turn is an option's rendered text, listed verbatim in
        # the prompt, and it parses back to the recorded answer code.
        answer = messages[2]["content"]
        if answer not in {opt.text for opt in item.options}:
            problems["answer_not_an_option"].append(ident)
        if item.match_answer(answer) != str(row["answer_code"]):
            problems["answer_does_not_match_code"].append(ident)
        option_lines = (messages[1]["content"].rsplit("\nOptions :\n", 1)[-1]
                        .split("\n\n")[0].split("\n"))
        if any(_CODED_OPTION.match(l) for l in option_lines) or \
                option_lines != [opt.render() for opt in item.options]:
            problems["target_option_has_code"].append(ident)

        used = ([] if condition == "C0"
                else [tuple(k) for k in json.loads(row["context_used"])])
        cosines = dict(
            zip(
                [tuple(k) for k in json.loads(row["context_keys"])],
                json.loads(row["context_cosines"]),
                strict=True,
            )
        )
        for nkey in used:
            if nkey == key:
                problems["context_contains_target"].append((ident, nkey))
            if cosines.get(nkey, 0.0) >= max_cosine:
                problems["context_above_max_cosine"].append((ident, nkey,
                                                             cosines.get(nkey)))
            if split.is_test_item(nkey):
                problems["context_is_test_item"].append((ident, nkey))

        # The rendered block must carry exactly the context items claimed, one
        # line each, and the target's own wording must never appear among them.
        user = messages[1]["content"]
        block = user.split("\n\n")[0] if condition != "C0" else ""
        lines = [l for l in block.split("\n") if l.startswith("- ")] if used else []
        if len(lines) != len(used):
            problems["context_line_count"].append((ident, len(used), len(lines)))
        if len(set(lines)) != len(lines):
            problems["duplicate_context_line"].append(ident)
        target_label = _strip(item.context_text)
        for nkey in used:
            nspec = spec_for(specs, nkey, language)
            if nspec is None:
                problems["prompt_not_in_response_language"].append((ident, nkey))
                continue
            if target_label and _strip(nspec.context_text) == target_label:
                problems["context_label_matches_target_label"].append((ident, nkey))

        # Wording hygiene, read back from the rendered text: no column name in
        # front of a wording, no mis-encoded sequence, context lines in the
        # language of the prompt.
        question = next((l for l in user.split("\n") if l.startswith("Question : ")), "")
        question = question[len("Question : "):]
        if strip_variable_prefix(question, key[1]) != question:
            problems["variable_prefix_in_wording"].append((ident, key))
        for nkey, line in zip(used, lines):
            wording = line[2:].rsplit(" → ", 1)[0]
            if strip_variable_prefix(wording, nkey[1]) != wording:
                problems["variable_prefix_in_wording"].append((ident, nkey))
            guessed = guess_language(wording)
            if guessed is not None and guessed != item.language:
                problems["context_line_wrong_language"].append((ident, nkey))
        if any(has_mojibake(m["content"]) for m in messages):
            problems["mojibake"].append(ident)
    return problems
