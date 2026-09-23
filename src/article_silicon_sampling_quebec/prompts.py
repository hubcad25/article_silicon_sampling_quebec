"""Single prompt template shared by every arm of the experiment — step 3.1.

Design constraint (`docs/plan_article.md` §6.3, non-negotiable): the non
fine-tuned roleplay arm and the fine-tuned arms must see the **exact same
template**, so that no prompt difference confounds the direct/indirect
supervision contrast. Nothing here may therefore depend on a model-specific
detail: no chat template, no special token, no system-prompt trick. This module
emits plain OpenAI-style ``messages`` and stops there.

What the template renders
-------------------------
``system``  the persona — canonical SES dimensions (`corpus.ses`) plus an
            explicit ``Population: Québec / Canada`` field (§3.3, neutralises
            the language x population confound).
``user``    the optional context block (C1 or C2) + the target item, verbatim,
            with its options.
``assistant`` the **code** of the chosen modality (training target only).

Refusal merge. The modalities rendered here are the ones of
``items.parquet``, where ``corpus.perimeter.merge_refusal_options`` has already
collapsed refusal / "prefer not to answer" / NA / skipped into a single code —
they are instrument behaviour, not opinion. "Don't know" is *not* merged: it is
a real state of opinion and a valid target (plan §0). Raw microdata codes must
therefore be passed through :meth:`ItemSpec.canonical_code` before they are
used as an answer, as a C1 context answer, or as a bin of a C2 / validation
distribution — otherwise the prompt offers one partition of the item and the
observed distribution reports another.

Three conditions (§2.2), one retrieval policy
---------------------------------------------
============ ==================================================================
``C0``       persona + target item
``C1``       persona + the same respondent's real answers to the k nearest items
``C2``       persona + the distributions observed *in the respondent's stratum*
             on the k nearest items
============ ==================================================================

Retrieval is identical at training and at inference time: **the k nearest
available neighbours below the anti-leak cut** (`nearest_context_items`). The
spread of context quality comes for free from the corpus — some items have a
0.94 neighbour, others a 0.60 one — which is exactly the axis §2.3 sweeps.

Leak protection
---------------
``build_messages`` raises if a context item is the target item itself. In C1
that would hand the model the answer; in C2 the target's own stratum
distribution. Both are silent, plausible, and fatal — hence loud here.

The subtler leak is a *different* item that asks the same thing.
``cps21_votechoice`` and ``cps21_v_advance`` sit at cosine 0.967 in the same
survey, over the same respondents: with one of them held out, the other is a
legal context item and C1 hands the model the answer verbatim. Hence the
second rule, enforced in the retrieval policy rather than left to the caller:
**no item at cosine >= CONTEXT_MAX_COSINE (0.95) of the target may ever be a
context item**, in C1 or C2, at training or at inference. It costs a few
neighbours and removes a whole class of silent inflation.

No network access, no I/O: callers pass already-loaded rows.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Option",
    "ItemSpec",
    "Persona",
    "ContextAnswer",
    "ContextDistribution",
    "PromptTemplate",
    "CONDITIONS",
    "PERSONA_DIMENSIONS",
    "NATIONAL_SURVEYS",
    "TRUNCATION_LENGTHS",
    "CONTEXT_MAX_COSINE",
    "item_text",
    "parse_options",
    "parse_code_map",
    "nearest_context_items",
    "example_rng",
]

CONDITIONS = ("C0", "C1", "C2")

#: Anti-leak cut on the retrieval policy: an item at or above this cosine to
#: the target is a quasi-duplicate of it, not a context for it. Same value as
#: the ``quasi_duplicate`` bin boundary of ``split.DISTANCE_BINS``, and kept in
#: sync with ``split.CONTEXT_MAX_COSINE`` (duplicated so this module keeps no
#: dependency on the split).
CONTEXT_MAX_COSINE = 0.95

#: Canonical SES dimensions, in the fixed render order. Field order is never a
#: nuisance variable: it is the same in every example, whatever the subset kept.
PERSONA_DIMENSIONS: tuple[str, ...] = (
    "age", "gender", "education", "region_qc", "region", "income", "language",
)

#: Surveys whose sample is national rather than Quebec-only (mirrors
#: ``corpus.strata.NATIONAL_SURVEYS``; duplicated so this module stays free of
#: any data dependency).
NATIONAL_SURVEYS: frozenset[str] = frozenset({
    "cecd_elxn_can_2011", "cecd_sante_can_usa",
    "ces_2019_online", "ces_2019_phone", "ces_2021", "ces_2025",
})

#: Stata caps a variable label at 80 characters, so ``question_text`` is
#: stored truncated wherever the catalogue read it from a ``.dta``: a spike of
#: 381 items at exactly 80, and a second one at 79 (69 items where ~30 are
#: expected) — the same cut after a trailing space was stripped. Both are
#: treated as truncated: ``scripts/09_extract_ces_full_wording.py`` now
#: recovers the real wording of the four CES from the questionnaires
#: themselves, so what is still 79 or 80 characters long here is an item no
#: source covers, and ``display_label`` remains its only fallback. The ~30
#: genuinely 79-character questions pay for that with a paraphrased label.
TRUNCATION_LENGTHS: tuple[int, ...] = (79, 80)

# Field labels. Two languages, same fields, same order, same punctuation —
# the FR/EN pair must not differ by anything but the words.
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "population": "Population",
        "age": "Âge",
        "gender": "Genre",
        "education": "Scolarité",
        "region_qc": "Région",
        "region": "Province",
        "income": "Revenu du ménage",
        "language": "Langue maternelle",
    },
    "en": {
        "population": "Population",
        "age": "Age",
        "gender": "Gender",
        "education": "Education",
        "region_qc": "Region",
        "region": "Province",
        "income": "Household income",
        "language": "First language",
    },
}

_TEXT: dict[str, dict[str, str]] = {
    "fr": {
        "persona_head": "Tu es un répondant à un sondage d'opinion.",
        "persona_none": "Tu es un répondant à un sondage d'opinion.",
        "c1_head": "Tes réponses à d'autres questions du sondage :",
        "c2_head": "Réponses observées dans ton groupe à d'autres questions :",
        "question": "Question",
        "options": "Options",
        "instruction": "Réponds uniquement par le numéro de l'option choisie.",
        "population_qc": "Québec",
        "population_ca": "Canada",
    },
    "en": {
        "persona_head": "You are a respondent to an opinion survey.",
        "persona_none": "You are a respondent to an opinion survey.",
        "c1_head": "Your answers to other questions in the survey:",
        "c2_head": "Answers observed in your group to other questions:",
        "question": "Question",
        "options": "Options",
        "instruction": "Answer with the number of the chosen option only.",
        "population_qc": "Quebec",
        "population_ca": "Canada",
    },
}


# --------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    """One response modality, verbatim. ``code`` is what the model must emit."""

    code: str
    label: str

    def render(self) -> str:
        return f"{self.code}) {self.label}"


@dataclass(frozen=True)
class ItemSpec:
    """A corpus item, ready to render. Wording and options are never rewritten.

    ``text`` is the verbatim wording used for a *target* item; ``short_text``
    is the compact label used when the item appears as *context* (§3.4 keeps
    the verbatim rule for target items only).
    """

    survey_id: str
    variable: str
    text: str
    options: tuple[Option, ...]
    language: str = "fr"
    short_text: str | None = None
    #: Raw microdata code -> surviving code, from the refusal merge of
    #: ``corpus.perimeter.merge_refusal_options`` (``code_map`` in
    #: items.parquet). Every raw answer must pass through
    #: :meth:`canonical_code` before it is compared to ``options``, in the
    #: prompt AND in the observed distribution, or the two describe different
    #: partitions of the same item.
    code_map: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.survey_id, self.variable)

    def canonical_code(self, code: Any) -> str | None:
        """Fold a raw microdata code onto the modality actually offered."""
        if code is None:
            return None
        code = str(code)
        for raw, kept in self.code_map:
            if raw == code:
                return kept
        return code

    @property
    def context_text(self) -> str:
        return self.short_text or self.text

    def option_label(self, code: Any) -> str | None:
        code = self.canonical_code(code)
        if code is None:
            return None
        for opt in self.options:
            if opt.code == code:
                return opt.label
        return None

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *,
                 prefer_label_when_truncated: bool = True,
                 truncation_lengths: Sequence[int] = TRUNCATION_LENGTHS) -> "ItemSpec":
        """Build from one row of ``data/items.parquet``."""
        return cls(
            survey_id=row["survey_id"],
            variable=row["variable"],
            text=item_text(row,
                           prefer_label_when_truncated=prefer_label_when_truncated,
                           truncation_lengths=truncation_lengths),
            options=parse_options(row["options"]),
            language=row.get("language") or "fr",
            short_text=(row.get("display_label") or None),
            code_map=parse_code_map(row.get("code_map")),
        )


@dataclass(frozen=True)
class Persona:
    """A respondent profile: canonical SES labels + the population field.

    ``fields`` maps a canonical dimension id to its already-rendered label
    (what ``SesCrosswalk.profile_labels`` returns). ``population`` is derived
    from the survey unless given explicitly.
    """

    fields: Mapping[str, str]
    survey_id: str | None = None
    population: str | None = None

    def resolved_population(self, language: str) -> str:
        if self.population is not None:
            return self.population
        texts = _TEXT[language]
        national = self.survey_id in NATIONAL_SURVEYS
        return texts["population_ca"] if national else texts["population_qc"]


@dataclass(frozen=True)
class ContextAnswer:
    """C1 unit: this respondent's real answer to a neighbouring item."""

    item: ItemSpec
    code: str
    label: str

    @classmethod
    def from_item(cls, item: ItemSpec, code: Any) -> "ContextAnswer":
        label = item.option_label(code)
        if label is None:
            raise ValueError(f"code {code!r} not an option of {item.key}")
        return cls(item=item, code=item.canonical_code(code), label=label)


@dataclass(frozen=True)
class ContextDistribution:
    """C2 unit: the stratum's observed distribution on a neighbouring item.

    ``shares`` holds ``(label, proportion)`` pairs in the item's option order,
    proportions summing to ~1. ``n`` is the cell size backing them — computed
    by the generator with the double masking of §3.3 (respondent excluded,
    target item excluded).
    """

    item: ItemSpec
    shares: tuple[tuple[str, float], ...]
    n: int | None = None


# --------------------------------------------------------------------------
# corpus-row helpers
# --------------------------------------------------------------------------

def item_text(row: Mapping[str, Any], *,
              prefer_label_when_truncated: bool = True,
              truncation_lengths: Sequence[int] = TRUNCATION_LENGTHS) -> str:
    """Verbatim wording of a target item.

    ``question_text`` is authoritative, except where the catalogue stored it
    truncated: a truncated stem is a different question, so ``display_label``
    is preferred there. Caveat carried into the report: ``display_label`` is a
    generated summary and is occasionally written in the other language.
    """
    text = row.get("question_text") or ""
    label = row.get("display_label") or ""
    if prefer_label_when_truncated and label and len(text) in tuple(truncation_lengths):
        return label
    return text or label


def parse_code_map(code_map: Any) -> tuple[tuple[str, str], ...]:
    """Parse the ``code_map`` column (JSON string, dict, or already a tuple)."""
    if code_map is None:
        return ()
    if isinstance(code_map, str):
        code_map = json.loads(code_map) if code_map.strip() else {}
    if isinstance(code_map, dict):
        return tuple((str(k), str(v)) for k, v in sorted(code_map.items()))
    return tuple((str(k), str(v)) for k, v in code_map)


def parse_options(options: Any) -> tuple[Option, ...]:
    """Parse the ``options`` column (JSON string or already-decoded list)."""
    if isinstance(options, str):
        options = json.loads(options)
    out = []
    for opt in options or ():
        if isinstance(opt, Option):
            out.append(opt)
        else:
            out.append(Option(code=str(opt["code"]), label=str(opt["label"])))
    return tuple(out)


def nearest_context_items(index, key: tuple[str, str], *, k: int = 6,
                          same_survey_only: bool = True,
                          eligible: Iterable[tuple[str, str]] | None = None,
                          max_cosine: float = CONTEXT_MAX_COSINE,
                          ) -> list[tuple[tuple[str, str], float]]:
    """The retrieval policy — one policy, training and inference alike.

    The ``k`` nearest **available** neighbours strictly **below**
    ``max_cosine``. There is no lower threshold: variance across examples
    comes from the corpus itself, not from a cutoff — some targets have a 0.94
    neighbour, some a 0.60 one, and that spread is the §2.3 axis.

    The upper cut is the anti-leak rule (see the module docstring). An item at
    or above ``CONTEXT_MAX_COSINE`` of the target asks the same question of the
    same people; injecting it as context hands the model the answer, in C1
    literally and in C2 through the stratum distribution. Pass
    ``max_cosine=1.01`` to disable it — only ever legitimate in a diagnostic,
    never in a dataset.

    ``same_survey_only`` defaults to True because both C1 and C2 need an
    empirical counterpart among the respondents of the survey that asked the
    item. Fewer than ``k`` neighbours is a legitimate outcome and is returned
    as is.
    """
    frame = index.neighbors(key, k=None, exclude_same_survey=False)
    allowed = None if eligible is None else {tuple(e) for e in eligible}
    out: list[tuple[tuple[str, str], float]] = []
    for row in frame.iter_rows(named=True):
        nkey = (row["neighbor_survey_id"], row["neighbor_variable"])
        if nkey == tuple(key):
            continue
        if same_survey_only and nkey[0] != key[0]:
            continue
        if allowed is not None and nkey not in allowed:
            continue
        if float(row["cosine"]) >= max_cosine:
            continue
        out.append((nkey, float(row["cosine"])))
        if len(out) == k:
            break
    return out


def example_rng(seed: int, example_id: str) -> random.Random:
    """Per-example RNG: same (seed, example_id) -> same SES dropout, always.

    Deriving the stream from the example id rather than from a running counter
    keeps the dataset reproducible under shuffling, sharding and resumption.
    """
    return random.Random(f"{seed}:{example_id}")


# --------------------------------------------------------------------------
# the template
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """The single template, parameterised by condition. Model-agnostic.

    Parameters
    ----------
    condition
        ``"C0"``, ``"C1"`` or ``"C2"``.
    k
        Number of context items (§0 decision: 6).
    ses_dropout
        ``"uniform_subset"`` (default), ``"bernoulli"`` or ``"none"``. See
        :meth:`keep_dimensions`.
    dropout_rate
        Per-field drop probability, used by ``"bernoulli"`` only.
    min_fields
        Floor on the number of SES fields kept. ``Population`` is never
        dropped: it is not an SES dimension, it is the fix for the
        language x population confound of §3.3.
    persona_dimensions
        Which canonical dimensions may appear, in render order.
    max_options_in_context
        Cap on modalities shown per C2 context item (``None`` = all). Purely a
        token-budget lever; it never touches the *target* item's options.
    show_cell_n
        Append ``(n=…)`` to each C2 context line.
    """

    condition: str = "C0"
    k: int = 6
    ses_dropout: str = "uniform_subset"
    dropout_rate: float = 0.3
    min_fields: int = 1
    persona_dimensions: tuple[str, ...] = PERSONA_DIMENSIONS
    max_options_in_context: int | None = None
    show_cell_n: bool = True
    percent_decimals: int = 0
    include_instruction: bool = True
    field_labels: Mapping[str, Mapping[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.ses_dropout not in ("uniform_subset", "bernoulli", "none"):
            raise ValueError(f"unknown ses_dropout {self.ses_dropout!r}")

    # ---------------- SES dropout (§2.7) ----------------

    def keep_dimensions(self, available: Sequence[str],
                        rng: random.Random | None = None) -> tuple[str, ...]:
        """Pick the SES subset shown in this example, in canonical order.

        ``uniform_subset`` draws the *number* of fields uniformly over
        ``[min_fields, D]``, then the subset uniformly. That is the default
        because §2.7's extra axis is performance **as a function of persona
        richness**: a uniform draw over richness gives every granularity the
        same number of training examples. Independent per-field dropout would
        make the count binomial and starve both extremes — with 5 dimensions
        and p=0.3, the 1-field and 5-field personas get ~3 % and ~17 % of the
        mass, which is precisely the out-of-distribution regime §2.7 warns
        about, only reintroduced through the back door.
        """
        ordered = [d for d in self.persona_dimensions if d in set(available)]
        if not ordered or self.ses_dropout == "none":
            return tuple(ordered)
        rng = rng or random.Random()
        floor = min(self.min_fields, len(ordered))
        if self.ses_dropout == "uniform_subset":
            size = rng.randint(floor, len(ordered))
            kept = set(rng.sample(ordered, size))
        else:  # bernoulli
            kept = {d for d in ordered if rng.random() >= self.dropout_rate}
            if len(kept) < floor:
                kept = set(rng.sample(ordered, floor))
        return tuple(d for d in ordered if d in kept)

    # ---------------- rendering ----------------

    def render_persona(self, persona: Persona, language: str,
                       rng: random.Random | None = None,
                       dimensions: Sequence[str] | None = None) -> str:
        labels = (self.field_labels or _FIELD_LABELS)[language]
        texts = _TEXT[language]
        kept = (tuple(dimensions) if dimensions is not None
                else self.keep_dimensions(list(persona.fields), rng))
        lines = [texts["persona_head"],
                 f"{labels['population']} : {persona.resolved_population(language)}"]
        for dim in kept:
            value = persona.fields.get(dim)
            if value:
                lines.append(f"{labels[dim]} : {value}")
        return "\n".join(lines)

    def render_target(self, item: ItemSpec, language: str) -> str:
        texts = _TEXT[language]
        block = [f"{texts['question']} : {item.text}", f"{texts['options']} :"]
        block += [opt.render() for opt in item.options]
        if self.include_instruction:
            block += ["", texts["instruction"]]
        return "\n".join(block)

    def render_context(self, context: Sequence[Any], language: str) -> str:
        if self.condition == "C0" or not context:
            return ""
        texts = _TEXT[language]
        if self.condition == "C1":
            lines = [texts["c1_head"]]
            for ans in context:
                lines.append(f"- {ans.item.context_text} → {ans.label}")
        else:
            lines = [texts["c2_head"]]
            for dist in context:
                lines.append(f"- {dist.item.context_text} : "
                             f"{self._render_shares(dist)}")
        return "\n".join(lines)

    def _render_shares(self, dist: ContextDistribution) -> str:
        shares = list(dist.shares)
        if self.max_options_in_context is not None:
            shares = sorted(shares, key=lambda s: -s[1])[:self.max_options_in_context]
        parts = [f"{label} {share * 100:.{self.percent_decimals}f} %"
                 for label, share in shares]
        rendered = ", ".join(parts)
        if self.show_cell_n and dist.n is not None:
            rendered = f"{rendered} (n={dist.n})"
        return rendered

    # ---------------- assembly ----------------

    def build_messages(self, persona: Persona, target: ItemSpec,
                       context: Sequence[Any] = (),
                       rng: random.Random | None = None,
                       dimensions: Sequence[str] | None = None,
                       ) -> list[dict[str, str]]:
        """The two prompt messages. Identical for the roleplay and FT arms."""
        language = target.language if target.language in _TEXT else "fr"
        if self.condition == "C0":
            context = ()
        else:
            context = tuple(context)[: self.k]
            for entry in context:
                if entry.item.key == target.key:
                    raise ValueError(
                        f"context leak: {target.key} used as its own context "
                        f"in {self.condition}"
                    )
        blocks = [self.render_context(context, language),
                  self.render_target(target, language)]
        user = "\n\n".join(b for b in blocks if b)
        return [
            {"role": "system",
             "content": self.render_persona(persona, language, rng, dimensions)},
            {"role": "user", "content": user},
        ]

    def build_example(self, persona: Persona, target: ItemSpec, answer_code: Any,
                      context: Sequence[Any] = (),
                      rng: random.Random | None = None,
                      dimensions: Sequence[str] | None = None,
                      ) -> dict[str, list[dict[str, str]]]:
        """A full chat training example; the assistant turn is the raw code."""
        messages = self.build_messages(persona, target, context, rng, dimensions)
        messages.append({"role": "assistant", "content": str(answer_code)})
        return {"messages": messages}
