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
            with its options listed **without codes** (``- <label>``).
``assistant`` the **text** of the chosen modality, exactly as listed
            (training target only).

Why text and not the code. The same content carries many codes across the
corpus ("Ne sais pas" alone has 18: -9, 3, 8, 98, 99, 998…), so a code target
ties what is learnt to one questionnaire's numbering and blocks transfer to
unseen items; code / symbol biases would also inflate the fine-tuned vs
roleplay contrast; and C1 context lines already show answers as label text.
Every rendered label is therefore unique within its item
(:meth:`ItemSpec.__post_init__` merges duplicate labels), and a model output
is mapped back to an option by :meth:`ItemSpec.match_answer` — exact or
normalised match only, never a fuzzy guess.

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
import re
import unicodedata
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
    "numeric_scale",
    "context_line_key",
    "deduplicate_context",
    "build_item_specs",
    "guess_language",
    "strip_variable_prefix",
    "strip_question_number",
    "repair_mojibake",
    "has_mojibake",
    "clean_wording",
    "shorten_wording",
    "question_text_truncated",
    "label_speaks",
    "normalise_answer",
    "strip_code_prefixes",
    "CONTEXT_MAX_CHARS",
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
#: Since ``items.parquet`` carries ``question_text_source``, this length rule
#: only drives rows *without* that column: see :func:`question_text_truncated`.
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
        "persona_head_year": "Tu es un répondant à un sondage d'opinion mené en {year}.",
        "scale_of": "sur",
        "c1_head": "Tes réponses à d'autres questions du sondage :",
        "c2_head": "Réponses observées dans ton groupe à d'autres questions :",
        "question": "Question",
        "options": "Options",
        "instruction": "Réponds uniquement par le texte exact de l'option choisie.",
        "population_qc": "Québec",
        "population_ca": "Canada",
    },
    "en": {
        "persona_head": "You are a respondent to an opinion survey.",
        "persona_none": "You are a respondent to an opinion survey.",
        "persona_head_year": "You are a respondent to an opinion survey conducted in {year}.",
        "scale_of": "out of",
        "c1_head": "Your answers to other questions in the survey:",
        "c2_head": "Answers observed in your group to other questions:",
        "question": "Question",
        "options": "Options",
        "instruction": "Answer with the exact text of the chosen option only.",
        "population_qc": "Quebec",
        "population_ca": "Canada",
    },
}


# --------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    """One response modality, verbatim.

    ``code`` identifies the modality in the microdata and is never shown to the
    model. ``text`` is what the prompt lists and what the model must emit.
    """

    code: str
    label: str

    @property
    def text(self) -> str:
        """The label as rendered and as expected back: mojibake repaired,
        whitespace collapsed, otherwise verbatim (``1 - greatly
        deteriorated`` and a bare ``2`` stay as they are)."""
        return " ".join(repair_mojibake(self.label).split())

    def render(self) -> str:
        return f"- {self.text}"


#: Stripped from both ends of a label / model output by :func:`normalise_answer`.
_ANSWER_LEAD = re.compile(r"^(?:[\s\"'«»“”‘’`]|[-*•]\s)+")
_ANSWER_TRAIL = re.compile(r"[\s\"'«»“”‘’`.,;:!?]+$")


def normalise_answer(text: str | None) -> str:
    """The key an answer is compared on when it is not an exact match.

    NFKC, casefold, whitespace collapsed, then surrounding quotes stripped, a
    leading list bullet (``- ``, copied from the option list) dropped, and
    trailing punctuation (``.``, ``!``, ``:``…) removed. Nothing else: no
    stemming, no edit distance. Option labels are made unique on this very key
    (:meth:`ItemSpec.__post_init__`), so a normalised match is unambiguous.
    """
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = " ".join(text.split())
    text = _ANSWER_LEAD.sub("", text)
    text = _ANSWER_TRAIL.sub("", text)
    return text


#: A label that starts with its own option code: ``(1) Liberal (Grits)``
#: (ces_2019_phone), ``1. Liberal Party`` (ces_2025), ``(3)`` alone.
_CODE_PREFIX = re.compile(r"^\(\s*(-?\d+)\s*\)\s*|^(-?\d+)[.)](?:\s+|$)")


def strip_code_prefixes(options: Sequence[Option]) -> tuple[Option, ...]:
    """Drop a label's leading copy of its own code; keep numbers that are content.

    With text answers the code must never reach the model (109 items of
    ces_2019_phone and all 317 of ces_2025 carry it). The prefix is removed
    only when its number *is* the option's code, so ``1 - greatly
    deteriorated`` or a party named ``1er choix`` are untouched.

    On a numbered scale the number is the answer, not a code: a point left
    empty by the strip (``(3)``) becomes the bare number ``3``, and a labelled
    point of the same consecutive run keeps it as ``0 - No interest at all`` —
    the ``N - anchor`` form the rest of the corpus already uses. Options outside
    the run (``(-9) Don't know``) just lose the prefix. Idempotent.
    """
    rests: dict[str, str | None] = {}
    for opt in options:
        m = _CODE_PREFIX.match(opt.label)
        if m and (m.group(1) or m.group(2)) == opt.code:
            rest = opt.label[m.end():].strip()
            rests[opt.code] = "" if rest == opt.code else rest
        else:
            rests[opt.code] = None
    if all(r is None for r in rests.values()):
        return tuple(options)
    bare = [int(c) for c, r in rests.items() if r == "" and re.fullmatch(r"-?\d+", c)]
    run: set[int] = set()
    if bare:
        numeric = {int(c) for c in rests if re.fullmatch(r"-?\d+", c)}
        lo = hi = min(bare)
        while lo - 1 in numeric:
            lo -= 1
        while hi + 1 in numeric:
            hi += 1
        run = set(range(lo, hi + 1))
    out = []
    for opt in options:
        rest = rests[opt.code]
        if rest is None:
            out.append(opt)
        elif re.fullmatch(r"-?\d+", opt.code) and int(opt.code) in run:
            out.append(Option(opt.code, f"{opt.code} - {rest}" if rest else opt.code))
        else:
            out.append(Option(opt.code, rest or opt.code))
    return tuple(out)


def merge_duplicate_labels(options: Sequence[Option],
                           ) -> tuple[tuple[Option, ...], dict[str, str]]:
    """Collapse options whose rendered labels coincide (:func:`normalise_answer`).

    Two options with the same label cannot be told apart by a text answer
    (``eeq_2018`` q53/q54 offer ``Je ne sais pas`` as both 3 and 98). Same rule
    as the refusal merge: the first code in questionnaire order is kept, the
    others are mapped onto it. Returns the surviving options and the
    ``dropped code -> kept code`` map.
    """
    kept: list[Option] = []
    first: dict[str, str] = {}
    remap: dict[str, str] = {}
    for opt in options:
        key = normalise_answer(opt.text)
        if key in first:
            if opt.code != first[key]:
                remap[opt.code] = first[key]
            continue
        first[key] = opt.code
        kept.append(opt)
    return tuple(kept), remap


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
    #: Year the survey was fielded (``year`` in items.parquet). It frames the
    #: interview, not the respondent: the same age bracket in 1998 and in 2025
    #: is two different birth cohorts, and an opinion on sovereignty or on
    #: immigration is only readable against the year it was collected. See
    #: :meth:`PromptTemplate.resolve_year` for the override chain.
    year: int | None = None
    #: Raw microdata code -> surviving code, from the refusal merge of
    #: ``corpus.perimeter.merge_refusal_options`` (``code_map`` in
    #: items.parquet). Every raw answer must pass through
    #: :meth:`canonical_code` before it is compared to ``options``, in the
    #: prompt AND in the observed distribution, or the two describe different
    #: partitions of the same item.
    code_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # Duplicate-label merge (see :func:`merge_duplicate_labels`), folded
        # into ``code_map`` so that ``canonical_code`` — hence the prompt, the
        # training target and every observed distribution — uses the merged
        # partition. Idempotent: rebuilding a spec from its own fields is a no-op.
        # Code prefixes go first, so that two options differing only by their
        # code (``(1) Yes`` / ``(2) Yes``) are caught by the merge.
        stripped = strip_code_prefixes(self.options)
        if stripped != self.options:
            object.__setattr__(self, "options", stripped)
        options, remap = merge_duplicate_labels(self.options)
        if not remap:
            return
        code_map = {raw: remap.get(kept, kept) for raw, kept in self.code_map}
        code_map.update(remap)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "code_map", tuple(sorted(code_map.items())))

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
    def wording(self) -> str:
        """The target wording as rendered (:func:`clean_wording`).

        ``text`` stays the corpus string; the variable-name prefix and the
        mis-encoded sequences are removed at render time only.
        """
        return clean_wording(self.text, self.variable)

    @property
    def context_text(self) -> str:
        """The one-line wording of this item on a context line.

        ``short_text``, when set, is an explicit override (a corpus-level
        disambiguation, or a same-language ``display_label`` standing in for
        an unusable wording — see :func:`build_item_specs`). Otherwise the line
        is :func:`shorten_wording` of the item's own wording, which is in the
        item's language by construction.
        """
        if self.short_text:
            return " ".join(clean_wording(self.short_text, self.variable).split())
        return shorten_wording(self.wording)

    def option_label(self, code: Any) -> str | None:
        code = self.canonical_code(code)
        if code is None:
            return None
        for opt in self.options:
            if opt.code == code:
                return opt.label
        return None

    def answer_text(self, code: Any) -> str | None:
        """The training target for a raw code: the option's rendered text,
        exactly the string listed in the prompt (``None`` if not offered)."""
        code = self.canonical_code(code)
        for opt in self.options:
            if opt.code == code:
                return opt.text
        return None

    def match_answer(self, text: str | None) -> str | None:
        """Map a model output back to an option code, or ``None``.

        Exact match on the rendered text first, then a match on
        :func:`normalise_answer`. Never a fuzzy guess: anything else — a code,
        a paraphrase, two options — is ``None`` and must be counted as an
        invalid answer by the caller, not silently reassigned.
        """
        if text is None:
            return None
        text = str(text)
        for opt in self.options:
            if text == opt.text:
                return opt.code
        key = normalise_answer(text)
        if not key:
            return None
        for opt in self.options:
            if key == normalise_answer(opt.text):
                return opt.code
        return None

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *,
                 prefer_label_when_truncated: bool = True,
                 truncation_lengths: Sequence[int] = TRUNCATION_LENGTHS) -> "ItemSpec":
        """Build from one row of ``data/items.parquet``.

        The context line is derived from the wording, not from
        ``display_label`` (a generated summary, in the wrong language on 30 %
        of the English context lines it used to produce). ``display_label`` is
        kept as ``short_text`` only where the wording is unusable — empty or
        Stata-truncated — **and** the label speaks the item's language. Use
        :func:`build_item_specs` for a whole table: it also resolves the
        collisions a per-row shortening cannot see.
        """
        label = row.get("display_label") or None
        text = row.get("question_text") or ""
        unusable = not text.strip() or question_text_truncated(row)
        short = label if unusable and label_speaks(row, label) else None
        return cls(
            survey_id=row["survey_id"],
            variable=row["variable"],
            text=item_text(row,
                           prefer_label_when_truncated=prefer_label_when_truncated,
                           truncation_lengths=truncation_lengths),
            options=parse_options(row["options"]),
            language=row.get("language") or "fr",
            short_text=short,
            year=(int(row["year"]) if row.get("year") is not None else None),
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
    #: Survey year, when the caller knows it without an item at hand (the
    #: inference harness builds personas before it picks a question). Wins over
    #: the target item's own year, loses to ``PromptTemplate.year_override``.
    year: int | None = None

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
    is preferred there — **but only when it speaks the item's language**
    (:func:`label_speaks`). ``display_label`` is a generated summary and is
    sometimes written in the other language; a French label on an English
    item would make the target itself bilingual, which is worse than a cut
    stem.

    Truncation is read from ``question_text_source`` when the row has it
    (:func:`question_text_truncated`): since the full CES wording was
    recovered, a 79- or 80-character question is usually a complete one, and
    the length rule alone replaced ~40 intact questions by their summary.
    `truncation_lengths` still drives rows without that column.
    """
    text = row.get("question_text") or ""
    label = row.get("display_label") or ""
    if "question_text_source" in row and row.get("question_text_source") is not None:
        truncated = question_text_truncated(row)
    else:
        truncated = len(text) in tuple(truncation_lengths)
    if prefer_label_when_truncated and truncated and label_speaks(row, label):
        return label
    return text or label  # an empty wording leaves nothing else to show


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


# --------------------------------------------------------------------------
# wording hygiene — render-time only, the corpus is never rewritten
# --------------------------------------------------------------------------

#: Default length cap of a context line, in characters. A context line names
#: the question; it does not re-ask it. 160 keeps 94 % of the corpus wording
#: whole (median 105) and still cuts the CES phone stems that run to 1 700.
CONTEXT_MAX_CHARS = 160

#: Longest battery row label kept verbatim at the end of a shortened line.
BATTERY_ROW_MAX_CHARS = 90

#: ``question_text_source`` of a stem read from a Stata label the extractor
#: could not replace with the questionnaire wording (Stata caps labels at 80
#: *bytes*, so a multi-byte character ends the stem a few characters early).
STATA_TRUNCATED_SOURCE = "stata_label_truncated"
STATA_LABEL_BYTES = 80

_FR_WORDS = frozenset(
    "le la les des du de est et vous votre vos au aux pour que qui une un sur "
    "dans pas ou ce cette il ils elle elles son sa ses leur leurs plus avec par "
    "êtes avez quel quelle quels quelles selon entre comme été était très sont "
    "être faire nous notre mais si ne se lequel laquelle lesquels diriez "
    "pensez croyez parti gouvernement élection chef".split()
)
_EN_WORDS = frozenset(
    "the of and to is are you your in for with do does how what which would "
    "that this be it have has should were was who about than by from their "
    "they not any following think party government election leader whether "
    "people did".split()
)
_WORD = re.compile(r"[a-zà-öø-ÿœ]+(?:'[a-zà-öø-ÿœ]+)?")
_ELISION = re.compile(r"^(?:l|d|qu|j|n|s|c|m|t)'")
_FR_ACCENT = re.compile(r"[éèêàçùûôîâëïœ]")


def guess_language(text: str | None) -> str | None:
    """``"fr"``, ``"en"`` or ``None`` (no evidence either way).

    A deliberately small, deterministic heuristic — function words, French
    elisions and French diacritics — used for one decision only: whether a
    generated ``display_label`` may stand in for a wording, which it may only
    when it speaks the item's language. It is not a language model; on a short
    noun phrase with no function word it abstains, and an abstention never
    licenses the fallback. Diacritics count only on lower-case words, so a
    proper noun ("Yves-François Blanchet") is not French evidence.
    """
    if not text:
        return None
    text = unicodedata.normalize("NFC", str(text)).replace("’", "'")
    fr = en = 0
    for index, raw in enumerate(re.findall(r"\S+", text)):
        lowered = raw.lower()
        words = _WORD.findall(lowered)
        for word in words:
            if _ELISION.match(word):
                fr += 1
                word = word.split("'", 1)[1]
            if word in _FR_WORDS:
                fr += 1
            elif word in _EN_WORDS:
                en += 1
        if _FR_ACCENT.search(lowered) and (index == 0 or not raw[:1].isupper()):
            fr += 1
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return None


def strip_variable_prefix(text: str, variable: str) -> str:
    """Drop a leading copy of the variable name: ``q6_02. [..] Êtes-vous…``.

    Some questionnaires print the column name in front of the wording
    (``PROV1.``, ``rts_q1.``, ``q6_02.``). The model must not learn to read a
    column identifier as part of a question: at inference, on a new question,
    there is none. The prefix is removed only when it *is* the variable name,
    followed by ``.``, ``--``, ``:`` or a space. An identifier-like variable
    (with a digit or an underscore) matches case-insensitively; a purely
    alphabetic one only with its exact case and a punctuation separator, so
    that ``vote`` never eats the first word of ``Vote déclaré``. Everything
    after the prefix — interviewer instructions included — is left verbatim.
    """
    if not text or not variable:
        return text or ""
    identifier = bool(re.search(r"[\d_]", variable))
    sep = r"(?:\s*(?:\.|--|:)\s*|\s+)" if identifier else r"(?:\.|--|:)\s*"
    pattern = re.compile(rf"^\s*{re.escape(variable)}{sep}",
                         0 if not identifier else re.IGNORECASE)
    match = pattern.match(text)
    if not match or not text[match.end():].strip():
        return text
    return text[match.end():]


#: A questionnaire number in front of the wording: ``Q13.  Dans la campagne…``,
#: ``7a. S'il y avait…``, ``S11.  Combien…``, ``3)  Et êtes-vous…``. An optional
#: upper-case letter prefix, one to three digits, an optional sub-item letter,
#: then ``.``, ``)``, ``:`` or ``--`` and whitespace before a letter, a quote or
#: a bracket. ``18 ans`` or ``1.5 million`` never match: no separator + space.
_QUESTION_NUMBER = re.compile(
    r"^\s*[A-Z]{0,3}\d{1,3}[a-z]?\s*(?:\.|\)|:|--)\s+(?=[^\W\d_]|[\"«“'(\[¿¡])"
)


def strip_question_number(text: str) -> str:
    """Drop a leading questionnaire number (``Q10.``, ``7a.``, ``25.``).

    Same reason as :func:`strip_variable_prefix`: the number is layout, not
    wording, and a new question at inference carries none. Unlike a variable
    name it cannot be checked against anything, so the pattern is narrow —
    see ``_QUESTION_NUMBER``. Text that would be left empty is kept whole.
    """
    if not text:
        return text or ""
    match = _QUESTION_NUMBER.match(text)
    if not match or not text[match.end():].strip():
        return text
    return text[match.end():]


#: A letter followed by a spacing grave accent is a dead-key sequence that was
#: never composed (``A` chaque élection``). Only the vowels French puts a grave
#: on are repaired: ``à è ù`` and their capitals — no other reading exists.
_SPACING_GRAVE = re.compile(r"([AEUaeu])`")
_GRAVE = {"A": "À", "E": "È", "U": "Ù", "a": "à", "e": "è", "u": "ù"}
#: UTF-8 bytes decoded as cp1252/latin-1: ``Ã©`` for ``é``, ``â€™`` for ``’``.
_DOUBLE_ENCODED = re.compile(r"Ã[\x80-\xbfŒ-™]|â€|Â[\xa0-\xbf]")


def has_mojibake(text: str | None) -> bool:
    """Whether :func:`repair_mojibake` would change `text`."""
    return bool(text) and repair_mojibake(text) != text


def repair_mojibake(text: str | None) -> str:
    """Repair mis-encoded sequences that admit exactly one reading.

    Two families, both unambiguous: an uncomposed dead-key grave
    (``A``` -> ``À``), and UTF-8 read as cp1252 (``Ã©`` -> ``é``), the latter
    only when the *whole* string round-trips — a partial match is left alone
    rather than guessed at.
    """
    if not text:
        return text or ""
    if _DOUBLE_ENCODED.search(text):
        for codec in ("cp1252", "latin-1"):
            try:
                text = text.encode(codec).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return _SPACING_GRAVE.sub(lambda m: _GRAVE[m.group(1)], text)


def clean_wording(text: str | None, variable: str = "") -> str:
    """The wording as rendered: variable prefix and question number off,
    mojibake repaired."""
    return repair_mojibake(
        strip_question_number(strip_variable_prefix(text or "", variable)))


_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[\"«“'(\[A-ZÀ-ÖØ-Þ0-9¿¡])")
#: `` - `` (Qualtrics battery rows, CES) and `` / `` (row or interviewer line,
#: CECD / EEQ): what follows the last one is a battery row label.
_ROW_SEPARATOR = re.compile(r"\s+[-–—/]\s+")
_MIN_HEAD_CHARS = 40


def _cut_on_word(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    cut = text[: max(budget - 1, 1)]
    space = cut.rfind(" ")
    if space >= budget // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-–—/") + "…"


def _head(text: str, budget: int) -> str:
    """Leading whole sentences up to `budget`, else a word-boundary cut.

    Sentences are added until the head is at least ``_MIN_HEAD_CHARS`` long,
    so a lone ``Q9.`` or ``Et vous ?`` is never the whole line.
    """
    head = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{head} {sentence}" if head else sentence
        if len(candidate) > budget:
            break
        head = candidate
        if len(head) >= _MIN_HEAD_CHARS:
            return head
    if head and len(head) >= min(_MIN_HEAD_CHARS, budget):
        return head
    return _cut_on_word(text, budget)


def shorten_wording(text: str, max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """A one-line, deterministic short form of a wording, for a context line.

    Whitespace is collapsed (a context line is one line). A wording within
    `max_chars` is kept whole. Longer, it keeps its first sentence(s) — or a
    word-boundary cut marked ``…`` — **plus, for a battery sub-item, the row
    label**: in ``How important … Canadian... - Big Corporations`` or
    ``… énoncés suivants. / Sans l'action du gouvernement…`` the discriminant
    is at the *end*, and a head-only cut would give every row of the battery
    the same line. The row is what follows the last `` - `` / `` / ``, kept
    verbatim when it is at most ``BATTERY_ROW_MAX_CHARS`` long. A row label
    written in front (``[J'ai l'habitude de voter] Êtes-vous…``) is part of the
    head and survives the cut on its own.

    Per-item only: two wordings can still shorten to the same line when their
    discriminant sits mid-sentence. :func:`build_item_specs` resolves those
    collisions over the corpus.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    separators = list(_ROW_SEPARATOR.finditer(text))
    if separators:
        last = separators[-1]
        row = text[last.end():]
        stem = text[: last.start()]
        glue = f" {last.group(0).strip()} "
        budget = max_chars - len(row) - len(glue)
        if row and len(row) <= BATTERY_ROW_MAX_CHARS and budget >= 20 and stem:
            return f"{_head(stem, budget)}{glue}{row}"
    return _head(text, max_chars)


def question_text_truncated(row: Mapping[str, Any]) -> bool:
    """Whether the row's ``question_text`` is a Stata label cut at 80 bytes.

    Decided from ``question_text_source``, which the wording extractor fills:
    only a stem it could not recover *and* that reaches the byte cap is
    truncated. Rows without the column (legacy callers) fall back to the
    length heuristic of :data:`TRUNCATION_LENGTHS`.
    """
    text = row.get("question_text") or ""
    if "question_text_source" in row and row.get("question_text_source") is not None:
        return (row["question_text_source"] == STATA_TRUNCATED_SOURCE
                and len(text.encode("utf-8")) >= STATA_LABEL_BYTES - 4)
    return len(text) in TRUNCATION_LENGTHS


def label_speaks(row: Mapping[str, Any], label: str | None) -> bool:
    """Whether ``display_label`` may stand in for the wording of this row.

    ``display_label`` is an LLM summary, and 30 % of the context lines of the
    English prompts once came out in French because of it. It may replace a
    wording only when its language is *positively* the item's; an abstention
    of :func:`guess_language` does not count.
    """
    if not label:
        return False
    language = row.get("language")
    if language is None:
        return True  # legacy row: nothing to check against
    return guess_language(label) == language


#: A modality label that is nothing but its own position on the scale.
_BARE_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?$")

#: ``"1- Fortement en désaccord"``, ``"5 - greatly improved"``,
#: ``"0 Le plus à gauche"`` — a scale point that carries its anchor wording.
_ANCHORED_POINT = re.compile(r"^(-?\d+)\s*(?:[-–—:.)]+\s*|\s+)(\S.*)$")


@dataclass(frozen=True)
class NumericScale:
    """The numbered scale an item's modalities describe, when they do.

    ``low``/``high`` are the extreme positions, ``low_anchor``/``high_anchor``
    the wording printed beside them (``None`` when the questionnaire only
    numbered the point). ``anchored`` says whether at least one end is named,
    i.e. whether the scale's *direction* is recoverable from the options alone.
    """

    low: int
    high: int
    low_anchor: str | None = None
    high_anchor: str | None = None

    @property
    def anchored(self) -> bool:
        return bool(self.low_anchor or self.high_anchor)


def numeric_scale(item: "ItemSpec") -> NumericScale | None:
    """Recover the 1-7 / 0-10 scale behind an item's modalities, if any.

    Agree/disagree and left/right batteries label the two ends and leave every
    intermediate point as a bare digit, so a context line rendered from the
    label alone reads ``habitude personnelle de voter → 5`` — a number with no
    scale attached. Recovering the scale here lets
    :meth:`PromptTemplate.context_label` put it back.

    Returns ``None`` unless the numbered modalities form a run of at least
    three consecutive integers: two points, or a gappy set, is a coding scheme
    rather than a scale, and annotating it would invent information.
    """
    points: dict[int, str | None] = {}
    for opt in item.options:
        label = str(opt.label).strip()
        if _BARE_NUMBER.match(label):
            points.setdefault(int(float(label.replace(",", "."))), None)
            continue
        match = _ANCHORED_POINT.match(label)
        if match:
            position = int(match.group(1))
            if points.get(position) is None:
                points[position] = match.group(2).strip()
    numbers = sorted(points)
    if len(numbers) < 3 or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return None
    low, high = numbers[0], numbers[-1]
    return NumericScale(low=low, high=high,
                        low_anchor=points[low], high_anchor=points[high])


def context_line_key(text: str) -> str:
    """Normalised form two context wordings are compared on."""
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def build_item_specs(rows: Iterable[Mapping[str, Any]],
                     ) -> dict[tuple[str, str], "ItemSpec"]:
    """Every row as an :class:`ItemSpec`, context lines resolved corpus-wide.

    :func:`shorten_wording` works one item at a time and so cannot know that
    two items of a survey end up on the same line. Two passes, per survey,
    fix that deterministically:

    1. *Shortening collisions.* Items whose wordings differ but shorten to
       the same line get their **whole** wording back (the discriminant sat
       mid-sentence: ``… how intelligent they are. - Blanchet`` vs
       ``… how trustworthy they are. - Blanchet``).
    2. *Identical wordings.* Items whose verbatim wording is the same as a
       sibling's (``Do you think Canada should admit:`` for immigrants,
       refugees, students…) cannot say on a context line which question was
       answered: that wording is unusable *as a context line*, so their
       ``display_label`` takes over — only where it speaks the item's
       language (:func:`label_speaks`). Where it does not, the wording stays
       and :func:`deduplicate_context` keeps one of the twins.

    The inference harness must build its specs through here too, or the same
    table renders different context lines at training and at inference.
    """
    specs: dict[tuple[str, str], ItemSpec] = {}
    labels: dict[tuple[str, str], tuple[str | None, bool]] = {}
    for row in rows:
        spec = ItemSpec.from_row(row)
        specs[spec.key] = spec
        label = row.get("display_label") or None
        labels[spec.key] = (label, label_speaks(row, label))

    def groups() -> list[list[tuple[str, str]]]:
        buckets: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key in sorted(specs):
            line = context_line_key(specs[key].context_text)
            buckets.setdefault((key[0], line), []).append(key)
        return [keys for keys in buckets.values() if len(keys) > 1]

    for keys in groups():
        wordings = {context_line_key(" ".join(specs[k].wording.split())) for k in keys}
        if len(wordings) > 1:
            for key in keys:
                spec = specs[key]
                if spec.short_text is None:
                    specs[key] = _replace_short(spec, " ".join(spec.wording.split()))
    for keys in groups():
        for key in keys:
            label, speaks = labels[key]
            if speaks and label and specs[key].short_text != label:
                specs[key] = _replace_short(specs[key], label)
    return specs


def _replace_short(spec: "ItemSpec", short: str) -> "ItemSpec":
    return ItemSpec(**{**spec.__dict__, "short_text": short})


def deduplicate_context(target: "ItemSpec | None",
                        context: Sequence[Any]) -> tuple[Any, ...]:
    """Drop context entries whose rendered wording repeats the target's or each other's.

    Enforced here, on the *rendering*, rather than on the embedding, because
    the cosine cut of :func:`nearest_context_items` works on the embedding of a
    text that is itself sometimes degenerate: the twelve slots of the
    ``ces_2021`` participation and internet-voting batteries share one generic
    ``display_label`` and one 80-character truncated stem, so a neighbour can
    sit at cosine 0.93 and still render as the exact string the target is asked
    with. Left alone that produces six identical context lines with six
    different answers, one of which reads as the target's own question:
    ambiguity at best, a copyable answer at worst.

    It lives in the template so the inference harness gets it for free — the
    defect would otherwise come back the day stratum context is injected at
    inference, where no dataset generator stands in the way.
    """
    seen = ({context_line_key(target.wording), context_line_key(target.text),
             context_line_key(target.context_text)}
            if target is not None else set())
    seen.discard("")
    kept = []
    for entry in context:
        rendered = context_line_key(entry.item.context_text)
        if rendered and rendered in seen:
            continue
        seen.add(rendered)
        kept.append(entry)
    return tuple(kept)


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
    year_override
        Render every prompt as if the survey had been fielded that year. The
        corpus stops in 2025 and the product will ask about 2026: the year is
        a knob of the harness, never a column to rewrite.
    annotate_numeric_labels
        Render a bare numeric context modality as ``5 sur 7 (1 = …, 7 = …)``
        instead of ``5``. See :meth:`context_label`.
    dedup_context
        Drop context entries repeating the target's wording or each other's.
        See :func:`deduplicate_context`.
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
    #: Forces the interview year, whatever the item or the persona says. The
    #: product will ask "and in 2026?" of a corpus trained up to 2025; that
    #: question is a render-time override, not a corpus edit.
    year_override: int | None = None
    #: Put a bare numeric context modality back on its scale (correctif 2).
    annotate_numeric_labels: bool = True
    #: Drop a context entry whose wording repeats the target's or a kept one.
    dedup_context: bool = True

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

    def resolve_year(self, persona: Persona | None = None,
                     target: "ItemSpec | None" = None) -> int | None:
        """The year that frames the interview: override > persona > item.

        The corpus spans 1998-2025 and the template carried no year at all,
        which made three things silently wrong: training averaged 27 years of
        opinion into one timeless distribution; per-cell validation compared a
        prediction to a cell of *one* survey the model could not identify; and
        the age bracket lost its referent, "45-54" being a 1944-53 cohort in
        1998 and a 1971-80 one in 2025. It is not an SES dimension and is never
        dropped by :meth:`keep_dimensions` — like ``Population``, it is the
        frame the rest is read against.
        """
        if self.year_override is not None:
            return int(self.year_override)
        if persona is not None and persona.year is not None:
            return int(persona.year)
        if target is not None and target.year is not None:
            return int(target.year)
        return None

    def context_label(self, item: "ItemSpec", label: str, language: str) -> str:
        """The modality as it appears on a context line, scale included.

        A bare number is the label of an unanchored point of a 1-7 or 0-10
        battery: on its own it says nothing, and dropping that information in
        C1 biases the very arm meant to demonstrate the value of context.
        """
        label = repair_mojibake(str(label))
        if not self.annotate_numeric_labels:
            return label
        text = label.strip()
        if not _BARE_NUMBER.match(text):
            return text
        scale = numeric_scale(item)
        if scale is None:
            return text
        rendered = f"{text} {_TEXT[language]['scale_of']} {scale.high}"
        anchors = [f"{pos} = {anchor}"
                   for pos, anchor in ((scale.low, scale.low_anchor),
                                       (scale.high, scale.high_anchor))
                   if anchor]
        return f"{rendered} ({', '.join(anchors)})" if anchors else rendered

    def render_persona(self, persona: Persona, language: str,
                       rng: random.Random | None = None,
                       dimensions: Sequence[str] | None = None,
                       year: int | None = None) -> str:
        labels = (self.field_labels or _FIELD_LABELS)[language]
        texts = _TEXT[language]
        kept = (tuple(dimensions) if dimensions is not None
                else self.keep_dimensions(list(persona.fields), rng))
        head = (texts["persona_head_year"].format(year=year) if year is not None
                else texts["persona_head"])
        lines = [head,
                 f"{labels['population']} : {persona.resolved_population(language)}"]
        for dim in kept:
            value = persona.fields.get(dim)
            if value:
                lines.append(f"{labels[dim]} : {value}")
        return "\n".join(lines)

    def render_target(self, item: ItemSpec, language: str) -> str:
        texts = _TEXT[language]
        block = [f"{texts['question']} : {item.wording}", f"{texts['options']} :"]
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
                label = self.context_label(ans.item, ans.label, language)
                lines.append(f"- {ans.item.context_text} → {label}")
        else:
            lines = [texts["c2_head"]]
            for dist in context:
                lines.append(f"- {dist.item.context_text} : "
                             f"{self._render_shares(dist, language)}")
        return "\n".join(lines)

    def _render_shares(self, dist: ContextDistribution, language: str = "fr") -> str:
        shares = list(dist.shares)
        if self.max_options_in_context is not None:
            shares = sorted(shares, key=lambda s: -s[1])[:self.max_options_in_context]
        parts = [f"{self.context_label(dist.item, label, language)} "
                 f"{share * 100:.{self.percent_decimals}f} %"
                 for label, share in shares]
        rendered = ", ".join(parts)
        if self.show_cell_n and dist.n is not None:
            rendered = f"{rendered} (n={dist.n})"
        return rendered

    # ---------------- assembly ----------------

    def select_context(self, target: ItemSpec,
                       context: Sequence[Any] = ()) -> tuple[Any, ...]:
        """The context entries actually rendered: leak check, dedup, then k.

        Public because the dataset generator must record exactly what the
        prompt shows — ``context_used`` and ``n_context`` would otherwise
        describe a block that was never rendered. Idempotent, so calling it
        upstream and letting :meth:`build_messages` call it again is safe.
        """
        if self.condition == "C0":
            return ()
        context = tuple(context)
        for entry in context:
            if entry.item.key == target.key:
                raise ValueError(
                    f"context leak: {target.key} used as its own context "
                    f"in {self.condition}"
                )
        if self.dedup_context:
            context = deduplicate_context(target, context)
        return context[: self.k]

    def build_messages(self, persona: Persona, target: ItemSpec,
                       context: Sequence[Any] = (),
                       rng: random.Random | None = None,
                       dimensions: Sequence[str] | None = None,
                       ) -> list[dict[str, str]]:
        """The two prompt messages. Identical for the roleplay and FT arms."""
        language = target.language if target.language in _TEXT else "fr"
        context = self.select_context(target, context)
        blocks = [self.render_context(context, language),
                  self.render_target(target, language)]
        user = "\n\n".join(b for b in blocks if b)
        year = self.resolve_year(persona, target)
        return [
            {"role": "system",
             "content": self.render_persona(persona, language, rng, dimensions,
                                            year=year)},
            {"role": "user", "content": user},
        ]

    def build_example(self, persona: Persona, target: ItemSpec, answer_code: Any,
                      context: Sequence[Any] = (),
                      rng: random.Random | None = None,
                      dimensions: Sequence[str] | None = None,
                      ) -> dict[str, list[dict[str, str]]]:
        """A full chat training example.

        The assistant turn is the chosen option's rendered text — the exact
        string listed in the prompt — looked up after the refusal and
        duplicate-label merges (:meth:`ItemSpec.answer_text`).
        """
        answer = target.answer_text(answer_code)
        if answer is None:
            raise ValueError(f"code {answer_code!r} is not an option of {target.key}")
        messages = self.build_messages(persona, target, context, rng, dimensions)
        messages.append({"role": "assistant", "content": answer})
        return {"messages": messages}
