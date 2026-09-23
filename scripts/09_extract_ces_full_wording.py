"""Recover the *full* question wording of the four CES surveys — step 3.2/3.3.

Why
---
``question_text`` in the normalized catalogue comes from the Stata **variable
label** of the ``.dta`` files. Stata caps a variable label at **80 characters**,
so the wording is truncated at the source for 450 of the 1 895 corpus items
(440 of them in the four CES). The exact wording is the object of the study, so
a truncated stem is not an acceptable input, and the current fallback
(``display_label``, an LLM-authored summary) is worse: it is a paraphrase, it is
sometimes written in the other language, and for ``ces_2021`` whole batteries
share one generic label.

Sources (read-only, in the sibling product repo ``mvp_moteur_recherche_sondages``)
--------------------------------------------------------------------------------
``ces_2021``          the two Qualtrics ``.qsf`` files — the questionnaire
                      itself, JSON, no length cap. Best possible source.
``ces_2019_online``   Online Survey Technical Report and Codebook v1.1 (PDF)
``ces_2025``          2025 Technical Report and Codebook v1 (PDF)
``ces_2019_phone``    the CATI questionnaire PDFs (CPS, PES, PES web)

Nothing is ever written to that repo. A variable that cannot be located in its
source is simply **absent** from the output — no label is ever invented.

Piping
------
The ``.qsf`` wording carries Qualtrics piping: ``${e://Field/x}`` (an embedded
data field) and ``${q://QIDn/ChoiceTextEntryValue/k}`` (text the respondent
typed earlier). An embedded field is resolved **only** when the survey flow
defines it with a single, literal value; a field whose value depends on a branch
(``pid_en`` takes six party names) and every ``q://`` reference are left
verbatim, because picking one of them would fabricate a question that no
respondent saw. Both counts are reported.

Output
------
``data/ces_full_wording.json`` ::

    {"ces_2021": {"cps21_demsat": {"question_text_full": "...",
                                   "source": "qsf",
                                   "confidence": "high"}}, ...}

``confidence`` grades the variable <-> question match against the truncated
Stata label, read from the normalized catalogue (never from
``data/items.parquet``, which this script feeds — reading back its own override
would leave nothing to grade on a second run):

``high``    the truncated label is a prefix of the recovered wording, or is
            contained in it verbatim once normalised — allowing for the word
            the 80-character cut left half-written. The match is certain;
``medium``  >= 85 % of the label's words are in the recovered wording but not
            contiguously — a label the CES team re-ordered or abridged when
            they built the ``.dta``, or one whose piping renders differently
            on the two sides;
``low``     weaker than that. **Dropped**: the variable is reported as
            unrecovered rather than paired with a wording that may not be its
            own. The ``display_label`` fallback keeps those items usable.

Run::

    .venv/bin/python scripts/09_extract_ces_full_wording.py
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from article_silicon_sampling_quebec.corpus import catalogue  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

PRODUCT_DATA = (REPO_ROOT.parent / "mvp_moteur_recherche_sondages" / "data").resolve()
OUT_JSON = REPO_ROOT / "data" / "ces_full_wording.json"
ITEMS_PARQUET = REPO_ROOT / "data" / "items.parquet"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "ces_wording_cache"

#: Lengths at which a Stata-sourced ``question_text`` is considered truncated.
#: 80 is the hard cap; 79 is the same cap after a trailing space was stripped.
TRUNCATION_LENGTHS = (79, 80)

QSF_FILES = [
    PRODUCT_DATA / "ces_2021" / "CES_CPS_2021_-_Public_Release.qsf",
    PRODUCT_DATA / "ces_2021" / "CES_PES_2021_-_Public_Release.qsf",
]

CODEBOOK_PDFS = {
    "ces_2019_online": [
        PRODUCT_DATA / "ces_2019_online"
        / "2019 Canadian Election Study - Online Survey Technical Report and "
          "Codebook v1.1.pdf",
    ],
    "ces_2025": [
        PRODUCT_DATA / "ces_2025"
        / "2025 Canadian Election Study Technical Report and Codebook v1.pdf",
    ],
}

PHONE_PDFS = [
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Campaign Period Survey EN v2.pdf",
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Post-election survey EN.pdf",
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Post-election survey (web) EN.pdf",
]


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_TAG_RE = re.compile(r"</?(?:br|p|div|li|tr|td|h[1-6])\b[^>]*>", re.I)
_WS_RE = re.compile(r"\s+")
_PIPE_E_RE = re.compile(r"\$\{e://Field/([^}]+)\}")
_PIPE_ANY_RE = re.compile(r"\$\{[^}]*\}")
#: The same piping, in the flattened form the Stata labels carry.
_PIPE_BRACKET_RE = re.compile(r"\[(?:Field|QID)[^\]]*\]")


def clean(text: str) -> str:
    """HTML -> plain text, entities decoded, whitespace collapsed."""
    if not text:
        return ""
    text = _BLOCK_TAG_RE.sub(" ", text)
    # Inline tags carry no space of their own: ``a <u>duty</u>.`` must not
    # become ``a duty .``.
    text = _TAG_RE.sub("", text)
    text = html.unescape(text).replace("\u00a0", " ")
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Aggressive fold used only for *matching*, never for storage."""
    text = clean(text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^a-z0-9àâäçéèêëîïôöùûüœ]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def pdf_text(path: Path) -> str:
    """``pdftotext -layout``, cached under data/raw/ces_wording_cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", path.stem)[:80] + ".txt")
    if not cached.exists():
        subprocess.run(
            ["pdftotext", "-layout", str(path), str(cached)],
            check=True, capture_output=True,
        )
    # pdftotext marks page breaks with a form feed glued to the first
    # character of the page — which would hide a variable sitting at the top
    # of a page from the line-start regexes below.
    return cached.read_text(encoding="utf-8", errors="replace").replace("\x0c", "")


# --------------------------------------------------------------------------
# source 1 — Qualtrics .qsf (ces_2021)
# --------------------------------------------------------------------------

def _flow_embedded_values(payload: dict) -> dict[str, set[str]]:
    """Every literal value the survey flow assigns to each embedded field."""
    values: dict[str, set[str]] = defaultdict(set)

    def walk(node) -> None:
        if isinstance(node, dict):
            for entry in node.get("EmbeddedData") or []:
                field = entry.get("Field")
                value = entry.get("Value")
                if field and isinstance(value, str) and value.strip():
                    values[field].add(value.strip())
            for child in node.get("Flow") or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)
    return values


def resolve_piping(text: str, field_values: dict[str, set[str]]) -> tuple[str, bool]:
    """Substitute what is unambiguously resolvable; leave the rest verbatim.

    Returns ``(text, still_has_piping)``.
    """

    def sub(match: re.Match) -> str:
        field = match.group(1)
        candidates = field_values.get(field, set())
        literal = {v for v in candidates if "${" not in v}
        if len(literal) == 1:
            return next(iter(literal))
        return match.group(0)

    text = _PIPE_E_RE.sub(sub, text)
    return text, bool(_PIPE_ANY_RE.search(text))


def ordered_choice_keys(payload: dict) -> list[str]:
    """Matrix statement keys in the order Qualtrics numbers the export columns.

    The ``.dta`` sub-variables of a battery are ``tag_1 … tag_n`` in the
    questionnaire's **statement order**, not in the internal choice-id order:
    ``cps21_issue_handle`` has ChoiceOrder ``[1..7, 9, 8]`` and its ninth
    column is Economy (choice id 8), which is exactly what the Stata label
    ``"… - Ec"`` says. Sorting by choice id would silently shift the last two
    statements of that battery onto each other's variable.
    """
    choices = payload.get("Choices") or {}
    order = [str(k) for k in (payload.get("ChoiceOrder") or [])]
    keys = [k for k in order if k in choices]
    keys += [k for k in choices if k not in set(keys)]
    return keys


def extract_qsf() -> tuple[dict[str, list[str]], Counter]:
    """(variable -> candidate wordings) for ces_2021, from the two .qsf files."""
    out: dict[str, list[str]] = defaultdict(list)
    stats = Counter()
    for path in QSF_FILES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        flow = next(
            (e for e in doc["SurveyElements"] if e["Element"] == "FL"), None
        )
        field_values = _flow_embedded_values(flow["Payload"]) if flow else {}

        for element in doc["SurveyElements"]:
            if element["Element"] != "SQ":
                continue
            payload = element["Payload"]
            tag = payload.get("DataExportTag")
            if not tag:
                continue
            stem_raw = payload.get("QuestionText_Unsafe") or payload.get(
                "QuestionText"
            ) or ""
            stem_raw, piped = resolve_piping(stem_raw, field_values)
            stem = clean(stem_raw)
            if not stem:
                continue
            stats["piping_unresolved" if piped else "piping_clean"] += 1

            if payload.get("QuestionType") in ("Matrix", "Slider"):
                for position, key in enumerate(ordered_choice_keys(payload), 1):
                    choice = payload["Choices"][key]
                    row_raw, row_piped = resolve_piping(
                        choice.get("Display") or "", field_values
                    )
                    row = clean(row_raw)
                    if not row:
                        continue
                    out[f"{tag}_{position}"].append(f"{stem} - {row}")
                    if row_piped:
                        stats["piping_unresolved_matrix_row"] += 1
            else:
                out[tag].append(stem)
    return out, stats


# --------------------------------------------------------------------------
# source 2 — CES online codebook PDFs (ces_2019_online, ces_2025)
# --------------------------------------------------------------------------

def _var_pattern(prefixes: tuple[str, ...]) -> re.Pattern:
    alt = "|".join(prefixes)
    # A stem can carry a small indent in the 2025 codebook; anything deeper
    # is a matrix answer-column header, not a variable line.
    return re.compile(rf"^ {{0,3}}((?:{alt})_[A-Za-z0-9_]+)(\s+)(\S.*)$")


_OPTION_RE = re.compile(r"^\s*(?:o|❍|❑|●)\s+\S")
_MATRIX_ROW_RE = re.compile(r"^(\s*)\(([A-Za-z0-9_]+)\)(\s|$)")
#: A row of radio buttons: standalone ``o`` glyphs, one per answer column.
_RADIO_RE = re.compile(r"(?<![^\s])o(?![^\s])")
#: An answer column header carries its code — "Never (1)", "Often (3)".
_ANSWER_HEADER_RE = re.compile(r"\(\d+\)")


def end_of_left_column(lines: list[str], at: int) -> int:
    """Column where the grid's first answer column starts, near line ``at``.

    Taken from the nearest row of radio glyphs: every statement of the battery
    is typeset to the left of it.
    """
    for offset in range(0, 6):
        for k in (at - offset, at + offset):
            if 0 <= k < len(lines):
                marks = [m.start() for m in _RADIO_RE.finditer(lines[k])]
                if len(marks) >= 3:
                    return marks[0]
    return 0
_STOP_PREFIXES = (
    "Display This Question", "Display this question", "Display This Choice",
    "Display this choice", "Page Break", "Start of Block", "End of Block",
    "NOTE:", "Note:", "If ", "Or ", "And ",
)
#: Choice-level display logic is printed inside the option list as
#: ``<some question> = <answer>``; when it lands right under a stem it would
#: otherwise be glued onto the wording.
_DISPLAY_LOGIC_RE = re.compile(r"\s(?:=|!=)\s")


def extract_codebook(text: str, prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    """(variable -> candidate wordings) from a CES online codebook PDF.

    Format, stable across 2019 and 2025: ``<variable> <question text>``, the
    text wrapping over following lines, then a blank line and the option list.
    The English block comes first and the French translation repeats the same
    variable name right after; a codebook NOTE can also open a line with a
    variable name followed by prose. Every hit is therefore kept as a
    *candidate* and the caller picks the one that actually matches the Stata
    label (:func:`grade`), which settles EN/FR and question/prose at once.

    Matrix batteries print the stem on the variable line and then one row per
    sub-item, the sub-variable in parentheses *underneath* its row label and in
    the same columns. Row labels are recovered by slicing the preceding lines
    to the column span of the ``(variable_n)`` token, which is what keeps the
    answer-column headers ("Liberal Party (1)", further right) out of them.
    """
    var_re = _var_pattern(prefixes)
    lines = text.split("\n")
    stems: dict[str, str] = {}
    out: dict[str, list[str]] = defaultdict(list)
    current_tag: str | None = None
    current_line = -1

    i = 0
    while i < len(lines):
        line = lines[i]
        match = var_re.match(line)
        if match:
            tag, gap, rest = match.group(1), match.group(2), match.group(3)
            # A codebook NOTE mentions variables mid-sentence; a real entry
            # starts the line and is followed by the question, not by prose
            # punctuation.
            if rest.startswith((",", ".", ";", ")", "=")):
                i += 1
                continue
            # The codebook's own variable index prints two variable names per
            # line; that is not a question.
            if re.fullmatch(r"[A-Za-z0-9_]+", rest.strip()):
                i += 1
                continue
            body = [rest.strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    break
                if var_re.match(nxt) or _MATRIX_ROW_RE.match(nxt):
                    break
                if _OPTION_RE.match(nxt) and nxt.strip().startswith(("o ", "❍", "❑")):
                    break
                if nxt.lstrip().startswith(_STOP_PREFIXES):
                    break
                if _DISPLAY_LOGIC_RE.search(nxt):
                    break
                if nxt.strip().startswith("_____"):
                    break
                # A wrapped question line starts at column 0 like the variable
                # line itself; anything indented is the answer-column header of
                # a matrix, which must not be glued onto the stem.
                if len(nxt) - len(nxt.lstrip()) > 4:
                    break
                body.append(nxt.strip())
                j += 1
            stem = clean(" ".join(body))
            if stem:
                stems[tag] = stem
                out[tag].append(stem)
                current_tag, current_line = tag, i
            i = j
            continue

        # A page break can leave the variable alone on its line, the wording
        # starting a few blank lines (and a page number) further down.
        alone = re.fullmatch(rf" {{0,3}}((?:{'|'.join(prefixes)})_[A-Za-z0-9_]+)\s*",
                             line)
        if alone:
            j = i + 1
            while j < len(lines) and (not lines[j].strip()
                                      or lines[j].strip().isdigit()):
                j += 1
            body = []
            while j < len(lines) and lines[j].strip() and not var_re.match(
                    lines[j]) and not _OPTION_RE.match(lines[j]):
                body.append(lines[j].strip())
                j += 1
            stem = clean(" ".join(body))
            if stem:
                tag = alone.group(1)
                stems[tag] = stem
                out[tag].append(stem)
                current_tag, current_line = tag, i
            i = j
            continue

        row = _MATRIX_ROW_RE.match(line)
        if row and current_tag and row.group(2).startswith(current_tag + "_"):
            sub = row.group(2)
            start = line.index("(" + sub + ")")
            # The statement text sits in the left-hand column of the grid,
            # which ends where the first answer column starts. Clipping to the
            # width of the ``(variable)`` token instead would cut long
            # statements mid-word; clipping any wider would drag the answer
            # headers ("Never (1)", "Sometimes (2)", …) into the wording.
            width = max(end_of_left_column(lines, i), start + len(sub) + 4)
            label_lines: list[str] = []
            k = i - 1
            while k > current_line:
                prev = lines[k]
                if _MATRIX_ROW_RE.match(prev) or var_re.match(prev):
                    break
                if _ANSWER_HEADER_RE.search(prev[width:]):
                    break  # the grid's own column headers
                piece = prev[:width].strip()
                if not piece:
                    if label_lines:
                        break
                    k -= 1
                    continue
                label_lines.insert(0, piece)
                k -= 1
            label = clean(" ".join(label_lines))
            if label:
                out[sub].append(f"{stems[current_tag]} - {label}")
        i += 1

    return out


# --------------------------------------------------------------------------
# source 3 — CES 2019 phone CATI questionnaires
# --------------------------------------------------------------------------

_PHONE_HEAD_RE = re.compile(r"^\s*([QP])(\d+)(?:\s+Show if\b.*)?\s*$")
_PHONE_SUBITEM_RE = re.compile(r"^\s*(\d+)\.\s+(\S.*?)\s*(?:\*.*)?$")
_PHONE_OPTION_RE = re.compile(r"^\s*[❍❑]")
#: CATI directives addressed to the interviewer, printed inside the question
#: block. They are instrument plumbing, not wording the respondent ever heard.
_PHONE_DIRECTIVE_RES = tuple(re.compile(pattern, re.I) for pattern in (
    r"\s*Repeat scale if needed\.?\s*$",
    r"\s*probe for one response only\.?\s*$",
    r"\s*name, enter party name\..*$",
    r"\s*\(Show if[^)]*\)\s*$",
))


def strip_cati_directives(text: str) -> str:
    changed = True
    while changed:
        changed = False
        for pattern in _PHONE_DIRECTIVE_RES:
            stripped = pattern.sub("", text).strip()
            if stripped != text:
                text, changed = stripped, True
    return text


def extract_phone(texts: list[str]) -> dict[str, list[str]]:
    """(variable -> full wording) from the CATI questionnaire PDFs.

    A question is a line holding only its number (``Q27``, ``P33``), then the
    wording, then either an option list (``❍ 1 …``) or a numbered statement
    list for a battery. Battery statements map to the ``_a``, ``_b``, …
    suffixes of the ``.dta`` in their printed order.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            head = _PHONE_HEAD_RE.match(lines[i])
            if not head:
                i += 1
                continue
            var = f"{head.group(1).lower()}{head.group(2)}"
            body: list[str] = []
            subs: list[str] = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if _PHONE_HEAD_RE.match(nxt):
                    break
                stripped = nxt.strip()
                if not stripped or stripped.isdigit():
                    j += 1
                    continue
                if _PHONE_OPTION_RE.match(nxt) or stripped.startswith("____"):
                    j += 1
                    continue
                sub = _PHONE_SUBITEM_RE.match(nxt)
                if sub and body:
                    subs.append(clean(sub.group(2)).rstrip(" *"))
                    j += 1
                    continue
                if re.match(r"^[QP]\d+\b", stripped):
                    # "P3 Show if P2 voted" — the next question, printed
                    # inline; everything after it belongs to that question.
                    break
                if stripped.startswith(("Minimum:", "Levels marked", "Do Not Read",
                                        "Do not read", "Read list", "If needed:",
                                        "Interviewer", "PROGRAMMER")):
                    j += 1
                    continue
                if subs:  # prose after the statement list belongs to no stem
                    j += 1
                    continue
                body.append(stripped)
                j += 1
            stem = strip_cati_directives(clean(" ".join(body)))
            if stem:
                out[var].append(stem)
                # A CATI question can print two conditional wordings
                # ("(if NOTQc) … (if Quebec) …"); the .dta then splits them
                # into two variables (q77eng / q77fr). Offer each branch as a
                # candidate — which one belongs to which variable is settled
                # by the Stata label, not guessed here.
                variants = [clean(v) for v in re.split(r"\(if [^)]*\)", stem) if
                            clean(v)]
                if len(variants) > 1:
                    out[var].extend(variants)
                for n, sub in enumerate(subs):
                    out[f"{var}_{chr(ord('a') + n)}"].append(f"{sub}: {stem}")
            i = j
    return out


# --------------------------------------------------------------------------
# matching and grading
# --------------------------------------------------------------------------

def strip_stata_prefix(survey_id: str, variable: str, text: str) -> str:
    """The phone ``.dta`` labels are prefixed ``q6 -- `` — drop that."""
    if survey_id == "ces_2019_phone":
        text = re.sub(rf"^{re.escape(variable)}\s*--\s*", "", text)
    return text


#: Stata labels of a "select one, or specify" question carry this Qualtrics
#: export suffix; it is not part of the wording.
_SELECTED_CHOICE_RE = re.compile(r"\s*-\s*Selected Ch(?:oice)?.*$")
#: ``[Field-x`` left unterminated by the 80-character cut.
_CUT_BRACKET_RE = re.compile(r"\[(?:Field|QID)[^\]]*$")


def grade(label: str, full: str) -> str:
    """Confidence of the variable <-> question match (see module docstring).

    The decisive test is *truncation itself*: the stored label is the first 80
    characters of the real wording, so the recovered wording must start with
    it, up to the final word the cut left half-written. That is a much sharper
    test than token overlap, and in particular it is the only one that catches
    a battery whose sub-items got shifted by one — the stems are identical
    there, only the trailing row label differs.
    """
    label = _SELECTED_CHOICE_RE.sub("", label)
    lab = normalize(_PIPE_BRACKET_RE.sub(" ", _CUT_BRACKET_RE.sub(" ", label)))
    ful = normalize(_PIPE_ANY_RE.sub(" ", full))
    if not lab:
        return "low"
    lab_tokens = lab.split()
    head, tail = " ".join(lab_tokens[:-1]), lab_tokens[-1]

    def cut_match(haystack: str, at: int) -> bool:
        """``head`` matched at ``at``; does the cut last word follow?"""
        rest = haystack[at + len(head):].lstrip()
        return rest.startswith(tail) or not tail

    if ful.startswith(head) and cut_match(ful, 0):
        return "high"
    at = ful.find(head)
    if head and at >= 0 and cut_match(ful, at):
        # The label is a middle slice of the wording: the CES phone codebook
        # labels the *second* paragraph of a two-paragraph question.
        return "high"
    core = lab_tokens[:-1] or lab_tokens
    ful_tokens = set(ful.split())
    overlap = sum(1 for t in core if t in ful_tokens) / len(core)
    if overlap >= 0.85:
        return "medium"
    return "low"


def stata_labels(survey_id: str) -> dict[str, str]:
    """The truncated ``question_text`` of the normalized catalogue.

    Read from the catalogue, never from ``data/items.parquet``: this script
    *feeds* that table, and reading back its own overridden output would leave
    nothing to grade on a second run.
    """
    return {
        item["variable"]: item.get("question_text") or ""
        for item in catalogue.survey_items(survey_id)
    }


def corpus_keys() -> set[tuple[str, str]]:
    """The (survey_id, variable) pairs that reach the experiment.

    Only the *keys*: the wording in that file may already be an override this
    script wrote, so it is never read back as a label.
    """
    if not ITEMS_PARQUET.exists():
        return set()
    frame = pl.read_parquet(ITEMS_PARQUET, columns=["survey_id", "variable"])
    return set(zip(frame["survey_id"], frame["variable"], strict=True))


def main() -> None:
    in_corpus = corpus_keys()

    sources: dict[str, tuple[str, dict[str, list[str]]]] = {}
    qsf, qsf_stats = extract_qsf()
    sources["ces_2021"] = ("qsf", qsf)
    for survey_id, paths in CODEBOOK_PDFS.items():
        prefixes = (("cps19", "pes19", "pes10") if "2019" in survey_id
                    else ("cps25", "pes25", "cses_module6"))
        merged: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            for key, values in extract_codebook(pdf_text(path), prefixes).items():
                merged[key].extend(values)
        sources[survey_id] = ("codebook_pdf", merged)
    sources["ces_2019_phone"] = (
        "codebook_pdf", extract_phone([pdf_text(p) for p in PHONE_PDFS])
    )

    table: dict[str, dict[str, dict]] = {}
    per_survey: dict[str, Counter] = defaultdict(Counter)
    rejected: dict[str, list[tuple[str, str]]] = defaultdict(list)
    misses: dict[str, list[str]] = defaultdict(list)

    rank = {"high": 0, "medium": 1, "low": 2}
    for survey_id, (source_name, mapping) in sources.items():
        entries: dict[str, dict] = {}
        for variable, raw_label in stata_labels(survey_id).items():
            label = strip_stata_prefix(survey_id, variable, raw_label)
            truncated = len(raw_label) in TRUNCATION_LENGTHS
            candidates = list(mapping.get(variable) or [])
            if not candidates:
                # ``q77eng`` / ``q77fr`` are two .dta columns of the single
                # CATI question Q77; fall back on the longest question number
                # the variable name extends.
                prefixes = sorted((k for k in mapping if variable.startswith(k)),
                                  key=len, reverse=True)
                if prefixes:
                    candidates = list(mapping[prefixes[0]])
            scope = "corpus" if (survey_id, variable) in in_corpus else "catalogue"
            if truncated:
                per_survey[survey_id][f"{scope}_truncated"] += 1
            if not candidates:
                if truncated:
                    per_survey[survey_id][f"{scope}_missing"] += 1
                    misses[survey_id].append(variable)
                continue
            # Several blocks of a codebook can open with the same variable
            # name (the English question, its French translation, a NOTE about
            # the split sample). The Stata label arbitrates: keep the reading
            # it actually matches, and if none does, keep nothing.
            full = min(candidates, key=lambda c: (rank[grade(label, c)], -len(c)))
            confidence = grade(label, full)
            if confidence == "low":
                if truncated:
                    per_survey[survey_id][f"{scope}_rejected"] += 1
                    rejected[survey_id].append((variable, full))
                continue
            entries[variable] = {
                "question_text_full": full,
                "source": source_name,
                "confidence": confidence,
            }
            if truncated:
                per_survey[survey_id][f"{scope}_{confidence}"] += 1
            else:
                per_survey[survey_id]["other"] += 1
        table[survey_id] = entries

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )

    # ---------------- report ----------------
    print(f"qsf piping: {qsf_stats['piping_clean']} clean, "
          f"{qsf_stats['piping_unresolved']} still carrying a placeholder "
          f"({qsf_stats['piping_unresolved_matrix_row']} matrix rows)\n")

    for scope, title in (("corpus", "items of data/items.parquet"),
                         ("catalogue", "catalogue items outside the corpus")):
        header = (f"{'survey_id':<18}{'trunc':>7}{'high':>7}{'medium':>8}"
                  f"{'rejected':>10}{'missing':>9}")
        print(f"\ntruncated question_text recovered — {title}")
        print(header)
        print("-" * len(header))
        total = Counter()
        for survey_id in ("ces_2019_online", "ces_2019_phone",
                          "ces_2021", "ces_2025"):
            c = per_survey[survey_id]
            row = {k: c[f"{scope}_{k}"] for k in
                   ("truncated", "high", "medium", "rejected", "missing")}
            total.update(row)
            print(f"{survey_id:<18}{row['truncated']:>7}{row['high']:>7}"
                  f"{row['medium']:>8}{row['rejected']:>10}{row['missing']:>9}")
        print("-" * len(header))
        print(f"{'TOTAL':<18}{total['truncated']:>7}{total['high']:>7}"
              f"{total['medium']:>8}{total['rejected']:>10}{total['missing']:>9}")

    print(f"\nnon-truncated CES items also recorded: "
          f"{sum(c['other'] for c in per_survey.values())}")

    print("\nrejected candidates (found in the source, but the Stata label "
          "does not match — no wording recorded):")
    for survey_id, entries in rejected.items():
        for variable, full in entries:
            print(f"  {survey_id}/{variable}: {full[:90]!r}")

    print("\nunmatched truncated variables:")
    for survey_id, variables in misses.items():
        print(f"  {survey_id}: {len(variables)} -> {sorted(variables)}")

    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
