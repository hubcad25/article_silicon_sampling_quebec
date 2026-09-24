"""Extract French question wording and option labels for the four CES surveys.

Why
---
The corpus stores CES items in English only, but 16 800 CES respondents
answered in French. To render each item in the language the respondent
actually answered in, we need, for every CES item of ``data/items.parquet``,
the French question wording and the French label of every response option,
keyed by the same numeric codes used in ``options``.

This script is extraction + diagnostic only: it reads ``data/items.parquet``
(never writes it), reads French sources already on disk (CES codebook PDFs
cached as text under ``data/raw/ces_wording_cache``, and the ces_2021
Qualtrics ``.qsf`` files and ces_2019_phone questionnaire PDFs in the
sibling read-only repo ``../mvp_moteur_recherche_sondages``), and writes:

- ``data/ces_french_wording.json``
- ``data/ces_french_wording_coverage.csv``

Sources
-------
``ces_2021``          the two Qualtrics ``.qsf`` files. Best source: no
                      length cap, ``Language["FR-CA"]`` holds French
                      QuestionText/Choices/Answers keyed by the same choice
                      ids as the English payload, and ``RecodeValues`` maps
                      a raw choice id to the code actually stored in the
                      ``.dta`` (== the code in ``items.parquet.options``).
``ces_2019_online``,
``ces_2025``          codebook text caches: the English question+options
                      block, then the French translation under the *same*
                      variable name. A simple question prints its options as
                      ``   o Label (code)``; a battery ("grid") question
                      prints a stem once, then a column-header row of answer
                      labels wrapped over several lines and offset above a
                      row of radio-button columns, and then one row per
                      sub-item, its sub-variable ``(tag_n)`` printed under a
                      row label truncated to the row's own column span.
``ces_2019_phone``    the French CATI questionnaire PDFs (`pdftotext
                      -layout`); a question is ``Q<n>``/`P<n>`` alone on a
                      line, wording follows, options are
                      ``<glyph> <code>  <label>`` (code is a *prefix* here,
                      unlike the online codebooks). -8/-9 (Refused/Don't
                      know) print as their own lines and get a French label
                      like every other code.

Nothing is ever written to the sibling repo. A wording or option label that
cannot be located with a reasonable confidence is left absent rather than
guessed.

Run::

    .venv/bin/python scripts/17_extract_ces_french_wording.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from article_silicon_sampling_quebec.prompts import guess_language  # noqa: E402

PRODUCT_DATA = (REPO_ROOT.parent / "mvp_moteur_recherche_sondages" / "data").resolve()
CACHE_DIR = REPO_ROOT / "data" / "raw" / "ces_wording_cache"
ITEMS_PARQUET = REPO_ROOT / "data" / "items.parquet"
HELDOUT_JSON = REPO_ROOT / "data" / "split" / "heldout_items.json"
OUT_JSON = REPO_ROOT / "data" / "ces_french_wording.json"
OUT_CSV = REPO_ROOT / "data" / "ces_french_wording_coverage.csv"

# --------------------------------------------------------------------------
# reuse the well-tested text/PDF helpers of script 09 (import only)
# --------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "_w09", REPO_ROOT / "scripts" / "09_extract_ces_full_wording.py"
)
w09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w09)

clean = w09.clean
pdf_text_cached = w09.pdf_text
_var_pattern = w09._var_pattern
_MATRIX_ROW_RE = w09._MATRIX_ROW_RE
_STOP_PREFIXES = w09._STOP_PREFIXES
_DISPLAY_LOGIC_RE = w09._DISPLAY_LOGIC_RE
resolve_piping = w09.resolve_piping
_flow_embedded_values = w09._flow_embedded_values
ordered_choice_keys = w09.ordered_choice_keys

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
CODEBOOK_PREFIXES = {
    "ces_2019_online": ("cps19", "pes19", "pes10"),
    "ces_2025": ("cps25", "pes25", "cses_module6"),
}
PHONE_PDFS_FR = [
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Campaign Period Survey FR v2.pdf",
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Post-election Survey FR.pdf",
    PRODUCT_DATA / "ces_2019_phone"
    / "2019 Canadian Election Study - Phone Survey Post-election Survey (web) FR.pdf",
]


def pdf_text(path: Path) -> str:
    """Like w09.pdf_text but tolerant of a cache dir that predates this
    script (falls back to running pdftotext directly if the cached .txt for
    a FR phone PDF is missing)."""
    return pdf_text_cached(path)


# --------------------------------------------------------------------------
# source 1 -- Qualtrics .qsf (ces_2021), French side
# --------------------------------------------------------------------------

def extract_qsf_fr() -> dict[str, dict]:
    """(variable -> {question_text_fr, options_fr}) for ces_2021."""
    out: dict[str, dict] = {}
    for path in QSF_FILES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        flow = next((e for e in doc["SurveyElements"] if e["Element"] == "FL"), None)
        field_values = _flow_embedded_values(flow["Payload"]) if flow else {}

        for element in doc["SurveyElements"]:
            if element["Element"] != "SQ":
                continue
            payload = element["Payload"]
            tag = payload.get("DataExportTag")
            if not tag:
                continue
            # a handful of DataExportTags in the source .qsf carry a stray
            # space ("cps21_ candidateref") that becomes a second underscore
            # in the .dta / items.parquet variable name.
            tag = tag.replace(" ", "_")
            lang_fr = (payload.get("Language") or {}).get("FR-CA") or {}
            if not lang_fr:
                continue
            recode = payload.get("RecodeValues") or {}

            def to_code(raw_key: str) -> int | str:
                mapped = recode.get(raw_key, raw_key)
                try:
                    return int(mapped)
                except (TypeError, ValueError):
                    return mapped

            stem_raw = lang_fr.get("QuestionText") or ""
            stem_raw, piped = resolve_piping(stem_raw, field_values)
            stem = clean(stem_raw)
            if not stem:
                continue

            qtype = payload.get("QuestionType")
            if qtype in ("Matrix", "Slider"):
                answers_fr = lang_fr.get("Answers") or {}
                choices_fr = lang_fr.get("Choices") or {}
                options_fr = {}
                for k, v in answers_fr.items():
                    text, _ = resolve_piping(v.get("Display") or "", field_values)
                    text = clean(text)
                    if text:
                        options_fr[to_code(k)] = text
                for position, key in enumerate(ordered_choice_keys(payload), 1):
                    choice = choices_fr.get(key) or {}
                    row_raw, row_piped = resolve_piping(
                        choice.get("Display") or "", field_values
                    )
                    row = clean(row_raw)
                    if not row:
                        continue
                    out[f"{tag}_{position}"] = {
                        "question_text_fr": f"{stem} - {row}",
                        "options_fr": {str(c): lab for c, lab in options_fr.items()},
                        "source": "qsf",
                        "piping_unresolved": bool(piped or row_piped),
                    }
            else:
                choices_fr = lang_fr.get("Choices") or {}
                options_fr = {}
                for k, v in choices_fr.items():
                    text, _ = resolve_piping(v.get("Display") or "", field_values)
                    text = clean(text)
                    if text:
                        options_fr[to_code(k)] = text
                out[tag] = {
                    "question_text_fr": stem,
                    "options_fr": {str(c): lab for c, lab in options_fr.items()},
                    "source": "qsf",
                    "piping_unresolved": piped,
                }
    return out


# --------------------------------------------------------------------------
# source 2 -- CES online codebook PDFs (ces_2019_online, ces_2025)
# --------------------------------------------------------------------------

_OPTION_START_RE = re.compile(r"^\s*(?:o|❍|❑|●)\s+(\S.*)$")
#: the code paren at the end of an option line -- allowing a
#: "(code): some_open_text_field_name" suffix on a "please specify" option.
_TRAILING_CODE_RE = re.compile(r"\((-?\d+)\)(?:\s*:\s*\S+)?\s*$")
#: an "Other (please specify) (7) ______________" free-text fill-in blank
#: after the code paren, with no ":" -- strip it before testing for the
#: trailing code so it does not defeat the match above.
_TRAILING_BLANK_RE = re.compile(r"[\s_]+$")
_LEADING_GLYPH_RE = re.compile(r"^[⊗✓•]+\s*")
_RADIO_MARK_RE = re.compile(r"(?<!\S)o(?!\S)")


def parse_simple_options(lines: list[str], var_re: re.Pattern) -> dict[int, str]:
    """``   o Label (code)`` lines, wrapping across up to a few lines."""
    options: dict[int, str] = {}
    cur: list[str] | None = None
    for line in lines:
        if not line.strip():
            cur = None
            continue
        m = _OPTION_START_RE.match(line)
        if m:
            cur = [m.group(1)]
        elif cur is not None and not var_re.match(line) and not _MATRIX_ROW_RE.match(line):
            cur.append(line.strip())
        else:
            cur = None
            continue
        text = _TRAILING_BLANK_RE.sub("", " ".join(cur))
        code_m = _TRAILING_CODE_RE.search(text)
        if code_m:
            code = int(code_m.group(1))
            label = _TRAILING_CODE_RE.sub("", text).strip()
            label = _LEADING_GLYPH_RE.sub("", label).strip()
            if label:
                options[code] = clean(label)
            cur = None
    return options


def _col_positions(marker_line: str) -> list[int]:
    return [m.start() for m in _RADIO_MARK_RE.finditer(marker_line)]


def parse_grid_header(header_lines: list[str], marker_line: str) -> dict[int, str] | None:
    """Column headers of a battery grid -> {code: label}, from the FR block.

    Each answer column is identified by the x-position of its radio glyph on
    ``marker_line`` (a row of ``(tag_n)  o  o  o ...``). Header words above it
    wrap over several lines; every non-space run on those lines is assigned to
    its nearest column by x-position and concatenated top-to-bottom, which is
    the header's own reading order. The trailing ``(code)`` printed at the end
    of each column's header settles the code -- never the column's left-right
    position alone, so a battery whose columns are not code-ordered is not
    silently mis-paired.
    """
    cols = _col_positions(marker_line)
    if len(cols) < 2:
        return None
    runs: list[tuple[int, int, int, str]] = []
    for li, line in enumerate(header_lines):
        for m in re.finditer(r"\S+", line):
            # the token's *centre*, not its left edge: columns can sit as
            # little as 2 chars apart (narrow French words), and a run's
            # start is closer to the column on its left almost by
            # construction, which biases every tie there.
            center = m.start() + len(m.group()) // 2
            runs.append((center, m.start(), li, m.group()))
    if not runs:
        return None
    buckets: dict[int, list[tuple[int, int, str]]] = {c: [] for c in cols}
    for center, pos, li, tok in runs:
        nearest = min(cols, key=lambda c: abs(c - center))
        buckets[nearest].append((li, pos, tok))

    # Note: a narrow column can word-wrap a single word with no hyphen (e.g.
    # "Beauc" / "oup" for "Beaucoup"), which this join renders as two space-
    # separated fragments instead of one word. Guessing which adjacent
    # fragment pairs are really one split word (vs. two real short words,
    # e.g. "Un peu") needs a lexicon this script does not have; gluing
    # fragments together on a length heuristic alone was tried and produced
    # worse damage (real words merged, e.g. "Parti" + "libéral" ->
    # "Partilibéral") than it fixed, so labels are left as space-joined
    # fragments and this artifact is reported as a known limitation instead.
    result: dict[int, str] = {}
    for c in cols:
        toks = [t for _, _, t in sorted(buckets[c])]
        joined = " ".join(toks)
        code_m = _TRAILING_CODE_RE.search(joined)
        if not code_m:
            continue
        code = int(code_m.group(1))
        label = _TRAILING_CODE_RE.sub("", joined).strip()
        if label:
            result[code] = clean(label)
    return result or None


def extract_codebook_fr(text: str, prefixes: tuple[str, ...]) -> dict[str, dict]:
    """(variable -> {text, options}) candidates, FR-selected, from a codebook.

    Structurally mirrors ``w09.extract_codebook`` (stem detection, matrix row
    detection) but also captures the option list / grid header of each
    occurrence, and keeps only the candidate whose stem is judged French by
    :func:`guess_language` (ties broken by length, longest wins).
    """
    var_re = _var_pattern(prefixes)
    alone_re = re.compile(rf" {{0,3}}((?:{'|'.join(prefixes)})_[A-Za-z0-9_]+)\s*$")
    lines = text.split("\n")
    stems: dict[str, str] = {}
    grid_options: dict[str, dict[int, str]] = {}
    candidates: dict[str, list[dict]] = defaultdict(list)
    current_tag: str | None = None
    current_line = -1

    def is_boundary(idx: int) -> bool:
        return bool(var_re.match(lines[idx]) or alone_re.match(lines[idx]))

    def process_stem(tag: str, body: list[str], j: int, at: int) -> int:
        """``body`` already holds the stem's first fragment; ``j`` is the next
        unread line. Consumes the wrapped stem and its option block/grid,
        records a candidate, returns the line index to resume scanning at."""
        nonlocal current_tag, current_line
        while j < n:
            nxt = lines[j]
            if not nxt.strip():
                break
            if is_boundary(j) or _MATRIX_ROW_RE.match(nxt):
                break
            if _OPTION_START_RE.match(nxt):
                break
            if nxt.lstrip().startswith(_STOP_PREFIXES):
                break
            if _DISPLAY_LOGIC_RE.search(nxt):
                break
            if nxt.strip().startswith("_____"):
                break
            if len(nxt) - len(nxt.lstrip()) > 4:
                break
            body.append(nxt.strip())
            j += 1
        stem = clean(" ".join(body))
        if not stem:
            return j
        stems[tag] = stem
        current_tag, current_line = tag, at

        options: dict[int, str] | None = None
        k = j
        while k < n and not lines[k].strip():
            k += 1
        # a simple option list can be interleaved with "Display This
        # Question" branch conditions (cps19_2nd_choice repeats one option's
        # display logic before each "o Label (code)" line), so the window is
        # bounded by the next variable boundary, not by whether the *very
        # first* following line looks like an option.
        window_end = min(n, j + 400)
        end = j
        while end < window_end and not is_boundary(end):
            end += 1
        simple_opts = parse_simple_options(lines[j:end], var_re)
        if simple_opts:
            options = simple_opts
        else:
            r = k
            found_row = -1
            while r < window_end:
                if is_boundary(r) and lines[r].strip() != lines[at].strip():
                    break
                m2 = _MATRIX_ROW_RE.match(lines[r])
                if m2 and m2.group(2).startswith(tag + "_"):
                    found_row = r
                    break
                r += 1
            if found_row >= 0:
                idx = found_row - 1
                while idx > j and lines[idx].strip():
                    idx -= 1
                header_lines = lines[j:idx]
                options = parse_grid_header(header_lines, lines[found_row])
                if options:
                    grid_options[tag] = options
        candidates[tag].append({"text": stem, "options": options})
        return j

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        match = var_re.match(line)
        if match:
            tag, gap, rest = match.group(1), match.group(2), match.group(3)
            if rest.startswith((",", ".", ";", ")", "=")):
                i += 1
                continue
            if re.fullmatch(r"[A-Za-z0-9_]+", rest.strip()):
                i += 1
                continue
            j = process_stem(tag, [rest.strip()], i + 1, i)
            i = j if j > i else i + 1
            continue

        alone = alone_re.match(line) if not var_re.match(line) else None
        if alone:
            tag = alone.group(1)
            j = i + 1
            while j < n and (not lines[j].strip() or lines[j].strip().isdigit()):
                j += 1
            if j < n and lines[j].strip() and not is_boundary(j) and not _OPTION_START_RE.match(lines[j]):
                jj = process_stem(tag, [], j, i)
                if jj > j:
                    i = jj
                    continue
            i += 1
            continue

        row = _MATRIX_ROW_RE.match(line)
        if row and current_tag and row.group(2).startswith(current_tag + "_"):
            sub = row.group(2)
            start = line.index("(" + sub + ")")
            width = max(_end_of_left_column(lines, i), start + len(sub) + 4)
            label_lines: list[str] = []
            k = i - 1
            while k > current_line:
                prev = lines[k]
                if _MATRIX_ROW_RE.match(prev) or var_re.match(prev):
                    break
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
                opts = grid_options.get(current_tag)
                candidates[sub].append({
                    "text": f"{stems[current_tag]} - {label}",
                    "options": opts,
                })
        i += 1

    out: dict[str, dict] = {}
    for tag, cands in candidates.items():
        fr_cands = [c for c in cands if guess_language(c["text"]) == "fr"]
        pool = fr_cands or cands  # keep something even if language guess abstains
        best = max(pool, key=lambda c: len(c["text"]))
        opt_source = best["options"]
        if opt_source is None:
            with_opts = [c for c in pool if c["options"]]
            if with_opts:
                opt_source = max(with_opts, key=lambda c: len(c["options"])).get("options")
        out[tag] = {
            "question_text_fr": best["text"],
            "options_fr": ({str(c): lab for c, lab in opt_source.items()}
                            if opt_source else {}),
            "source": "codebook",
        }
    return out


def _end_of_left_column(lines: list[str], at: int) -> int:
    for offset in range(0, 6):
        for k in (at - offset, at + offset):
            if 0 <= k < len(lines):
                marks = [m.start() for m in _RADIO_MARK_RE.finditer(lines[k])]
                if len(marks) >= 3:
                    return marks[0]
    return 0


# --------------------------------------------------------------------------
# source 3 -- CES 2019 phone CATI questionnaire (French PDFs)
# --------------------------------------------------------------------------

_PHONE_HEAD_RE = re.compile(r"^\s*([QP])(\d+)(?:\s+.*)?\s*$")
_PHONE_OPTION_RE = re.compile(r"^\s*[❍❑]\s*(-?\d+)\s+(.+?)\s*(?:\*.*)?$")
_PHONE_MULTI_MARK_RE = re.compile(r"❍\s*(-?\d+)")
_PHONE_SUBITEM_RE = re.compile(r"^\s*(\d+)\.\s+(\S.*?)\s*(?:\*.*)?$")
_PHONE_DIRECTIVE_RES = tuple(re.compile(pattern, re.I) for pattern in (
    r"\s*Repeat scale if needed\.?\s*$", r"\s*Répétez l.échelle.*$",
    r"\s*probe for one response only\.?\s*$", r"\s*sondez pour une seule réponse\.?\s*$",
    r"\s*name, enter party name\..*$",
    r"\s*\(Show if[^)]*\)\s*$", r"\s*\(Montrer si[^)]*\)\s*$",
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


_WIDE_GAP_RE = re.compile(r"\S(\s{6,})\S")


def _phone_endpoint_labels(lines: list[str], marker_idx: int) -> tuple[str, str, int]:
    """The two end-anchor labels of a 0..N phone scale.

    They sit on one or more lines directly above the row of bare ``❍ <code>``
    marks, as two clusters separated by a wide gap (themselves possibly
    wrapped over several lines, one fragment per cluster per line). Returns
    ``(left_text, right_text, n_lines_consumed)`` -- the caller pops that many
    already-collected lines back out of the question's running ``body``.
    """
    block: list[str] = []
    idx = marker_idx - 1
    saw_gap = False
    blanks_seen = 0
    while idx >= 0 and len(block) < 6:
        ln = lines[idx]
        if not ln.strip():
            blanks_seen += 1
            if blanks_seen > 1 or block:
                break
            idx -= 1
            continue
        has_gap = bool(_WIDE_GAP_RE.search(ln))
        short = len(ln.strip()) <= 35
        if has_gap:
            saw_gap = True
        elif not short:
            break
        block.insert(0, ln)
        idx -= 1
    if not saw_gap or not block:
        return "", "", 0

    gap_line = max(block, key=lambda ln: max(
        (len(m.group(1)) for m in _WIDE_GAP_RE.finditer(ln)), default=0))
    gap_match = max(_WIDE_GAP_RE.finditer(gap_line), key=lambda m: len(m.group(1)))
    boundary = gap_match.start(1) + len(gap_match.group(1)) // 2

    left_parts, right_parts = [], []
    for ln in block:
        for tok_m in re.finditer(r"\S+", ln):
            pos = tok_m.start()
            (left_parts if pos < boundary else right_parts).append(tok_m.group())
    left_text = clean(" ".join(left_parts))
    right_text = clean(" ".join(right_parts))
    return left_text, right_text, len(block)


def extract_phone_fr(texts: list[str]) -> dict[str, dict]:
    """(variable -> {question_text_fr, options_fr}) from the FR CATI PDFs."""
    out: dict[str, dict] = {}
    for text in texts:
        lines = text.split("\n")
        n = len(lines)
        i = 0
        while i < n:
            head = _PHONE_HEAD_RE.match(lines[i])
            if not head:
                i += 1
                continue
            var = f"{head.group(1).lower()}{head.group(2)}"
            body: list[str] = []
            subs: list[str] = []
            options: dict[int, str] = {}
            j = i + 1
            while j < n:
                nxt = lines[j]
                if _PHONE_HEAD_RE.match(nxt):
                    break
                stripped = nxt.strip()
                if not stripped or stripped.isdigit():
                    j += 1
                    continue
                multi = _PHONE_MULTI_MARK_RE.findall(nxt)
                if len(multi) >= 3:
                    # a numeric anchor scale ("0 ... 10"): one line carries
                    # every code as a bare glyph+number, no per-code text.
                    # Only the two end codes are ever labelled, on the plain
                    # text line(s) just above (two clusters separated by a
                    # wide gap, themselves possibly wrapped); the interior
                    # codes are the number itself -- no translation needed.
                    codes = [int(c) for c in multi]
                    left_text, right_text, n_consumed = _phone_endpoint_labels(lines, j)
                    if n_consumed:
                        del body[len(body) - n_consumed:]
                    if left_text:
                        options[codes[0]] = left_text
                    if right_text:
                        options[codes[-1]] = right_text
                    for c in codes[1:-1]:
                        options.setdefault(c, str(c))
                    j += 1
                    continue
                opt = _PHONE_OPTION_RE.match(nxt)
                if opt:
                    code = int(opt.group(1))
                    label = clean(opt.group(2))
                    if label:
                        options[code] = label
                    j += 1
                    continue
                sub = _PHONE_SUBITEM_RE.match(nxt)
                if sub and body:
                    subs.append(clean(sub.group(2)).rstrip(" *"))
                    j += 1
                    continue
                if re.match(r"^[QP]\d+\b", stripped):
                    break
                if stripped.startswith(("Minimum:", "Levels marked", "Do Not Read",
                                        "Do not read", "Read list", "If needed:",
                                        "Interviewer", "PROGRAMMER", "Ne pas lire",
                                        "Lire la liste", "INTERVIEWEUR")):
                    j += 1
                    continue
                if options or subs:
                    j += 1
                    continue
                body.append(stripped)
                j += 1
            stem = strip_cati_directives(clean(" ".join(body)))
            if stem:
                out[var] = {
                    "question_text_fr": stem,
                    "options_fr": {str(c): lab for c, lab in options.items()},
                    "source": "pdf",
                }
                for m, sub_label in enumerate(subs):
                    letter = chr(ord("a") + m)
                    out[f"{var}_{letter}"] = {
                        "question_text_fr": f"{sub_label}: {stem}",
                        "options_fr": {str(c): lab for c, lab in options.items()},
                        "source": "pdf",
                    }
            i = j
    return out


# --------------------------------------------------------------------------
# assembly, validation, coverage
# --------------------------------------------------------------------------

def load_items() -> pl.DataFrame:
    df = pl.read_parquet(ITEMS_PARQUET)
    return df.filter(pl.col("survey_id").str.starts_with("ces_"))


def load_test_keys() -> set[tuple[str, str]]:
    if not HELDOUT_JSON.exists():
        return set()
    data = json.loads(HELDOUT_JSON.read_text(encoding="utf-8"))
    return {(it["survey_id"], it["variable"]) for it in data.get("items", [])}


def build_table(items: pl.DataFrame, sources: dict[str, dict[str, dict]]) -> dict:
    table: dict[str, dict] = {sid: {} for sid in sources}
    coverage_rows = []
    test_keys = load_test_keys()

    for row in items.iter_rows(named=True):
        survey_id, variable = row["survey_id"], row["variable"]
        src = sources.get(survey_id, {})
        entry = src.get(variable)
        options_en = json.loads(row["options"]) if isinstance(row["options"], str) else row["options"]
        codes_en = {int(o["code"]) for o in options_en} if options_en else set()

        if entry is None:
            status = "missing"
            n_matched = 0
            missing_codes = sorted(codes_en)
            table[survey_id][variable] = {
                "question_text_fr": None, "options_fr": {}, "source": None,
                "status": status, "notes": "no FR candidate found",
            }
        else:
            options_fr = dict(entry.get("options_fr") or {})
            notes = []
            if survey_id == "ces_2019_phone":
                # -8 (Refused) / -9 (Don't know) are standard CES phone
                # missing-data codes: the label is fixed ("Refus" /
                # "Ne sais pas", verified verbatim across dozens of
                # questions) but the two codes are not always the ones
                # printed next to *this* question (the .dta's own
                # code_map can route a question-specific "don't know
                # enough" choice into -8/-9 instead) -- so when the
                # printed PDF gave us no label for them, fall back to the
                # fixed text rather than reporting them missing.
                fallback = {"-8": "Refus", "-9": "Ne sais pas"}
                for code, label in fallback.items():
                    if int(code) in codes_en and code not in options_fr:
                        options_fr[code] = label
                        notes.append(f"code {code} label from fixed CES phone "
                                     f"convention, not this item's own PDF block")
            fr_codes = {int(k) for k in options_fr}
            missing_codes = sorted(codes_en - fr_codes)
            n_matched = len(codes_en) - len(missing_codes)
            qtext = entry.get("question_text_fr") or ""
            lang = guess_language(qtext)
            if lang == "en":
                notes.append("question_text_fr looks English")
            if not qtext:
                notes.append("no question_text_fr")
            if missing_codes:
                notes.append(f"missing FR label for codes {missing_codes}")
            if not qtext or missing_codes or lang == "en":
                status = "wording_only" if qtext and lang != "en" else (
                    "missing" if not qtext else "wording_only")
            else:
                status = "complete"
            table[survey_id][variable] = {
                "question_text_fr": qtext or None,
                "options_fr": options_fr,
                "source": entry.get("source"),
                "status": status,
                "notes": "; ".join(notes),
            }

        coverage_rows.append({
            "survey_id": survey_id,
            "variable": variable,
            "is_test_item": (survey_id, variable) in test_keys,
            "is_context_only": row["is_context_only"],
            "status": table[survey_id][variable]["status"],
            "n_options_en": len(codes_en),
            "n_options_fr_matched": n_matched,
            "missing_codes": ",".join(str(c) for c in missing_codes),
            "source": table[survey_id][variable]["source"] or "",
        })
    return table, coverage_rows


def main() -> None:
    items = load_items()

    print("extracting ces_2021 from .qsf ...")
    ces2021 = extract_qsf_fr()

    print("extracting ces_2019_online from codebook cache ...")
    online_text = pdf_text(CODEBOOK_PDFS["ces_2019_online"][0])
    ces2019_online = extract_codebook_fr(online_text, CODEBOOK_PREFIXES["ces_2019_online"])

    print("extracting ces_2025 from codebook cache ...")
    c2025_text = pdf_text(CODEBOOK_PDFS["ces_2025"][0])
    ces2025 = extract_codebook_fr(c2025_text, CODEBOOK_PREFIXES["ces_2025"])

    print("extracting ces_2019_phone from FR CATI PDFs ...")
    phone_texts = [pdf_text(p) for p in PHONE_PDFS_FR]
    ces2019_phone = extract_phone_fr(phone_texts)

    sources = {
        "ces_2021": ces2021,
        "ces_2019_online": ces2019_online,
        "ces_2025": ces2025,
        "ces_2019_phone": ces2019_phone,
    }

    table, coverage_rows = build_table(items, sources)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True),
                        encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        # QUOTE_ALL: missing_codes is a comma-joined list ("1,2,3") that CSV
        # readers with default quoting only wrap in quotes when it actually
        # contains a comma, so a column that is sometimes "8" (no quotes,
        # read back as int) and sometimes "1,2,3" (quoted, read back as str)
        # trips a naive reader's type inference. Quoting every field keeps
        # the column's dtype consistent regardless of value.
        writer = csv.DictWriter(f, fieldnames=[
            "survey_id", "variable", "is_test_item", "is_context_only", "status",
            "n_options_en", "n_options_fr_matched", "missing_codes", "source",
        ], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(coverage_rows)

    # ---------------- report ----------------
    import polars as pl2
    cov = pl2.DataFrame(coverage_rows)
    print("\ncoverage per survey:")
    print(cov.group_by("survey_id").agg(
        pl2.col("status").eq("complete").sum().alias("complete"),
        pl2.col("status").eq("wording_only").sum().alias("wording_only"),
        pl2.col("status").eq("missing").sum().alias("missing"),
        pl2.len().alias("n"),
    ).sort("survey_id"))

    test_cov = cov.filter(pl2.col("is_test_item"))
    print(f"\ntest items (n={len(test_cov)}):")
    print(test_cov.group_by("status").agg(pl2.len().alias("n")))
    print("\ntest item detail:")
    for r in test_cov.sort(["survey_id", "variable"]).iter_rows(named=True):
        print(f"  {r['survey_id']:<18}{r['variable']:<22}{r['status']:<14}"
              f"matched {r['n_options_fr_matched']}/{r['n_options_en']}"
              f"{'  missing=' + r['missing_codes'] if r['missing_codes'] else ''}")

    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
