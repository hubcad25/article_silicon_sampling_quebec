#!/usr/bin/env python3
"""Extract CES 2021 codebook questions from PDF to Markdown/JSONL.

This script uses `pdftotext` (Poppler) to convert the PDF, then applies
heuristics to recover variable blocks with labels/questions and coded options.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SECTION_CPS = "Campaign Period Survey Questions"
SECTION_PES = "Post-Election Suvey Questions"

VAR_LINE_RE = re.compile(
    r"^\s*((?:cps21|pes21)_[A-Za-z0-9_]+(?:_[A-Za-z0-9_]+)*)\b(?:\s+(.*\S))?\s*$"
)
OPTION_LINE_RE = re.compile(r"^\s*o\s+(.+\S)\s*$")
CODED_OPTION_RE = re.compile(r"^\s*(.+?)\s*\((-?\d+)\)\s*$")

SKIP_PROMPT_PREFIXES = (
    "If ",
    "Note:",
    "Skip to ",
    "Project Title:",
    "Principal Investigator:",
    "Co-Investigators:",
)


@dataclass
class Entry:
    section: str
    variable: str
    label: str
    question: str
    options: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CES codebook questions/labels from PDF into Markdown and JSONL."
    )
    parser.add_argument(
        "--input-pdf",
        default="data/raw/ces_2021/ces_2021_codebook.pdf",
        help="Path to codebook PDF.",
    )
    parser.add_argument(
        "--output-md",
        default="data/raw/ces_2021/ces_2021_codebook_questions.md",
        help="Output Markdown file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/raw/ces_2021/ces_2021_codebook_questions.jsonl",
        help="Output JSONL file.",
    )
    return parser.parse_args()


def ensure_pdftotext() -> None:
    if shutil.which("pdftotext") is None:
        raise SystemExit(
            "Missing dependency: `pdftotext` (Poppler). Install poppler-utils and retry."
        )


def pdf_to_lines(pdf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.replace("\f", "").rstrip() for line in proc.stdout.splitlines()]


def find_section_bounds(lines: list[str]) -> tuple[int, int]:
    cps_indices = [i for i, line in enumerate(lines) if SECTION_CPS in line]
    pes_indices = [i for i, line in enumerate(lines) if SECTION_PES in line]
    if not cps_indices or not pes_indices:
        raise SystemExit("Could not find CPS/PES question sections in the extracted text.")
    start = cps_indices[-1]
    split = pes_indices[-1]
    if split <= start:
        raise SystemExit("Unexpected section order in extracted codebook text.")
    return start, split


def is_heading_candidate(line: str) -> tuple[bool, str, str]:
    if not line or "|" in line:
        return False, "", ""
    match = VAR_LINE_RE.match(line)
    if not match:
        return False, "", ""
    variable = match.group(1)
    remainder = (match.group(2) or "").strip()

    if remainder:
        starts_ok = remainder[0].isupper() or remainder[0].isdigit() or remainder[0] in {'"', "'", "[", "("}
        if not starts_ok:
            return False, "", ""

    return True, variable, remainder


def compact(text: str, max_len: int = 480) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def filter_prompt_line(line: str) -> bool:
    if not line:
        return False
    if line in {"Timing", "First Click", "Last Click", "Page Submit", "Click Count"}:
        return False
    if line.isdigit():
        return False
    if VAR_LINE_RE.match(line):
        return False
    if any(line.startswith(prefix) for prefix in SKIP_PROMPT_PREFIXES):
        return False
    return True


def finalize_entry(section: str, variable: str, remainder: str, block: list[str]) -> Entry | None:
    label = remainder
    options: list[str] = []
    prompt_lines: list[str] = []

    for line in block:
        opt_match = OPTION_LINE_RE.match(line)
        if opt_match:
            options.append(compact(opt_match.group(1), max_len=200))
            continue

        coded_match = CODED_OPTION_RE.match(line)
        if coded_match and len(line) <= 180:
            option_text = f"{coded_match.group(1).strip()} ({coded_match.group(2)})"
            if options and not re.search(r"\(-?\d+\)$", options[-1]):
                options[-1] = compact(f"{options[-1]} {option_text}", max_len=200)
                continue
            options.append(compact(option_text, max_len=200))
            continue

        if filter_prompt_line(line):
            prompt_lines.append(line)

    if not label and prompt_lines:
        label = prompt_lines[0]
        prompt_lines = prompt_lines[1:]

    question = compact(" ".join(prompt_lines[:4])) if prompt_lines else ""
    label = compact(label, max_len=220)

    if not label and not question and not options:
        return None

    return Entry(section=section, variable=variable, label=label, question=question, options=options)


def parse_entries(lines: list[str], start: int, split: int) -> list[Entry]:
    entries: list[Entry] = []
    i = start

    while i < len(lines):
        line = lines[i].strip()
        is_heading, variable, remainder = is_heading_candidate(line)
        if not is_heading:
            i += 1
            continue

        section = "CPS" if i < split else "PES"
        j = i + 1
        block: list[str] = []
        while j < len(lines):
            nxt = lines[j].strip()
            next_is_heading, _, _ = is_heading_candidate(nxt)
            if next_is_heading:
                break
            if nxt:
                block.append(nxt)
            j += 1

        entry = finalize_entry(section=section, variable=variable, remainder=remainder, block=block)
        if entry is not None:
            entries.append(entry)

        i = j

    return entries


def write_markdown(entries: list[Entry], input_pdf: Path, output_md: Path) -> None:
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cps_count = sum(1 for entry in entries if entry.section == "CPS")
    pes_count = sum(1 for entry in entries if entry.section == "PES")

    lines: list[str] = []
    lines.append("# CES 2021 Codebook (PDF -> Markdown)")
    lines.append("")
    lines.append(f"- Source PDF: `{input_pdf}`")
    lines.append(f"- Generated (UTC): `{generated}`")
    lines.append(f"- Variables extracted: `{len(entries)}` (`CPS={cps_count}`, `PES={pes_count}`)")
    lines.append("")

    current_section = ""
    for entry in entries:
        if entry.section != current_section:
            current_section = entry.section
            section_title = "Campaign Period Survey (CPS)" if current_section == "CPS" else "Post-Election Survey (PES)"
            lines.append(f"## {section_title}")
            lines.append("")

        lines.append(f"### `{entry.variable}`")
        if entry.label:
            lines.append(f"- Label: {entry.label}")
        if entry.question:
            lines.append(f"- Question: {entry.question}")
        if entry.options:
            lines.append("- Options:")
            for option in entry.options:
                lines.append(f"  - {option}")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(entries: list[Entry], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "section": entry.section,
                        "variable": entry.variable,
                        "label": entry.label,
                        "question": entry.question,
                        "options": entry.options,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    input_pdf = Path(args.input_pdf)
    output_md = Path(args.output_md)
    output_jsonl = Path(args.output_jsonl)

    if not input_pdf.exists():
        raise SystemExit(f"Input PDF not found: {input_pdf}")

    ensure_pdftotext()
    lines = pdf_to_lines(input_pdf)
    start, split = find_section_bounds(lines)
    entries = parse_entries(lines, start=start, split=split)

    if not entries:
        raise SystemExit("No codebook entries parsed. Check extraction heuristics.")

    write_markdown(entries, input_pdf=input_pdf, output_md=output_md)
    write_jsonl(entries, output_jsonl=output_jsonl)

    print(f"Wrote {len(entries)} entries to {output_md} and {output_jsonl}")


if __name__ == "__main__":
    main()
