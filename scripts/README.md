# Scripts

Project scripts (download, cleaning, orchestration).

## Download CES 2021

Script: `scripts/download_ces_2021.R`

Install R dependency once:

```r
install.packages("ces")
```

Run from project root:

```bash
Rscript scripts/download_ces_2021.R
```

Optional custom output directory:

```bash
Rscript scripts/download_ces_2021.R data/raw/ces_2021
```

## Convert CES 2021 codebook PDF to Markdown

Script: `scripts/extract_codebook_pdf_to_md.py`

Prerequisite (CLI dependency):

```bash
sudo apt-get install -y poppler-utils
```

Run from project root:

```bash
python3 scripts/extract_codebook_pdf_to_md.py
```

Outputs:

- `data/raw/ces_2021/ces_2021_codebook_questions.md`
- `data/raw/ces_2021/ces_2021_codebook_questions.jsonl`
