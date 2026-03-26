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

## Prepare CES 2021 data

Script: `scripts/02_prepare_data.py`

Filters Quebec respondents, selects relevant SES and attitude variables, applies value labels from Stata, adds thematic domain and train/test split assignments. Loads bilingual (EN/FR) question metadata from codebook.

Outputs:
- `data/processed/questions.parquet` — metadata for each question (EN + FR versions, thematic domain, split)
- `data/processed/respondents.parquet` — respondents × questions matrix (text labels), with survey_language column

Run from project root:

```bash
python scripts/02_prepare_data.py
```

## Generate question embeddings

Script: `scripts/03_generate_question_embeddings.py`

Run from the project root:

```bash
python scripts/03_generate_question_embeddings.py
```

The default model is `sentence-transformers/all-MiniLM-L12-v2` pinned to Hugging Face revision `936af83a2ecce5fe87a09109ff5cbcefe073173a` for reproducibility. Override with `--model` or `--revision` if needed.

## Simulate Condition 1: Cold LLM baseline

Script: `scripts/04_simulate_condition1.py`

Generates N samples per test question by prompting the LLM with no respondent context (cold baseline). Uses LiteLLM + OpenRouter for reproducibility and cost efficiency.

Configuration:
- Model: `openrouter/meta-llama/llama-3.1-70b-instruct` (override with `--model`)
- Samples per question: 500 (configurable with `--n-samples`)
- Temperature: 0.7 (for stochasticity)
- API key: Set `OPENROUTER_API_KEY` in `.env`

Run from project root:

```bash
python scripts/04_simulate_condition1.py --n-samples 500 --temperature 0.7
```

Output: `data/results/condition1_samples.parquet` — question × sample_idx → simulated response

## Build RAG similarity index

Script: `scripts/05_build_similarity_index.py`

Run from the project root:

```bash
python scripts/05_build_similarity_index.py --top-k 5
```

Default behavior builds nearest neighbors from `test` questions to `train` questions and writes `data/processed/rag_similarity.parquet`.
