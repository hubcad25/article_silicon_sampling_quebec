# article_silicon_sampling_quebec

Working repo for an academic article on silicon sampling applied to Quebec public opinion data.

## Research question

Two mechanisms for LLM-based opinion simulation are compared: retrieval-augmented generation (RAG), which conditions a general-purpose model on empirically similar survey responses at inference time, and supervised fine-tuning (SFT), which trains a model to internalize population-level opinion patterns. We evaluate which mechanism — alone or combined — best reproduces observed response distributions in a held-out validation design applied to Quebec public opinion data.

## Positioning relative to Argyle et al. (2023)

Argyle et al. (*Political Analysis*, 2023) demonstrate that GPT-3, when conditioned on sociodemographic backstories, reproduces human response distributions with high algorithmic fidelity. Their study is limited to U.S. politics, English-language context, and uses the LLM "cold" (no empirical prior, no fine-tuning).

This article extends their framework in three ways:

1. **Empirical priors via RAG** — instead of conditioning the LLM on demographics alone, we inject real responses from semantically similar survey questions before simulation. We test whether semantic proximity of injected questions moderates predictive accuracy.

2. **Fine-tuning as an alternative mechanism** — we fine-tune an open-source LLM on a thematic subset of CES questions and test generalization to held-out thematic domains. Fine-tuned weights are published for reproducibility (HuggingFace).

3. **Non-American, francophone validation** — we validate against public Quebec/Canadian Election Studies (CES 2021) data, filling an explicit gap acknowledged by Argyle et al.

> **Note on media context**: injecting real-time news as a conditioning layer was considered but excluded from the core design. With a single survey wave (CES 2021), there is no temporal variation to validate against. Media context is flagged as a natural extension for future work using multi-wave data.

## Experimental design

The study follows a **2 × 4 × 4 factorial design** for the fine-tuning (SFT) conditions, compared against 3 baseline conditions.

### Baselines (No Fine-Tuning)
| # | Condition | Context at inference | Target |
|---|---|---|---|
| 1 | LLM Cold | None (Zero-shot) | Baseline baseline |
| 2 | LLM + Demographics | SES profile only | Argyle et al. (2023) replication |
| 3 | LLM + Demographics + RAG | SES profile + Retrieved Q&A | Extension of Argyle (Empirical Prior) |

### SFT Experimental Matrix (32 conditions)
We systematicallly vary three dimensions to evaluate SFT performance:

1.  **Generalization Target**: 
    - **Q** (Question): New questions, known respondents (Internalization of personal styles).
    - **R** (Respondent): New respondents, known questions (Internalization of population patterns).
2.  **Model Size**: 0.5B, 1B, 8B, 70B.
3.  **Context Size (n_ctx)**: 10, 15, 25, 50 context questions.

| Dim | Levels | Models / Values |
|---|---|---|
| **Generalization** | 2 | Questions (Q), Respondents (R) |
| **Model Size** | 4 | 0.5B (Qwen2.5), 1B (Llama3.2), 8B (Llama3.1), 70B (Llama3.1) |
| **Context (n_ctx)** | 4 | 10, 15, 25, 50 |

**Condition 6 (SFT + RAG)** is maintained as an extension to test if RAG provides marginal gains on top of the best-performing SFT model.

## Data

Single survey: **Canadian Election Study 2021 (CES 2021)**. No multi-survey harmonization required.

Processed data format (no SQL database needed):
- `data/processed/respondents.parquet` — respondents × variables matrix
- `data/processed/questions.parquet` — one row per question: variable name, label, full text, embedding vector

## Validation design: intra-wave held-out questions

- Split CES 2021 questions by **thematic domain** (economy, environment, immigration, etc.)
- Fine-tune on a subset of domains; evaluate on held-out domains
- For RAG conditions: mask target question, inject responses to semantically closest questions from the same respondent
- Aggregate individual-level simulations to distribution level
- Primary comparison unit: **distributions** (not individual predictions), stratified by sociodemographic subgroups

> **Note on semantic proximity**: performance will be reported as a function of semantic distance between the target question and injected RAG questions, to distinguish genuine generalization from near-duplicate retrieval.

## LLM infrastructure

**To be determined.** Open-source model (Mistral or Llama 3) preferred for reproducibility. Deployment options under consideration:
- Self-hosted API (own inference server)
- Third-party API with fixed model version (Together.ai, Groq, etc.)

Need to review how comparable studies have handled this (not the first to face this choice). Constraint: model version must be fixed and citable for reproducibility.

## Tech Stack

- **Manuscript**: Quarto (`.qmd`)
- **Data**: Canadian Election Study 2021 (CES 2021)
- **Code**: Python/R (English); fine-tuning via script (`scripts/finetune.py`), not notebooks
- **Embeddings**: sentence-transformers or OpenAI embeddings (TBD)
- **Fine-tuning**: `transformers` + `peft` + `trl` (HuggingFace ecosystem, QLoRA); cloud GPU (provider TBD)
- **LLM inference**: open-source model (Mistral 7B or Llama 3), deployment TBD (see LLM infrastructure)

## Project structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── paper/
├── scripts/
├── src/article_silicon_sampling_quebec/
└── tests/
```

- `scripts/` for executable data/pipeline scripts
- `src/` for reusable Python code
- `paper/` for Quarto manuscript files
- `data/` split by stage (raw/interim/processed)

## Python environment (quick start)

A virtual environment (`.venv`) is included in this repo. Activate it with:

```bash
source .venv/bin/activate  # on Linux/macOS
# or on Windows:
.venv\Scripts\activate
```

To set up a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Potential outlets

- *Political Analysis*
- *Canadian Journal of Political Science*
- *Journal of Information Technology & Politics*

## Status

- [ ] Literature review (Argyle 2023, Santurkar 2023, Tjuatja 2024, fine-tuning comparables)
- [ ] Methodology section
- [ ] Data pipeline (CES 2021 → respondents.parquet + questions.parquet)
- [ ] Embeddings for all CES 2021 questions
- [ ] Thematic domain split (train/test split by topic, not random)
- [ ] Simulation pipeline — conditions 1, 2, 3 (cold / demographics / RAG)
- [ ] Regenerate SFT datasets with n_ctx=10 and n_ctx=15 (conditions 4A/4B, 5A/5B)
- [ ] Fine-tuning pipeline — conditions 4A, 4B (question generalization, 10/15-ctx)
- [ ] Fine-tuning pipeline — conditions 5A, 5B (respondent generalization, 10/15-ctx)
- [ ] Combined pipeline — condition 6 (fine-tuned + RAG, respondent gen)
- [ ] Validation study (held-out questions, distribution-level metrics)
- [ ] Subgroup analysis (sociodemographic stratification)
- [ ] Writing (Quarto)
