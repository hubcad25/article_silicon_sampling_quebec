# article_silicon_sampling_quebec

Working repo for an academic article on silicon sampling applied to Quebec public opinion data.

## Research question

Can LLM-based silicon sampling — conditioned on real survey data and updated with real-time media context — reliably simulate Quebec public opinion on unsurveyed topics?

## Positioning relative to Argyle et al. (2023)

Argyle et al. (*Political Analysis*, 2023) demonstrate that GPT-3, when conditioned on sociodemographic backstories, reproduces human response distributions with high algorithmic fidelity. Their study is limited to U.S. politics, English-language context, and uses the LLM "cold" (no empirical prior, no temporal updating).

This article extends their framework in three ways:

1. **Empirical priors via RAG** — instead of conditioning the LLM on demographics alone, we inject real response distributions from semantically similar survey questions as Bayesian priors before simulation. This makes the approach scientifically defensible beyond pure LLM outputs.

2. **Real-time media context** — we inject recent news articles to update priors on emerging issues, enabling simulation on topics not yet surveyed. We test whether this temporal updating improves algorithmic fidelity.

3. **Non-American, francophone validation** — we validate against public Quebec/Canadian Election Studies (CES/QES) data, filling an explicit gap acknowledged by Argyle et al.

## Architecture (Article implementation)

The simulation pipeline follows three steps:

1. **Semantic RAG** on harmonized CES/QES data → identifies closest existing questions and respondents
2. **Bayesian prior construction** → built from real response distributions, updated with targeted media context
3. **Silicon sampling** → LLM simulates responses in the persona of real respondents, calibrated by updated priors

## Validation Design: Intra-wave prediction

We use a "held-out question" approach:
- Select a specific survey wave.
- Mask a target question for a subset of respondents.
- Simulate the response using sociodemographics, other survey responses (via RAG), and the media context surrounding the survey period.
- Compare simulated distributions against the ground truth.

## Tech Stack

- **Manuscript**: Quarto (`.qmd`)
- **Data**: Canadian Election Study (CES) & Étude électorale du Québec (EEQ)
- **Code**: Python/R (English)
- **Pipeline**: Inspired by Opubliq simulator architecture

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

Simple and practical layout:
- `scripts/` for executable data/pipeline scripts
- `src/` for reusable Python code
- `paper/` for Quarto manuscript files
- `data/` split by stage (raw/interim/processed)

## Python environment (quick start)

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

- [ ] Literature review
- [ ] Methodology section
- [ ] Data pipeline (CES/QES)
- [ ] Validation study (Intra-wave)
- [ ] Writing (Quarto)
