---
title: "Can 100 Real Interviews Rescue a Synthetic Subgroup Estimate?"
author: "Hubert Cadieux"
date: "October 19, 2026"
lang: en
---

# The short version

We simulated a cheap, quota-free survey by drawing 50 to 400 people from CES 2025, then combined
their answers with approach C, the dossier model that directly predicts distributions for 15
region-by-age groups. The full CES 2025 sample is treated as the answer key. Each condition is
repeated 100 times; the score is total variation (TV), or roughly the share of answers put in the
wrong category. Lower is better.

The hybrid works. With **100 biased real interviews**, the best version scores **0.129 on repeated
questions and 0.147 on new questions**, compared with 0.358 and 0.388 for the small sample alone.
That is about as accurate as **1,200–1,275 ordinary random interviews**, or roughly **825–890
interviews in a survey balanced across the 15 groups**. This is an accuracy comparison, not yet a
validated cost estimate, but it supports the practical premise behind the “$15k precision for a
$2–3k collection” pitch.

# What was compared

- **Sample only:** weighted answers from the small real sample, post-stratified to the population
  shares of the 15 groups. A group with no answers receives the sample's national distribution.
- **C only:** the mean of three existing approach-C draws; the model was not called again.
- **Bayesian hybrid:** C is a Dirichlet prior and the real answers update it. Prior strength was
  selected on the repeated questions; **k = 20** won for every sample size and sampling mechanism.
- **National level + C gaps:** the real sample supplies the national answer distribution, while C
  supplies the differences among groups.

The biased collection deliberately gives twice the selection chance to respondents reporting high
political interest (8–10 on a 0–10 scale). It uses no age or region quotas and does not correct for
political interest afterward.

# Results

## Repeated questions

**Random collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.395 | 0.361 | 0.294 | 0.218 |
| C only | 0.151 | 0.151 | 0.151 | 0.151 |
| Bayesian hybrid (k=20) | **0.145** | 0.139 | 0.130 | 0.117 |
| National level + C gaps | 0.164 | **0.126** | **0.102** | **0.087** |

**Biased collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.397 | 0.358 | 0.293 | 0.226 |
| C only | 0.151 | 0.151 | 0.151 | 0.151 |
| Bayesian hybrid (k=20) | **0.144** | 0.138 | 0.129 | 0.117 |
| National level + C gaps | 0.166 | **0.129** | **0.105** | **0.089** |

## New questions, including the two Trump questions

**Random collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.405 | 0.390 | 0.333 | 0.257 |
| C only | 0.213 | 0.213 | 0.213 | 0.213 |
| Bayesian hybrid (k=20) | 0.200 | 0.190 | 0.174 | 0.153 |
| National level + C gaps | **0.190** | **0.148** | **0.119** | **0.099** |

**Biased collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.405 | 0.388 | 0.337 | 0.264 |
| C only | 0.213 | 0.213 | 0.213 | 0.213 |
| Bayesian hybrid (k=20) | 0.200 | 0.190 | 0.175 | 0.156 |
| National level + C gaps | **0.187** | **0.147** | **0.121** | **0.099** |

# The Trump-question sensitivity check

The two questions about Canada–US relations after Trump's election are exactly where a pre-2025
model cannot know the current national level. At n=100 in the biased collection:

| Method | All 8 new questions | Other 6 new questions |
|---|---:|---:|
| Sample only | 0.388 | 0.460 |
| C only | 0.213 | **0.134** |
| Bayesian hybrid (k=20) | 0.190 | 0.136 |
| National level + C gaps | **0.147** | 0.170 |

Without the Trump questions, C is already unusually strong and a noisy national estimate can hurt
at n=100. With the Trump questions included, real data supply the missing 2025 level and the
level-plus-gaps hybrid improves sharply. This is the clearest case for collecting a small current
sample rather than relying on the model alone.

# Three conclusions

1. **The hybrid beats the sample alone from n=50 onward**, the smallest size tested, for both
   repeated and new questions.
2. **At n=100 with biased selection, the best hybrid is worth about 1,200–1,275 random real
   respondents, or about 825–890 respondents in a balanced stratified survey**, depending on the
   question series.
3. **This particular selection bias survives only weakly:** at n=400 it adds about 0.007 TV to the
   sample alone on new questions, but 0.003 or less to the hybrids; this does not establish that
   stronger or differently targeted selection bias would be harmless.

The calibration extends the sample-only curve to n=800 and n=1,600 and interpolates on log sample
size. The simulated sample overlaps the CES 2025 answer key by design, so the largest-sample results
are slightly optimistic. The optional Quebec age-by-gender variant was not run because the existing
C outputs distinguish region and age, but not gender. Full repetition-level results are in
`data/analysis/hybrid_small_sample.csv`; the reproducible analysis is
`scripts/44_hybrid_small_sample.py`.
