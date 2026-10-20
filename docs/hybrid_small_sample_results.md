---
title: "Can 100 Real Interviews Rescue a Synthetic Subgroup Estimate?"
author: "Hubert Cadieux"
date: "October 20, 2026"
lang: en
geometry: margin=2.1cm
fontsize: 10pt
mainfont: Fira Sans
mainfontoptions:
- Scale=0.95
header-includes:
- \usepackage{float}
- \usepackage{array}
- \usepackage{needspace}
- \newcommand{\notesubtitle}{100 real interviews + approach C · a short simulation}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · CES 2025 · hybrid test}
---

# The short version

We simulated a cheap, quota-free survey by drawing 50 to 400 people from CES 2025, then combined
their answers with approach C, the dossier model that directly predicts distributions for 15
region-by-age groups (C trained by example, first training run; the best C so far). The full CES
2025 sample is the answer key. Each condition is repeated 100 times; the score is total variation
(TV), roughly the share of answers put in the wrong category. Lower is better.

The hybrid works. With **100 biased real interviews**, the best version scores **0.106 on repeated
questions and 0.115 on new questions**, against 0.315 and 0.322 for the small sample alone. That is
about as accurate as a **balanced, representative survey of 500–650 people** (about ±16–17 points
per group). With **400 biased interviews**, it reaches **0.081 and 0.089**, about a representative
survey of **830–1,060 people**, the size of a standard national poll (about ±12–13 points per group).

*Correction (October 20).* A first version reported much higher equivalences ("1,200 interviews").
Several CES 2025 questions were asked of only 25–50% of respondents, so a simulated "400-person"
collection held only 100–200 answers on those questions, which made the sample alone look far worse
than a real survey where everyone answers. Each question's sample is now drawn among the respondents
who were asked it. All numbers below use the corrected simulation.

# What was compared

- **Sample only:** weighted answers from the small real sample, post-stratified to the population
  shares of the 15 groups. A group with no answers receives the sample's national distribution.
- **C only:** the mean of three existing approach-C draws; the model was not called again.
- **Bayesian hybrid:** C is a Dirichlet prior and the real answers update it. Prior strength was
  selected on the repeated questions; **k = 20** won for every sample size and sampling mechanism.
- **National level + C gaps:** the real sample, post-stratified by region and age, supplies the
  national answer distribution; C supplies the differences among groups.

The biased collection deliberately gives twice the selection chance to respondents reporting high
political interest (8–10 on a 0–10 scale). It uses no age or region quotas and does not correct for
political interest afterward. This is a mild bias; a real quota-free collection can be much worse.

# Results

## Repeated questions

**Random collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.387 | 0.314 | 0.235 | 0.167 |
| C only | 0.151 | 0.151 | 0.151 | 0.151 |
| Bayesian hybrid (k=20) | 0.142 | 0.135 | 0.124 | 0.108 |
| National level + C gaps | **0.132** | **0.104** | **0.088** | **0.079** |

```{=latex}
\needspace{7\baselineskip}
```
**Biased collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.382 | 0.315 | 0.238 | 0.169 |
| C only | 0.151 | 0.151 | 0.151 | 0.151 |
| Bayesian hybrid (k=20) | 0.142 | 0.134 | 0.123 | 0.107 |
| National level + C gaps | **0.132** | **0.106** | **0.090** | **0.081** |

## New questions, including the two Trump questions

**Random collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.389 | 0.322 | 0.243 | 0.171 |
| C only | 0.213 | 0.213 | 0.213 | 0.213 |
| Bayesian hybrid (k=20) | 0.197 | 0.185 | 0.165 | 0.140 |
| National level + C gaps | **0.138** | **0.113** | **0.096** | **0.087** |

```{=latex}
\needspace{7\baselineskip}
```
**Biased collection**

| Method | n=50 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| Sample only | 0.388 | 0.322 | 0.247 | 0.174 |
| C only | 0.213 | 0.213 | 0.213 | 0.213 |
| Bayesian hybrid (k=20) | 0.197 | 0.185 | 0.167 | 0.141 |
| National level + C gaps | **0.141** | **0.115** | **0.098** | **0.089** |

## What it is worth, in survey terms

Equivalent size = the balanced, representative survey (everyone answers every question) with the
same TV, read on the simulated sample-only curve for balanced surveys; the conversion of the
October 14 note (TV ≈ 0.66 / √n per group) gives about 10–15% less. Margin of error per group at 95%,
for a proportion near 50%.

| Biased collection + C | TV, repeated / new | Worth a representative survey of | Margin per group |
|---|---|---|---|
| 100 interviews | 0.106 / 0.115 | 500–650 | ±16–17 pts |
| 400 interviews | 0.081 / 0.089 | 830–1,060 | ±12–13 pts |
| *For reference: standard poll of 1,000, balanced* | *≈ 0.08* | *1,000* | *±12 pts* |

# The Trump-question sensitivity check

The two questions about Canada–US relations after Trump's election are exactly where a pre-2025
model cannot know the current national level. At n=100 in the biased collection:

| Method | All 8 new questions | Other 6 new questions |
|---|---:|---:|
| Sample only | 0.322 | 0.373 |
| C only | 0.213 | 0.134 |
| Bayesian hybrid (k=20) | 0.185 | 0.131 |
| National level + C gaps | **0.115** | **0.127** |

Without the Trump questions, C alone is already strong and 100 interviews add little. With them,
real data supply the missing 2025 level and the hybrid improves sharply: the small sample matters
most when opinion has moved since the model's data.

# Four conclusions

1. **The hybrid beats the sample alone at every size tested**, from n=50, for repeated and new
   questions.
2. **100 biased interviews + C are worth a representative survey of about 500–650 people; 400 + C,
   about 830–1,060**, the size of a standard national poll.
3. **This mild selection bias barely survives the mix** (at most 0.003 TV against random
   collection); a stronger or topic-related bias remains untested.
4. **The level-plus-gaps hybrid levels off near 0.07–0.08** as the sample grows: that floor is the
   error of C's own group differences. Beyond about 800 interviews the Bayesian hybrid, which lets
   real data override C cell by cell, becomes better (0.06–0.07 at n=1,600). Better group
   differences from C are what would lower the floor.

The simulated sample overlaps the CES 2025 answer key by design, so the largest-sample results are
slightly optimistic. The optional Quebec age-by-gender variant was not run because the existing C
outputs distinguish region and age, but not gender. Full repetition-level results are in
`data/analysis/hybrid_small_sample.csv`; the reproducible analysis is
`scripts/44_hybrid_small_sample.py`.
