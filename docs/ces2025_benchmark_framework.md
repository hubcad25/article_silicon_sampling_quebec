---
title: "A Common CES 2025 Benchmark for Silicon-Sampling Models"
author: "Hubert Cadieux"
date: "September 30, 2026"
lang: en
geometry: margin=2.3cm
fontsize: 10pt
mainfont: Fira Sans
mainfontoptions:
- Scale=0.95
monofont: DejaVu Sans Mono
monofontoptions:
- Scale=0.82
header-includes:
- \usepackage{float}
- \floatplacement{figure}{H}
- \newcommand{\notesubtitle}{10 items · 5 regions × 3 age groups · 150 subgroup TVs per model}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · comparison framework}
---

```{=latex}
\begin{encadre}
```
**In brief.** Any model, whatever it was trained on, is scored the same way: it receives the
same frozen set of synthetic respondents with the same prompt, answers ten CES 2025 questions, and
one script turns its answers into **150 total-variation (TV) scores** (15 subgroups × 10 items).
The only differences between two runs are the model weights. This replaces the 21-subgroup protocol
of September 30.
```{=latex}
\end{encadre}
```

# 1. What is fixed and what varies

| Fixed for every model | Free to vary |
|---|---|
| The 10 target items and their answer options (§2) | Base model, fine-tuning data, training recipe |
| The 15 subgroups and the CES 2025 human distributions (§3) | Number of training examples |
| The inference file: profiles and injected answers (§4) | |
| The prompt template (§5), temperature 1.0, one answer per call | |
| Parsing, scoring and bootstrap code (§6–7) | |

**Training rule.** No CES 2025 data may be used for training or tuning, in any form. The
prompt template is frozen *before* training, so a new model can be trained on exactly the format it
will see at inference. Each model is delivered with a one-line manifest: base model, training
sources, number of examples, and whether the anchor source survey (§4) was part of training.

# 2. Target items

Ten items from the CES 2025 Campaign Period Survey (prefix `cps25_`). Wording and options are taken
from the codebook in the respondent's language; the order of options is the codebook order.

| # | Item | Variable | Scale |
|---|---|---|---|
| 1 | Spending: affordable housing | `spend_afford_h` | S |
| 2 | Spending: national childcare | `spend_nation_c` | S |
| 3 | Spending: defence | `spend_defence` | S |
| 4 | Spending: reconciliation with Indigenous Peoples | `spend_rec_indi` | S |
| 5 | Medical assistance in dying | `pos_life` | A |
| 6 | Government should help build pipelines | `pos_energy` | A |
| 7 | Jobs before environment | `pos_jobs` | A |
| 8 | Number of immigrants Canada should admit | `imm` | I |
| 9 | Satisfaction with democracy | `demsat` | D |
| 10 | Personal finances over the past year | `own_fin_retro` | F |

S: less / about the same / more. A: five-point agree–disagree. I: more / fewer / about the same.
D: four-point satisfaction. F: better / same / worse. **Every scale keeps its "don't know / prefer
not to answer" option (DK)**, both in the prompt and in scoring.

# 3. Subgroups and human distributions

**Sample.** All CES 2025 respondents aged 18+, living in one of the ten provinces, with a positive
`cps25_weight_general_all` ($w_i$). No quality exclusions. Because CES 2025 is never used for
training, the whole sample is evaluation data: there is no development/evaluation split.

**Subgroups.** 5 regions × 3 age groups = 15 cells, fixed before seeing any result and never merged.

- Region (`cps25_province`): BC · Prairies (AB, SK, MB) · ON · QC · Atlantic (NB, NL, NS, PE).
- Age (`cps25_age_in_years`): 18–34 · 35–54 · 55+.

**Human distribution.** For item $j$, cell $g$ and answer code $k$, among respondents $H_{jg}$ who
were asked the item and gave any recorded answer (DK included; blank or "not asked" excluded):
$$p_{jgk} = \frac{\sum_{i \in H_{jg}} w_i\,\mathbf{1}(y_{ij}=k)}{\sum_{i \in H_{jg}} w_i}.$$

**Cell sizes.** Raw $n$ for items asked of everyone (defence, reconciliation, immigration,
democracy, finances), effective $n_{\text{eff}} = (\sum w_i)^2/\sum w_i^2$, and the smallest item
$n$ (housing and childcare were asked of half the sample; the three agree–disagree items of a
third). No cell falls below 30. The last column is the pool of Democracy Checkup 2024 respondents
from which profiles are drawn (§4).

| Cell | 18–34 $n$ / $n_{\text{eff}}$ / min | 35–54 $n$ / $n_{\text{eff}}$ / min | 55+ $n$ / $n_{\text{eff}}$ / min | DC 2024 pool (18–34 · 35–54 · 55+) |
|---|---|---|---|---|
| BC | 600 / 493 / 193 | 871 / 744 / 286 | 1,267 / 993 / 430 | 316 · 353 · 490 |
| Prairies | 934 / 759 / 286 | 1,245 / 1,045 / 407 | 1,389 / 1,096 / 455 | 347 · 466 · 661 |
| ON | 1,918 / 1,565 / 602 | 2,528 / 2,163 / 799 | 3,191 / 2,521 / 1,035 | 730 · 964 · 1,300 |
| QC | 1,240 / 1,031 / 413 | 1,600 / 1,340 / 543 | 2,066 / 1,635 / 670 | 544 · 601 · 753 |
| Atlantic | 277 / 224 / 90 | 404 / 340 / 127 | 589 / 470 / 182 | 107 · 169 · 278 |


# 4. The frozen inference file

The file is built once, with a fixed seed, and handed unchanged to every model.

**Profiles.** For each of the 15 cells, draw **1,000 respondents** from the anchor source survey
(the Democracy Checkup 2024), with replacement and with probability proportional to their survey weight. A
respondent belongs to a cell through their province and their **age in 2025** (age in 2024 + 1; the survey has no one who is 18 in 2025, a negligible gap).
Each profile carries: age in 2025, gender, province, language, education and household income.

**Injected answers (anchors).** Each profile also carries that same real respondent's answers to
**the same five anchor questions**, chosen from varied themes (table below). The anchors are
identical for all profiles, all target items and all models. None is a target item or a close
equivalent of one. Fixed anchors were preferred to random or semantically nearest questions:
they give every model and every item exactly the same information, and they avoid near-duplicates
that would turn the task into copying a past answer. A respondent who skipped an anchor is shown
the other anchors only.

| # | Theme | DC 2024 variable | Question (short) | Options |
|--|----------|--------------------|--------------------------------|--------------------|
| 1 | Ideology | `dc24_lr_self_1` | Left–right self-placement | 0 (left) to 10 (right) |
| 2 | Partisanship | `dc24_party_id` | Federal party identification | Liberal, Conservative, NDP, Bloc, Green, PPC, other, none |
| 3 | Redistribution | `dc24_inequality_gap` | How much should be done to reduce the gap between rich and poor | Much more … much less (5) |
| 4 | Moral traditionalism | `dc24_pos_family_val` | Fewer problems with more emphasis on traditional family values | Strongly agree … strongly disagree (4) |
| 5 | Language and identity | `dc24_pos_bilingualis` | We have gone too far in pushing bilingualism | Strongly agree … strongly disagree (4) |

All five were asked of every DC 2024 respondent, with more than 98% substantive answers. Items
too close to a target were ruled out: confidence in government (demsat), job creation by the
private sector (jobs vs environment), and the DC 2024 versions of immigration, democracy
satisfaction and jobs vs environment. Anchors are shown with the respondent's exact option label.


**Size.** 15 cells × 1,000 profiles × 10 items = **150,000 calls per model**. Each profile answers
all ten items, each in a separate call.

# 5. Prompt

One template, in English and French; a profile gets the version matching its language. Only the
chat-template wrapping (special tokens) may differ between models. Temperature 1.0, no system
prompt beyond the template, no retries on a completed answer (transport failures are retried).

```text
You are a {age}-year-old {gender} living in {province}, Canada. It is 2025.
Your first language is {language}. Your highest level of education is {education}.
Your household income is {income}.

In 2024, you answered these survey questions:
- {anchor_question_1} {answer_1}
- ...
- {anchor_question_5} {answer_5}

Answer the following question from a 2025 survey. Reply with one option exactly as written.

{item_wording}
{option_1}
...
Don't know / Prefer not to answer
```

# 6. Scoring

**Parsing.** A reply is mapped to an option code by exact then normalized text match. A reply that
matches no option is **invalid**. Invalid replies are removed, and the invalid rate is reported per
model, item and cell. DK is a valid answer and a regular category.

**Generated distribution.** $q_{jgk}$ is the share of valid replies in cell $g$ for item $j$ that
fall on code $k$. All profiles have equal weight, since survey weights were applied when drawing them.

**Subgroup TV (150 per model).**
$$TV_{jg} = \tfrac{1}{2}\sum_k |p_{jgk} - q_{jgk}|.$$
TV is the share of answers that must be moved to another option to turn one distribution into the
other: 0 means identical, 1 means disjoint. Zero-probability options are allowed; no smoothing.

**Summaries.** For each model: the 15 × 10 matrix; per item, the mean over the 15 cells; overall,
the mean of the 150 TVs (**headline score**). As a secondary score, the same average weighted by
each cell's share of CES 2025 weight.

# 7. Reference points and uncertainty

Every model is reported next to two references, computed once.

- **Human floor.** Each CES 2025 cell is split at random into two halves; TV between the two
  halves' weighted distributions, averaged over 200 random splits. It shows how much TV comes from
  sampling noise alone. Each half has only half the respondents, so this floor is conservative.
- **Untrained model.** The base model with the same file and prompt.

**Uncertainty.** 2,000 bootstrap replicates. Human respondents are resampled within cells with their
weights; profiles are resampled within cells with all their answers. The same resampled profile IDs
are used for every model, so differences between models are **paired**. We report percentile 95%
intervals for each model's score and for each pairwise difference. Cells with fewer than 30
respondents (raw or effective) are flagged but kept.

# 8. What a run delivers

One CSV per model, one row per call: `model_id`, `profile_id`, `cell`, `item`, `raw_response`,
`code` (empty if invalid). The scoring script reads only this file and the frozen inputs, and
writes the 150 TVs, the summaries and the intervals.
