---
title: "CES 2025 Benchmark: What We Did and What We Found"
author: "Hubert Cadieux"
date: "October 8, 2026"
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
- \newcommand{\notesubtitle}{Six fine-tuned Llama 3.3 70B models · 150 subgroup scores each · what it means}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · CES 2025 benchmark}
---

```{=latex}
\begin{encadre}
```
**In brief.** Fine-tuning roughly halves the error of the untrained model, and the best recipe
(fixed context, 50k examples) gets subgroups right to within about 18 answers out of 100. But
**most of that error is about where Canada as a whole stands in 2025, not about how subgroups
differ.** Simply giving every subgroup the real national distribution does twice as well. The
models do pick up subgroup differences, in the right direction but at about a third of their real
size. Combined with a real national figure, they beat the national figure alone, but only by a little.
```{=latex}
\end{encadre}
```

# 1. What we did

**Training.** Six Llama 3.3 70B models, fine-tuned on Azure AI Foundry on 8 Canadian surveys
(CES 2019 and 2021, Democracy Checkup 2019–2024). No CES 2025 data was used anywhere. Each
training example is one real respondent answering one question, after seeing their own answers to
a few other questions. Two things vary:

- **Context.** *Fixed*: the other questions always cover the same themes (ideology, party, trust,
  redistribution…). *Semantic*: the other questions are the ones closest in meaning to the target.
- **Size.** 20k, 50k or 100k examples, each set containing the smaller one.

**Test.** Every model answered the same frozen set: 15,000 synthetic respondents drawn from the
Democracy Checkup 2024 (1,000 in each of 5 regions × 3 age groups, each with a real person's
profile and 7 of their answers), 10 CES 2025 questions, 150,000 calls per model. We compare each of
the 150 subgroup × question answer distributions with the real CES 2025 one.

**Practical lessons.**

- Azure's training-data check rejects files with items it flags as hate or unfairness. We had to
  screen every item with Azure Content Safety. As a result, **most immigration items were removed
  from training**, along with two of the fixed context questions (family values, religious symbols).
- Inference ran in cloud containers, several models in parallel. Fewer than 1% of replies were
  invalid, for every model. The untrained model also follows the format well.

**How to read the score.** Total variation (TV) is the share of answers that would have to move
to another option to match the real distribution. TV = 0.18 means about **18 answers out of 100**
are in the wrong place. Lower is better.

# 2. Overall results

| Model | Mean TV (150 pairs) | Out of place, per 100 |
|------------------------------|-----------|-----------|
| \cellcolor{ref}Human floor (two halves of CES 2025) | \cellcolor{ref}0.062 | \cellcolor{ref}6 |
| \cellcolor{ref}Real national distribution for every subgroup | \cellcolor{ref}0.085 | \cellcolor{ref}9 |
| \cellcolor{best}**Fixed, 50k** | \cellcolor{best}**0.178** | \cellcolor{best}**18** |
| Semantic, 50k | 0.188 | 19 |
| Fixed, 100k | 0.191 | 19 |
| Semantic, 100k | 0.195 | 20 |
| Fixed, 20k | 0.200 | 20 |
| Semantic, 20k | 0.214 | 21 |
| Untrained Llama 3.3 70B | 0.325 | 33 |
| \cellcolor{ref}Most common answer for everyone | \cellcolor{ref}0.540 | \cellcolor{ref}54 |

The grey rows are built from the CES 2025 data itself. They are not forecasts; they
show what a model would have to beat.

- **Training matters most:** −15 points from untrained to Fixed 50k.
- **Fixed beats semantic** at every size (1.4, 1.1 and 0.4 points at 20k, 50k, 100k).
- **50k beats both 20k and 100k.** The 100k drop is small (1.4 points) and comes from one training
  run per recipe, so it may not hold on a repeat.

All these gaps are statistically clear on the test set (paired bootstrap), but the intervals do
not include training-run variability.

# 3. By region and age group

Mean TV over the 10 questions and the other dimension (e.g. BC = BC 18–34, 35–54 and 55+).

| Model | BC | Prairies | ON | QC | Atlantic | 18–34 | 35–54 | 55+ |
|----------------------|------|------|------|------|------|------|------|------|
| \cellcolor{ref}Human floor | \cellcolor{ref}0.067 | \cellcolor{ref}0.058 | \cellcolor{ref}0.040 | \cellcolor{ref}0.048 | \cellcolor{ref}0.099 | \cellcolor{ref}0.074 | \cellcolor{ref}0.061 | \cellcolor{ref}0.052 |
| \cellcolor{ref}National distribution | \cellcolor{ref}0.078 | \cellcolor{ref}0.092 | \cellcolor{ref}0.074 | \cellcolor{ref}0.088 | \cellcolor{ref}0.094 | \cellcolor{ref}0.099 | \cellcolor{ref}0.066 | \cellcolor{ref}0.091 |
| **Fixed, 50k** | **0.172** | **0.172** | **0.169** | **0.178** | **0.196** | **0.182** | **0.175** | **0.176** |
| Semantic, 50k | 0.177 | 0.172 | 0.178 | 0.206 | 0.209 | 0.188 | 0.186 | 0.191 |
| Fixed, 100k | 0.175 | 0.191 | 0.183 | 0.195 | 0.212 | 0.189 | 0.193 | 0.192 |
| Semantic, 100k | 0.179 | 0.200 | 0.191 | 0.197 | 0.209 | 0.189 | 0.202 | 0.195 |
| Fixed, 20k | 0.181 | 0.189 | 0.194 | 0.216 | 0.219 | 0.201 | 0.196 | 0.203 |
| Semantic, 20k | 0.203 | 0.199 | 0.210 | 0.228 | 0.229 | 0.210 | 0.209 | 0.223 |
| Untrained | 0.329 | 0.296 | 0.331 | 0.301 | 0.369 | 0.333 | 0.330 | 0.312 |

- **The error is spread evenly.** For the best model, every region and age group sits between
  0.17 and 0.18, except the Atlantic (0.20). The Atlantic also has the noisiest real data (smallest
  samples, highest human floor), so part of its gap is measurement noise.
- **Quebec is where context format matters most.** Semantic context costs 2.8 points there at 50k,
  against 0 to 1.3 points in the other regions. We have no explanation for this yet.
- **No age group is left behind.** Young, middle-aged and older respondents are all within one
  point of each other.

# 4. By question

For Fixed 50k: the subgroup score (as above) and the error on the **national** distribution alone
(the model's 15 subgroups pooled with their real population weights, compared with CES 2025 for
Canada as a whole).

| Question | Subgroup TV | National-level TV | Human floor |
|------------------------------|---------|---------|---------|
| Spending: affordable housing | 0.10 | 0.09 | 0.05 |
| Satisfaction with democracy | 0.12 | 0.10 | 0.05 |
| Medical assistance in dying | 0.14 | 0.08 | 0.09 |
| Jobs before environment | 0.15 | 0.11 | 0.10 |
| Spending: defence | 0.17 | 0.16 | 0.04 |
| Pipelines | 0.17 | 0.13 | 0.10 |
| Spending: national childcare | 0.18 | 0.18 | 0.06 |
| Personal finances, past year | 0.21 | 0.18 | 0.04 |
| Spending: reconciliation | 0.24 | 0.24 | 0.05 |
| \cellcolor{worse}**Immigration levels** | \cellcolor{worse}**0.30** | \cellcolor{worse}**0.30** | \cellcolor{worse}0.04 |

On most questions, the national-level error is nearly as large as the subgroup error. The model
is mostly wrong about **Canada**, and the subgroups inherit it. Immigration is the extreme case:
it is the topic training saw least, and opinion on it shifted sharply over 2023–2025. Defence,
reconciliation and personal finances are also topics where 2025 may differ from the training years.

# 5. Where the error comes from

We split each model's error into two parts: the **level** (the national distribution) and the
**differences** between subgroups (each subgroup's gap from the national figure).

| Model | Subgroup TV | National-level TV | Size of gaps (1 = real) | Gap correlation | Real national + model gaps |
|---------------|------|------|------|------|------|
| **Fixed, 50k** | **0.178** | **0.157** | 0.31 | 0.42 | 0.082 |
| Semantic, 50k | 0.188 | 0.161 | 0.28 | 0.39 | 0.086 |
| Fixed, 100k | 0.191 | 0.172 | 0.35 | 0.47 | **0.078** |
| Semantic, 100k | 0.195 | 0.178 | 0.30 | 0.41 | 0.082 |
| Fixed, 20k | 0.200 | 0.184 | 0.26 | 0.38 | 0.084 |
| Semantic, 20k | 0.214 | 0.194 | 0.25 | 0.33 | 0.089 |
| Untrained | 0.325 | 0.296 | 0.52 | 0.35 | 0.113 |
| \cellcolor{ref}Real national only (no gaps) | \cellcolor{ref}0.085 | \cellcolor{ref}0 | \cellcolor{ref}0 | \cellcolor{ref}— | \cellcolor{ref}0.085 |

*Size of gaps* is the slope of the model's subgroup gaps on the real ones; *gap correlation* says
whether the model puts the gaps in the right direction. The last column gives each subgroup the
real national distribution plus the model's gap for that subgroup.

- **The level is the main problem.** For Fixed 50k, the national error alone (0.157) is almost the
  whole subgroup error (0.178).
- **Differences are right in direction, too small in size.** Model gaps correlate 0.42 with the
  real ones (0.61 for age), but they are about a third of the real size: the model flattens
  subgroups toward each other. The untrained model has larger gaps but they point the right way
  less often.
- **The model's gaps add real but modest information.** Real national + model gaps (0.078–0.082)
  beats the real national alone (0.085) for every trained model, but closes only a tenth (Fixed 50k)
  to a third (Fixed 100k) of the distance to the human floor (0.062). We have no interval on this comparison yet.
- **The best recipe depends on the job.** Fixed 50k has the best level; Fixed 100k has the best
  differences. Training size seems to help differences but not the level.

# 6. What it means for what comes next

1. **The bottleneck is time, not people.** Training on 2019–2024 teaches the model how Canadians
   differ, but not where opinion stands in 2025. More data of the same kind will not fix that; the
   100k models show it.
2. **The most promising use is "national figure + model for subgroups."** When a recent national
   poll exists, the model should supply only the subgroup gaps. That is the honest test of what
   it adds, and it should become the second headline score of the benchmark.
3. **Make the gaps bigger and sharper.** The gaps are too small. Worth testing: richer or more
   context answers per respondent, and checking whether flattening comes from the training
   targets or from sampling at temperature 1.0.
4. **Update the level.** Ways to try: add recent non-CES surveys (2025) to training, or give the
   model recent national figures in the prompt.
5. **Immigration** stays a blind spot until we find a way past Azure's training filter (other
   phrasings, another platform). Until then it should be reported separately.
6. **Repeat Fixed 50k and Fixed 100k** with a new seed before treating 50k vs 100k as settled.
