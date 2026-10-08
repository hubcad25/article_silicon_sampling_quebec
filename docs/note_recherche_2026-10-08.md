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
**In brief.** Fine-tuning roughly halves the error of the untrained model. The best recipe (fixed
context, 50k examples) is off by about 18 answers out of 100 per subgroup. But **simply reusing the
CES 2021 answers of the same subgroups does better (15 out of 100)**, on 8 of the 10 questions. The
old survey also reproduces how subgroups differ far better than the models do. On questions that
were already asked before, a past poll beats our models. Their possible value is for questions with
no past version, which this benchmark does not test yet.
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

**The 10 questions are not new.** All 10 were asked in CES 2021 with the same wording and options
(7 also in CES 2019). Those earlier versions were in the training pool, but rarely drawn: a few dozen
examples each out of 100k, and none for immigration.

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
| \cellcolor{ref}Real 2025 national distribution for every subgroup | \cellcolor{ref}0.085 | \cellcolor{ref}9 |
| \cellcolor{best}**CES 2021, same subgroup, unchanged** | \cellcolor{best}**0.151** | \cellcolor{best}**15** |
| **Fixed, 50k** | **0.178** | **18** |
| Semantic, 50k | 0.188 | 19 |
| Fixed, 100k | 0.191 | 19 |
| Semantic, 100k | 0.195 | 20 |
| Fixed, 20k | 0.200 | 20 |
| Semantic, 20k | 0.214 | 21 |
| Untrained Llama 3.3 70B | 0.325 | 33 |
| \cellcolor{ref}Most common answer for everyone | \cellcolor{ref}0.540 | \cellcolor{ref}54 |

**Two kinds of reference.** The grey rows are built from CES 2025 itself: they know the answer
and only show what is reachable. **CES 2021** is a real forecast: it uses only data available
before 2025, like the models. It is the bar the models have to clear.

- **Training matters:** −15 points from untrained to Fixed 50k.
- **But the old survey wins:** CES 2021 beats Fixed 50k by 2.6 points (95% interval 2.6 to 3.8),
  and beats every model.
- **Among models,** fixed beats semantic at every size, and 50k beats both 20k and 100k. The
  100k drop is small (1.4 points) and comes from one training run per recipe, so it may not hold.

Model comparisons use a paired bootstrap; intervals do not include training-run variability.

# 3. By region and age group

Mean TV over the 10 questions and the other dimension (e.g. BC = BC 18–34, 35–54 and 55+).

| Model | BC | Prairies | ON | QC | Atlantic | 18–34 | 35–54 | 55+ |
|----------------------|------|------|------|------|------|------|------|------|
| \cellcolor{ref}Human floor | \cellcolor{ref}0.067 | \cellcolor{ref}0.058 | \cellcolor{ref}0.040 | \cellcolor{ref}0.048 | \cellcolor{ref}0.099 | \cellcolor{ref}0.074 | \cellcolor{ref}0.061 | \cellcolor{ref}0.052 |
| \cellcolor{ref}National distribution | \cellcolor{ref}0.078 | \cellcolor{ref}0.092 | \cellcolor{ref}0.074 | \cellcolor{ref}0.088 | \cellcolor{ref}0.094 | \cellcolor{ref}0.099 | \cellcolor{ref}0.066 | \cellcolor{ref}0.091 |
| **CES 2021** | **0.140** | **0.140** | **0.155** | **0.166** | **0.156** | **0.157** | **0.140** | **0.158** |
| Fixed, 50k | 0.172 | 0.172 | 0.169 | 0.178 | 0.196 | 0.182 | 0.175 | 0.176 |
| Semantic, 50k | 0.177 | 0.172 | 0.178 | 0.206 | 0.209 | 0.188 | 0.186 | 0.191 |
| Fixed, 100k | 0.175 | 0.191 | 0.183 | 0.195 | 0.212 | 0.189 | 0.193 | 0.192 |
| Semantic, 100k | 0.179 | 0.200 | 0.191 | 0.197 | 0.209 | 0.189 | 0.202 | 0.195 |
| Fixed, 20k | 0.181 | 0.189 | 0.194 | 0.216 | 0.219 | 0.201 | 0.196 | 0.203 |
| Semantic, 20k | 0.203 | 0.199 | 0.210 | 0.228 | 0.229 | 0.210 | 0.209 | 0.223 |
| Untrained | 0.329 | 0.296 | 0.331 | 0.301 | 0.369 | 0.333 | 0.330 | 0.312 |

- **CES 2021 wins in every region and age group.** The gap is smallest in Quebec and Ontario
  (1.2 and 1.4 points) and largest in the Atlantic (4 points).
- **The model's error is spread evenly.** For Fixed 50k, every region and age group sits between
  0.17 and 0.18, except the Atlantic (0.20). The Atlantic also has the noisiest real data (smallest
  samples, highest human floor), so part of its gap is measurement noise.
- **Quebec is where context format matters most.** Semantic context costs 2.8 points there at 50k,
  against 0 to 1.3 points in the other regions. We have no explanation for this yet.

# 4. By question

Subgroup TV for Fixed 50k and CES 2021, and the error on the **national** distribution alone
(the 15 subgroups pooled with their real 2025 population weights, compared with CES 2025 for
Canada as a whole).

| Question | Fixed 50k | CES 2021 | Fixed 50k, national | CES 2021, national |
|------------------------------|---------|---------|---------|---------|
| Spending: affordable housing | 0.10 | **0.08** | 0.09 | 0.06 |
| Satisfaction with democracy | 0.12 | **0.06** | 0.10 | 0.04 |
| Medical assistance in dying | 0.14 | **0.09** | 0.08 | 0.07 |
| Jobs before environment | 0.15 | **0.13** | 0.11 | 0.11 |
| Spending: defence | **0.17** | 0.31 | 0.16 | 0.33 |
| Pipelines | 0.17 | **0.16** | 0.13 | 0.14 |
| Spending: national childcare | 0.18 | **0.09** | 0.18 | 0.08 |
| Personal finances, past year | 0.21 | **0.13** | 0.18 | 0.13 |
| Spending: reconciliation | 0.24 | **0.17** | 0.24 | 0.17 |
| Immigration levels | **0.30** | 0.31 | 0.30 | 0.31 |

- **Defence is the one clear model win.** Opinion moved sharply between 2021 and 2025 (CES 2021
  is off by 31 points). The model is closer, possibly because its training also covered more
  recent surveys (Democracy Checkup up to 2024).
- **Immigration fails for both.** Opinion shifted a lot since 2021, and the model never saw the
  question in training. Neither source has a recent reading.
- **Elsewhere, the old survey is closer**, often by a wide margin (childcare, democracy,
  personal finances).
- **For both, most of the error is national.** The national-level error is close to the subgroup
  error: getting Canada's overall 2025 position right is the main difficulty.

# 5. Level versus differences

We split each error into two parts: the **level** (the national distribution) and the
**differences** between subgroups (each subgroup's gap from the national figure).

| Model | Subgroup TV | National-level TV | Size of gaps (1 = real) | Gap correlation | Real national + its gaps |
|---------------|------|------|------|------|------|
| \cellcolor{best}**CES 2021** | \cellcolor{best}**0.151** | \cellcolor{best}**0.141** | \cellcolor{best}**0.69** | \cellcolor{best}**0.71** | \cellcolor{best}**0.064** |
| Fixed, 50k | 0.178 | 0.157 | 0.31 | 0.42 | 0.082 |
| Semantic, 50k | 0.188 | 0.161 | 0.28 | 0.39 | 0.086 |
| Fixed, 100k | 0.191 | 0.172 | 0.35 | 0.47 | 0.078 |
| Semantic, 100k | 0.195 | 0.178 | 0.30 | 0.41 | 0.082 |
| Fixed, 20k | 0.200 | 0.184 | 0.26 | 0.38 | 0.084 |
| Semantic, 20k | 0.214 | 0.194 | 0.25 | 0.33 | 0.089 |
| Untrained | 0.325 | 0.296 | 0.52 | 0.35 | 0.113 |
| \cellcolor{ref}Real national only (no gaps) | \cellcolor{ref}0.085 | \cellcolor{ref}0 | \cellcolor{ref}0 | \cellcolor{ref}— | \cellcolor{ref}0.085 |

*Size of gaps* is the slope of the subgroup gaps on the real ones; *gap correlation* says whether
they point the right way. The last column gives each subgroup the real 2025 national distribution
plus that source's gap for the subgroup. No intervals yet for this table.

- **On the level, CES 2021 is slightly ahead** (0.141 vs 0.157 for Fixed 50k). Neither knows
  well where Canada stands in 2025, and they fail on different questions (defence vs the rest).
- **On differences, CES 2021 is far ahead.** Its gaps are 70% of their real size and correlate
  0.71 with them. The models' gaps are about a third of the real size, with a correlation of 0.4:
  they flatten subgroups toward each other. With the real 2025 national figure, CES 2021 gaps land
  almost on the human floor (0.064 vs 0.062).
- **More data helps differences, not level.** From 20k to 100k (fixed), gap correlation rises from
  0.38 to 0.47, while the level is best at 50k.

# 6. What it means for what comes next

1. **The models do not yet know subgroups better than a four-year-old survey.** That is the honest
   reading. Subgroup differences are stable over time and a past poll captures them well; our
   models learn them only partly.
2. **The right comparison is on questions with no past version.** That is where silicon sampling
   would add something a past poll cannot. The benchmark should add such items (new 2025 questions,
   or questions we hold out of training on purpose) and score them separately.
3. **On repeated questions, models could complement a past poll, not replace it.** Defence shows
   models can carry more recent information. Worth testing: past poll for subgroup gaps, model
   (trained on the latest data) for the shift in level.
4. **The flattening is the main thing to fix.** Gaps a third of their real size limit any use.
   Worth testing: more and richer context answers per respondent, and whether the targets or
   sampling at temperature 1.0 cause it.
5. **Immigration** stays a blind spot until we find a way past Azure's training filter.
6. **Repeat Fixed 50k and Fixed 100k** with a new seed before treating 50k vs 100k as settled.
