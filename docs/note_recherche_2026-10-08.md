---
title: "CES 2025 Benchmark: First Results"
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
- \newcommand{\notesubtitle}{Six fine-tuned models · 150 subgroup scores each · what to keep}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · CES 2025 benchmark}
---

```{=latex}
\begin{encadre}
```
**In brief.** Fine-tuning works: our best model is about **twice as close** to real CES 2025
subgroups as the same model untrained. The best recipe is **fixed context questions with 50k
training examples**. More data (100k) did not help. We are still about 12 points above the
human noise floor, and immigration is the weakest item by far.
```{=latex}
\end{encadre}
```

# 1. What we tested

Six Llama 3.3 70B models, fine-tuned on 8 Canadian surveys (CES 2019 and 2021, Democracy Checkup
2019–2024). None of them saw any CES 2025 data. They differ on two points:

- **Context format.** At training, each example shows 9 other answers from the same respondent.
  *Fixed*: the same 9 themes every time (ideology, party, trust…). *Semantic*: the 9 questions
  closest in meaning to the target.
- **Training size.** 20k, 50k or 100k examples. Each larger set contains the smaller one.

Every model answered the same frozen set: 15,000 synthetic respondents from the Democracy Checkup
2024 (5 regions × 3 age groups), 10 CES 2025 questions, 150,000 calls. We compare the answer
distribution of each of the 150 subgroup × question pairs with the real CES 2025 one.

**How to read the score.** Total variation (TV) is the share of answers that would have to change
to match the real distribution. A TV of 0.18 means about **18 answers out of 100** are in the
wrong place. Lower is better.

# 2. Results

| Model | Mean TV (150 pairs) | Answers out of place, per 100 |
|------------------------------|-----------|-----------|
| Human floor (two halves of CES 2025) | 0.062 | 6 |
| \cellcolor{best}**Fixed, 50k** | \cellcolor{best}**0.178** | \cellcolor{best}**18** |
| Semantic, 50k | 0.188 | 19 |
| Fixed, 100k | 0.191 | 19 |
| Semantic, 100k | 0.195 | 20 |
| Fixed, 20k | 0.200 | 20 |
| Semantic, 20k | 0.214 | 21 |
| Untrained Llama 3.3 70B | 0.325 | 33 |
| Same answer for everyone (most common one) | 0.540 | 54 |

**Three takeaways.**

1. **Training matters most.** Going from untrained to the best model removes 15 out-of-place answers
   per 100 (0.325 → 0.178). That is more than half the distance to the human floor.
2. **Fixed context beats semantic context** at every size: by 1.4 points at 20k, 1.1 at 50k and
   0.4 at 100k. Fixed context is also
   simpler: the same questions for every item, and no risk of showing a near-copy of the target.
3. **50k is the sweet spot.** 20k → 50k helps (about −2.5 points). 50k → 100k hurts slightly for
   both formats (+1.4 points fixed, +0.7 semantic).

Each of these comparisons is statistically clear on our test set (paired bootstrap, 95% intervals
exclude zero). Invalid replies are rare (under 1% for every model).

## Where the best model struggles

| Question | Fixed, 50k | Untrained | Human floor |
|------------------------------|---------|---------|---------|
| Spending: affordable housing | 0.10 | 0.15 | 0.05 |
| Satisfaction with democracy | 0.12 | 0.28 | 0.05 |
| Medical assistance in dying | 0.14 | 0.34 | 0.09 |
| Jobs before environment | 0.15 | 0.40 | 0.10 |
| Spending: defence | 0.17 | 0.23 | 0.04 |
| Pipelines | 0.17 | 0.43 | 0.10 |
| Spending: national childcare | 0.18 | 0.40 | 0.06 |
| Personal finances, past year | 0.21 | 0.25 | 0.04 |
| Spending: reconciliation | 0.24 | 0.45 | 0.05 |
| \cellcolor{worse}**Immigration levels** | \cellcolor{worse}**0.30** | \cellcolor{worse}0.32 | \cellcolor{worse}0.04 |

**Immigration** barely improves with training. It is also the topic most absent from training:
Azure's content check rejected most immigration items. Regions and age groups behave alike (0.17–0.18), except
the Atlantic provinces (0.20), which also have the smallest real samples.

# 3. Caveats

- **One training run per recipe.** The intervals cover the test data, not training luck. We do not yet
  know how much a second run of the same recipe would move the score. Gaps of about one point (50k vs
  100k, fixed vs semantic) need a repeat before we call them real.
- **One base model.** All six use Llama 3.3 70B. The ranking may differ for smaller models.

# 4. Next steps

1. **Keep "Fixed, 50k"** as our reference model.
2. **Add Justin's model** (Qwen3-4B). It cannot run in our Azure setup (no GPU access there), so it
   would run on his Mac with the same frozen file and the same scoring script.
3. **Repeat Fixed 50k and Fixed 100k** with a new seed to check that the size effect holds.
4. **Immigration**: find a way to get immigration questions past the training filter, or report it as
   a known blind spot.
