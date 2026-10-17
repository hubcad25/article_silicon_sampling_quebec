---
title: "Silicon Sampling on CES 2025: Three Ways to Extend the Model"
author: "Hubert Cadieux & Justin Savoie"
date: "October 16, 2026"
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
- \newcommand{\notesubtitle}{New questions, longer context, and a dossier-based analyst · results so far}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · CES 2025 · joint summary}
- \definecolor{hgood}{HTML}{8FD3A8}
- \definecolor{hmid}{HTML}{FFF2B3}
- \definecolor{hbad}{HTML}{F29A8E}
- \newcommand{\tbd}{\textcolor{muted}{\textit{tbd}}}
---

# 1. What we tested

The test: predict how 15 subgroups (5 regions × 3 age groups) answered CES 2025 questions, using
only data from before 2025. The score is **TV**, the share of answers placed in the wrong spot,
averaged over subgroups and questions — lower is better. As a point of reference, we also compute
the **floor**: how much a second real CES 2025 poll of the same size would miss by, just from
ordinary sampling luck. No model can beat the floor; it is not a competitor, it is the ceiling of
what is achievable.

We now score two sets of questions. The **repeated set** (10 questions) were all asked in CES 2021
too, so a model could, in principle, get credit for having seen them before. The **new set** (8
questions) were never asked in any of our 8 training surveys — US relations with the US since
Trump's election; who is to blame for worse relations; protecting weaker nations from foreign
aggression; whether safe injection sites make neighbourhoods unsafe; whether campus protests have
gone too far; returning land and resources to Indigenous peoples; worry about one's own housing
security; and whether government should guarantee decent housing for all. These are the closest
thing we have to what a real client would ask: a question nobody has polled before.

```{=latex}
\begin{center}\small
\begin{tabular}{lp{8.6cm}p{4.2cm}}
\toprule
Short label & Question & Answers \\
\midrule
US relations & Has the relationship between Canada and the US become better or worse after Trump's election? & Better / worse / same \\
Who is to blame & Who is responsible for worsening relations between the US and Canada? & Canada / US / both \\
Protect weaker nations & Protecting weaker nations from foreign aggression, as a foreign policy goal & Very / somewhat / not important \\
Safe injection sites & Safe injection sites make neighbourhoods unsafe. & Agree, 5 points \\
Campus protests & Student protests on university campuses have gone too far. & Agree, 5 points \\
Indigenous land & Return control over land and natural resources to Indigenous peoples. & Support, 4 points \\
Housing worry & How worried are you about the security of your housing situation in the next few years? & Worry, 4 points \\
Housing for all & The government has a responsibility to provide decent housing for all. & Agree, 5 points \\
\bottomrule
\end{tabular}
\end{center}
```

**One caveat:** "Who is to blame" was asked to only part of the sample, probably those who said
relations got worse. The model answers it for everyone, which may unfairly inflate its error on
that question.

We are testing three different ways of getting a model to produce these answers:

```{=latex}
\begin{center}\small
\begin{tabular}{lp{5.6cm}p{5.6cm}}
\toprule
Approach & What the model sees & What it outputs \\
\midrule
A. Role-play, short context & 7 of one real person's own past answers & That one person's guessed answer \\
B. Role-play, long context & About 16 of one real person's own past answers & That one person's guessed answer \\
C. Dossier-to-distribution & A write-up of how Canadians and each subgroup answered similar past questions & The full percentage table for all 15 subgroups, in one go \\
\bottomrule
\end{tabular}
\end{center}
```

A and B repeat this one-person-at-a-time guess 15,000 times (once per fake respondent) and tally
the results into percentages. C skips the role-play: the model acts as an analyst and writes the
table directly. For repeated questions, any past version of the question worded exactly the same is
removed before the model sees anything, so even "repeated" questions are new to it.

# 2. Results

```{=latex}
\begin{center}\small
\begin{tabular}{l>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{3.4cm}}
\toprule
 & Repeated (10) & New (8) & Same error as a real poll of… per subgroup \\
\midrule
A. Role-play, 7 answers & \cellcolor{hgood!69!hmid}0.178 & \cellcolor{hgood!55!hmid}0.243 & 14 (repeated) · \tbd (new) \\
B. Role-play, 16 answers & \tbd & \tbd & \tbd \\
C. Dossier, untrained Llama & \cellcolor{hgood!65!hmid}0.196 & \cellcolor{hgood!50!hmid}0.264 & \tbd \\
C. Dossier, trained by example & \tbd & \tbd & \tbd \\
C. Dossier, trained by reward & \tbd & \tbd & \tbd \\
\textcolor{muted}{C. Dossier, untrained GPT-5.5 (flagged)} & \textcolor{muted}{0.106} & \textcolor{muted}{0.144} & \textcolor{muted}{see note below} \\
\midrule
CES 2021 reused & \cellcolor{hgood!74!hmid}0.151 & \textcolor{muted}{n/a} & 20 (repeated) \\
National mode, for everyone & \cellcolor{hbad!100!hmid}0.540 & \cellcolor{hbad!100!hmid}0.544 & — \\
\midrule
{\color{muted}Floor (two halves of CES 2025)} & \textcolor{muted}{0.062} & \textcolor{muted}{0.074} & — \\
\bottomrule
\end{tabular}
\end{center}
```

Reading the last column: 0.151 is worth about 20 real respondents per subgroup (300 in total), 0.10
about 40 (600), 0.08 about 65 (1,000, a standard poll).

**About the GPT-5.5 line:** a recent, general-purpose model scores well on both sets with no
training at all. We show it, muted, as a caution rather than a result. Its training data run into
2026 and may already contain 2025 opinion data, or even CES 2025 itself — on the national-level
defence spending question it is almost perfect, which is a sign of this, not of skill. We cannot
tell how much of its score is genuine reasoning versus having effectively seen the answer key.

```{=latex}
\needspace{14\baselineskip}
```

**New questions, one by one** (role-play with 7 answers, the untrained dossier, and the floor; the
rest fill in as training finishes):

```{=latex}
\begin{center}\small
\begin{tabular}{l>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.9cm}c}
\toprule
New question & A. 7 answers & B. 16 answers & C. Dossier (untrained) & {\color{muted}Floor} \\
\midrule
US relations & \cellcolor{hbad!82!hmid}0.416 & \tbd & \cellcolor{hbad!88!hmid}0.445 & {\color{muted}0.030} \\
Who is to blame & \cellcolor{hbad!96!hmid}0.479 & \tbd & \cellcolor{hbad!81!hmid}0.409 & {\color{muted}0.033} \\
Protect weaker nations & \cellcolor{hgood!70!hmid}0.172 & \tbd & \cellcolor{hbad!64!hmid}0.331 & {\color{muted}0.045} \\
Safe injection sites & \cellcolor{hgood!70!hmid}0.173 & \tbd & \cellcolor{hgood!55!hmid}0.240 & {\color{muted}0.104} \\
Campus protests & \cellcolor{hgood!60!hmid}0.220 & \tbd & \cellcolor{hgood!59!hmid}0.222 & {\color{muted}0.102} \\
Indigenous land & \cellcolor{hgood!77!hmid}0.139 & \tbd & \cellcolor{hgood!56!hmid}0.239 & {\color{muted}0.109} \\
Housing worry & \cellcolor{hgood!63!hmid}0.202 & \tbd & \cellcolor{hgood!80!hmid}0.124 & {\color{muted}0.099} \\
Housing for all & \cellcolor{hgood!76!hmid}0.143 & \tbd & \cellcolor{hgood!85!hmid}0.099 & {\color{muted}0.070} \\
\midrule
\textbf{Mean} & \cellcolor{hgood!55!hmid}\textbf{0.243} & \tbd & \cellcolor{hgood!50!hmid}\textbf{0.264} & {\color{muted}\textbf{0.074}} \\
\bottomrule
\end{tabular}
\end{center}
```

The two US–Trump questions (0.416 and 0.479 for the role-play model) are the clear outliers in both
A and the untrained dossier. On the other six new questions, the role-play model averages about
0.175 — the same as its score on repeated questions.

# 3. What we learn so far

- **New questions are not the problem, 2025 events are.** The role-play model scores about the same
  on new and repeated questions, except for the two questions about US relations under Trump. Those
  ask about something that happened in 2025, after the model's training data ends — it cannot know
  the answer, no matter how the question is asked. Questions that are merely *new* in wording, but
  not about events after the model's knowledge cutoff, are no harder than ones it has seen before.
- **An untrained analyst is already about as good as a trained respondent.** With no training at
  all, simply handing the model a written summary of related past results (approach C) scores close
  to the fine-tuned role-play model (approach A) on both question sets. That is a strong signal that
  training the dossier approach is worth doing.
- **A model's own knowledge cutoff limits how it can be judged.** Models whose training data reach
  into 2025 or 2026 may already have absorbed 2025 opinion data, or the survey itself. A good score
  from such a model is not trustworthy evidence of genuine prediction, which is why we flag the
  GPT-5.5 result rather than count it as a result.
- **Still pending:** whether a longer, 16-answer context (B) meaningfully reduces error over the
  7-answer version (A), and whether training the dossier approach — by example or by reward —
  improves on its untrained score. Both are running.

# 4. Next steps

- Finish scoring B at 50,000 training examples, then repeat at 100,000 and 150,000, each size
  trained twice, to see whether more data helps and how much results vary run to run.
- Finish training C by example and by reward, and compare both against the untrained dossier and
  against A and B.
- Once all rows are filled in, decide whether to keep investing in longer role-play context (B) or
  move resources into the dossier approach (C), based on which gets closest to the floor for the
  least effort.
