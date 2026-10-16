---
title: "Silicon Sampling on CES 2025: Questions Never Asked Before"
author: "Hubert Cadieux & Justin Savoie"
date: "TBD"
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
- \newcommand{\notesubtitle}{Questions never asked before}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling · CES 2025 · joint summary}
- \definecolor{hgood}{HTML}{8FD3A8}
- \definecolor{hmid}{HTML}{FFF2B3}
- \definecolor{hbad}{HTML}{F29A8E}
- \newcommand{\tbd}{{\color{muted}\textit{tbd}}}
---

# 1. What changed

Our first 10 questions had all been asked in CES 2021. A 2021 poll reused as is beat every model
on them (0.151 against 0.178). That rival only exists for repeated questions: a client will ask
questions no past poll has asked. So we added a **second set of 8 CES 2025 questions that appear in
none of our 8 training surveys** (no shared variable, and no past question close in wording). Same
test as before: 15 subgroups (5 regions × 3 ages), mean **TV**, the share of answers in the wrong
place (lower is better). Same fake respondents (Democracy Checkup 2024 profiles) as for the first set.

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

- **One caveat:** "Who is to blame" was asked to only part of the sample (18,000 of 20,000),
  probably those who said relations got worse. The model answers it for everyone.
- "Housing for all" comes from the post-election wave, the others from the campaign wave.

# 2. Results

```{=latex}
\begin{center}\small
\begin{tabular}{l>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{3.4cm}}
\toprule
 & Repeated questions (10) & New questions (8) & Same error as a real poll of… per subgroup \\
\midrule
Fine-tuned Llama 70B & \cellcolor{hgood!15!hmid}0.178 & \tbd & 14 (repeated) · \tbd \\
Dossier, off-the-shelf model & \tbd & \tbd & \tbd \\
Item encoder (Justin) & \tbd & \tbd & \tbd \\
CES 2021 reused & \cellcolor{hgood!32!hmid}0.151 & {\color{muted}n/a} & 20 (repeated) \\
National mode, for everyone & \cellcolor{hbad!100!hmid}0.540 & \tbd & — \\
\midrule
{\color{muted}Floor (two halves of CES 2025)} & {\color{muted}0.062} & \tbd & — \\
\bottomrule
\end{tabular}
\end{center}
```

Reading the last column: 0.10 is worth about 40 real respondents per subgroup (600 in total), 0.08
about 65 (1,000, a standard poll).

```{=latex}
\begin{center}\small
\begin{tabular}{l>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.9cm}c}
\toprule
New question & Fine-tuned Llama 70B & Dossier & Item encoder & {\color{muted}Floor} \\
\midrule
US relations & \tbd & \tbd & \tbd & \tbd \\
Who is to blame & \tbd & \tbd & \tbd & \tbd \\
Protect weaker nations & \tbd & \tbd & \tbd & \tbd \\
Safe injection sites & \tbd & \tbd & \tbd & \tbd \\
Campus protests & \tbd & \tbd & \tbd & \tbd \\
Indigenous land & \tbd & \tbd & \tbd & \tbd \\
Housing worry & \tbd & \tbd & \tbd & \tbd \\
Housing for all & \tbd & \tbd & \tbd & \tbd \\
\midrule
\textbf{Mean} & \tbd & \tbd & \tbd & \tbd \\
\bottomrule
\end{tabular}
\end{center}
```

**What to read in it** (to fill once the numbers are in):

- **If the fine-tuned model scores about the same on new questions as on repeated ones (≈ 0.18):**
  it does not lean on having seen the questions; its value is real for new questions, where no
  past poll can compete.
- **If it does clearly worse on new questions:** part of its 0.178 came from questions close to
  its training. The product case then rests on the dossier or the item encoder.
- **If the dossier beats the item encoder:** the language model adds something beyond "similar
  questions, similar gaps". That justifies training it (section 3).
- **If the item encoder wins or ties:** a plain statistical model does the job; no need to train
  a language model to write distributions.

```{=latex}
\newpage
```

# 3. Next steps

## B. A fine-tuned model that knows each person better (in progress)

Same model and same task, with 15 to 20 of the person's answers instead of 7. Training at 50,000,
100,000 and 150,000 examples, each size twice, to see how much results vary from one training run
to the next. Scored on both sets. Results in the next note.

## C. From a dossier to a distribution: train it, if it earns it

We train the dossier approach **only if** the off-the-shelf dossier beats the item encoder on the
new questions. Every question of our 8 surveys then becomes a practice exercise (its dossier built
without it), with practice and test questions on different topics. Two ways to learn: by example
(shown the right table), or by reward (a small program scores each proposed table).

If the item encoder wins, we stop C and put the effort into B and the product option already
measured: one real national figure plus the model's subgroup gaps (0.082 on the repeated set).
