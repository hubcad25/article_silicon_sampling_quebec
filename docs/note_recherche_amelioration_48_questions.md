---
title: "Peut-on faire mieux que 0,219?"
author: "Hubert Cadieux"
date: "29 septembre 2026"
lang: fr
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
- \usepackage{graphicx}
- \floatplacement{figure}{H}
- \newcommand{\notesubtitle}{48 nouvelles questions · 250 tirages}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling au Québec · troisième test}
---

```{=latex}
\begin{encadre}
```
**Question.** Peut-on battre le résultat de **0,219** obtenu avec **Entraîné 8k**?

**Réponse.** **Pas sur ces nouvelles questions.** Le meilleur résultat est **0,260** avec **Avec
répondants 20k**, contre 0,304 pour **Entraîné 8k** sur les mêmes 48 questions. Le gain est de 0,044
(IC 95 % : [0,023 ; 0,067]), mais le résultat demeure au-dessus du 0,219 obtenu sur les 12 questions
initiales.
```{=latex}
\end{encadre}
```

# Test

Les 48 questions gelées restantes donnent **885 couples question × sous-groupe**. Chaque bras produit
250 réponses par couple, à température 1,0.

| Bras | Modèle | Information ajoutée |
|---|---|---|
| **Entraîné 8k** | standard 8k | aucune |
| **Entraîné 20k** | standard 20k | aucune |
| **Indices retirés 20k** | entraîné avec réponses voisines, 20k | aucune |
| **Avec répondants 20k** | entraîné avec réponses voisines, 20k | réponses d'une personne réelle |

On teste aussi, sans nouvelle inférence :

- la moyenne des quatre modèles;
- la moyenne des trois modèles sans réponses individuelles;
- une version régularisée d'**Entraîné 8k**;
- une version régularisée de la moyenne des quatre modèles.

La régularisation réduit les écarts prédits entre sous-groupes vers la moyenne de la question. La
part de l'écart conservée est fixée sur les 12 questions initiales : 0,20 pour **Entraîné 8k** et 0,55
pour la moyenne des quatre modèles.

# Résultats

| Méthode | Variation totale | Écart avec Entraîné 8k | IC 95 % |
|---|---:|---:|---|
| Entraîné 8k | 0,304 | réf. | — |
| Entraîné 20k | 0,285 | −0,020 | [−0,035 ; −0,005] |
| Indices retirés 20k | 0,290 | −0,014 | [−0,038 ; 0,007] |
| Avec répondants 20k | **0,260** | **−0,044** | **[−0,067 ; −0,023]** |
| Moyenne sans répondants | 0,280 | −0,025 | [−0,037 ; −0,013] |
| Moyenne des quatre modèles | 0,269 | −0,035 | [−0,050 ; −0,022] |
| Entraîné 8k régularisé | 0,301 | −0,003 | [−0,005 ; −0,001] |
| Moyenne régularisée | 0,269 | −0,036 | [−0,050 ; −0,022] |

**Meilleur résultat : 0,260 avec Avec répondants 20k.**

## Ce qui fait une différence

- **20k contre 8k :** le passage à 20k réduit la variation totale de 0,020
  ([−0,035 ; −0,005]). Le volume supplémentaire aide clairement.
- **Répondants réels, à modèle constant :** leur ajout réduit la variation totale de 0,030 par
  rapport à **Indices retirés 20k** ([−0,046 ; −0,015]). C'est l'amélioration la plus convaincante.
- **Combinaison des modèles :** les deux moyennes améliorent **Entraîné 8k**, mais la moyenne des
  quatre (0,269) reste derrière **Avec répondants 20k** (0,260). La différence entre les deux n'est
  toutefois pas nette ([−0,021 ; 0,002] pour Avec répondants 20k moins la moyenne).
- **Régularisation des écarts entre sous-groupes :** elle aide légèrement **Entraîné 8k** (−0,003),
  mais ne change pratiquement pas la moyenne des quatre (−0,000; [−0,001 ; 0,001]).

## Où est-ce que ça marche?

On retient d'abord la meilleure méthode sur l'ensemble des 48 questions. On vérifie ensuite si son
gain tient selon le thème et selon la proximité de la question avec le corpus d'entraînement. On ne
choisit pas une méthode différente après coup pour chaque catégorie.

| Thème | Questions | Entraîné 8k | Méthode retenue | Écart |
|---|---:|---:|---:|---:|
| Partis et vote | 9 | 0,422 | 0,327 | −0,095 |
| Démocratie et engagement | 9 | 0,276 | 0,238 | −0,037 |
| Valeurs sociales | 8 | 0,259 | 0,272 | +0,013 |
| Santé | 6 | 0,239 | 0,219 | −0,020 |
| État et économie | 6 | 0,301 | 0,269 | −0,032 |
| Économie perçue | 5 | 0,277 | 0,204 | −0,073 |
| Identité et fédéralisme | 5 | 0,328 | 0,252 | −0,075 |

| Proximité avec l'entraînement | Questions | Entraîné 8k | Méthode retenue | Écart |
|---|---:|---:|---:|---:|
| Isolée (< 0,70) | 10 | 0,322 | 0,262 | −0,060 |
| Loin (0,70–0,775) | 9 | 0,300 | 0,263 | −0,038 |
| Modérée (0,775–0,85) | 10 | 0,296 | 0,255 | −0,040 |
| Proche (0,85–0,95) | 10 | 0,329 | 0,288 | −0,042 |
| Quasi-doublon (≥ 0,95) | 9 | 0,272 | 0,230 | −0,042 |

**Lecture :** le gain demeure dans les cinq niveaux de proximité et il est même le plus grand sur les
questions isolées. Il n'augmente donc pas simplement avec la proximité du corpus. Le passage de 8k à
20k aide lui aussi surtout les questions isolées; son avantage est presque nul dans la catégorie
« loin ». **Avec répondants 20k** améliore six thèmes sur sept, mais recule de 0,013 sur les valeurs
sociales.

# Décision

**Méthode retenue : Avec répondants 20k.**

**Gain par rapport à Entraîné 8k : 0,044, soit une réduction de 14,6 % de la variation totale
([0,023 ; 0,067]).**

**Pourquoi :** c'est la plus faible variation totale observée; son avantage est net à la fois contre
**Entraîné 8k** et contre le même modèle 20k privé des réponses individuelles. Il améliore aussi le
résultat dans toutes les catégories de proximité. Sa faiblesse sur les valeurs sociales et l'absence
d'avantage clair sur la moyenne des quatre modèles doivent néanmoins rester visibles.

Les contrastes sont appariés sur les mêmes questions et sous-groupes. Les intervalles à 95 % sont
obtenus par bootstrap des 48 questions. Chaque question reçoit le même poids après calcul de la
moyenne de ses sous-groupes. La divergence KL, qui mène aux mêmes conclusions générales, est
conservée dans les résultats détaillés.
