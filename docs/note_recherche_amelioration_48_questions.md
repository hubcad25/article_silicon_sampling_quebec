---
title: "Peut-on faire mieux que 0,219?"
author: "Hubert Cadieux"
date: "À compléter"
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

**Réponse.** À compléter.
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

La régularisation réduit les écarts prédits entre sous-groupes vers la moyenne de la question. Son
intensité est fixée sur les 12 questions initiales : 0,20 pour **Entraîné 8k** et 0,55 pour la moyenne
des quatre modèles.

# Résultats

| Méthode | Variation totale | Écart avec Entraîné 8k | IC 95 % |
|---|---:|---:|---|
| Entraîné 8k | — | — | — |
| Entraîné 20k | — | — | — |
| Indices retirés 20k | — | — | — |
| Avec répondants 20k | — | — | — |
| Moyenne sans répondants | — | — | — |
| Moyenne des quatre modèles | — | — | — |
| Entraîné 8k régularisé | — | — | — |
| Moyenne régularisée | — | — | — |

**Meilleur résultat : à compléter.**

## Ce qui fait une différence

- **20k contre 8k :** à compléter.
- **Répondants réels, à modèle constant :** à compléter.
- **Combinaison des modèles :** à compléter.
- **Régularisation des écarts entre sous-groupes :** à compléter.

## Où est-ce que ça marche?

On retient d'abord la meilleure méthode sur l'ensemble des 48 questions. On vérifie ensuite si son
gain tient selon le thème et selon la proximité de la question avec le corpus d'entraînement. On ne
choisit pas une méthode différente après coup pour chaque catégorie.

| Thème | Questions | Entraîné 8k | Méthode retenue | Écart |
|---|---:|---:|---:|---:|
| Partis et vote | 9 | — | — | — |
| Démocratie et engagement | 9 | — | — | — |
| Valeurs sociales | 8 | — | — | — |
| Santé | 6 | — | — | — |
| État et économie | 6 | — | — | — |
| Économie perçue | 5 | — | — | — |
| Identité et fédéralisme | 5 | — | — | — |

| Proximité avec l'entraînement | Questions | Entraîné 8k | Méthode retenue | Écart |
|---|---:|---:|---:|---:|
| Isolée (< 0,70) | 10 | — | — | — |
| Loin (0,70–0,775) | 9 | — | — | — |
| Modérée (0,775–0,85) | 10 | — | — | — |
| Proche (0,85–0,95) | 10 | — | — | — |
| Quasi-doublon (≥ 0,95) | 9 | — | — | — |

**Lecture : à compléter.** Le gain demeure-t-il sur les questions isolées? Le 20k aide-t-il surtout
loin du corpus? Une méthode échoue-t-elle dans un thème précis?

# Décision

**Méthode retenue : à compléter.**

**Gain par rapport à Entraîné 8k : à compléter.**

**Pourquoi : à compléter.**
