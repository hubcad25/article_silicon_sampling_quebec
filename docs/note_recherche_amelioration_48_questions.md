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

# Décision

**Méthode retenue : à compléter.**

**Gain par rapport à Entraîné 8k : à compléter.**

**Pourquoi : à compléter.**

Les contrastes sont appariés sur les mêmes questions et sous-groupes. Les intervalles à 95 % sont
obtenus par bootstrap des 48 questions. La divergence KL est conservée dans les résultats détaillés
comme vérification secondaire.
