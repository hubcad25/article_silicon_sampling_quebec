---
title: "Estimer l'opinion d'un sous-groupe sur une nouvelle question"
subtitle: "Protocole d'évaluation du pilote"
author: "Hubert Cadieux"
date: "Septembre 2026"
lang: fr
geometry: margin=2.2cm
fontsize: 10pt
header-includes:
- \usepackage{float}
- \usepackage{graphicx}
- \usepackage{pdflscape}
- \floatplacement{figure}{H}
---

## En bref

Cette étude teste si un modèle de langage peut reproduire la **distribution des réponses** à une
nouvelle question dans un groupe sociodémographique. Il ne s'agit pas de prédire chaque personne.
Pour chaque question et chaque cellule, le modèle répond 100 fois; la distribution synthétique est
ensuite comparée à celle de répondants réels jamais utilisés pour le fine-tuning.

Trois questions guident l'expérience :

1. Le fine-tuning sur des réponses de sondage améliore-t-il le modèle de base?
2. Des réponses à des questions voisines améliorent-elles l'estimation d'une nouvelle question?
3. Les modèles de langage font-ils mieux qu'un modèle statistique utilisant la même information?

## Données

| | |
|---|---|
| Sondages | 17 sondages québécois et canadiens, de 1998 à 2025 |
| Répondants | 108 199 |
| Questions au catalogue | 1 778, dont 1 633 utilisables comme cibles d'opinion |
| Questions de test | 60 questions gelées avant le fine-tuning |
| Pilote | 12 questions dans 3 blocs thématiques |
| Unités évaluées | 275 couples question × cellule sociodémographique |
| Langues | français et anglais |

Les 12 questions pilotes portent sur l'identité québécoise et le fédéralisme, les valeurs sociales,
ainsi que les partis et le vote. Elles couvrent différents degrés de proximité avec les questions
d'entraînement, du thème isolé au quasi-doublon.

```{=latex}
\begin{landscape}
\begin{figure}[H]
\centering
\includegraphics[width=0.88\linewidth]{../data/analysis/inference/figures/schema_protocole.png}
\caption{Construction des modèles, conditions d'inférence et références d'évaluation.}
\end{figure}
\end{landscape}
```

Le partage des données est fait par répondant. Les 78 055 répondants de la branche d'entraînement
peuvent servir au fine-tuning; les 30 144 autres n'y participent jamais. Ces derniers sont ensuite
séparés, dans chaque sondage et chaque cellule, en 14 947 répondants de contexte et 15 197
répondants d'évaluation.

### Deux modèles fine-tunés

Les modèles C0 et C1 sont entraînés sur les **mêmes 8 000 couples répondant-question**. Seule
l'information dans le prompt change :

- **C0** reçoit le profil sociodémographique et la question cible;
- **C1** reçoit aussi les réponses individuelles du même répondant à un maximum de six questions
  voisines.

Dans les deux cas, la cible d'entraînement est la réponse individuelle réellement donnée. Les 60
questions de test et les 30 144 répondants gelés sont exclus du fine-tuning.

### Conditions d'inférence

Toutes les conditions reçoivent le même profil sociodémographique, la même question cible et les
mêmes options de réponse.

- **R** interroge le modèle de base, sans fine-tuning ni contexte supplémentaire.
- **A-C0** interroge le modèle C0 sans contexte supplémentaire.
- **B0-C1** interroge le modèle C1 sans lui montrer de contexte. Cette condition isole ce que C1 a
  appris pendant le fine-tuning.
- **B-C1** ajoute les distributions de la cellule sur un maximum de six questions voisines. Elles
  sont calculées avec les 30 144 répondants gelés; les répondants d'évaluation y contribuent donc.
- **BS-C1** ajoute les mêmes types de distributions, mais les calcule seulement avec les 14 947
  répondants de contexte. Les personnes utilisées pour le contexte et l'évaluation sont alors
  distinctes.

Le contexte ne contient jamais la réponse ni la distribution de la question cible. Il contient
uniquement des distributions observées sur des questions voisines du même sondage.

### Benchmark non-LLM

La condition **S** sera un logit conditionnel régularisé avec embeddings textuels figés. Elle recevra
le profil sociodémographique, le texte de la question et celui des options, puis produira directement
une probabilité pour chaque option. Elle sera entraînée sur les mêmes 8 000 exemples que C0 et C1,
sans aucune réponse aux questions de test.

S est un benchmark statistique : il permet de déterminer si un modèle de langage apporte quelque
chose au-delà d'une méthode discriminative entraînée sur la même information. Il n'est pas encore
implémenté et aucun résultat ne lui est attribué dans cette version de la note.

## Évaluation

Chaque condition LLM produit 100 réponses pour chaque couple question × cellule, à quatre
températures : 0,3; 0,7; 1,0 et 1,3. Ces réponses forment une distribution synthétique. Le modèle
statistique S produira directement une distribution de probabilités.

Toutes les conditions seront comparées à la même référence : la distribution pondérée de la
question cible chez les **15 197 répondants d'évaluation**, dans les mêmes cellules. La température
1,0 est l'analyse principale; les autres températures servent à vérifier la robustesse.

La fidélité sera résumée par une distance entre la distribution prédite et la distribution observée :
plus cette distance est faible, meilleure est la reproduction. Les comparaisons entre conditions sont
appariées sur les mêmes questions et les mêmes cellules. Les taux de réponses invalides, la sensibilité
à la température et l'aplatissement des différences entre cellules seront présentés en annexe.

La référence **humain-humain** compare la distribution de la question cible dans la moitié contexte
à celle de la moitié évaluation. Elle mesure le désaccord attribuable à l'échantillonnage. Elle ne
constitue pas une méthode pour répondre à une nouvelle question, puisqu'elle utilise déjà des réponses
à cette question.

## Comparaisons

| Comparaison | Question posée |
|---|---|
| **A-C0 vs R** | Le fine-tuning direct améliore-t-il le modèle de base? |
| **B0-C1 vs A-C0** | L'entraînement avec des réponses voisines change-t-il le modèle, même sans contexte à l'inférence? |
| **B-C1 vs B0-C1** | Le contexte agrégé améliore-t-il C1 lorsque contexte et évaluation se chevauchent? |
| **BS-C1 vs B0-C1** | Le contexte agrégé améliore-t-il C1 sur des répondants indépendants? |
| **B-C1 vs BS-C1** | Quelle part du gain apparent vient du chevauchement des répondants? |
| **Meilleure condition LLM vs S** | Le LLM ajoute-t-il quelque chose au benchmark statistique? |

La comparaison B-C1 contre BS-C1 ne porte pas sur deux modèles différents : les deux utilisent le
même modèle C1. Seule la provenance des distributions de contexte change.

## Portée

Ce pilote fournit un premier test sur 12 questions; il ne représente pas encore l'ensemble des 60
questions gelées. Après le second partage, seulement 144 des 275 couples question-cellule conservent
au moins 30 réponses d'évaluation. Les résultats absolus des petites cellules seront donc interprétés
avec prudence, et les règles d'agrégation devront être fixées avant d'examiner les comparaisons.

Le modèle C1 est entraîné avec des réponses **individuelles**, alors que B-C1 et BS-C1 reçoivent des
distributions **agrégées** à l'inférence. Cette différence correspond au produit envisagé, mais elle
reste un changement de format entre l'entraînement et l'utilisation.

## État du travail

- [x] Split des questions et des répondants gelé avant le fine-tuning
- [x] Modèles C0 et C1 entraînés sur les mêmes 8 000 exemples
- [x] Inférences R, A-C0, B0-C1 et B-C1 terminées
- [x] Second partage contexte-évaluation gelé
- [ ] Inférence BS-C1
- [ ] Benchmark statistique S
- [ ] Analyse comparative sur la moitié d'évaluation

Aucun résultat substantif n'est présenté ici avant que BS-C1, le benchmark S et les règles finales
d'analyse soient disponibles.
