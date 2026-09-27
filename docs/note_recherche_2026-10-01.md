---
title: "Améliorer l'estimation de l'opinion d'un sous-groupe"
author: "Hubert Cadieux"
date: "1er octobre 2026"
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
- \newcommand{\notesubtitle}{Plan du deuxième test · modèles 20k · répondants réels · modèle statistique}
- \input{note_style.tex}
---

```{=latex}
\begin{encadre}
```
**Document de travail.** Cette note préparera le deuxième brief de résultats. Elle répondra à trois
questions laissées ouvertes par le premier test : est-ce que davantage d'exemples d'entraînement
améliorent les estimations, est-ce que les indices deviennent utiles lorsqu'ils sont présentés comme
les réponses d'une vraie personne, et comment les modèles de langage se comparent-ils à un modèle
statistique entraîné sur les mêmes données?
```{=latex}
\end{encadre}
```

# 1. Point de départ

Le premier brief, daté du 27 septembre, établit trois résultats sur 12 questions et 275
sous-groupes :

- **Entraîné 8k** réduit fortement la distance aux vrais répondants par rapport à **Non entraîné**;
- **Entraîné avec indices 8k** ne bat ni **Entraîné 8k** ni **Indices retirés 8k**;
- une température de 1,0 donne le meilleur compromis et est retenue pour la suite.

Les modèles **Entraîné 20k** et **Entraîné avec indices 20k** ont depuis terminé leur entraînement.
La prochaine étape ne répète pas le balayage de température : toutes les nouvelles inférences sont
faites à température 1,0, sur les mêmes 275 couples question × sous-groupe et avec 100 réponses par
couple.

# 2. Les trois questions du deuxième test

## Est-ce que plus d'entraînement aide?

Le passage de 8 000 à 20 000 exemples est évalué séparément pour les deux entraînements :

- **Entraîné 20k** contre **Entraîné 8k**;
- **Indices retirés 20k** contre **Indices retirés 8k**.

La première comparaison mesure l'effet du volume pour le modèle qui n'a jamais reçu de réponses
voisines. La seconde le mesure pour le modèle entraîné à lire les réponses voisines, mais sans lui en
donner à l'inférence.

## Les indices fonctionnent-ils dans le format appris?

La nouvelle condition est appelée **Entraîné avec répondants**. Chaque réponse simulée reçoit les
réponses individuelles d'un répondant réel du même sous-groupe à des questions voisines, dans le
format utilisé à l'entraînement. La distribution du sous-groupe émerge des réponses simulées pour
plusieurs personnes plutôt que d'un résumé en pourcentages.

Cette condition est évaluée avec les modèles 8k et 20k. Les comparaisons principales sont :

- **Entraîné avec répondants 8k − Indices retirés 8k** : effet des réponses individuelles;
- **Entraîné avec répondants 8k − Entraîné avec indices 8k** : nouveau format contre pourcentages;
- **Entraîné avec répondants 8k − Entraîné 8k** : meilleur montage contre modèle simple;
- **Entraîné avec répondants 20k − Indices retirés 20k** : effet des réponses individuelles à 20k;
- **Entraîné avec répondants 20k − Entraîné 20k** : meilleur montage 20k contre modèle simple 20k;
- **Entraîné avec répondants 20k − Entraîné avec répondants 8k** : effet du volume dans le nouveau
  montage.

Le diagnostic prévu dans l'ADR 0005 sera d'abord calculé sur les résultats existants afin de vérifier
si les pourcentages poussaient le modèle dans la bonne direction tout en exagérant les différences et
en concentrant trop les réponses. Ce diagnostic ne demande aucun nouvel appel au modèle.

## Les modèles de langage battent-ils le modèle statistique?

Le **modèle statistique** reçoit le profil sociodémographique, la question et ses options. Il est
entraîné sur les mêmes 8 000 exemples que **Entraîné 8k** et produit directement une distribution,
sans échantillonner 100 réponses. Il sera comparé au plancher humain et à toutes les conditions 8k,
sans choisir après coup le modèle de langage qui lui est le plus favorable.

# 3. Travail restant

## Quatre campagnes d'inférence

| Nouvelle campagne | Modèle utilisé | Information ajoutée à l'inférence | Appels |
|---|---|---|---:|
| **Entraîné 20k** | entraîné sans réponses voisines, 20k | aucune | 27 500 |
| **Indices retirés 20k** | entraîné avec réponses voisines, 20k | aucune | 27 500 |
| **Entraîné avec répondants 8k** | entraîné avec réponses voisines, 8k | réponses individuelles | 27 500 |
| **Entraîné avec répondants 20k** | entraîné avec réponses voisines, 20k | réponses individuelles | 27 500 |
| **Total** | | | **110 000** |

Il n'est pas nécessaire de refaire **Entraîné avec indices** avec le modèle 20k pour répondre aux
questions principales. La comparaison entre les pourcentages et les réponses individuelles est déjà
isolée avec le modèle 8k. Une campagne 20k avec les anciens pourcentages ne serait ajoutée que pour
tester explicitement une interaction entre volume d'entraînement et ancien format des indices.

## Traitements sans nouvelle inférence

1. terminer l'ajustement et les prédictions du **modèle statistique**;
2. intégrer au brief le diagnostic maintenant calculé des anciens indices;
3. intégrer les quatre nouvelles campagnes à la même analyse de variation totale, de concentration
   et d'aplatissement des différences entre sous-groupes;
4. produire les intervalles et contrastes appariés selon les règles déjà utilisées dans le premier
   brief.

# 4. Structure prévue du brief final

## Diagnostic des anciens indices en pourcentages

À température 1,0, l'ajout des indices déplace les probabilités dans la direction des écarts humains
pour **47,1 % [43,4 % ; 50,7 %]** des unités question × cellule × option. La pente des écarts entre
cellules est de **0,144 [0,059 ; 0,215]** pour **Entraîné avec indices**, contre
**0,087 [0,033 ; 0,127]** pour **Indices retirés**. Les indices augmentent donc un peu la dispersion,
mais les deux conditions demeurent très loin de reproduire toute l'ampleur des écarts humains
(pente de 1).

L'entropie moyenne est de **1,188 [0,929 ; 1,460]** nat chez les répondants et de
**1,268 [1,027 ; 1,506]** avec les indices. La différence de **+0,080 [0,021 ; 0,142]** indique que
les réponses synthétiques sont moins concentrées, et non davantage. Ainsi, les trois prédictions de
l'hypothèse 3 de l'ADR 0005 ne sont pas vérifiées : la direction n'est pas meilleure que le hasard,
les différences restent fortement aplaties et BS a une entropie supérieure à celle des répondants.

Les calculs détaillés et le texte reproductible se trouvent dans
`data/analysis/second_brief/diagnostic_anciens_indices_*`. Les intervalles à 95 % reposent sur un
bootstrap apparié par question; les pentes sont calculées après centrage par question et option.

## En bref

Trois résultats à remplir après l'analyse : effet du passage à 20k, effet des réponses individuelles
et comparaison avec le modèle statistique.

## Est-ce que 20k bat 8k?

Tableau des distances moyennes, contrastes appariés et résultats question par question pour
**Entraîné** et **Indices retirés**.

## Est-ce que les répondants réels rendent les indices utiles?

Comparaison de **Entraîné avec répondants** avec **Indices retirés**, **Entraîné avec indices** et
**Entraîné**, puis examen de la concentration et des différences entre sous-groupes.

## LLM ou modèle statistique?

Comparaison des erreurs, de la calibration, du coût et de la simplicité opérationnelle. La différence
entre une distribution exacte du modèle statistique et une distribution estimée à partir de 100
réponses du LLM sera indiquée explicitement.

## Quel montage retenir?

Le brief se terminera par une décision avant l'évaluation des 48 questions gelées restantes : modèle
retenu, format des indices retenu et pertinence ou non d'augmenter encore le volume d'entraînement.

# 5. Tableaux et figures à préparer

- tableau principal : plancher humain, modèle statistique et toutes les conditions 8k et 20k retenues;
- figure des contrastes appariés pour les six comparaisons préannoncées;
- figure par question montrant l'effet de 20k et celui des répondants réels;
- tableau de concentration des réponses;
- diagnostic des anciens indices : direction, amplification et entropie;
- diagnostic d'aplatissement entre sous-groupes, recalculé pour **Entraîné avec répondants**.

# 6. État d'avancement

| Élément | État au 1er octobre |
|---|---|
| **Entraîné 20k** | entraînement terminé; inférence à faire |
| **Indices retirés 20k** | entraînement terminé; inférence à faire |
| **Entraîné avec répondants 8k** | protocole fixé; implémentation et inférence à faire |
| **Entraîné avec répondants 20k** | protocole fixé; implémentation et inférence à faire |
| **Modèle statistique** | implémentation en cours; exécution complète à terminer |
| Diagnostic des anciens indices | terminé; résultats intégrés ci-dessus |
