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
- \newcommand{\notesubtitle}{Résultats saillants · décision · prochaines étapes}
- \input{note_style.tex}
- \fancyhead[L]{\footnotesize\color{muted} Silicon sampling au Québec · deuxième test}
---

```{=latex}
\begin{encadre}
```
**Conclusion.** Retenir **Entraîné 8k** pour la prochaine étape. Sur les 12 questions testées, passer
à 20k exemples ou ajouter les réponses de personnes réelles n'améliore pas clairement les
estimations. Le modèle 8k affiche la meilleure distance moyenne, avec le montage le plus simple et
le moins coûteux. Il faut maintenant confirmer ce choix sur les 48 questions gelées restantes.
```{=latex}
\end{encadre}
```

# 1. Résultats saillants

La **variation totale** mesure l'écart entre les répartitions simulée et observée. Une valeur de
0,219 signifie qu'il faudrait, en moyenne, **réaffecter au minimum environ 22 réponses sur 100** à
une autre option pour reproduire la répartition observée. Il ne s'agit pas d'un taux d'erreur
individuel. La divergence KL, moins directement interprétable, est conservée comme mesure de
robustesse dans les résultats détaillés.

| Résultat | Ce que montrent les données | Implication |
|---|---|---|
| **20k ne bat pas 8k** | Entraîné 20k exige 22,4 réaffectations sur 100, contre 21,9 pour Entraîné 8k. L'écart est incertain : +0,5 [−1,9 ; +3,1]. | Ne pas investir davantage dans le volume d'entraînement pour l'instant. |
| **Les réponses individuelles n'ajoutent pas de gain démontré** | À 20k, le modèle avec répondants et le modèle simple sont à égalité : 22,4 réaffectations sur 100 chacun. À 8k, le modèle avec répondants demeure légèrement derrière le modèle simple. | Mettre en pause les montages avec indices, plus complexes et plus coûteux. |
| **Le LLM entraîné bat le modèle statistique** | Entraîné 8k exige environ 22 réaffectations sur 100, contre 28 pour le modèle statistique : un avantage de **6 réponses sur 100** [2 ; 9]. | Conserver le modèle statistique comme référence, mais pas comme méthode principale. |

Le plancher humain se situe à environ **19 réaffectations sur 100**. **Entraîné 8k** n'en est séparé
que par environ 3 réponses sur 100, mais ce plancher reflète lui-même le bruit de deux échantillons de
répondants.

## Ce qui explique l'échec des indices

Présenter les indices sous forme de réponses individuelles corrige en partie la faiblesse des anciens
pourcentages, sans dépasser le modèle simple. Le diagnostic des anciens indices montre pourquoi :
ils font bouger les probabilités, mais pas plus souvent dans la bonne direction que dans la mauvaise
(47,1 %), et reproduisent trop peu les écarts réels entre sous-groupes. Leur signal est donc peu
exploitable, plutôt que trop concentré sur une réponse dominante.

# 2. Décision et prochaines étapes

## Montage retenu

**Entraîné 8k**, à température 1,0, sans indices. C'est le meilleur résultat moyen observé (0,219),
avec le montage le plus simple et le moins coûteux parmi les variantes entraînées.

## Priorités

1. **Évaluer Entraîné 8k sur les 48 questions gelées restantes** pour vérifier que son avantage se
   généralise au-delà des 12 premières questions.
2. **Terminer les diagnostics susceptibles de nuancer la décision** : résultats par question,
   concentration des réponses et aplatissement des différences entre sous-groupes.
3. **Ne pas lancer de nouvelle campagne 20k ou avec indices** à moins qu'un diagnostic révèle un gain
   ciblé sur certains types de questions ou de sous-groupes.
4. **Garder le modèle statistique dans l'évaluation finale** comme référence plus simple et sans
   échantillonnage, malgré sa précision moyenne inférieure.

## Limite à garder en tête

Ces conclusions portent sur 12 questions et 275 couples question × sous-groupe. Les intervalles
excluent un gain moyen important de plusieurs variantes, mais pas de petits effets ni des avantages
ciblés. La validation sur les questions gelées doit donc précéder toute conclusion générale.

# Annexe A — Résultats détaillés

<!-- DÉBUT RÉSULTATS AUTOMATIQUES -->

<!-- Généré par scripts/29_prepare_second_brief.py; ne pas modifier à la main. -->

## Résultats disponibles

Nouvelles campagnes complètes incluses : **Entraîné 20k, Indices retirés 20k, Entraîné avec répondants 8k, Entraîné avec répondants 20k**. Les quatre nouvelles campagnes sont complètes.

| Condition | Variation totale | IC 95 % | KL | Sous-groupes |
|---|---|---|---|---|
| Plancher humain | 0,186 | [0,153 ; 0,222] | — | 275 |
| Entraîné 8k | 0,219 | [0,186 ; 0,255] | 0,207 | 275 |
| Indices retirés 20k | 0,222 | [0,189 ; 0,257] | 0,209 | 275 |
| Entraîné avec répondants 20k | 0,224 | [0,192 ; 0,254] | 0,206 | 275 |
| Entraîné 20k | 0,224 | [0,181 ; 0,268] | 0,212 | 275 |
| Entraîné avec répondants 8k | 0,226 | [0,191 ; 0,259] | 0,207 | 275 |
| Indices retirés 8k | 0,242 | [0,203 ; 0,280] | 0,256 | 275 |
| Entraîné avec indices 8k | 0,246 | [0,201 ; 0,288] | 0,251 | 275 |
| Modèle statistique | 0,281 | [0,235 ; 0,330] | 0,277 | 275 |
| Non entraîné | 0,548 | [0,479 ; 0,618] | 1,628 | 275 |

## Contrastes préannoncés disponibles

| Contraste | Différence TV | IC 95 % | Différence KL |
|---|---|---|---|
| Entraîné 8k − Modèle statistique | -0,062 | [-0,089 ; -0,019] | -0,069 |
| Entraîné 20k − Entraîné 8k | 0,005 | [-0,019 ; 0,031] | 0,005 |
| Entraîné 20k − Modèle statistique | -0,057 | [-0,075 ; -0,018] | -0,064 |
| Indices retirés 20k − Indices retirés 8k | -0,020 | [-0,052 ; 0,010] | -0,047 |
| Entraîné avec répondants 20k − Entraîné 20k | -0,000 | [-0,033 ; 0,034] | -0,006 |
| Entraîné avec répondants 20k − Indices retirés 20k | 0,001 | [-0,023 ; 0,027] | -0,003 |
| Entraîné avec répondants 20k − Entraîné avec répondants 8k | -0,003 | [-0,027 ; 0,019] | -0,001 |
| Entraîné avec répondants 20k − Modèle statistique | -0,057 | [-0,093 ; -0,005] | -0,071 |
| Entraîné avec répondants 8k − Entraîné 8k | 0,007 | [-0,018 ; 0,039] | 0,000 |
| Entraîné avec répondants 8k − Indices retirés 8k | -0,016 | [-0,055 ; 0,025] | -0,049 |
| Entraîné avec répondants 8k − Entraîné avec indices 8k | -0,019 | [-0,043 ; 0,008] | -0,044 |
| Entraîné avec répondants 8k − Modèle statistique | -0,055 | [-0,092 ; 0,002] | -0,069 |

<!-- FIN RÉSULTATS AUTOMATIQUES -->

# Annexe B — Protocole en bref

- Analyse principale à température 1,0 sur 12 questions et 275 couples question × sous-groupe.
- Chaque condition LLM produit 100 réponses par couple; le modèle statistique produit directement
  une distribution.
- Quatre nouvelles campagnes complètes : **Entraîné 20k**, **Indices retirés 20k**, **Entraîné avec
  répondants 8k** et **Entraîné avec répondants 20k**, pour 110 000 appels au total.
- Le modèle statistique et **Entraîné 8k** utilisent les mêmes 8 000 exemples d'entraînement.
- Les comparaisons sont appariées et leurs intervalles à 95 % reposent sur un bootstrap par question.

# Annexe C — Diagnostic des anciens indices en pourcentages

À température 1,0, l'ajout des indices déplace les probabilités dans la direction des écarts humains
pour **47,1 % [43,4 % ; 50,7 %]** des unités question × cellule × option. La pente des écarts entre
cellules est de **0,144 [0,059 ; 0,215]** pour **Entraîné avec indices**, contre
**0,087 [0,033 ; 0,127]** pour **Indices retirés**. Les indices augmentent donc un peu la dispersion,
mais les deux conditions demeurent très loin de reproduire toute l'ampleur des écarts humains
(pente de 1).

L'entropie moyenne est de **1,188 [0,929 ; 1,460]** nat chez les répondants et de
**1,268 [1,027 ; 1,506]** avec les indices. La différence de **+0,080 [0,021 ; 0,142]** indique que
les réponses synthétiques sont moins concentrées, et non davantage.

Les trois prédictions de l'hypothèse 3 de l'ADR 0005 ne sont donc pas vérifiées. Les pourcentages font
bouger le modèle, mais ne l'aident pas à repérer de façon fiable ce qui distingue un sous-groupe. Le
format semble surtout fournir un signal peu exploitable.

Les calculs détaillés et le texte reproductible se trouvent dans
`data/analysis/second_brief/diagnostic_anciens_indices_*`. Les intervalles à 95 % reposent sur un
bootstrap apparié par question; les pentes sont calculées après centrage par question et option.
