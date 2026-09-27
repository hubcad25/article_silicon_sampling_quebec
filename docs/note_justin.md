---
title: "Estimer l'opinion d'un sous-groupe sur une question inédite"
subtitle: "Premiers résultats du pilote (Llama 3.3 70B fine-tuné, 12 items)"
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

J'ai fine-tuné Llama 3.3 70B sur des réponses individuelles de 17 sondages québécois et canadiens,
puis je lui ai demandé de reproduire la **distribution des réponses d'un sous-groupe** (âge × genre ×
scolarité) à des questions qu'il n'a jamais vues. Le pilote porte sur 12 questions et 275 sous-groupes.

- **Le fine-tuning fonctionne.** Il divise par 2,5 la distance aux vrais répondants par rapport au
  modèle de base (variation totale 0,22 contre 0,55) et comble environ 90 % de l'écart entre le modèle
  de base et un plancher « humain » (deux échantillons de vraies personnes : 0,19).
- **Ajouter de l'information sur le sous-groupe n'aide pas.** Donner au modèle les réponses du
  sous-groupe à des questions voisines ne fait pas mieux que le modèle le plus simple.
- **La température compte pour le modèle fine-tuné, presque pas pour le modèle de base**, comme dans
  ta note : forte baisse de 0,3 à 0,7, plateau ensuite.

Des runs deux fois et demie plus longs (20 000 exemples) sont en cours.

## 1. Ce que j'ai fait

### La question

Peut-on estimer comment un sous-groupe répondrait à une **nouvelle** question de sondage, sans
l'avoir jamais posée? L'unité évaluée n'est pas la personne mais la distribution d'un groupe : par
exemple, la répartition des réponses des femmes de 25 à 34 ans titulaires d'un bac à une question
sur la souveraineté.

### Les données

| | |
|---|---|
| Sondages | 17 sondages québécois et canadiens, 1998–2025, en français et en anglais |
| Répondants | 108 199 |
| Questions | 1 778 au catalogue, dont 1 633 utilisables comme cibles d'opinion |
| Test | 60 questions gelées avant tout entraînement ; ce pilote en utilise 12 |
| Unités évaluées | 275 couples question × cellule |

Une **cellule** est une combinaison âge × genre × scolarité (218 couples) ou âge × genre quand les
effectifs ne permettent pas plus fin (57 couples). Les 12 questions pilotes couvrent trois thèmes
(identité et fédéralisme, partis et vote, valeurs sociales), les deux langues, et des questions plus
ou moins proches de celles vues à l'entraînement.

### Le partage des répondants

Le partage est fait **par répondant**, dans chaque sondage. 78 055 répondants servent à
l'entraînement ; 30 144 n'y participent jamais. Ces derniers sont coupés en deux moitiés gelées :
14 947 répondants de **contexte** (qui peuvent nourrir le prompt) et 15 197 répondants
d'**évaluation** (la vérité à reproduire, jamais montrée au modèle).

### Deux modèles fine-tunés, cinq façons de les interroger

Les deux modèles sont entraînés sur les **mêmes 8 000 couples** répondant–question, avec la réponse
réelle du répondant comme cible (1 époque, Azure AI Foundry). Seul le prompt d'entraînement change :

- **FT-Profil** voit le profil sociodémographique et la question ;
- **FT-Voisins** voit en plus les réponses du même répondant à jusqu'à six questions voisines.

À l'inférence, chaque condition reçoit le profil de la cellule, la question et ses options, puis
répond 100 fois par cellule à quatre températures (0,3 ; 0,7 ; 1,0 ; 1,3).

| Condition | Modèle | Contexte ajouté dans le prompt |
|---|---|---|
| **Base** | Llama 3.3 70B, sans fine-tuning | aucun |
| **Profil** | FT-Profil | aucun |
| **Voisins** | FT-Voisins | aucun |
| **Voisins+Cellule** | FT-Voisins | distributions de la cellule sur jusqu'à 6 questions voisines, calculées sur les répondants de contexte |
| **Fuite** | FT-Voisins | mêmes distributions, calculées sur tous les répondants gelés (évaluation incluse) |

Le contexte ne contient jamais la question cible ni sa distribution. **Fuite** n'est pas une méthode :
c'est un contrôle qui mesure si réutiliser les répondants d'évaluation dans le contexte gonfle les
résultats.

![](../data/analysis/inference/figures/schema_protocole.png){width=100%}

### L'évaluation

Pour chaque question × cellule, les 100 réponses forment une distribution, comparée à la distribution
pondérée des répondants d'évaluation par la **variation totale**
($\tfrac{1}{2}\sum_k |p_k - q_k|$ : 0 = identique, 1 = disjoint). On fait la moyenne sur les cellules
de chaque question, puis sur les 12 questions. Les comparaisons entre conditions sont **appariées** sur
les mêmes cellules, avec un bootstrap sur les questions. L'analyse principale est à température 1,0 ;
le protocole a été fixé avant de regarder les résultats.

**Plancher humain.** Pour donner une échelle, on compare aussi, cellule par cellule, la distribution
des répondants de contexte à celle des répondants d'évaluation. C'est la distance entre deux
échantillons de vraies personnes du même profil, soit le bruit d'échantillonnage pur. Les cellules
étant petites (environ 40 répondants par moitié), ce plancher est élevé. Il est aussi un peu pessimiste :
il additionne le bruit de deux échantillons, alors qu'un modèle parfait ne subirait que celui de
l'échantillon d'évaluation.

Nos chiffres ne se comparent pas directement aux tiens : métrique différente (variation totale
plutôt que KL) et évaluation par cellule plutôt que sur la distribution nationale.

## 2. Résultats

### Le fine-tuning s'approche du plancher humain

| Condition | Variation totale | IC 95 % | Écart au plancher humain |
|---|---|---|---|
| Humain (plancher) | **0,186** | [0,153 ; 0,222] | — |
| **Profil** | **0,219** | [0,186 ; 0,257] | +0,033 [0,014 ; 0,051] |
| Voisins | 0,242 | [0,203 ; 0,283] | +0,056 [0,014 ; 0,100] |
| Voisins+Cellule | 0,246 | [0,202 ; 0,288] | +0,059 [0,025 ; 0,097] |
| Fuite | 0,240 | [0,195 ; 0,290] | +0,054 [0,013 ; 0,099] |
| Base | 0,548 | [0,478 ; 0,616] | +0,362 [0,304 ; 0,430] |

*Température 1,0 ; 12 questions, 275 cellules ; IC par bootstrap sur les questions.*

Profil comble environ 90 % de l'écart entre Base et le plancher humain :
(0,548 − 0,219) / (0,548 − 0,186). Il reste significativement au-dessus du plancher, et l'écart grandit
sur les cellules d'au moins 30 répondants, où le plancher est moins bruité (+0,052 [0,031 ; 0,070],
annexe B).

Le modèle de base échoue surtout parce qu'il **répond presque toujours la même chose** : son option
la plus fréquente reçoit 93 % des tirages même à température 1,0, contre 50 % chez les vrais
répondants. Profil retrouve la bonne dispersion (50,5 %). C'est plus marqué que ton bras roleplay
(environ 54 % de part modale) : un 70B instruit, sans fine-tuning, s'effondre sur une réponse.

![](../data/analysis/inference/figures/items_tv.png){width=90%}

Le résultat tient question par question : Profil est plus proche du plancher que de Base sur les 12
questions, et à moins de 0,04 du plancher sur 8 d'entre elles.

### Le contexte du sous-groupe n'aide pas

![](../data/analysis/inference/figures/contrastes_tv.png){width=85%}

- **Voisins+Cellule contre Voisins : +0,003 [−0,038 ; 0,040].** Montrer au modèle comment la cellule a
  répondu à des questions voisines ne change rien en moyenne.
- **Voisins+Cellule contre Profil : +0,026 [−0,005 ; 0,057].** Le montage le plus riche ne bat pas le
  plus simple ; la tendance est même légèrement défavorable.
- **Fuite contre Voisins+Cellule : −0,005 [−0,024 ; 0,012].** Réutiliser les répondants d'évaluation
  dans le contexte ne gonfle pas les résultats : la version propre ne perd rien.

Une piste : FT-Voisins a appris à partir des réponses **individuelles** d'un répondant, mais reçoit
à l'inférence des **distributions** de groupe. Ce changement de format peut l'empêcher d'exploiter
le contexte. Le contexte n'est pas inutile pour autant : il rend au modèle les différences entre
cellules que Profil aplatit (annexe D), sans que ça se traduise en distance moyenne.

### La température : même profil que dans ta note

![](../data/analysis/inference/figures/tv_temperature.png){width=85%}

Les modèles fine-tunés passent de 0,37–0,42 à 0,22–0,25 entre les températures 0,3 et 1,0, puis
plafonnent jusqu'à 1,3. Le modèle de base bouge peu (0,59 à 0,53). C'est le même contraste que tu observais
entre fine-tune et roleplay, sur un modèle 17 fois plus gros : la température agit sur un modèle
entraîné, beaucoup moins sur un modèle simplement prompté. Nous n'avons pas testé 0,0.

## 3. Limites et suite

- **Pilote de 12 questions.** Les 48 autres questions gelées ne sont pas encore évaluées.
- **Petites cellules.** La médiane est de 31 répondants d'évaluation par cellule ; seules 144 des 275
  en ont au moins 30. Les contrastes appariés restent valides, mais les niveaux absolus sont gonflés
  par le bruit, d'où le plancher humain.
- **Décalage entraînement–inférence** pour FT-Voisins (réponses individuelles contre distributions),
  qui pourrait expliquer l'absence de gain du contexte.
- **Un seul modèle de base** (Llama 3.3 70B), une seule époque, 8 000 exemples.
- **En cours :** les mêmes deux modèles entraînés sur 20 000 exemples. Ça testera directement ta
  conclusion selon laquelle entraîner plus longtemps dégrade les résultats.

\newpage

## Annexes

### A. Résultats par question (température 1,0)

| Question | Lang. | Proximité | Cell. | Humain | Profil | Voisins | V.+Cell. | Base |
|--------------------------------|-----|---------|------|------|-----|------|------|-----|
| Souveraineté, langue et culture (CES 2021) | en | quasi-doublon | 43 | 0,13 | 0,16 | 0,25 | 0,13 | 0,63 |
| Importance de l'ethnicité et de la langue (CES 2019) | en | loin | 34 | 0,21 | 0,19 | 0,16 | 0,21 | 0,50 |
| Partage des dépenses fédérales (EEQ 2012) | fr | isolée | 11 | 0,21 | 0,19 | 0,20 | 0,24 | 0,48 |
| Plus de pouvoirs ou indépendance (EEQ 2012) | fr | proche | 11 | 0,17 | 0,21 | 0,26 | 0,18 | 0,38 |
| Meilleur premier ministre (CECD 1998) | fr | loin | 9 | 0,12 | 0,16 | 0,14 | 0,12 | 0,37 |
| Meilleur parti : immigration (EEQ 2018) | fr | modérée | 8 | 0,17 | 0,24 | 0,32 | 0,34 | 0,67 |
| Parti de 2^e^ choix (CES 2019) | en | quasi-doublon | 54 | 0,21 | 0,25 | 0,24 | 0,31 | 0,77 |
| Meilleur parti : économie (CES 2019 tél.) | en | proche | 29 | 0,30 | 0,30 | 0,28 | 0,36 | 0,59 |
| Aide médicale à mourir et avortement (CES 2021) | en | loin | 43 | 0,13 | 0,15 | 0,33 | 0,23 | 0,58 |
| Signes religieux des enseignants (EEQ 2018) | fr | isolée | 8 | 0,09 | 0,16 | 0,15 | 0,26 | 0,39 |
| Impact économique des immigrants (CES 2019 tél.) | en | modérée | 15 | 0,23 | 0,27 | 0,21 | 0,22 | 0,55 |
| Pratique religieuse et société (EEQ 2008) | fr | quasi-doublon | 10 | 0,28 | 0,36 | 0,36 | 0,35 | 0,67 |

*Questions groupées par thème : identité et fédéralisme, partis et vote, valeurs sociales. Proximité : distance sémantique entre la question et la plus proche question d'entraînement.*
Les écarts entre conditions varient beaucoup d'une question à l'autre, mais ni la proximité ni la
langue n'expliquent de tendance nette sur 12 questions (Profil : 0,220 en français, 0,219 en anglais).

### B. Sensibilité : cellules d'au moins 30 répondants

| Condition | Variation totale | IC 95 % | Écart au plancher |
|---|---|---|---|
| Humain | 0,153 | [0,124 ; 0,185] | — |
| Profil | 0,205 | [0,171 ; 0,245] | +0,052 [0,031 ; 0,070] |
| Voisins | 0,240 | [0,189 ; 0,288] | +0,087 [0,048 ; 0,132] |
| Voisins+Cellule | 0,236 | [0,178 ; 0,295] | +0,082 [0,039 ; 0,129] |
| Base | 0,566 | [0,479 ; 0,642] | +0,412 [0,346 ; 0,483] |

*144 cellules, 11 questions : « Impact économique des immigrants » n'a aucune cellule d'au moins 30
répondants d'évaluation. Le protocole prévoit donc cette sensibilité comme descriptive, pas comme
substitut à l'analyse principale sur 12 questions.*

### C. Réponses invalides

Une réponse est invalide si elle ne correspond exactement à aucune option offerte. Aucun tirage n'est
perdu en silence : les erreurs d'API sont réessayées jusqu'à obtenir une réponse ou arrêter le run.

| Condition | T = 0,3 | T = 0,7 | T = 1,0 | T = 1,3 |
|---|---|---|---|---|
| Base | 0,0 % | 0,0 % | 0,0 % | 0,2 % |
| Profil | 0,0 % | 0,0 % | 0,4 % | 4,6 % |
| Voisins | 0,0 % | 0,0 % | 0,4 % | 4,8 % |
| Voisins+Cellule | 0,0 % | 0,0 % | 0,5 % | 5,2 % |
| Fuite | 0,0 % | 0,0 % | 0,5 % | 5,2 % |

À 1,3, les modèles fine-tunés commencent à produire du texte hors options, ce qui plaide pour 1,0.

### D. Aplatissement des différences entre cellules

Ratio entre la variance des distributions prédites d'une cellule à l'autre et la variance observée
(1 = le modèle différencie les groupes autant que la réalité ; 0 = même réponse pour tous). Médiane
sur les options, température 1,0, calculée seulement sur les cellules d'au moins 100 répondants
(3 questions).

| Base | Profil | Voisins | Voisins+Cellule | Fuite |
|---|---|---|---|---|
| 0,00 | 0,57 | 0,55 | 1,09 | 1,06 |

Le modèle de base donne la même distribution à tous les groupes. Les deux modèles sans contexte
compressent les écarts entre groupes de moitié environ ; le contexte de la cellule les rétablit.
À prendre avec prudence : 3 questions seulement.

### E. Détails techniques

- **Fine-tuning** : Llama-3.3-70B-Instruct sur Azure AI Foundry (serverless), 1 époque, lot de 64,
  multiplicateur de taux d'apprentissage 1, graine 20260924. La cible est le texte exact de l'option
  choisie (pas son code), pour que le modèle puisse transférer vers des questions inédites dont les
  codes diffèrent.
- **Langue** : chaque répondant voit la question, le contexte et le profil dans sa langue de
  passation ; une question sans version complète dans cette langue est exclue plutôt que traduite.
- **Gels** : le partage des questions et des répondants a été gelé avant le fine-tuning ; le partage
  contexte/évaluation et les règles d'analyse, avant de consulter les résultats.
- **Correspondance avec les fichiers** : Base = R, Profil = A (modèle C0), Voisins = B0,
  Voisins+Cellule = BS, Fuite = B (modèle C1).
