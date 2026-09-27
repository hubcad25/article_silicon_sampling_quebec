---
title: "Estimer l'opinion d'un sous-groupe sur une question inédite"
author: "Hubert Cadieux"
date: "Septembre 2026"
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
- \usepackage{pdflscape}
- \usepackage{afterpage}
- \newcommand{\notesubtitle}{Premiers résultats · Llama 3.3 70B fine-tuné · 12 questions, 275 sous-groupes}
- \input{note_style.tex}
---

```{=latex}
\begin{encadre}
```
**En bref.** Ce premier test vérifie si un modèle de langage peut reproduire la répartition des réponses
d'un sous-groupe sociodémographique à une question de sondage qu'il n'a jamais vue.

- **L'entraînement sur des réponses de sondage fonctionne.** Le modèle entraîné divise par 2,5 la
  distance aux vrais répondants et s'approche d'un plancher humain (la distance entre deux
  échantillons de vraies personnes).
- **Donner au modèle des indices sur le groupe n'aide pas**, du moins dans le format testé : lui
  montrer comment le sous-groupe a répondu à des questions voisines n'améliore pas la distance moyenne.
- **La température compte pour le modèle entraîné, presque pas pour le modèle non entraîné**, ce
  comme dans les tests de Justin. Une température de 1,0 est adéquate.
```{=latex}
\end{encadre}
```

# 1. Question et design

## La question

Peut-on estimer comment un sous-groupe répondrait à une **nouvelle** question de sondage, sans
l'avoir posée? L'unité évaluée n'est pas la personne mais la distribution d'un groupe : par exemple,
la répartition des réponses des femmes de 25 à 34 ans titulaires d'un baccalauréat à une question
sur la souveraineté.

## Données et partage

| | |
|---|---|
| Sondages | 17 sondages québécois et canadiens, 1998–2025, en français et en anglais |
| Répondants | 108 199 |
| Questions | 1 778 au catalogue, dont 1 633 utilisables comme cibles d'opinion |
| Test | 60 questions gelées avant tout entraînement ; ce premier test en évalue 12 |
| Unités évaluées | 275 couples question × sous-groupe |

Un **sous-groupe** (ou cellule) croise l'âge, le genre et la scolarité, ou seulement l'âge et le
genre quand les effectifs sont trop faibles.

Seules 12 des 60 questions de test sont évaluées ici, pour limiter le coût d'inférence avant
d'engager toute la batterie : quatre questions dans chacun de trois thèmes (identité et
fédéralisme, partis et vote, valeurs sociales), dans les deux langues. Les 48 autres restent
gelées pour la suite.

Le partage est fait **par répondant**. 78 055 répondants servent à l'entraînement ; 30 144 n'y
participent jamais et sont coupés en deux moitiés : 14 947 répondants de **contexte**, qui peuvent
servir d'indices, et 15 197 répondants d'**évaluation**, la vérité à reproduire. Le schéma de la
page suivante résume le protocole.

```{=latex}
\begin{landscape}
\thispagestyle{empty}
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth,height=0.92\textheight,keepaspectratio]{../data/analysis/inference/figures/schema_protocole.png}
\end{figure}
\end{landscape}
```

## Deux modèles, cinq conditions

Deux modèles sont entraînés à partir de Llama 3.3 70B, sur les **mêmes 8 000 exemples**
répondant–question, avec la réponse réelle du répondant comme cible :

- le **modèle standard** voit le profil du répondant et la question ;
- le **modèle à indices** voit en plus les réponses du même répondant à six questions voisines au
  plus. Il apprend ainsi à se servir d'indices sur la personne.

Le design pose deux questions : l'entraînement aide-t-il? et des indices sur le groupe aident-ils?
D'où trois conditions principales et deux contrôles.

| Condition | Modèle | Ce qu'il reçoit en plus du profil et de la question |
|---|---|---|
| **Non entraîné** | Llama 3.3 70B d'origine | rien |
| **Entraîné** | standard | rien |
| **Entraîné + indices** | à indices | comment le sous-groupe a répondu à six questions voisines au plus (répondants de contexte) |
| *Indices retirés* (contrôle) | à indices | rien ; isole l'effet de l'entraînement à indices |
| *Fuite* (contrôle) | à indices | les mêmes indices, calculés en incluant les répondants d'évaluation |

Les indices ne contiennent jamais la question cible ni sa distribution. Chaque condition répond
100 fois par sous-groupe, à quatre températures (0,3 ; 0,7 ; 1,0 ; 1,3).


## Évaluation

Les 100 réponses d'une condition forment une distribution, comparée à celle des répondants
d'évaluation du même sous-groupe par la **variation totale** : la part des réponses qu'il faudrait
déplacer pour passer d'une distribution à l'autre (0 = identiques, 1 = disjointes). La moyenne est
prise sur les sous-groupes de chaque question, puis sur les 12 questions. Les comparaisons entre
conditions sont appariées sur les mêmes sous-groupes, avec un bootstrap sur les questions.
L'analyse principale est à température 1,0.

**Plancher humain.** Pour situer les chiffres, la même distance est calculée entre les répondants de
contexte et ceux d'évaluation, sous-groupe par sous-groupe : deux échantillons de vraies personnes du
même profil. Avec environ 40 répondants par moitié, ce plancher est élevé. Il est aussi un peu
pessimiste, puisqu'il additionne le bruit de deux échantillons.

La métrique (variation totale plutôt que KL) et le niveau d'évaluation (sous-groupe plutôt que
distribution nationale) diffèrent de ceux de Justin : les niveaux absolus ne se comparent pas
directement aux siens.

# 2. Résultats

## L'entraînement rapproche le modèle du plancher humain

| Condition | Variation totale | IC 95 % | Écart au plancher humain |
|---|---|---|---|
| \cellcolor{ref}Humain (plancher) | \cellcolor{ref}0,186 | \cellcolor{ref}[0,153 ; 0,222] | \cellcolor{ref}— |
| Entraîné | \cellcolor{best}0,219 | \cellcolor{best}[0,186 ; 0,257] | \cellcolor{best}+0,033 [0,014 ; 0,051] |
| Entraîné + indices | 0,246 | [0,202 ; 0,288] | +0,059 [0,025 ; 0,097] |
| *Indices retirés* | 0,242 | [0,203 ; 0,283] | +0,056 [0,014 ; 0,100] |
| *Fuite* | 0,240 | [0,195 ; 0,290] | +0,054 [0,013 ; 0,099] |
| Non entraîné | 0,548 | [0,478 ; 0,616] | +0,362 [0,304 ; 0,430] |

*Température 1,0 ; 12 questions, 275 sous-groupes. Dans tous les tableaux : \colorbox{best}{vert} = meilleur résultat,
\colorbox{worse}{rouge} = résultat défavorable, \colorbox{ref}{gris} = plancher humain ou vrais répondants (référence).*

Le modèle entraîné comble environ 90 % de l'écart entre le modèle non entraîné et le plancher
humain. Il reste significativement au-dessus du plancher, et l'écart grandit sur les sous-groupes
d'au moins 30 répondants, où le plancher est moins bruité (annexe B).

Le modèle non entraîné échoue surtout parce qu'il **répond presque toujours la même chose** : son
option la plus fréquente reçoit 93 % des tirages, contre 50 % chez les vrais répondants. Le modèle
entraîné retrouve la bonne dispersion (51 %). L'effondrement est plus marqué que celui du bras
roleplay de Justin, autour de 54 %.

![](../data/analysis/inference/figures/items_tv.png){width=88%}

Le constat tient question par question : le modèle entraîné est plus proche du plancher que du
modèle non entraîné sur les 12 questions, et à moins de 0,04 du plancher sur 8 d'entre elles.

## Les indices sur le groupe n'améliorent pas la distance moyenne

![](../data/analysis/inference/figures/contrastes_tv.png){width=88%}

- **Effet des indices** (Entraîné + indices contre Indices retirés) : +0,003 [−0,038 ; 0,040]. Montrer
  au modèle comment le sous-groupe a répondu à des questions voisines ne change rien en moyenne.
- **Meilleur montage** (Entraîné + indices contre Entraîné) : +0,026 [−0,005 ; 0,057]. Le montage le
  plus riche ne bat pas le plus simple ; à température 0,7, il est même significativement moins bon.
- **Fuite** : −0,005 [−0,024 ; 0,012]. Inclure les répondants d'évaluation dans les indices ne gonfle
  pas les résultats.

Deux éléments nuancent ce constat. D'abord, le modèle à indices a appris à lire les réponses
**individuelles** d'une personne, alors qu'on lui donne ici des **distributions** de groupe ; ce
changement de format peut l'empêcher d'en tirer parti. Ensuite, les indices ont un effet visible
ailleurs : ils rétablissent les différences entre sous-groupes que le modèle entraîné aplatit de
moitié (annexe D), même si ça ne se traduit pas en distance moyenne.

## Selon la température

![](../data/analysis/inference/figures/tv_temperature.png){width=85%}

| Condition | T = 0,3 | T = 0,7 | T = 1,0 | T = 1,3 |
|---|---|---|---|---|
| Non entraîné | 0,590 | 0,568 | 0,548 | 0,532 |
| Entraîné | \cellcolor{best}0,366 | \cellcolor{best}0,239 | \cellcolor{best}0,219 | \cellcolor{best}0,220 |
| Entraîné + indices | 0,415 | 0,283 | 0,246 | 0,246 |
| *Indices retirés* | 0,369 | 0,266 | 0,242 | 0,243 |
| *Fuite* | 0,402 | 0,265 | 0,240 | 0,231 |
| \cellcolor{ref}Humain (plancher) | \cellcolor{ref}0,186 | \cellcolor{ref}0,186 | \cellcolor{ref}0,186 | \cellcolor{ref}0,186 |

*Variation totale moyenne. Part de l'option la plus fréquente et contrastes par température : annexe C.*

- **Les modèles entraînés dépendent fortement de la température** : à 0,3, ils concentrent trop leurs
  réponses (81 % sur l'option la plus fréquente pour le modèle entraîné) ; à 1,0, ils retrouvent la
  dispersion observée ; à 1,3, la distance ne bouge plus mais les réponses hors options apparaissent.
- **Le modèle non entraîné bouge à peine** : il reste concentré sur une option à toutes les températures.
- **Les conclusions ne dépendent pas de la température** : l'entraînement aide nettement partout,
  et les indices n'aident nulle part.

C'est le même contraste que Justin observait entre son fine-tune de 4 milliards de paramètres
et un modèle prompté : la température agit sur un modèle entraîné, beaucoup moins sur un modèle
simplement prompté. La température 0 n'a pas été testée ici.

## Ce qui se compare aux tests de Justin

Les deux projets posent la même question de fond (un modèle entraîné sur des réponses de sondage
fait-il mieux qu'un modèle simplement prompté?), mais avec des montages différents.

| | Tests de Justin | Ici |
|---|---|---|
| Modèle entraîné | Qwen3 4B, LoRA, sur un portable | Llama 3.3 70B, sur Azure |
| Données | 2 sondages canadiens (DC 2024, CES 2025) | 17 sondages québécois et canadiens, français et anglais |
| Questions de test | 10 | 12 (sur 60 gelées) |
| Ce qu'on reproduit | distribution nationale | distribution par sous-groupe |
| Métrique principale | KL | variation totale (KL en complément) |
| Comparaison « prompté » | roleplay d'un autre modèle | même modèle de base, non entraîné |

**Directement comparable**

- **Même volume d'entraînement.** Le checkpoint retenu par Justin (2 000 itérations × lots de 4) a
  vu 8 000 exemples, exactement comme nos modèles. Son checkpoint plus long (environ 17 400 exemples)
  correspond presque à nos runs de 20 000 en cours : ce sera une réplication directe de son résultat
  « entraîner plus longtemps dégrade ».
- **Effet de la température sur le modèle entraîné.** En KL, le modèle entraîné s'améliore d'un
  facteur 3,4 entre les températures 0,3 et 1,0 chez Justin, et de 3,3 ici. Même ampleur, sur un
  modèle 17 fois plus gros et une tâche plus difficile.
- **Concentration des réponses** (part de l'option la plus fréquente) : même statistique dans les deux
  notes.

| Température | 0,3 | 0,7 | 1,0 |
|---|---|---|---|
| Fine-tune de Justin | ≈ 58 % | ≈ 43 % | 39 % |
| Roleplay de Justin | ≈ 53 % | ≈ 54 % | 54 % |
| Entraîné (ici) | 81 % | 60 % | 51 % |
| Non entraîné (ici) | 98 % | 95 % | 93 % |

*Valeurs de Justin lues sur sa figure 2 (« ≈ »). Vrais répondants ici : 50 %.*

**Même direction, ampleur non comparable**

- **Entraîné contre prompté.** Chez Justin, le fine-tune est 5,7 fois meilleur que le roleplay en KL ;
  ici, 7,9 fois meilleur que le modèle non entraîné en KL (2,5 fois en variation totale). Le niveau
  d'évaluation et le modèle prompté diffèrent : seule la direction se compare.

**Ce qui diverge**

- **Le modèle prompté.** Le roleplay de Justin dispersait ses réponses et variait peu avec la
  température ; notre modèle non entraîné, lui, s'effondre sur une seule réponse et ne distingue pas
  les sous-groupes. Un modèle instruit sans entraînement, avec un gabarit de sondage strict, se
  comporte différemment d'un roleplay avec persona.
- **Les différences entre groupes.** Chez Justin, les modèles échantillonnés exagèrent les écarts entre
  groupes (rapport de 1,7 à 2,0) ; ici, le modèle entraîné les compresse de moitié (0,57) et seuls
  les indices les rétablissent. Les deux calculs ne sont pas identiques, et le nôtre repose
  sur 3 questions seulement : à confirmer avant d'en tirer une conclusion.

# 3. Ce qu'on retient et où investir

**Ce qui a marché**

- Entraîner le modèle sur des réponses individuelles de sondage : c'est l'essentiel du gain, sur
  toutes les questions, dans les deux langues et les trois thèmes.
- Répondre par le texte de l'option (plutôt qu'un code) : les réponses restent valides sur des
  questions inédites.
- Le protocole : questions et répondants gelés avant l'entraînement, aucun tirage perdu, un contrôle
  de fuite qui confirme que l'évaluation n'est pas contaminée.

**Ce qui n'a pas marché**

- Les indices sur le groupe, dans le format actuel : ils ne battent pas le modèle le plus simple.
- Le modèle non entraîné, même avec un bon prompt : il donne la même réponse à tout le monde.

**Où mettre l'effort, par ordre de rendement attendu**

1. **Plus de questions de test.** L'incertitude sur les comparaisons vient surtout du petit nombre
   de questions. Les 48 questions gelées restantes ne demandent que de l'inférence, avec les
   modèles déjà entraînés. C'est le moyen le moins cher de rendre les conclusions solides.
2. **De plus gros sous-groupes d'évaluation.** Avec une quarantaine de répondants par sous-groupe,
   le bruit d'échantillonnage est presque aussi grand que l'erreur du modèle : on ne peut plus voir
   une amélioration. Il faut privilégier des questions tirées de grands sondages, ou des sous-groupes
   plus larges, plutôt que plus de répondants à l'entraînement.
3. **Plus de données d'entraînement : l'angle le moins exploité, à trancher avec les runs en cours.**
   Les modèles ont vu 8 000 paires répondant–question, soit 6 995 répondants avec, le plus souvent,
   une seule de leurs réponses. La branche d'entraînement contient environ 6,4 millions de réponses
   valides : ces 8 000 paires en représentent à peine 0,1 %, et les runs de 20 000 en cours, 0,3 %.
   Deux directions restent intactes :
   - **le volume** : des dizaines ou des centaines de milliers de paires restent abordables (le coût
     d'entraînement croît linéairement, environ 7 $ US par tranche de 8 000 pour le modèle standard) ;
   - **la profondeur par répondant** : entraîner sur plusieurs réponses d'une même personne, pour que
     le modèle apprenne comment les opinions d'un individu se tiennent entre elles. C'est précisément
     ce que le modèle à indices cherche à exploiter.

   Justin trouvait toutefois qu'entraîner plus longtemps dégradait les résultats. Les runs de
   20 000 diront si le volume aide ici avant d'aller plus loin.
4. **Les indices, avec un format cohérent.** Une seule expérience ciblée : entraîner le modèle
   directement avec des distributions de groupe, comme il les reçoit à l'inférence. Les indices
   rétablissent déjà les écarts entre groupes ; c'est la piste la plus plausible pour la question
   centrale du projet.

**Ce qu'on peut laisser de côté**

- Balayer davantage la température : 1,0 fonctionne, et 1,3 n'apporte que des réponses invalides.
- Le modèle non entraîné comme méthode : il ne sert plus que de point de comparaison.
- La crainte de fuite entre indices et évaluation : le contrôle ne montre aucun effet.

**Limites.** 12 questions sur 60 ; sous-groupes petits (médiane de 31 répondants d'évaluation) ;
un seul modèle de base, une seule époque. Les règles d'analyse principales ont été fixées avant de
consulter les résultats ; le plancher humain, prévu au protocole, a été calculé ensuite.

\newpage

# Annexes

## A. Résultats par question (température 1,0)

| Question | Lang. | Proximité | Cell. | Humain | Entraîné | E. + ind. | Ind. ret. | Non entr. |
|--------------------------------|-----|---------|-----|------|------|------|------|------|
| Souveraineté, langue et culture (CES 2021) | en | quasi-doublon | 43 | \cellcolor{ref}0,13 | 0,16 | \cellcolor{best}0,13 | 0,25 | 0,63 |
| Importance de l'ethnicité et de la langue (CES 2019) | en | loin | 34 | \cellcolor{ref}0,21 | 0,19 | 0,21 | \cellcolor{best}0,16 | 0,50 |
| Partage des dépenses fédérales (EEQ 2012) | fr | isolée | 11 | \cellcolor{ref}0,21 | \cellcolor{best}0,19 | 0,24 | 0,20 | 0,48 |
| Plus de pouvoirs ou indépendance (EEQ 2012) | fr | proche | 11 | \cellcolor{ref}0,17 | 0,21 | \cellcolor{best}0,18 | 0,26 | 0,38 |
| Meilleur premier ministre (CECD 1998) | fr | loin | 9 | \cellcolor{ref}0,12 | 0,16 | \cellcolor{best}0,12 | 0,14 | 0,37 |
| Meilleur parti : immigration (EEQ 2018) | fr | modérée | 8 | \cellcolor{ref}0,17 | \cellcolor{best}0,24 | 0,34 | 0,32 | 0,67 |
| Parti de 2^e^ choix (CES 2019) | en | quasi-doublon | 54 | \cellcolor{ref}0,21 | 0,25 | 0,31 | \cellcolor{best}0,24 | 0,77 |
| Meilleur parti : économie (CES 2019 tél.) | en | proche | 29 | \cellcolor{ref}0,30 | 0,30 | 0,36 | \cellcolor{best}0,28 | 0,59 |
| Aide médicale à mourir et avortement (CES 2021) | en | loin | 43 | \cellcolor{ref}0,13 | \cellcolor{best}0,15 | 0,23 | 0,33 | 0,58 |
| Signes religieux des enseignants (EEQ 2018) | fr | isolée | 8 | \cellcolor{ref}0,09 | 0,16 | 0,26 | \cellcolor{best}0,15 | 0,39 |
| Impact économique des immigrants (CES 2019 tél.) | en | modérée | 15 | \cellcolor{ref}0,23 | 0,27 | 0,22 | \cellcolor{best}0,21 | 0,55 |
| Pratique religieuse et société (EEQ 2008) | fr | quasi-doublon | 10 | \cellcolor{ref}0,28 | 0,36 | \cellcolor{best}0,35 | 0,36 | 0,67 |

*Questions groupées par thème (identité et fédéralisme, partis et vote, valeurs sociales). En vert,
la meilleure condition de chaque question (le plancher humain, en gris, sert de référence).*

**Proximité.** Ressemblance entre la question de test et la question d'entraînement qui lui est
la plus semblable, mesurée par la similarité cosinus de leurs embeddings de texte. Elle indique à quel
point la question est « nouvelle » pour le modèle : *quasi-doublon* (0,95 et plus : pratiquement la
même question, posée ailleurs ou autrement), *proche* (0,85 à 0,95), *modérée* (0,775 à 0,85),
*loin* (0,70 à 0,775) et *isolée* (moins de 0,70 : aucune question semblable à l'entraînement). Les
60 questions de test ont été tirées en nombre égal dans chaque catégorie.

Question par question, aucune condition entraînée ne domine : le modèle entraîné n'est le meilleur
que sur 3 questions, contre 4 pour Entraîné + indices et 5 pour Indices retirés. Il gagne en moyenne
parce qu'il n'échoue jamais gravement : il n'est jamais à plus de 0,06 de la meilleure condition,
alors que les deux autres s'en écartent parfois de 0,11 à 0,18. Les écarts varient
beaucoup d'une question à l'autre, sans tendance nette selon la proximité ou la langue (modèle entraîné :
0,220 en français, 0,219 en anglais).

## B. Sensibilité : sous-groupes d'au moins 30 répondants

| Condition | Variation totale | IC 95 % | Écart au plancher |
|---|---|---|---|
| \cellcolor{ref}Humain | \cellcolor{ref}0,153 | \cellcolor{ref}[0,124 ; 0,185] | \cellcolor{ref}— |
| Entraîné | \cellcolor{best}0,205 | \cellcolor{best}[0,171 ; 0,245] | \cellcolor{best}+0,052 [0,031 ; 0,070] |
| Entraîné + indices | 0,236 | [0,178 ; 0,295] | +0,082 [0,039 ; 0,129] |
| *Indices retirés* | 0,240 | [0,189 ; 0,288] | +0,087 [0,048 ; 0,132] |
| Non entraîné | 0,566 | [0,479 ; 0,642] | +0,412 [0,346 ; 0,483] |

*144 sous-groupes, 11 questions : « Impact économique des immigrants » n'a aucun sous-groupe d'au
moins 30 répondants d'évaluation. Cette sensibilité est donc descriptive et ne remplace pas
l'analyse principale sur 12 questions.*

## C. Détail par température

**Part de l'option la plus fréquente** (vrais répondants : 50 % ; en vert, la plus proche de 50 %)

| Condition | T = 0,3 | T = 0,7 | T = 1,0 | T = 1,3 |
|---|---|---|---|---|
| Non entraîné | 98 % | 95 % | 93 % | 91 % |
| Entraîné | 81 % | 60 % | \cellcolor{best}51 % | \cellcolor{best}44 % |
| Entraîné + indices | 78 % | 56 % | 47 % | 42 % |
| *Indices retirés* | \cellcolor{best}66 % | \cellcolor{best}50 % | 43 % | 39 % |

**Contrastes appariés** (différence de variation totale, IC 95 % ; négatif = premier meilleur ;
en vert ou en rouge, les différences dont l'intervalle exclut zéro)

| Contraste | T = 0,3 | T = 0,7 | T = 1,0 | T = 1,3 |
|---|---|---|---|---|
| Entraîné − Non entraîné | \cellcolor{best}−0,22 [−0,27 ; −0,17] | \cellcolor{best}−0,33 [−0,35 ; −0,25] | \cellcolor{best}−0,33 [−0,35 ; −0,23] | \cellcolor{best}−0,31 [−0,34 ; −0,20] |
| Entraîné + indices − Entraîné | +0,05 [−0,01 ; 0,11] | \cellcolor{worse}+0,04 [0,00 ; 0,08] | +0,03 [−0,00 ; 0,06] | +0,03 [−0,00 ; 0,05] |
| Entraîné + indices − Indices retirés | +0,05 [−0,02 ; 0,12] | +0,02 [−0,03 ; 0,06] | +0,00 [−0,04 ; 0,04] | +0,00 [−0,03 ; 0,03] |
| Fuite − Entraîné + indices | −0,01 [−0,03 ; 0,00] | −0,02 [−0,03 ; 0,00] | −0,01 [−0,02 ; 0,01] | −0,01 [−0,03 ; 0,00] |

**Réponses invalides** (réponse qui ne correspond exactement à aucune option ; en rouge, 1 % et plus). Aucun tirage n'est
perdu en silence : les erreurs d'API sont réessayées jusqu'à obtenir une réponse, sinon le run s'arrête.

| Condition | T = 0,3 | T = 0,7 | T = 1,0 | T = 1,3 |
|---|---|---|---|---|
| Non entraîné | 0,0 % | 0,0 % | 0,0 % | 0,2 % |
| Entraîné | 0,0 % | 0,0 % | 0,4 % | \cellcolor{worse}4,6 % |
| Entraîné + indices | 0,0 % | 0,0 % | 0,5 % | \cellcolor{worse}5,2 % |
| *Indices retirés* | 0,0 % | 0,0 % | 0,4 % | \cellcolor{worse}4,8 % |

## D. Aplatissement des différences entre sous-groupes

Rapport entre la variance des distributions prédites d'un sous-groupe à l'autre et la variance
observée (1 = le modèle différencie les groupes autant que la réalité ; 0 = même réponse pour tous).
Médiane sur les options, température 1,0, sous-groupes d'au moins 100 répondants seulement
(3 questions). En vert, la valeur la plus proche de 1.

| Non entraîné | Entraîné | Entraîné + indices | *Indices retirés* | *Fuite* |
|---|---|---|---|---|
| 0,00 | 0,57 | 1,09 | 0,55 | \cellcolor{best}1,06 |

Le modèle non entraîné donne la même distribution à tous les groupes. Sans indices, les modèles
entraînés compressent les écarts entre groupes de moitié environ ; les indices les rétablissent.
À prendre avec prudence : 3 questions seulement.

## E. Détails techniques

- **Entraînement** : Llama-3.3-70B-Instruct sur Azure AI Foundry, 1 époque, lots de 64,
  multiplicateur de taux d'apprentissage 1, graine 20260924. La cible est le texte exact de l'option
  choisie, pour permettre le transfert vers des questions dont les codes diffèrent.
- **Langue** : chaque répondant voit la question, les indices et son profil dans sa langue de
  passation ; une question sans version complète dans cette langue est exclue plutôt que traduite.
- **Correspondance avec les fichiers d'analyse** : Non entraîné = R ; Entraîné = A (modèle C0) ;
  Entraîné + indices = BS, Indices retirés = B0, Fuite = B (modèle C1).

## F. Exemples de prompts

**Entraînement.** Le même couple répondant–question, tel que vu par les deux modèles. La réponse du
tour « assistant » est la cible apprise. Le profil est celui du répondant, avec les champs
disponibles dans son sondage.

*Modèle standard*

```text
[system]
Tu es un répondant à un sondage d'opinion mené en 2018.
Population : Québec
Âge : 18-24 ans / 25-34 ans
Région : Montréal
Province : Québec

[user]
Question : Lequel des chefs des principaux partis provinciaux ferait le meilleur premier ministre du Québec? Serait-ce…
Options :
- Philippe Couillard, du Parti libéral
- Jean-François Lisée, du Parti Québécois
- François Legault, de la Coalition Avenir Québec
- Manon Massé, de Québec solidaire
- Aucun d’entre eux
- Ne sais pas/Pas certain(e)

Réponds uniquement par le texte exact de l'option choisie.

[assistant — cible]
Aucun d’entre eux
```

*Modèle à indices* : même exemple, précédé des réponses **de ce répondant** à des questions voisines.

```text
[user]
Tes réponses à d'autres questions du sondage :
- Selon vous, quel chef de parti a fait la meilleure campagne électorale jusqu’à présent? → Ne sais pas/Pas certain(e)
- Et selon vous, quel parti est le plus susceptible de remporter l’élection du 1er octobre? → La Coalition Avenir Québec (CAQ)
- En pensant à ce que vous ressentez maintenant, si une élection PROVINCIALE était tenue demain, le candidat de quel parti appuieriez-vous probablement? → Le Parti libéral du Québec (PLQ)

Question : Lequel des chefs des principaux partis provinciaux ferait le meilleur premier ministre du Québec? […]
```

**Inférence.** Une question de test (EEQ 2012), pour le sous-groupe des hommes de 45 à 54 ans. Le
profil est celui du sous-groupe ; le modèle répond 100 fois par température.

*Non entraîné, Entraîné et Indices retirés* : exactement le même prompt, seul le modèle change.

```text
[system]
Tu es un répondant à un sondage d'opinion mené en 2012.
Population : Québec
Âge : 45-54 ans
Genre : Homme

[user]
Question : Si vous deviez choisir entre plus de pouvoirs pour le Québec et l'indépendance, lequel préféreriez-vous?
Options :
- Plus de pouvoirs pour le Québec
- Indépendance
- Ne sais pas
- Pas de réponse

Réponds uniquement par le texte exact de l'option choisie.
```

*Entraîné + indices* : le même prompt, précédé de la façon dont **le sous-groupe** a répondu à des
questions voisines, calculée sur les répondants de contexte (n = 45).

```text
[user]
Réponses observées dans ton groupe à d'autres questions :
- Si vous deviez choisir entre le statu quo et plus de pouvoirs pour le Québec, lequel préféreriez-vous? : Statu quo 26 %, Plus de pouvoirs pour le Québec 67 %, Ne sais pas 4 %, Pas de réponse 2 % (n=45)
- Lequel des énoncés suivants est plus proche de votre point de vue? : Le Québec devrait devenir indépendant, séparé du Canada 24 %, L’Assemblée nationale du Québec devrait avoir plus de pouvoirs 28 %, On devrait laisser les choses telles qu’elles sont 38 %, L’Assemblée nationale du Québec devrait avoir moins de pouvoirs 3 %, Il ne devrait pas y avoir de gouvernement provincial au Québec 5 %, Ne sais pas 0 %, Pas de réponse 2 % (n=45)
- Si vous deviez choisir entre le statu quo et l'indépendance, lequel préféreriez-vous? : Statu quo 63 %, Indépendance 31 %, Ne sais pas 2 %, Pas de réponse 4 % (n=45)
- Et s'il y avait un référendum avec trois options. Voteriez-vous pour: le statu quo, plus de pouvoirs pour le Québec, ou l'indépendance? : Statu quo 33 %, Plus de pouvoirs pour le Québec 46 %, Indépendance 17 %, Ne sais pas 0 %, Pas de réponse 4 % (n=45)
- Et si un référendum avait lieu vous demandant si vous voulez que l'Assemblée nationale du Québec ait beaucoup plus de pouvoirs, voteriez- vous OUI ou… : Oui 61 %, Non 24 %, Ne sais pas 13 %, Pas de réponse 2 % (n=45)
- Si un référendum sur l'indépendance avait lieu vous demandant si vous voulez que le Québec devienne un pays indépendant, voteriez-vous OUI ou voteriez-vous NON? : Oui 30 %, Non 61 %, Ne sais pas 2 %, Pas de réponse 7 % (n=45)

Question : Si vous deviez choisir entre plus de pouvoirs pour le Québec et l'indépendance, lequel préféreriez-vous?
[…]
```

*Fuite* : le même format, mais les pourcentages sont calculés sur tous les répondants tenus à
l'écart (n = 90), y compris ceux qui servent à l'évaluation.

**Ce que chaque condition a produit pour ce sous-groupe** (température 1,0, 100 réponses ; en vert,
la condition la plus proche des vrais répondants pour chaque réponse)

| Réponse | Vrais répondants | Non entraîné | Entraîné | Indices retirés | Entraîné + indices |
|---|---|---|---|---|---|
| Plus de pouvoirs pour le Québec | \cellcolor{ref}58 % | 100 % | 50 % | 45 % | \cellcolor{best}62 % |
| Indépendance | \cellcolor{ref}34 % | 0 % | \cellcolor{best}29 % | 26 % | 9 % |
| Ne sais pas | \cellcolor{ref}6 % | 0 % | \cellcolor{best}10 % | 15 % | 13 % |
| Pas de réponse | \cellcolor{ref}2 % | \cellcolor{best}0 % | 11 % | 14 % | 15 % |

Le modèle non entraîné donne la même réponse 100 fois sur 100. Le modèle entraîné retrouve l'ordre
de grandeur des deux grandes options. Avec les indices, le modèle sous-estime nettement
l'indépendance, alors que les indices montrent environ 30 % d'appui à l'indépendance dans ce groupe.
C'est un exemple où le modèle ne tire pas parti des distributions de groupe qu'on lui fournit.
