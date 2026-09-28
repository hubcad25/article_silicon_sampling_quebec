# ADR 0006 - Inférence sur les 48 questions pour améliorer le résultat de référence

- **Statut** : accepté, avant toute inférence sur les 48 questions restantes
- **Portée** : choix des modèles, des bras, du nombre de tirages et de l'exécution distante
- **Liens** : ADR 0003 (évaluation), ADR 0005 (répondants réels),
  `docs/note_recherche_2026-10-01.md`

## Contexte

Le pilote sur 12 questions et 275 cellules donne une variation totale de **0,219** pour
**Entraîné 8k**. C'est le meilleur résultat individuel observé, mais les écarts entre les meilleurs
bras sont faibles et incertains : rien ne permet de conclure que 8k est réellement préférable à
20k. Les 48 autres questions du partage gelé n'ont encore servi ni à choisir un modèle ni à régler
une méthode.

La prochaine note cherchera directement à améliorer ce résultat de référence. Elle ne sera pas une
note sur les ensembles : combiner les modèles sera seulement une des améliorations évaluées, avec le
volume d'entraînement, les réponses individuelles et la régularisation des différences entre
cellules.

Les 48 questions restantes forment **885 couples question × cellule** admissibles selon le seuil déjà
gelé de 30 répondants tenus à l'écart. Une campagne de 250 tirages représente donc 221 250 appels par
bras.

## Décision

### Questions évaluées

L'inférence porte exclusivement sur les lignes `pilot == false` de
`data/analysis/test_blocks.csv`, soit **48 questions et 885 cellules**. Les 12 questions du pilote
restent un jeu de développement : elles peuvent servir à fixer une transformation avant l'analyse,
mais elles ne sont pas mélangées aux résultats confirmatoires des 48 questions.

### Paramètres communs

- température : **1,0** seulement;
- tirages : **250 par bras et par cellule**;
- `top_p`, limite de sortie, gabarits et règles de correspondance : inchangés;
- mêmes cellules et mêmes profils pour tous les bras;
- aucune analyse intermédiaire des réponses avant la fin et la validation des quatre bras.

Il n'y aura pas de comparaison entre 100 et 250 tirages dans la note. Les 250 tirages constituent le
nouveau régime d'inférence; leur but est de réduire le bruit Monte-Carlo, pas de créer un nouvel axe
d'analyse.

### Modèles et bras conservés

| Campagne | Bras | Modèle | Contexte à l'inférence | Appels |
|---|---|---|---|---:|
| `c0-8k` | **A** | entraîné standard 8k | aucun | 221 250 |
| `c0-20k` | **A20** | entraîné standard 20k | aucun | 221 250 |
| `c1-20k` | **B020** | entraîné avec réponses voisines 20k | retirées | 221 250 |
| `c1-20k` | **BR20** | entraîné avec réponses voisines 20k | répondant réel de la moitié contexte | 221 250 |
| **Total** | | | | **885 000** |

**BR8 n'est pas relancé.** Sur le pilote, son ajout à l'ensemble change pratiquement rien, tandis
qu'il augmenterait le coût total de 25 %. Les anciens bras de pourcentages, le modèle de base non
entraîné et le modèle statistique ne sont pas relancés non plus : ils ne constituent pas des pistes
prioritaires pour battre 0,219.

### Estimations dérivées

Les bras bruts demeurent des conditions à part entière. Après leur achèvement, des conditions
supplémentaires seront calculées option par option, pour chaque question × cellule, sans nouvel appel
au modèle :

1. **Ensemble sans attitudes** : moyenne de A, A20 et B020;
2. **Ensemble avec attitudes** : moyenne de A, A20 et BR20;
3. **Ensemble complet** : moyenne de A, A20, B020 et BR20;
4. **Régularisation des écarts entre cellules** : rétrécissement des prédictions vers la moyenne
   prédite de la question, avec une intensité fixée sur les 12 questions du pilote avant d'ouvrir les
   résultats des 48 questions;
5. **Ensemble complet régularisé** : même transformation appliquée à l'ensemble complet, avec son
   intensité elle aussi fixée sur le pilote.

Ces estimations seront comparées aux quatre bras qui les composent. L'ensemble n'est donc pas traité
comme le remplacement automatique des modèles individuels, mais comme une amélioration candidate.

## Critère de réussite de la prochaine note

La référence confirmatoire est **A sur les mêmes 48 questions et avec les mêmes 250 tirages**, et non
le chiffre historique de 0,219 pris isolément. Le 0,219 sert de point de départ substantiel; la
comparaison statistique doit rester appariée sur les nouvelles questions et cellules.

La prochaine note répondra à une seule question : **une des méthodes préspécifiées améliore-t-elle
clairement Entraîné 8k?** Elle rapportera, pour chaque bras brut et chaque estimation dérivée :

- la variation totale moyenne;
- la différence appariée avec A;
- l'intervalle à 95 % obtenu par bootstrap des 48 questions;
- la divergence KL comme vérification secondaire.

Les contrastes diagnostiques **A20 − A** et **BR20 − B020** permettront respectivement d'interpréter
l'effet du volume et celui des réponses individuelles. Ils ne remplacent pas la question principale
d'amélioration par rapport à A.

## Exécution sur Azure Container Instances

L'inférence de production se fait de nouveau sur **Azure Container Instances (ACI)**. Les sorties
doivent être écrites dans un nouvel espace persistant, distinct du pilote, par exemple :

```text
final48-250/c0-8k/A.*
final48-250/c0-20k/A20.*
final48-250/c1-20k/B020.*
final48-250/c1-20k/BR20.*
```

La session d'implémentation doit, avant le lancement :

1. généraliser `build_item_cells()` afin de sélectionner explicitement `pilot`, `remaining` ou
   `all`, sans affaiblir les contrôles de couverture;
2. faire accepter et transmettre par `scripts/20_cloud.py` les paramètres `--draws 250`,
   `--temperatures 1.0`, le sous-ensemble `remaining` et un identifiant de sortie distinct;
3. inscrire dans chaque manifeste le sous-ensemble, les **48 questions**, les **885 cellules**, les
   250 tirages et la température 1,0;
4. vérifier qu'une reprise ne peut jamais mélanger les anciens fichiers à 100 tirages avec cette
   campagne;
5. ajouter les tests couvrant la sélection des 48 questions, les 885 cellules, le passage des
   paramètres à ACI et l'isolation des sorties;
6. exécuter un smoke test ACI avant les campagnes complètes.

Les deux campagnes C0 utilisent le même quota `GlobalStandard` et doivent être lancées
séquentiellement. La campagne C1, sur `DataZoneStandard`, peut tourner en parallèle avec l'une
d'elles. L'ordre recommandé est donc :

1. `c0-8k` / A et `c1-20k` / B020 + BR20 en parallèle;
2. `c0-20k` / A20 après la suppression du déploiement `c0-8k`;
3. validation des manifestes et de la couverture, puis téléchargement local des quatre sorties.

## Conséquences

- Aucun nouvel entraînement n'est requis.
- Quatre bras sont produits, mais ils permettent d'évaluer plusieurs améliorations sans inférence
  additionnelle.
- Le coût principal est celui des 885 000 appels; BR20 est le bras le plus lourd en préparation de
  contexte, mais partage le déploiement C1 avec B020.
- Les 48 questions restent intactes jusqu'à l'analyse finale. Toute règle d'ensemble ou de
  régularisation doit être écrite et figée à partir du pilote avant leur évaluation.
