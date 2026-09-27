# ADR 0002 - Preparation des distributions du pilote

- **Statut** : accepte (24 septembre 2026)
- **Portee** : entrees de l'analyse, avant le calcul des metriques

## Contexte

Les quatre bras d'inference produisent un CSV par tirage et un CSV de diagnostics par
paire item-cellule-temperature. Les campagnes completes ne sont pas encore toutes
disponibles, mais leur schema est fixe par `inference.RESULT_FIELDS`.

La note de Justin exige une divergence KL et un bootstrap des deux cotes, sans fixer
le lissage des probabilites nulles, le nombre de repetitions, la graine ni la methode
exacte de reechantillonnage pondere. Ces choix ne sont donc pas herites de la note.

## Decision

1. `scripts/21_build_distributions.py` ne calcule pas encore les metriques. Il prepare
   leurs entrees sans faire de choix statistique supplementaire.
2. `data/analysis/distributions.csv` suit le schema long defini dans
   `docs/eval_distributions.md`. La temperature de `arm = observed` est manquante dans
   le CSV; elle ne constitue pas une categorie.
3. Les diagnostics d'invalidite des conteneurs sont conserves. Une table consolidee,
   `data/analysis/distribution_diagnostics.csv`, est aussi derivee directement des
   tirages afin de garder `transport_n`, `effective_n`, `invalid_n`, `invalid_rate` et
   `coverage_ok` ensemble.
4. Les reponses observees valides sont gardees dans
   `data/analysis/observed_responses.csv`, avec leur identifiant de repondant, leur
   code canonique et leur poids. Cela permet un futur bootstrap sans reconstruire la
   verite terrain.
5. Le script echoue si un bras, une paire, une temperature ou un indice de tirage
   manque, ou si les metadonnees d'un tirage contredisent le bras et la paire attendus.
   Une campagne partielle ne produit pas une table d'analyse apparemment complete.
6. Une condition ou tous les tirages sont invalides reste dans les sorties avec
   `n = 0`, `invalid_rate = 1` et `share` manquant. Elle n'est ni omise ni transformee
   artificiellement en distribution uniforme.

## Decisions reportees

Le lissage KL, le bootstrap pondere, son nombre de repetitions, sa graine et la forme
des intervalles seront fixes avant le calcul des metriques. En particulier, le
pseudo-compte de 0,5 mentionne dans `docs/eval_distributions.md` demeure un exemple et
non une decision.

Ces choix sont fixes pour l'analyse reelle par l'ADR 0003. Les parametres ci-dessous
restent limites aux donnees factices historiques.

## Parametres exploratoires pour les donnees factices

Pour tester toute la chaine et preparer la structure de la mini-note avant l'arrivee
des vraies campagnes, `scripts/23_evaluate_distributions.py` emploie provisoirement :

- KL(observee || modele), avec un pseudo-compte de 0,5 sur les comptes du modele
  seulement, puis renormalisation;
- 1 000 repetitions, graine 20260924 et intervalle percentile a 95 %;
- reechantillonnage uniforme des repondants avec remise, en conservant le poids
  d'enquete attache a chaque repondant;
- reechantillonnage avec remise des tirages valides du modele;
- pour les contrastes, reechantillonnage des 12 items avec remise et meme selection
  d'items dans les deux bras compares.

Ces parametres servent uniquement aux sorties marquees `FAKE_DATA`. Ils seront revus
avant toute analyse des vraies donnees et ne constituent pas une decision scientifique
definitive.
