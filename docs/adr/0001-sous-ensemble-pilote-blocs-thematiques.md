# ADR 0001 — Sous-ensemble pilote du test : 3 blocs thématiques × 4 items

- **Statut** : accepté (24 septembre 2026)
- **Décideur** : Hubert Cadieux
- **Exécution déléguée** : première évaluation C0 8k vs C1 8k (voir « Ce qui est délégué »)

## Contexte

Le split de test est gelé et pré-enregistré (commit `eeb7e87`) : 60 items, 30 FR / 30 EN, 13 sondages,
soit environ 1 160 paires item × cellule (cellules n ≥ 30) et ~116 000 appels par modèle à N = 100.
C'est trop pour une première analyse ou une mini-note. On veut un pilote **petit, circonscrit**, qui permet
malgré tout de **comparer des blocs thématiques** entre eux et **C0 vs C1** à l'intérieur de chaque bloc.

## Décision

1. **Le split n'est pas modifié.** Le pilote est un filtre d'analyse sur les 60 items gelés. Les 48 autres
   items sont **non évalués à ce stade** (pas « écartés »).
2. **Regroupement des 60 items en 7 blocs thématiques**, d'après les libellés, avant tout résultat :
   `scripts/18_test_blocks.py` → `data/analysis/test_blocks.csv` (colonnes `block`, `pilot`).
3. **Pilote = 3 blocs × 4 items = 12 items**, choisis à la main (pas de tirage aléatoire) pour :
   couvrir des bins de distance différents, mélanger FR et EN (2 / 2 par bloc), et garder des items
   dont les cellules sont assez peuplées.

| Bloc | Item (sondage · variable) | Langue | Bin | Cellules n ≥ 30 | n ≥ 100 |
|---|---|---|---|---|---|
| identite_qc_federalisme | ces_2019_online · pes19_ethid (identité ethnique/linguistique) | EN | far | 34 | 1 |
| | eeq_2012 · Q44 (partage des dépenses fédérales) | FR | isolated | 11 | 0 |
| | eeq_2012 · Q55 (plus de pouvoirs vs indépendance) | FR | near | 11 | 0 |
| | ces_2021 · pes21_cultureQC (souveraineté, langue, culture) | EN | quasi_duplicate | 43 | 10 |
| valeurs_sociales | ces_2021 · pes21_abort2 (aide médicale à mourir, avortement) | EN | far | 43 | 10 |
| | eeq_2018 · q44 (symboles religieux, enseignants) | FR | isolated | 8 | 7 |
| | ces_2019_phone · p22_a (immigrants et économie) | EN | moderate | 15 | 0 |
| | eeq_2008 · q67 (pratique religieuse) | FR | quasi_duplicate | 10 | 0 |
| partis_vote | cecd_elxn_qc_1998 · meilpm (meilleur premier ministre) | FR | far | 9 | 0 |
| | eeq_2018 · q51a_8 (meilleur parti : immigration) | FR | moderate | 8 | 7 |
| | ces_2019_phone · q33 (meilleur parti : économie) | EN | near | 29 | 0 |
| | ces_2019_online · cps19_2nd_choice (2e choix de parti) | EN | quasi_duplicate | 54 | 30 |

**Total : 275 paires item × cellule (n ≥ 30), dont 65 à n ≥ 100 → ~27 500 appels par modèle à N = 100.**

**Réserves** (à utiliser seulement si un item pilote s'avère inutilisable, avec trace écrite du remplacement) :
eeq_2007 · q18a (identité Québécois/Canadien), eeq_2012 · Q31 (intention de vote fédérale),
ces_2021 · pes25_indig_favors, eeq_2018 · q34_c.

4. **Les items restent analysés séparément d'abord.** Chaque sortie porte `item_idx` et `block`. Le choix
   « items séparés ou agrégés par bloc » se fait à la visualisation, pas au calcul : on calcule toujours au
   niveau item × cellule, l'agrégation par bloc vient après.

## Justification

- **Pourquoi 3 blocs et pas 1** : un seul bloc ne permet aucune comparaison entre thèmes.
- **Pourquoi ces 3** : de natures contrastées — identité québécoise (sujet de l'article, surtout FR),
  valeurs sociales (équilibré FR/EN), préférences partisanes (le plus « prévisible » à partir de la
  sociodémo, donc une référence).
- **Pourquoi à la main** : le tirage à graine aurait pu laisser un bloc sans item bien peuplé ou sans FR/EN.
  Le choix est documenté ici, avant tout résultat.

## Limites (à déclarer dans la note)

- Pas de bin `moderate` dans identite_qc ni `near` dans valeurs_sociales ; 12 items ne permettent pas
  d'analyse fine bloc × bin.
- Beaucoup d'items FR (EEQ, 1998) ont ~8-11 cellules à n ≥ 30 et 0 à n ≥ 100 : contrastes C0 vs C1 seulement,
  **pas d'affirmation de précision absolue** sur ces items (seuils du plan, §3.5).
- Le pilote n'est pas représentatif des 60 items ni du corpus ; il valide le pipeline et donne un premier signal.

## Ce qui est délégué

Quelqu'un d'autre exécute le pilote. Points d'entrée et règles :

- **Cellules et répondants tenus à l'écart** : `data/split/heldout_respondents.parquet` (colonne `cell`,
  `__weight`) ; définition des cellules par sondage dans `data/strata_definition.json`. N'utiliser que les
  cellules n ≥ 30 comptées ci-dessus.
- **Items** : `data/analysis/test_blocks.csv`, filtrer `pilot == true`.
- **Modèles** : C0 8k (`...-c0-8k-txt`, déployé sous `c0-8k-txt`) et C1 8k (`ftjob-5dfdb814…`, suffixe
  `c1-8k-txt`, à déployer à sa réussite). Une seule condition d'inférence par modèle pour le pilote.
- **Appels** : toujours via `src/article_silicon_sampling_quebec/foundry.py` (`FoundryChat`) — réessaie les
  429, **ne saute jamais un tirage** ; `ItemSpec.match_answer` (`prompts.py`) pour apparier la sortie
  (non apparié = invalide, à compter, pas à écarter).
- **N** : 100 tirages par paire item × cellule, température comme validée au smoke test.
- **Déploiements Azure** : `--model-format Meta --sku-name GlobalStandard --sku-capacity 50`, abonnement
  sponsorisé `a54061e5…` (vérifier `az account show`), **supprimer chaque déploiement dès la fin de la
  campagne** (facturation horaire ; crédits expirés le 4 octobre 2026).
- **Split, items pilote et règles ci-dessus : ne pas les modifier** sans amender cet ADR.
- L'évaluation elle-même (KL, bootstrap des deux côtés, baselines) suit le plan §5 ; ce n'est pas décidé ici.

## Références

`docs/plan_article.md` (§3.5 strates, §4 split, § « Ensuite ») · `data/split/split_manifest.json` ·
`scripts/18_test_blocks.py` · `data/analysis/test_blocks.csv`
