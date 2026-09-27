# ADR 0001 — Sous-ensemble pilote du test : 3 blocs thématiques × 4 items

- **Statut** : accepté (24 septembre 2026), amendé (25 septembre 2026)
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

5. **Les campagnes complètes d'inférence sont exécutées dans Azure**, et non dans une session locale.
   Le runner doit écrire chaque tirage réussi dans un checkpoint persistant sur du stockage Azure, permettre
   une reprise sans doublons à partir d'une clé de tirage stable et conserver un manifeste qui épingle les
   entrées, le checkpoint du modèle et les paramètres d'inférence. Le poste local sert aux smoke tests, au
   déclenchement et au contrôle seulement. Une interruption du terminal ou du poste local ne doit pas arrêter
   la campagne. Les ressources de calcul et le déploiement du modèle doivent être supprimés automatiquement
   à la fin de la campagne, y compris après un échec.

6. **Stratégie d'inférence : quatre bras** (amendement du 25 septembre 2026, avant tout résultat C1).
   Commun à tous : les 275 paires item × cellule, le persona de la cellule (toutes ses dimensions, sans
   abandon SES), 100 tirages, T ∈ {0,3 · 0,7 · 1,0 · 1,3}, mêmes clés de tirage, appariement par
   `ItemSpec.match_answer`. Seuls le modèle et le bloc de contexte changent.

   | Bras | Modèle | Contexte dans le prompt | Appels |
   |---|---|---|---|
   | **A** | C0 | aucun | 110 000 |
   | **B** | C1 | distributions observées de la cellule sur les items voisins | 110 000 |
   | **B0** | C1 | aucun | 110 000 |
   | **R** | Llama-3.3-70B-Instruct de base (version 9, celle des fine-tunes) | aucun — jeu de rôle | 110 000 |

   - **B0** sépare l'effet de l'entraînement avec contexte de celui du contexte injecté : A vs B0 = effet
     du fine-tune C1 seul ; B0 vs B = effet du contexte de strate à modèle constant.
   - **Contexte de B** : politique de recherche de l'entraînement (`prompts.nearest_context_items`) — même
     sondage, cosinus < 0,95 à la cible, items de test exclus, dédoublonnage du gabarit, k = 6. Pour chaque
     voisin, distribution **pondérée** (`__weight`) des réponses valides des **répondants tenus à l'écart de
     la cellule** (jamais montrés à l'entraînement), rendue en pourcentages suivis de `(n=…)` ; un voisin
     avec moins de 10 réponses valides dans la cellule est sauté. Trace par paire :
     `<campagne>/B.context.csv`.
   - **Couverture mesurée** : 258 paires à 6 lignes de contexte, 8 à 4-5, **9 à 0** — `cecd_elxn_qc_1998 ·
     meilpm` n'a aucun voisin du même sondage dans l'index (top 50 du corpus). Politique identique à
     l'entraînement, donc on ne la contourne pas ; pour cet item, B = B0 par construction.
   - **Limites à déclarer** : (i) le modèle C1 a été entraîné sur les réponses individuelles d'un répondant,
     pas sur des distributions — B est hors distribution d'entraînement ; (ii) les distributions de contexte
     et la vérité terrain proviennent des mêmes répondants tenus à l'écart (sur des items différents), ce qui
     correspond au scénario d'usage mais partage leur bruit d'échantillonnage.
   - **Exécution** : une campagne par modèle, un déploiement, un conteneur ; les bras d'une campagne tournent
     en parallèle. `c0-8k` = A ; `c1-8k` = B + B0 ; `base` = R (quota distinct, 250 unités) ; idem pour 20k plus tard. Outil : `scripts/20_cloud.py`
     (`launch` / `status` / `logs` / `fetch` / `stop`). Capacité effective : 1 000 unités par déploiement
     (quota Llama-3.3-70B fine-tuné : 1 000 en GlobalStandard pour C0, 1 000 en DataZoneStandard pour C1).
7. **Bras BS : contexte et vérité sur deux moitiés disjointes** (amendement du 25 septembre 2026, avant
   tout résultat BS). En B, les distributions de contexte et la vérité terrain viennent des mêmes répondants
   tenus à l'écart. BS s'ajoute à B pour retirer ce recouvrement.
   - **Deuxième split** : les 30 144 répondants tenus à l'écart sont coupés en deux à l'intérieur de chaque
     sondage × cellule (`n // 2` contexte, le reste évaluation), graine 20260925 :
     `data/split/heldout_halves.csv` (14 947 contexte / 15 197 évaluation), produit par
     `scripts/26_split_heldout_halves.py`. **Gelé.**
   - **BS** = modèle C1, même gabarit et même politique de contexte que B, mais distributions calculées sur la
     **moitié contexte seulement**. Mêmes 275 paires, mêmes tirages et températures. Campagne `c1-8k`, bras `BS`.
   - **Évaluation de BS sur la moitié évaluation seulement.** Pour comparer BS aux autres bras, les évaluer
     eux aussi sur cette moitié (leurs prompts ne dépendent d'aucun répondant, donc aucun rerun).
   - **Couverture mesurée** : contexte — 242 paires à 6 lignes, 24 à 4-5, 9 à 0 (`meilpm`) ; n médian par ligne
     26 (B : 49). Vérité (moitié évaluation) — n médian 31 par paire, min 13 ; 144 / 275 paires à n ≥ 30.
     Les contrastes entre bras restent lisibles ; la précision absolue par cellule ne l'est pas. Si besoin,
     les cellules peuvent être agrégées après coup (retrait d'une dimension), sans rerun : le modèle d'une
     cellule agrégée est le mélange de ses cellules fines, pondéré par leur poids. Ce choix se consigne avant
     de regarder les résultats.

## Justification

- **Pourquoi 3 blocs et pas 1** : un seul bloc ne permet aucune comparaison entre thèmes.
- **Pourquoi ces 3** : de natures contrastées — identité québécoise (sujet de l'article, surtout FR),
  valeurs sociales (équilibré FR/EN), préférences partisanes (le plus « prévisible » à partir de la
  sociodémo, donc une référence).
- **Pourquoi à la main** : le tirage à graine aurait pu laisser un bloc sans item bien peuplé ou sans FR/EN.
  Le choix est documenté ici, avant tout résultat.
- **Pourquoi un runner cloud** : la grille complète exige plusieurs heures d'inférence. Une session locale
  n'offre pas une garantie suffisante de continuité et peut laisser un déploiement facturé après une coupure.
  Le stockage persistant sépare la durée de vie des résultats de celle du conteneur d'exécution.

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
  `c1-8k-txt`, réussi). Bras d'inférence : décision 6 (A pour C0 ; B et B0 pour C1).
- **Appels** : toujours via `src/article_silicon_sampling_quebec/foundry.py` (`FoundryChat`) — réessaie les
  429, **ne saute jamais un tirage** ; `ItemSpec.match_answer` (`prompts.py`) pour apparier la sortie
  (non apparié = invalide, à compter, pas à écarter).
- **N** : 100 tirages par paire item × cellule, température comme validée au smoke test.
- **Déploiements Azure** : `--model-format Meta --sku-name GlobalStandard --sku-capacity 50`, abonnement
  sponsorisé `a54061e5…` (vérifier `az account show`). La capacité peut être augmentée dans le quota disponible
  pour réduire la durée de la campagne, à condition de consigner la capacité effective. **Supprimer le runner
  et chaque déploiement dès la fin de la campagne**, même en cas d'échec (facturation horaire ; crédits expirés
  le 4 octobre 2026).
- **Split, items pilote et règles ci-dessus : ne pas les modifier** sans amender cet ADR.
- L'évaluation elle-même (KL, bootstrap des deux côtés, baselines) suit le plan §5 ; ce n'est pas décidé ici.

## Références

`docs/plan_article.md` (§3.5 strates, §4 split, § « Ensuite ») · `data/split/split_manifest.json` ·
`scripts/18_test_blocks.py` · `data/analysis/test_blocks.csv`
