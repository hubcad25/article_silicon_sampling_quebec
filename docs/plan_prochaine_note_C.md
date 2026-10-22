# Plan — prochaine note : fixer la recette de l'approche C

*Mis à jour le 20 octobre 2026. Écrit pour qu'un agent qui arrive sans contexte puisse exécuter.
Lire d'abord ce fichier, puis `docs/synthese_draft_next.md` (résultats de la ronde précédente) et
`docs/hybrid_small_sample_results.md` (test hybride).*

## 1. Où on en est (à savoir avant de toucher à quoi que ce soit)

**Le problème.** Prédire comment 15 sous-groupes (5 régions × 3 âges) répondent à des questions de
CES 2025, avec seulement des données d'avant 2025. Score : **TV** (variation totale), la part des
réponses mal placées, moyennée sur sous-groupes × questions. Plus bas = mieux. Plancher (deux moitiés
de CES 2025) : 0,062 sur les questions répétées, 0,074 sur les nouvelles.

**Deux séries de questions** (`benchmark/__init__.py`, `SETS`, choisies par `BENCH_SET=main|novel`) :
« répétées » (10 items aussi posés dans CES 2021) et « nouvelles » (8 items CES 2025 jamais posés dans
nos 8 sondages d'entraînement ; `scripts/39_novel_ces2025_items.py`).

**Trois approches testées** (ronde du 16-19 oct.) :
- A / B = faux répondants : le Llama 3.3 70B fine-tuné joue 15 000 répondants (profils Democracy
  Checkup 2024) avec 7 (A) ou ~16 (B) de leurs réponses en contexte. **Mis de côté** (décision Hubert).
- **C = dossier → distribution** : une requête par question ; le modèle reçoit un dossier (libellé,
  questions semblables du passé avec national + écart de chaque sous-groupe) et écrit directement le
  tableau 15 sous-groupes × options en JSON. Code : `benchmark/dossier.py` (build / run / score).

**Résultats C** (TV répétées / nouvelles) : sans entraînement Llama base 0,196 / 0,264 ;
**C par l'exemple 0,152 / 0,214** (meilleur ; égale CES 2021 réutilisé 0,151) ; C par récompense
(expert iteration) 0,202 / 0,216 ; gpt-5.5 sans entraînement 0,106 / 0,144 (signalé : ses données vont
en 2026, contamination possible, ne pas compter comme résultat).

**Cible produit** (`mvp_moteur_recherche_sondages/docs/vision_plateforme.md`, `docs/besoins_marche-2026-09-29.md`) :
un client apporte un petit échantillon sans quotas → pondération → C → mélange → marges d'erreur,
orchestré par l'agent de la plateforme. Test hybride (script 44) : 100 répondants biaisés + C ≈ un
sondage représentatif de 500-650 personnes ; 400 + C ≈ 830-1 060. **Dans l'hybride, le niveau national
vient de l'échantillon et C fournit les écarts entre groupes** ; le plafond de l'hybride (~0,07-0,08)
est l'erreur des écarts de C. Donc améliorer les écarts de C est la priorité.

## 2. But de la note

Trouver la meilleure recette de C (dossier, prompt, entraînement), mesurée sur une sortie proche du
produit. **Pas** de calibration des marges dans cette note (elle vient une fois la recette fixée).

## 3. Ce qui change dans le test

1. **Grille plus riche.** En plus de région × âge (15 cases) : sexe, scolarité, et deux croisements
   (âge × sexe, région × sexe). C prédit les écarts par variable et par croisement ; la grille complète
   se reconstruit par calage sur les parts de population (repondération en cascade). Les variables
   socio-démo des sondages sont harmonisées dans `data/crosswalks/ses_crosswalk.json` (`SesCrosswalk`).
2. **Plus de questions jamais posées.** Passer de 8 à ~25-30 items CES 2025, tirés des 119 candidats de
   `data/analysis/ces2025_novelty.csv` (script 39) : n ≥ ~5 000, pas de questions sur les partis/leaders,
   pas d'items sensibles au filtre de contenu (racisme, sexisme, nativisme). Rapporter avec et sans les
   items liés à l'actualité 2025 (ex. `tarrifs_1/2`).
   - **Fait (22 oct.)** : set `novel28` (`benchmark/__init__.py`) = 20 items CES + 8 items CSES
     (2 331 répondants, ~155 par cellule, rapportés à part).
   - **Routage** (remarque de Justin, note du 22 oct.) : pour chaque item, vérifier à qui la question
     a été posée. Vérifié dans les données : `tarrifs_2` (si `tarrifs_1` = « Worse »), `howvote1`
     (votants probables), `rent_security` (locataires, n = 5 713), `big_issue_gov` (si un enjeu nommé).
     La vérité se calcule déjà parmi les répondants concernés ; le dossier l'indique maintenant
     (`ASKED_ONLY`). Les blocs `block_*` / `pes_block` sont des sous-échantillons aléatoires : rien à faire.
   - **Set 3 de Justin** (8 items, `questions_set3_holdout8.json`) : `resident_2e` et `pes25_ai` sont
     déjà dans `novel28`. `foreign_grid_1` (cos 0,847), `internetvote_1` (0,851), `emb_vote16` (0,918)
     et `tieindia` (0,825) dépassent notre seuil anti-fuite de 0,80 ; `abort2` a déjà été posée
     (2019/2021) ; `tarrifs_3` n'est pas dans nos candidats (à vérifier).
3. **Trois mesures par variante**, chacune convertie en « vaut un sondage représentatif de n » et en marge
   d'erreur par groupe (conversion : TV ≈ 0,66 / √n par groupe ; marge ≈ ±98/√n points) :
   - **C seul** (client sans données) ;
   - **qualité des écarts** : C recalé sur le vrai niveau national (ce que l'hybride utilise) ;
   - **hybride** : 200 répondants biaisés + C, simulation du script 44.

## 4. Variantes (toutes sur Llama 3.3 70B, deux entraînements chacune)

| | Variante | Question posée |
|---|---|---|
| V0 | C par l'exemple actuel (`ces25-dossier-1k-s1/s2` dans `benchmark/models.json`) | Référence |
| V1 | Dossier enrichi : 15 questions semblables au lieu de 8, tailles d'échantillon par groupe, écarts par sexe et scolarité | Plus de preuves aide-t-il ? |
| V2 | Plus d'entraînement : plus de questions (études québécoises du corpus : EEQ, CECD) et 10 variantes de dossier par question au lieu de 3, soit ~5-10× plus d'exemples | Plafonne-t-on déjà ? |
| V3 | Exemple + récompense : (1) V0 écrit 8 tableaux par exercice, notés contre la vérité (`benchmark/dossier_grader.py`) ; (2) on réentraîne V0 sur ses meilleurs (`scripts/43_dossier_expert_iteration.py`, à adapter pour partir de V0) | La récompense aide-t-elle sur un bon départ ? |
| V5 | Contexte d'actualité : résumé thématique daté d'avant la collecte, ajouté au dossier, à l'entraînement comme au test (voir §4b) | Corrige-t-il les questions d'actualité sans nuire aux autres ? |

Références sans entraînement : C Llama de base et gpt-5.5 (signalé). Pour V5, deux de plus
(proposition de Justin) : (a) Llama de base + résumés d'actualité ; (b) V0 + résumés au test
seulement. Justin a déjà testé (a) en zéro-shot avec un briefing de 4 puces, sur la grille
région × âge : Set 2 0,247 → 0,211 (l'égal de notre C par l'exemple, 0,214), Set 1 inchangé
(0,219 → 0,225). V4 (national puis écarts) écartée :
dans l'hybride le niveau vient de l'échantillon.

## 4a. Décisions du 22 octobre

- **V0 réentraîné** : la sortie change (grille riche), l'ancien V0 ne sait pas produire sexe/scolarité.
  V0 = même recette, nouvelle grille, anti-fuite refait sur les ~25-30 nouveaux items.
- **Grille : une requête par variable ou croisement** (région, âge, sexe, scolarité, région × âge,
  âge × sexe, région × sexe). JSON plus court, ~5× plus d'exemples d'entraînement. Cellules complètes
  reconstruites par calage (pas d'interaction d'ordre 3).

## 4b. V5 : résumés d'actualité (fait)

- Thèmes : `docs/themes_actualite_v5.md` (17 thèmes + « aucun »), pas les clusters k-means du script 41
  (ils regroupent selon la forme des questions).
- Classement : `data/analysis/item_themes_v5.csv` (779 questions = 734 exercices + 45 candidats
  CES 2025 ; ≤ 2 thèmes ; ~50 % « aucun »). Fait par agents Sonnet, 2025 revu à la main.
- Résumés : `data/analysis/news_summaries_v5.csv`, 105 = thème × sondage, 80-150 mots, anglais,
  faits datés d'avant la date limite (début du terrain : DC selon `StartDate`, CES 2019 13 sept.,
  CES 2021 17 août, CES 2025 24 mars), aucun résultat de sondage. Recherche web + passe de révision ;
  2025 validé par Hubert (`docs/news_summaries_2025_review.md`).
- Limite : le logement (4) n'apparaît qu'au test 2025.

## 5. Mécanique (ce qui marche, ce qui a cassé)

- **Exercices d'entraînement de C** : `scripts/41_dossier_training_data.py` → `data/datasets_dossier/`.
  Chaque question des 8 sondages = un exercice (dossier construit sans elle → vrai tableau). Petites
  cellules (< 30 répondants) tirées vers le national.
- **Anti-fuite** : exclure de l'entraînement et des dossiers toute question à cosinus ≥ 0,80 ou de même
  radical qu'un item testé, et tout item libellé exactement comme lui.
- **Filtre de contenu Azure** avant toute soumission : `scripts/38_content_safety_screen.py --filter
  FICHIERS` (score par morceaux de 9 000 caractères, retire les lignes ≥ 4).
- **Soumission** : `scripts/37_submit_benchmark_finetunes.py --dir <dossier> --arms <nom> --sizes <N>
  --seeds 2 --epochs 3 --batch-size 8`. **Batch 8 obligatoire** pour les dossiers (≈5 000 tokens ; batch
  64 = « Out Of Memory »). Permission déjà accordée pour ce script dans `~/.claude/settings.json`.
- **Inférence C** : enregistrer le modèle dans `benchmark/models.json`, puis déployer → `python -m
  benchmark.dossier run --deployment <nom>` et `score` → supprimer le déploiement (try/finally avec
  `benchmark.backends`). **Toujours en ACI** (`benchmark/cloud.py`), même court : un run local a planté
  sur une coupure réseau en laissant un déploiement facturé à l'heure. Vérifier à la fin qu'aucun
  déploiement `bench-*` ne reste.
- **Récompense « officielle » Azure impossible** : o4-mini retiré (jobs échoués), gpt-5 sur invitation,
  Qwen/gpt-oss = GPU dédiés (quota 0). D'où l'expert iteration (V3).
- **Simulation hybride** : `scripts/44_hybrid_small_sample.py`. Plusieurs items CES 2025 sont posés à
  25-50 % des répondants : tirer chaque échantillon parmi ceux à qui la question a été posée (corrigé le
  20 oct. ; l'ancienne version gonflait l'équivalence à « 1 200 »).
- **Données** : CSV par défaut (`AGENTS.md`), Parquet seulement pour les embeddings.

## 6. Coût et durée estimés

- Dossiers et exercices : 1-2 jours ; résumés d'actualité (V5) : ~1 jour, **à faire valider par Hubert**
  (faits datés seulement, jamais de résultats d'opinion).
- 8 entraînements (V1, V2, V3, V5 × 2) : ~1 jour en parallèle (file d'attente Azure : 1-3 h) ; V3 demande
  un tour d'échantillonnage de plus (~15k requêtes, ACI).
- Inférence : ~30 items × 3 tirages × ~10 modèles ; négligeable.

## 7. Livrable

Une note de 3 pages au format de `docs/synthese_draft_next.md` (anglais, pandoc + `note_style.tex`,
lisible par quelqu'un qui n'a pas vu le code) : un tableau variantes × (C seul, qualité des écarts,
hybride), en TV et en « vaut n répondants » ; une recommandation de recette pour la suite (calibration
des marges, agent).
