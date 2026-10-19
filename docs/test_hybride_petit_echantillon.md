# Test : petit échantillon biaisé + approche C

*19 octobre 2026. Spécification d'un test court, à exécuter dans ce repo.*

## Question

Une petite collecte réelle **sans quotas** (n = 100 à 200, biaisée), complétée par l'approche C,
donne-t-elle une estimation par sous-groupe aussi bonne qu'un vrai sondage stratifié plus gros ?
Pitch visé : « la précision d'un sondage à 15 k$ pour le prix d'une collecte à 2-3 k$ ».

## Contexte (à lire)

- `docs/synthese_draft_next.md` : approche C (dossier → tableau de distributions par sous-groupe),
  résultats. C entraîné par l'exemple : TV moyen 0,152 (questions répétées), 0,214 (nouvelles).
- `benchmark/dossier.py` : construction des dossiers, appel du modèle, scoring.
- `benchmark/score.py` : `tv()`, données humaines CES 2025, poids `cps25_weight_general_all`.
- Note du 8 oct. (`docs/note_recherche_2026-10-08.md`) : « vrai national + écarts du modèle » = 0,082.

## Protocole

1. **Vérité.** Pour chaque question des deux séries (`benchmark/__init__.py`, `SETS`), la vérité
   par sous-groupe = distribution pondérée de **tous** les répondants CES 2025 du sous-groupe.
   Sous-groupes : les 15 cellules région × âge actuelles. Variante « local » si le temps le
   permet : une seule région (QC) découpée en âge × sexe.
2. **Petit échantillon réel.** Tirer n ∈ {50, 100, 200, 400} répondants CES 2025, deux façons :
   - **aléatoire** (référence) ;
   - **biaisé** : probabilité de tirage proportionnelle à l'intérêt politique ou à la force de
     l'opinion (ex. répondants aux positions extrêmes surreprésentés ×2), et sans quotas d'âge ni
     de région. Documenter le mécanisme exact.
   100 répétitions par (n, mécanisme), graine fixe.
3. **Estimation de C.** Les distributions de C par sous-groupe existent déjà :
   `benchmark/results_dossier/{main,novel}/responses_bench-ces25-dossier-1k-s1.jsonl`
   (parser avec `dossier.parse`, moyenne des tirages). Ne pas rappeler le modèle.
4. **Combinaisons à comparer**, par sous-groupe :
   - **Échantillon seul** : distribution pondérée du petit échantillon (post-stratifiée sur les
     parts de population des cellules). Cellule vide = distribution nationale de l'échantillon.
   - **C seul.**
   - **Hybride bayésien** : C comme a priori Dirichlet de force k (pseudo-répondants),
     mis à jour par les comptes réels de la cellule : `p = (k * p_C + n_cell * p_obs) / (k + n_cell)`.
     Tester k ∈ {5, 10, 20, 40} ; choisir k sur la série « répétées », rapporter sur « nouvelles ».
   - **Hybride niveau + écarts** : niveau national tiré de l'échantillon, écarts entre
     sous-groupes tirés de C (recentrer les cellules de C pour que leur moyenne pondérée égale le
     national observé ; renormaliser, tronquer à 0).
5. **Score.** TV moyen sur sous-groupes × questions, contre la vérité (étape 1). Rapporter aussi la
   taille équivalente : n réel aléatoire qui donne le même TV (courbe de l'étape 2, mécanisme
   aléatoire, échantillon seul).

## Livrables

- `scripts/44_hybrid_small_sample.py` (CSV de résultats dans `data/analysis/hybrid_small_sample.csv`).
- Un tableau : lignes = méthode, colonnes = n ; deux panneaux (aléatoire, biaisé) ; séries
  répétées et nouvelles séparées.
- Trois phrases de conclusion : à partir de quel n l'hybride bat l'échantillon seul ; combien de
  vrais répondants vaut l'hybride à n = 100 biaisé ; si le biais de sélection survit au mélange.

## Pièges

- Chevauchement voulu : le petit échantillon et la vérité viennent du même sondage (on simule une
  collecte locale). La vérité reste l'échantillon complet ; à n = 400, le chevauchement gonfle un peu
  la précision : le signaler.
- Le biais de sélection à l'intérieur d'une cellule ne se corrige pas par post-stratification ;
  c'est ce que le mécanisme « biaisé » doit révéler. Ne pas le corriger avec une variable que le
  client n'aurait pas.
- Questions Trump (`tarrifs_1`, `tarrifs_2`) : rapporter avec et sans.
- CSV seulement (politique du repo), pas de Parquet.
