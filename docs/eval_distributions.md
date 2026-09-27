# Évaluation du pilote : quoi coder pour les distributions

Contrat scientifique : `docs/adr/0001-sous-ensemble-pilote-blocs-thematiques.md` (décision 6). Métriques : `docs/plan_article.md` §5.

## Ce qui existe déjà

On pose 12 questions de test à 275 cellules sociodémographiques (âge × genre × …). Chaque paire question × cellule reçoit **100 tirages à chacune de 4 températures** (0,3 · 0,7 · 1,0 · 1,3), dans 4 bras :

| Bras | Modèle | Contexte dans le prompt | Campagne (dossier) |
|---|---|---|---|
| A | fine-tuné C0 8k | aucun | `c0-8k` |
| B | fine-tuné C1 8k | distributions de la cellule sur 6 questions voisines | `c1-8k` |
| B0 | fine-tuné C1 8k | aucun | `c1-8k` |
| R | Llama 3.3 70B de base (jeu de rôle) | aucun | `base` |
| BS | fine-tuné C1 8k | comme B, mais distributions calculées sur la **moitié contexte** des tenus à l'écart | `c1-8k` |

Récupérer les sorties (une fois les runs finis ; `status` pour voir où ils en sont) :

```bash
.venv/bin/python scripts/20_cloud.py status
.venv/bin/python scripts/20_cloud.py fetch c0-8k c1-8k base
# → data/analysis/inference/<campagne>/<bras>.csv  (+ .jsonl, .diagnostics.csv, .manifest.json)
```

Une ligne de `<bras>.csv` = un tirage. Colonnes utiles :

| Colonne | Sens |
|---|---|
| `arm`, `temperature` | bras, température |
| `item_idx`, `block`, `survey_id`, `variable` | la question (`item_idx` = clé de `data/analysis/test_blocks.csv`) |
| `cell` | la cellule, ex. `25_34\|woman\|bachelor` |
| `heldout_valid_n` | nombre de vrais répondants de la cellule qui ont répondu à la question |
| `raw_response` | texte brut du modèle |
| `matched_code` | code de l'option reconnue, **même codage que les microdonnées** ; vide si invalide |
| `valid` | `True` si la réponse correspond à une option |
| `n_context` | bras B seulement : lignes de contexte montrées (0 à 6) |

Chaque paire × température doit avoir exactement 100 lignes, invalides comprises : c'est vérifié par le runner et résumé dans `<bras>.diagnostics.csv`.

## Étape 1 — distribution observée (vérité terrain)

Pour chaque paire question × cellule : **proportions pondérées** des options chez les **répondants tenus à l'écart** de la cellule, en ne gardant que les réponses valides (celles qui tombent sur une option offerte).

- Répondants tenus à l'écart : `data/split/heldout_respondents.parquet` (`__survey_id`, `__respondent_id`, `__weight`, `cell`). Jamais vus à l'entraînement.
- Réponses : `blob.read_survey(survey_id, ["__respondent_id", variable])`.
- Codes : `item.canonical_code(normalise_code(valeur))` ; valide si le code est une des `item.options`. C'est exactement ce que fait le prompt, et `matched_code` est dans le même codage.
- Les 275 paires et leurs objets `ItemSpec` sont donnés par `build_item_cells()`. **Ne pas les recalculer à la main.**

Point de départ (réutilise le code existant) :

```python
import polars as pl
from article_silicon_sampling_quebec.corpus import blob
from article_silicon_sampling_quebec.dataset import normalise_code
from article_silicon_sampling_quebec.inference import HELDOUT_PATH, build_item_cells

pairs = build_item_cells()                       # 275 paires, gelées
held = pl.read_parquet(HELDOUT_PATH)
rows = []
for p in pairs:
    micro = blob.read_survey(p.survey_id, ["__respondent_id", p.variable]).with_columns(
        pl.col("__respondent_id").cast(pl.Utf8))
    cell = held.filter((pl.col("__survey_id") == p.survey_id) & (pl.col("cell") == p.cell)) \
               .join(micro, on="__respondent_id")
    offered = {o.code for o in p.item.options}
    totals, n = dict.fromkeys(offered, 0.0), 0
    for raw, w in zip(cell[p.variable], cell["__weight"]):
        code = p.item.canonical_code(normalise_code(raw))
        if code in offered:
            totals[code] += w
            n += 1
    assert n == p.heldout_valid_n                # garde-fou : même n que le run
    mass = sum(totals.values())
    rows += [{"item_idx": p.item_idx, "cell": p.cell, "code": c,
              "share": totals[c] / mass, "n": n} for c in totals]
```

Pour le bootstrap du côté observé (§5), il faut garder les réponses **individuelles** (code, poids) par cellule, pas seulement les proportions.

## Étape 2 — distribution du modèle

Pour chaque bras × paire × température : proportions de `matched_code` parmi les tirages **valides**.

- Toutes les options de l'item apparaissent, même à 0 (reprendre la liste des options de l'étape 1).
- Garder à côté `n_valide` (N effectif) et `taux_invalide`. On ne jette rien en silence : un taux d'invalides élevé sur un item est un résultat à rapporter.

## Sortie attendue : une table longue

`data/analysis/distributions.csv`, une ligne par option :

| arm | item_idx | cell | temperature | code | share | n |
|---|---|---|---|---|---|---|
| observed | 4 | 25_34\|woman\|bachelor | | 1 | 0.12 | 38 |
| A | 4 | 25_34\|woman\|bachelor | 0.7 | 1 | 0.09 | 99 |

`arm = observed` pour la vérité (sans température), `n` = n valide. Tout le reste (KL, bootstrap, graphiques) part de cette table.

## Ensuite : métriques (plan §5, en bref)

- **KL(observée ‖ modèle)** par paire × bras × température. Il faut lisser les zéros du modèle (une option jamais tirée donne une KL infinie). **Choix à faire et à consigner**, par exemple un lissage additif de 0,5 sur les comptes.
- **Bootstrap des deux côtés** : rééchantillonner les répondants de la cellule (avec leurs poids) et les 100 tirages.
- **Contrastes entre bras** : bootstrap **apparié par item**.
- **Références** : marginale triviale (distribution observée de l'item, toutes cellules confondues), marginale par cellule, R (jeu de rôle).
- **Diagnostics** : aplatissement (variance entre cellules, synthétique / observée), nombre d'options distinctes, part de l'option modale, N effectif.

## Bras BS : évaluer sur la moitié évaluation

Les répondants tenus à l'écart sont coupés en deux dans chaque sondage × cellule : `data/split/heldout_halves.csv` (colonne `half` = `context` ou `eval` ; ADR 0001, décision 7).

- **BS s'évalue uniquement contre la moitié `eval`** : sa distribution observée se calcule comme à l'étape 1, en ne gardant que les répondants `half == "eval"`.
- **Tous les bras** sont évalués sur cette même moitié `eval`, y compris B. B conserve son contexte construit sur les 30 144 répondants gelés; BS utilise seulement la moitié `context` (ADR 0003).
- **Taille de la moitié `eval`** : n médian 31 par paire, minimum 13 ; 144 paires sur 275 ont n ≥ 30. Les comparaisons entre bras tiennent, mais pas la précision cellule par cellule. Si on agrège des cellules après coup (retrait d'une dimension), la distribution du modèle d'une cellule agrégée est la moyenne de ses cellules fines, pondérée par leur poids. À consigner avant de regarder les résultats.
- `evaluation.py` construit une vérité unique sur la moitié `eval` et connaît les cinq bras.

## À savoir

- **B = B0 pour l'item `meilpm`** (`cecd_elxn_qc_1998`, 9 cellules) : aucune question voisine n'est disponible, donc pas de contexte. À exclure des contrastes B vs B0.
- **Contexte de B** : il vient des mêmes répondants tenus à l'écart que la vérité, mais sur d'autres questions. Ce que chaque paire a vu est dans `c1-8k/B.context.csv`, avec le cosinus de chaque voisin. **Analyser le gain de B selon ce cosinus** : un voisin quasi identique rend la tâche triviale.
- **Petites cellules** : beaucoup d'items en français ont 8 à 11 cellules, toutes à n < 100. Ça permet de comparer les bras entre eux, mais pas d'affirmer une précision absolue.
- **Ne pas modifier** les 12 items, les 275 cellules ni le split. Toute nouvelle règle d'analyse se consigne dans l'ADR avant de regarder les résultats.
