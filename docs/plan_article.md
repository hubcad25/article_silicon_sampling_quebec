# Plan de recherche — Échantillonnage silicone par strate, ancré sur un corpus de sondages

**Version** : 2026-09-24 · **Statut** : split gelé ; datasets régénérés au format **réponse = texte de l'option** ; runs 1-2 (format code) écartés, runs à relancer

---

## ⏩ REPRENDRE ICI — état au 24 septembre 2026

**Échéance dure : 4 octobre 2026** (expiration des crédits Azure).

### Fait depuis le 21 septembre
| | |
|---|---|
| **Phase 2 — split gelé** | commit `eeb7e87` (pré-enregistrement). 60 items de test (6 par cellule bin × langue, 30 FR / 30 EN), 30 144 répondants tenus à l'écart, 5 paires ≥ 0,97 signées. **Ne plus y toucher.** |
| Wording complet des CES | 440/440 items tronqués récupérés depuis les `.qsf` et codebooks (`data/ces_full_wording.json`, script 09) |
| Corpus | 1 778 items, dont 145 **contexte-seulement** (comportement déclaré : participation, dons, pétitions… — jamais cible) → 1 573 cibles d'entraînement |
| **Phase 3.1 — gabarit** | `prompts.py`. **Année du sondage** en en-tête du bloc system (`… mené en 2018.`), jamais abandonnée ; `year_override` pour l'inférence. Échelles annotées en contexte (`→ 4 sur 7 (1 = …, 7 = …)`). Déduplication du contexte au rendu (`select_context`). |
| **Phase 3.2 — générateur** | `dataset.py` + `scripts/16_generate_dataset.py` → `data/datasets/` (gitignoré, régénérable). C0 et C1 × 8 000 et 20 000 paires + validation 500. Stratifié sondage/langue/thème (thème = k-means sur embeddings, la colonne `themes` du catalogue est vide). 8k = préfixe du 20k ; C0 et C1 = mêmes paires. **4 runs ≈ 12,5 M tokens ≈ 69 $ US.** |

### Décisions prises (ne pas rouvrir)
- **C2 abandonné** : l'entraînement est purement par répondant ; la strate n'existe qu'à l'inférence (contexte de strate + marges d'erreur injectés à l'inférence seulement).
- **Pas de traduction** : le wording est la variable indépendante.
- La langue du répondant est déjà un champ du persona (`Langue maternelle`), tiré du crosswalk.
- Choix de vote déclaré de l'élection en cours = cible valide ; participation déclarée = contexte seulement.

### Correctifs de rendu — FAITS (24 septembre)
Vérifiés sur le corpus et sur les JSONL régénérés (tests : `tests/test_wording_hygiene.py`, 238 verts).
1. **Langue du contexte** : ligne dérivée du `question_text` (raccourci déterministe, libellé de batterie
   conservé en fin de ligne) ; `display_label` seulement si sa langue = celle de l'item. **0 / 87 824**
   lignes de contexte dans la mauvaise langue (C1 20k).
2. **Identifiants** : nom de variable (`q6_02.`, `p33 --`) **et numéro de questionnaire** (`Q13.`, `7a.`,
   `25.` — ~70 items CECD/EEQ non couverts par la première version) retirés au rendu. 122 énoncés touchés,
   0 cible restante avec préfixe.
3. **Mojibake** : seul cas du corpus (`rts_q1`, ``A` ``) réparé ; 0 dans les fichiers.

Datasets régénérés, audit OK, régénération byte-identique. Coût révisé : **4 runs ≈ 14,0 M tokens ≈ 77 $ US**
(C0 8k 7,5 $ · C0 20k 18,7 $ · C1 8k 14,5 $ · C1 20k 36,2 $).

### 🚀 Run 1 lancé — 24 septembre
`ftjob-be22287553734e58bf9d0c8ae0b3e4b2` · C0 8k · Llama-3.3-70B-Instruct-9 · 1 époque, batch 64, LR ×1,
seed 20260924 · suffixe `c0-8k`. Fichiers : train `file-2aa8d829…`, valid `file-083865c3…`.
**Solde avant : CA$ 907** (relevé portail). Solde après : _à relever (le portail peut accuser quelques heures de retard)_.
**Succeeded** — 13:40 soumis, 13:48 running, 14:49 terminé (~61 min de calcul). `trained_tokens` = **1 336 671**
(estimé 1 358 206). Modèle : `Llama-3.3-70B-Instruct-9.ft-be22287553734e58bf9d0c8ae0b3e4b2-c0-8k` (non déployé).
Loss : train 4,68 → ~0,8 dès le pas 16 puis plateau 0,66-0,82 ; moyenne 0,816 ; **eval 0,716** (fin). Métriques :
`logs/ft_c0_8k_metrics.csv`. Suivi : `python scratch/watch_job.py <job>`.

### 🚀 Run 2 lancé — 24 septembre
`ftjob-61f3591bd70a4c1c89bad9775842c6d5` · C0 20k · mêmes hyperparamètres et seed que le run 1 · suffixe `c0-20k`.
Train `file-760dd8aa…`, même fichier de validation. Lancé **sans attendre le delta du run 1** : le solde
portail (CA$ 907) n'avait pas bougé plusieurs heures après, et l'API de consommation renvoie des champs
vides sur l'abonnement sponsorisé. Pire cas (×3 le tarif Qwen) sur les 3 runs restants ≈ CA$ 290 < solde.
Le delta couvrira les runs 1+2 ensemble : répartir au prorata des `trained_tokens`.

### 🔁 Changement de format — 24 septembre : le modèle répond par le **texte** de l'option, plus par son code
**Décision (ne pas rouvrir).** Les options de l'item cible sont listées **sans code** (`- Le Parti Québécois`, ordre du
questionnaire, hygiène habituelle : mojibake réparé, espaces réduits) ; consigne FR « Réponds uniquement par le texte
exact de l'option choisie. » / EN « Answer with the exact text of the chosen option only. » ; le tour assistant est le
libellé rendu de l'option choisie (après `canonical_code`, fusion des refus comprise).
Pourquoi : un même contenu porte des codes différents d'un item à l'autre (« Ne sais pas » : 18 codes ; PLQ : 5), une
cible-code empêche le transfert vers des items inédits — la question centrale — et ses biais de symbole gonfleraient le
contraste FT vs roleplay ; les lignes de contexte C1 montraient déjà des libellés.
- **Runs 1 et 2 écartés** (format code) : run 1 `ftjob-be22…` (C0 8k, succeeded, non déployé) ; run 2 `ftjob-61f3…`
  (C0 20k) **annulé le 24 sept**. Les 4 runs sont à relancer sur les nouveaux fichiers.
- **Split pré-enregistré inchangé** (`data/split/` non touché) ; paires, personas, abandon SES, contexte C1, graines :
  identiques à l'octet près au build précédent (vérifié : seuls la liste d'options, la consigne et le tour assistant changent).
- **Unicité des libellés** : fusion au rendu des options de même libellé (insensible casse/espaces/ponctuation de bord,
  1er code gardé, les autres y sont repliés via `code_map` → prompt et distributions observées sur la même partition).
  Seuls cas du corpus : `eeq_2018` q53 et q54 (`Je ne sais pas` en 3 **et** 98 → 98 replié sur 3). Ce sont des items
  contexte-seulement ; ⚠️ le code 3 y pèse 41 % / 20 % des réponses, à vérifier au questionnaire (3 ≠ NSP ?).
- **Parseur d'évaluation** : `ItemSpec.match_answer(texte) → code | None` — correspondance exacte, puis normalisée
  (NFKC, casse, espaces, guillemets, puce `- ` recopiée, ponctuation finale). Jamais de correspondance floue : pas de
  correspondance = réponse invalide, comptée comme telle.
- **Métadonnées** : `data/datasets/pairs.csv` (compagnon de `pairs.parquet`), aligné ligne à ligne sur les JSONL :
  `split` (validation / train), `row_index` (ligne dans son fichier ; 8k = préfixe du 20k), `answer_code`, `answer_label`…
- Audit OK (0 violation, 0 avertissement) ; sur les fichiers : 0 tour assistant hors des libellés listés, 0 ligne
  d'option `N) `, 0 ligne de contexte dans la mauvaise langue, 0 cible à préfixe, 0 mojibake ; régénération
  byte-identique (11 artefacts, deux exécutions).

**Tokens et coût (tokenizer Llama-3 exact, 5,50 $/M)** — les options sans code coûtent un peu moins :

| Fichier | Tokens (texte) | $ US | Avant (code) |
|---|---|---|---|
| C0 8k | 1 344 243 | 7,39 | 1 358 206 · 7,47 |
| C0 20k | 3 360 638 | 18,48 | 3 395 115 · 18,67 |
| C1 8k | 2 617 866 | 14,40 | 2 631 829 · 14,48 |
| C1 20k | 6 553 796 | 36,05 | 6 588 273 · 36,24 |
| **4 runs** | **13 876 543** | **76,32** | 13 973 423 · 77,06 |

Validation : C0 84 039 tok (0,46 $), C1 162 211 tok (0,89 $).

### Test rapide du modèle C0 8k — 24 septembre (validation, pas test) — *format code, écarté*
Déployé 16:11 → supprimé 16:16. Format appris : **100 % de codes valides** sur les appels aboutis ; pas
d'effondrement à T=1 (5-6 modalités sur 20 tirages). Taux de réussite **non concluants** : ~1/3 des appels en
429 comptés comme erreurs par la 1re version du script (corrigée : `scratch/quick_eval.py`, 3 fils, 429 exclus,
réponses sauvegardées dans `logs/quick_eval_<dep>.json`).
**Règle pour la phase 5 : aucun tirage n'est jamais sauté.** Tout appel passe par
`foundry.FoundryChat` : les 429 et erreurs passagères sont réessayés (backoff exponentiel plafonné à
60 s, `Retry-After` respecté, jitter) ; après 15 min d'attente pour un même appel, le run **s'arrête**
avec `CallFailed` plutôt que de perdre le tirage — un tirage perdu biaise la distribution et réduit N en silence.

### Ensuite
1. Premier run **au format texte** : **C0 à 8 000 exemples** sur `Llama-3.3-70B-Instruct-9` (recette API plus bas).
   L'évaluation lit les sorties avec `ItemSpec.match_answer` (sortie non appariée = invalide, jamais réattribuée).
   **Relever le solde Azure credits juste avant et après** (CA$ 907,63 au dernier relevé) — seul
   moyen de connaître le prix réel du fine-tuning 70B, non publié.
2. Ajuster puis lancer les 3 autres runs **en séquence**.
3. Phase 5 : évaluation (déploiement, 80 tirages par cellule, balayage de température, **supprimer
   le déploiement immédiatement après**).

---

## 0. État d'avancement — 21 septembre 2026

**Échéance dure : 4 octobre 2026**, date d'expiration des crédits Azure. 13 jours.

### Fait

| Étape | Résultat |
|---|---|
| **0** Inventaire Azure | abonnement sponsorisé `a54061e5…` (info@opubliq.com) ; **quota GPU = 0 partout** ; seule voie = fine-tuning serverless Foundry |
| **1.1-1.2** Accès corpus + table des items | `blob.py`, `catalogue.py`, `data/items.parquet` |
| **1.3** Crosswalks SES | 7 dimensions canoniques, 17/17 sondages, 17 cas `needs_review` |
| **1.4** Dimensions de strate | définition à paliers (voir §3.5) |
| **1.5** Index de similarité | axe de distance validé à la main, seuils 0,95 / 0,70 |
| **Correctif catalogue** | bug de typage CES corrigé dans le mvp, réingéré en production |
| **Smoke test** | température validée sur déploiement fine-tuné |

### Corpus après correctif

**2 070 items**, 17 sondages, ~108 000 répondants, 1998-2025.
Équilibre **FR 951 / EN 1 119** (46/54) — il était de 68/32 avant correction du bug de typage.
424 items `multiple` **écartés** : cases à cocher et colonnes de mention, incompatibles avec une cible catégorielle unique. Extension possible, documentée comme limite.

### Budget réel

| | |
|---|---|
| Solde | **CA$ 907,63** (US$ 644,14 — le montant est en USD) |
| Consommation du produit existant | ~CA$ 6,65/jour → ~CA$ 87 d'ici le 4 oct |
| **Allouable à la recherche** | **~CA$ 820**, dont CA$ 100 de réserve |

L'API de consommation renvoie **tous les montants à zéro** sur un abonnement sponsorisé. Le seul instrument de mesure est le solde relevé manuellement dans le portail Azure credits, avec 8 à 24 h de décalage.

**Ancrage mesuré** : 200 exemples C0 = 35 000 tokens d'entraînement, soit **175 tokens/exemple**
(~600 avec k=8 items de contexte). Budget révisé du design complet :

| Poste | Volume | Coût estimé |
|---|---|---|
| Entraînement, 4 runs (C0/C1 × 2 durées) | ~11 M tokens | **~60 $** |
| Inférence, ~400 k appels × ~600 tok d'entrée | 240 M tokens | ~240-310 $ |
| Hébergement horaire | quelques dizaines d'heures | ~50-100 $ |
| **Total** | | **~500-600 $ sur 907 $** |

**Coûts par exemple — mesurés au tokenizer Llama-3 exact** (27 000 prompts réels, `data/prompt_token_report.csv`) :

| Condition | FR | EN | Global |
|---|---|---|---|
| C0 | 181 | 153 | **167** |
| C1 (k=6) | 302 | 242 | **272** |
| ~~C2~~ (abandonné) | 567 | 518 | 542 |

**Taille des datasets : ~13 000 paires par condition**, PAS 10⁵. Avec C2 abandonné et seulement
C0/C1, l'entraînement tombe à ~60 $ — la marge libérée finance une grille d'évaluation plus large. À 10⁵ paires, les 6 runs feraient
294 M tokens ≈ 1 620 $, soit deux fois l'enveloppe. Les 38,5 M tokens budgétés correspondent à
~13 k paires. Deux économies mesurées si besoin : exclure les items à > 15 options, et plafonner
le contexte C2 à 4 modalités (C2 : 542 → 458).

**Incertitude principale, à facteur 2-3** : le prix d'entraînement du 70B n'est pas publié ; le
tarif Qwen (5,50 $/M) sert d'approximation. D'où une règle ferme pour la phase 4 :

> **Lancer les 6 entraînements en séquence, jamais en parallèle.** Le delta de solde après le
> premier run donne le tarif réel et permet d'ajuster `k` ou le nombre de runs avant de s'engager.

### Modèle — décision forcée par l'inventaire

Seuls **4 modèles à poids ouverts** sont fine-tunables (eastus2 et canadaeast) :

| Modèle | Supervisé | Raisonnement forcé |
|---|---|---|
| **Llama-3.3-70B-Instruct v9** | ✅ | non |
| qwen3-32b | ❌ `rl_environment` seulement | — |
| gpt-oss-20b | ✅ | ✅ 188 tokens/appel |
| Ministral-3B | ✅ (non vérifié) | non |

**Rien entre 20B et 70B.** Qwen3-32B est éliminé : il refuse la supervision directe, qui est le cœur du papier. gpt-oss-20b impose un chain-of-thought qu'on voulait justement contrôler, et multiplie par 40 le coût en tokens de sortie. **Llama-3.3-70B est donc la cible**, sous réserve de son prix d'entraînement, absent de la grille publique.

Repli propriétaire si le 70B est trop cher : `gpt-4.1-mini` (poids fermés, faiblesse de reproductibilité).

**Llama-3.3-70B validé par smoke test** (job réussi, 35 000 tokens, 56 min) :
- fine-tuning supervisé fonctionnel ;
- **aucun canal `reasoning`** — 2 tokens de sortie par réponse contre 188 pour gpt-oss-20b. Le CoT
  reste une variable qu'on contrôle, et le poste « tokens de sortie » de l'évaluation devient
  négligeable ;
- **pas de `DeveloperTier`** — uniquement `GlobalStandard`, facturé à l'heure au tarif plein ;
- **le débit est plafonné par la capacité provisionnée** : à `sku-capacity 1`, 59 appels sur 60
  rejetés en 429. À dimensionner pour la phase 5, qui demande ~400 000 appels.

La courbe de dispersion par température **n'a pas été mesurée sur le 70B** (le déploiement a été
supprimé avant, priorité à l'arrêt de la facturation horaire). Elle l'a été sur gpt-oss-20b, même
pile de service — voir plus bas.

### Recette technique validée

```python
POST {endpoint}/openai/fine_tuning/jobs?api-version=2025-04-01-preview
{ "model": "Llama-3.3-70B-Instruct-9",
  "trainingType": "globalStandard",          # g minuscule, sinon retombe sur "Standard" et échoue
  "method": {"type":"supervised","supervised":{"hyperparameters":{"n_epochs":1}}} }
```

Déploiement d'un 70B fine-tuné (vérifié 24 sept.) : `az cognitiveservices account deployment create -g rg-opubliq-sondages -n info-4552-resource --model-name <ft> --model-version 1 --model-format **Meta** --sku-name **GlobalStandard** --sku-capacity 50` — prêt en < 1 min. `--model-format OpenAI` → erreur trompeuse « no hosting capacity » ; `DeveloperTier` refusé pour ce modèle. À capacité 50, ~1/3 des appels en 429 avec 8 fils ; **supprimer le déploiement immédiatement après chaque campagne** — facturation horaire, et l'abonnement bascule sur la carte de crédit après le 4 octobre.

Plancher de durée : **~40 min par job**, file d'attente comprise, quelle que soit la taille du dataset.

### Résultat du smoke test — la température fonctionne

200 exemples réels (EEQ 2014), 1 époque, gpt-oss-20b, 60 tirages par température :

| T | Catégories | Part modale |
|---|---|---|
| 0,0 | 1 | **100 %** |
| 0,7 | 3 | 91,7 % |
| 1,0 | 5 | 70,0 % |
| 1,3 | 6 | 45,0 % + **25 % de réponses vides** |

Réplique la courbe de Justin : effondrement en décodage glouton, récupération sous échantillonnage. **Plage utile 0,7-1,0.** T=1,3 est inutilisable et confirme que le compteur de N effectif par item est obligatoire.

### Décisions prises (21 septembre)

1. **« Je ne sais pas » / « Je préfère ne pas répondre » → gardées comme cibles valides.** Ce sont
   de vraies modalités et la distribution humaine les contient. Deux réserves à documenter : les
   modalités à faible masse (« préfère ne pas répondre » est souvent < 2 %) déstabilisent le KL
   dans les petites cellules et **exigent un lissage** ; et le taux de « ne sais pas » dépend en
   partie du mode d'administration (téléphone vs web), donc le modèle en apprend un peu de la
   méthodologie d'enquête et pas seulement de l'opinion.
2. **`k = 6` items de contexte** (C1, et le contexte injecté à l'inférence). Ramène les exemples à ~450 tokens, l'entraînement à
   ~160 $, et **libère de quoi financer aussi le balayage de la richesse du persona** (§2.7) —
   l'arbitrage entre les deux n'a plus lieu d'être.

### Décision encore ouverte

**Le thème du leave-one-theme-out**, à choisir et geler en même temps que le split. Candidats avec
assez d'items pour que le test soit informatif : immigration, environnement, laïcité/identité.
L'immigration est la mieux couverte et la plus proche de vrais cas d'usage.


---

## 1. Question de recherche

Un modèle de langue fine-tuné sur des microdonnées de sondage peut-il produire, **pour une
strate sociodémographique donnée et une question jamais posée**, une distribution d'opinion
proche de la distribution réellement observée ?

Et surtout : **qu'est-ce qu'il faut lui donner en contexte pour y arriver ?**

Trois contributions visées :

1. **Le facteur « type de contexte »** — comparer trois façons de conditionner le modèle
   (rien / les autres réponses du même répondant / les distributions observées dans sa strate
   sur des questions voisines). La troisième est la seule disponible à l'inférence sur un sujet
   nouveau ; c'est l'architecture du produit.
2. **La courbe couverture-performance** — l'erreur en fonction de la distance sémantique entre
   la question cible et le corpus d'entraînement. Dit où la méthode cesse de fonctionner, et
   fournit un indicateur de confiance exploitable en production.
3. **La validation par strate** — comparer des distributions *par cellule sociodémo* avec
   intervalles des deux côtés, plutôt que des marginales nationales. C'est le seul test qui
   vérifie que le persona conditionne réellement quelque chose.

### Positionnement

S'inscrit dans le prolongement de la note de Justin Savoie (2026-08-27, `litt/research-note.pdf`),
qui établit sur données canadiennes que (a) la supervision directe bat le roleplay d'un modèle
plus gros (5,66×), mais (b) **uniquement sous décodage échantillonné** — le même checkpoint passe
de 0,053 à 3,618 de KL entre T=1,0 et T=0,0. Ses contraintes méthodologiques sont reprises
intégralement ici (§6). Ce qu'on ajoute : un corpus deux ordres de grandeur plus large et plus
divers, l'ancrage empirique par strate, et l'axe de distance sémantique.

---

## 2. Design expérimental

### 2.1 Unité d'analyse : la strate, pas l'individu

Un **persona = une cellule sociodémographique**, pas une personne. On interroge le modèle
N fois (N ≈ 100) sur la même cellule à température élevée ; l'ensemble des tirages forme la
distribution prédite pour cette cellule. On agrège ensuite les cellules par post-stratification
sur les marges réelles pour obtenir une estimation de population, déclinable par strate.

Conséquence directe : la contrepartie empirique existe — c'est la distribution observée parmi
les vrais répondants de cette cellule. La validation est distribution-contre-distribution, avec
**incertitude des deux côtés** (la cellule observée a sa propre erreur d'échantillonnage,
souvent grande à n < 50).

### 2.2 Facteur principal — contexte (décision révisée, 22 septembre)

**L'entraînement est purement par répondant. La strate n'existe qu'à l'inférence.**
Un exemple d'entraînement = un répondant réel, et la cible est toujours **sa** réponse, jamais une
moyenne de cellule — s'entraîner sur des moyennes ferait sortir la modale et détruirait la
dispersion, qui est précisément ce que le produit vend.

| Bras | Contexte injecté | Cible |
|---|---|---|
| **C0** | rien | la réponse de ce répondant |
| **C1** | ses **propres** réponses aux 6 items les plus proches | la réponse de ce répondant |

**À l'inférence**, on interroge une cellule sociodémo et on peut lui injecter les **attitudes
observées de cette strate** sur les items voisins, **avec leurs marges d'erreur** (le `n` et le
`n_eff` de Kish sont disponibles par cellule ; une cellule à n=127 et une à n=18 ne méritent pas
la même confiance, et sans l'indiquer le modèle les traite pareil).

C1 n'est donc pas seulement un plafond : c'est lui qui **apprend au modèle à lire un bloc de
contexte**. Un checkpoint C0 n'en a jamais vu, donc lui en injecter à l'inférence serait franchement
hors distribution. C1 est le support de l'ancrage empirique.

> **Ce qui a été abandonné** : une troisième condition d'entraînement (« C2 ») injectant des
> distributions de strate dans le prompt d'**entraînement**. Coût mesuré 521 tokens/exemple contre
> 155 et 266, ~40 % du budget d'entraînement, et la pièce d'ingénierie la plus délicate (double
> masquage des priors). Elle n'ajoutait rien : le besoin est à l'inférence, pas à l'entraînement.
> Limite à déclarer : le test d'ancrage empirique se fait donc hors distribution d'entraînement.

### 2.3 Axe « distance sémantique » (variable continue, pas deux buckets)

Pour chaque item de test, on mesure la distance cosinus à son plus proche voisin dans
l'entraînement (embeddings `text-embedding-3-large`, déjà dans l'index AI Search). On rapporte
la performance **en fonction** de cette distance.

À l'entraînement, la distance des items de contexte (C1) est **échantillonnée** : chaque
exemple tire un bin (aucun / proche / moyen / lointain). Deux bénéfices — le modèle reste robuste
sur tout l'axe, et la distance devient balayable à l'évaluation **avec un seul checkpoint** au
lieu d'un checkpoint par régime.

Test extrême complémentaire : **leave-one-theme-out** — retirer un thème entier de
l'entraînement (ex. immigration) et évaluer dessus. Simule le cas réel « sujet jamais sondé »,
et met l'ancrage empirique en difficulté maximale (plus de voisin proche à récupérer).

### 2.4 Variables balayées à l'inférence (gratuites en entraînement)

- **Température** : {0,3 · 0,7 · 1,0 · 1,3}. Non négociable — c'est le premier facteur d'effet
  selon la note de Justin, et il ne se transfère pas d'un régime à l'autre. Rapportée avec
  chaque résultat, jamais fixée par défaut.
- **N tirages par cellule** : 100 en production ; vérifier par simulation que l'erreur Monte
  Carlo est petite devant l'erreur d'échantillonnage de la cellule observée.

### 2.5 Durée d'entraînement : une courbe, pas une constante

Justin observe une dégradation monotone (2 000 itérations battent 4 351 d'un facteur 2,48). Son
setup portait sur **28 items** — c'est du surapprentissage sur un ensemble étroit, et sa val loss
montait dès l'itération 1 000. Notre corpus (~3 160 items, ~108 k répondants, 1998-2025,
contexte tiré aléatoirement) est bien plus divers : le point de retournement devrait être plus
loin, et C1 augmente encore la diversité effective par exemple.

On ne copie donc pas son « 2 000 itérations » — **on mesure la courbe**. Et avec la bonne
métrique : il note lui-même s'être fait avoir en écartant une val loss montante au motif qu'elle
était dominée par les tokens de gabarit. On évalue donc le **KL distributionnel sur un set de
validation à intervalles réguliers pendant l'entraînement**, pas la loss. Si notre courbe ne se
retourne pas là où la sienne se retourne, l'interaction durée × diversité est un résultat en soi.

### 2.6 Hors périmètre pour la V1

- **Chain-of-thought** : reporté. Aucune justification n'existe dans les données (on entraînerait
  sur les rationalisations d'un autre LLM) ; le CoT concentre la sortie alors que le produit vit
  de la dispersion ; et il multiplie par 20-50× un coût d'inférence déjà en centaines de milliers
  de générations. Éventuellement testé à l'inférence seule sur le checkpoint sans-CoT.

  > **Contrainte découverte au smoke test** : `gpt-oss-20b` est un modèle à raisonnement — il émet
  > ~188 tokens de `reasoning` avant chaque réponse, donc il **impose** le CoT. C'est l'une des
  > raisons de lui préférer Llama-3.3-70B, qui répond directement (voir §0).
- **Comparaison multi-modèles de base** : un seul modèle, choisi selon le compute disponible.
  L'objet d'étude, ce sont les conditions, pas le classement des modèles.

---

### 2.7 Richesse du persona — granularité variable à l'inférence

L'entraînement se fait sur des **répondants individuels** avec leur profil SES complet ; la strate
n'existe qu'à l'inférence, définie par ce qu'on met dans le prompt. Un persona à 5 champs donne une
cellule fine (petit n), un persona à 2 champs une cellule grossière (gros n) — **même checkpoint**.

**Condition nécessaire** : entraîner avec **abandon aléatoire des champs SES** (chaque exemple tire
un sous-ensemble des dimensions). Sans ça, un modèle toujours entraîné sur 5 champs est hors
distribution quand on lui en donne 2, et la comparaison entre granularités mesure cet artefact.

Bénéfice, à coût nul en entraînement : un axe supplémentaire — **la performance en fonction de la
richesse du persona**, mesurable avec un seul checkpoint. Seul coût : l'inférence se multiplie par
le nombre de granularités testées.

**Arbitrage budgétaire ouvert** : `k` plus grand (contexte plus riche) OU balayage de la richesse
du persona. Les deux ne tiennent pas ensemble dans CA$ 820.

## 3. Données

### 3.1 Source

Le corpus du moteur de recherche (`../mvp_moteur_recherche_sondages`), deux rails déjà construits
et découplés :

- **catalogue** → index Azure AI Search `survey-questions` : `variable`, `question_text`,
  `response_options {code,label}`, `var_type`, `is_ordinal`, `is_sociodemo`, `sociodemo_type`,
  `concepts`, `themes`, embeddings 3072d.
- **microdonnées** → 1 Parquet par sondage dans le Blob `survey-responses` (raw-first : 1 ligne =
  1 répondant, colonnes = variables RAW, + `__respondent_id` / `__survey_id` / `__weight`),
  interrogé en DuckDB.

Jointure catalogue ↔ microdonnées triviale : `variable` = nom de colonne.

### 3.2 Périmètre retenu

Sur les 23 sondages normalisés (5 769 questions), on **écarte les sondages gouvernementaux
techniques** dont le vocabulaire noierait le mix thématique :

| Écarté | Items | Motif |
|---|---|---|
| `govcan_parca_2024` | 1 239 | satisfaction envers Parcs Canada, très spécialisé |
| `govcan_habit_2024` | 837 | habitation, très spécialisé |
| `govcan_06822_wave{1,2,3}_2024` | 502 | idem |
| `medaillon_organismes_qualitatif` | 0 fermé | qualitatif |

**Reste : 17 sondages · 2 070 items cibles · ~108 000 répondants · 1998 → 2025.**
(3 162 questions catalogées dans le périmètre ; 2 070 après filtres, une fois le bug de typage corrigé.) Mix thématique
sain (élections QC et CAN, charte/identité/laïcité, santé, opinion générale).

Items retenus : fermés `single` / `scale`, non-sociodémo. Les items sociodémo servent à construire
les personas, pas de cibles.

### 3.3 Langue : bilingue assumé, pas de traduction

On conserve le wording natif. Trois raisons : traduire corromprait la variable la plus
déterminante du design (le libellé exact détermine la distribution des réponses) ; les modèles
de base sont solidement bilingues sur ce type de tâche courte ; couper l'anglais coûterait
**1 119 items sur 2 070** et ~83 k des 108 k répondants (CES 2019/2021/2025).

Le risque réel n'est pas la langue mais le **confond langue × population** (anglais ≈ échantillon
canadien, français ≈ québécois). Neutralisé en mettant la population comme **champ explicite du
persona** (`Population : Québec` / `Canada`) plutôt que de la laisser se cacher dans la langue.
Résultats rapportés séparément FR / EN.

À vérifier : dans les CES, la langue est un attribut **du répondant**, pas du sondage — voir si le
catalogue normalisé a conservé les deux wordings ou seulement l'anglais.

### 3.4 Harmonisation SES — non bloquante pour l'entraînement

Les Parquet sont raw-first : l'index dit *quelle* variable est l'âge, pas comment ses niveaux se
comparent. L'hétérogénéité est réelle et large :

```
age        cecd_charte qage → 7 tranches · cecd_elxn_qc_1998 age → 5 tranches
           ces_2019 cps19_yob → année de naissance brute · ces_2019_phone q2 → codes -9/-8/-7
education  3 niveaux (1998) · 5 (2007) · 9 (sante_can_usa, anglais) · 12 (CES21/25) · 13-14 (eeq)
```

Mais l'entraînement peut se faire en **labels natifs verbatim** — le modèle lit du texte.
L'harmonisation n'est requise que pour trois choses, toutes en aval : définir des strates
comparables entre sondages, post-stratifier sur les marges du recensement, et donner un
vocabulaire d'entrée au produit.

**Compromis retenu** : harmoniser le **rendu du persona** (gabarit canonique sur ~5 dimensions,
un crosswalk par sondage) sans toucher aux items cibles, qui restent verbatim. Ça évite que le
modèle voie 14 vocabulaires pour la même personne. Volume : 17 sondages × ~5 dimensions.
Ne bloque pas un dataset v0 en labels natifs.

### 3.5 Choix des dimensions de strate — empirique

Contrainte double : **mesurable au recensement** (sinon pas de post-stratification ni de produit)
et **cellules assez peuplées** (sinon la distribution observée n'est pas estimable). Candidates :
âge, genre, scolarité, région, revenu, langue.

**Résultat (étape 1.4 faite).** C'est le **sondage** qui contraint, pas le corpus empilé : la
contrepartie empirique d'un item n'existe que chez les répondants du sondage qui l'a posé.

Définition retenue : **`age × gender × education`, seuil n ≥ 50, par sondage**, avec paliers :

| Palier | Sondages | Dimensions | Cellules ≥ 50 |
|---|---|---|---|
| A | les 5 gros (CES + sante_can_usa) | 3 | **246** |
| B | 10 sondages QC/CAN ≤ 2 500 | `age × gender` | **97** |
| C | eeq_2018, provincial_qc_2018 | `gender × education` | **16** |

**359 cellules au total**, dont 197 à n ≥ 100. Le gabarit de persona reste identique partout ;
seules varient les cellules retenues pour la validation.

Écartées avec preuves : **income** (11,3 % de non-réponse, absent de 2 sondages), **language**
(57,4 % du corpus assigné seulement), **region_qc** (84,7 % du corpus ne l'a pas — la contrainte
recensement ne la disqualifiait pas, les données si).

Un répondant n'entre dans une cellule que si **chaque** dimension se résout à un niveau unique :
les mappings `coarse` et `missing` sont comptés à part, jamais imputés ni répartis.

**Deux seuils, deux usages** : n ≥ 30 pour les contrastes entre conditions (l'erreur de la cellule
observée est commune aux bras et s'annule au bootstrap apparié) ; n ≥ 100 pour les affirmations
de précision absolue et le diagnostic d'aplatissement. Le plancher à ~30 est technique : sous ce
seuil les catégories vides rendent le KL non estimable sans convention arbitraire.

---

## 4. Split et pré-enregistrement

**Split factoriel items × répondants** : les items de test sont évalués sur des répondants
également retenus hors entraînement. Un item nouveau répondu par des répondants déjà vus est un
test plus facile, et on veut la version honnête.

Sélection des items de test **stratifiée sur l'axe de distance** : des items avec quasi-doublon
dans l'entraînement, des items isolés, et tout l'entre-deux — c'est la courbe du §2.3 qui est
l'objet, il faut donc couvrir la plage.

**Pré-enregistrement** — geler la liste des items de test **et la règle qui l'a produite** dans un
fichier commité et horodaté, **avant tout entraînement** : `data/heldout_items.json` + le script
générateur + un commit git daté. Pour une version citable, pousser le hash sur OSF.

Pourquoi : chez Justin, l'item encoder faisait 0,297 de KL moyen sur ses 30 items mais **0,510 sur
les 10 pré-enregistrés**. Choisir les items après avoir vu les résultats aurait flatté le fine-tune
d'un facteur 1,7, de façon indétectable de l'extérieur. Avec ~3 160 items, la tentation de
« choisir un test set représentatif » après coup est d'autant plus forte.

---

## 5. Métriques

**Niveau cellule** (primaire) : divergence KL entre distribution prédite (N tirages) et observée
(vrais répondants de la cellule, pondérés), avec **bootstrap des deux côtés** — bootstrap des
répondants réels de la cellule pour son erreur d'échantillonnage, bootstrap Monte Carlo des
tirages synthétiques. Le test devient « la prédite tombe-t-elle dans l'IC de l'observée ».

**Niveau population** : post-stratification des cellules sur les marges réelles, puis erreur sur
la marginale.

**Contrastes entre bras** : bootstrap **apparié par item** (les bras partagent l'ensemble d'items ;
la difficulté d'item est la source de dispersion dominante). Reprise directe du protocole de Justin.

**Diagnostics obligatoires** :
- **aplatissement par sous-groupe** — ratio entre variance inter-cellules synthétique et observée ;
  < 1 signale l'effacement de la variation démographique réelle. Chez Justin, 0,22 en décodage
  glouton contre 1,7-2,0 sous échantillonnage.
- **diversité de sortie** — nombre de catégories distinctes et part de la catégorie modale.
- **N effectif par item** — nombre d'appels *réellement* parsés, pas la taille nominale de la
  population synthétique. Sa note documente deux runs terminés sans erreur, plausibles et faux,
  où 14,2 % et 4,3 % des appels avaient échoué en se concentrant sur quelques items. Gate de run
  sur ce compteur.

**Baselines** :
- **marginale triviale** — tirage dans la distribution observée de l'item. Connaît la réponse,
  donc pas un concurrent : c'est l'échelle. On rapporte nos résultats en multiples de ce plancher.
- **marginale par cellule** — tirage dans la distribution observée *de la cellule*. Plancher plus
  exigeant, propre à notre design par strate.
- **roleplay non fine-tuné** — même gabarit de prompt exactement, balayé en température.

---

## 6. Contraintes méthodologiques héritées (non négociables)

1. Décodage **échantillonné**, jamais glouton ; température **balayée**, jamais ponctuelle ;
   configuration de décodage rapportée avec chaque résultat.
2. Ne **jamais transférer** une conclusion sur le décodage entre modèle promptté et modèle
   entraîné — le paramètre y fait un travail catégoriquement différent.
3. Bras roleplay et bras fine-tuné avec un **gabarit de prompt identique**, pour qu'aucune
   différence de prompt ne confonde le contraste supervision directe / indirecte.
4. Split de test **pré-enregistré** avant tout entraînement.
5. **N effectif par item** journalisé et gaté.
6. Sélection de checkpoint sur une **métrique distributionnelle**, pas sur la loss masquée.

---

## 7. Infrastructure

**Contrainte de calendrier** : crédits Azure à dépenser avant octobre 2026 — ~2 semaines utiles
au 17 septembre.

**Confirmé** : quota GPU à **0 vCPU** dans les six régions testées, pour les 28 familles NC/ND/NV,
y compris en low-priority. Aucune VM GPU ne peut démarrer. `canadaeast` n'offre d'ailleurs aucun
SKU N-series — une demande de quota devrait viser `canadacentral` ou `eastus2`.

**La voie est donc le fine-tuning serverless Azure AI Foundry**, qui ne consomme aucun quota de VM.
Modèle cible : **Llama-3.3-70B-Instruct v9** (voir §0 pour la décision et ses raisons).
Une demande d'augmentation de quota GPU reste un plan B, au délai incompatible avec le 4 octobre.

**Hygiène obligatoire** : `spendingLimit = Off` et une carte de crédit est attachée. Après le
4 octobre, tout ce qui tourne encore est facturé pour vrai. Aucun déploiement ne doit survivre à
sa campagne d'inférence.

Budget d'entraînement : plusieurs **runs courts** plutôt qu'un long (§2.5) — ce qui tombe bien
avec l'échéance. Ordre de grandeur : 4 checkpoints = {C0, C1} × 2 durées, un seul modèle de
base, puis balayage de température et de contexte injecté à l'inférence.

---

## 8. Étapes

### Phase 0 — Inventaire Azure ✅ FAIT

- **0.1** — *Agent dédié, tâche unique, lecture seule* : inventaire `az` — abonnement et type
  d'offre, crédits et date d'expiration, dépense 30 jours, **quotas GPU par région × famille de
  VM** (NC/ND/NV, A100/H100), workspaces Azure ML, comptes et déploiements Cognitive Services /
  Foundry, providers enregistrés. Aucune création, aucune clé affichée.
- **0.2** — Si quota nul : déposer la demande d'augmentation (famille, région, vCPU) **le jour
  même**. Chemin critique.
- **0.3** — Trancher Azure ML compute cluster vs Foundry serverless, et figer le modèle de base.

### Phase 1 — Socle de données ✅ FAIT

- **1.1** — Accès programmatique au corpus depuis ce dépôt : lecture des Parquet du Blob
  `survey-responses` via DuckDB, lecture du catalogue via l'index AI Search. Pas de duplication
  des données dans ce dépôt.
- **1.2** — Filtre de périmètre : 17 sondages, items fermés non-sociodémo → table
  `(survey_id, variable, question_text, options, theme, embedding)`.
- **1.3** — Crosswalks SES : schéma canonique de niveaux + mapping par sondage sur ~5 dimensions,
  avec gestion des non-réponses (-9/-8/-7) et bracketing des continues. Revue manuelle.
- **1.4** — Choix empirique des dimensions de strate (§3.5) : table n par cellule pour 2/3/4
  dimensions, décision documentée.
- **1.5** — Index de similarité item ↔ item et calcul des bins de distance.

### Phase 2 — Split pré-enregistré ⬅️ **PROCHAINE ÉTAPE** `[bloque toute la phase 3]`

- **2.1** — Script de sélection des items de test, stratifié sur la distance ; sélection des
  répondants de test.
- **2.2** — Gel : `data/heldout_items.json` + script + commit horodaté. **Aucun entraînement
  avant ce commit.**
- **2.3** — Choix du thème pour le leave-one-theme-out, gelé de la même façon.

### Phase 3 — Génération des datasets

- **3.1** — Gabarit de prompt unique, partagé par tous les bras (fine-tune et roleplay).
- **3.2** — Générateur paramétré C0 / C1, échantillonnage à la volée, **pas de gros `.jsonl`
  matérialisés** (le dépôt traîne des fichiers de 618 Mo : précisément le symptôme à éviter).
  Datasets par condition de taille modeste et fixe (~10⁵ paires).
- **3.3** — Contexte injecté à l'**inférence** : priors de strate avec **double masquage**
  (exclusion du répondant et de l'item cible), plus la marge d'erreur par cellule (`n`, `n_eff`).
  Tests unitaires sur le masquage — c'est là que se cache la fuite.
- **3.4** — Échantillonnage du bin de distance des items de contexte.

### Phase 4 — Entraînement

- **4.1** — Portage de `scripts/finetune.py` vers la cible Azure retenue.
- **4.2** — Évaluation distributionnelle périodique pendant l'entraînement (§2.5).
- **4.3** — Runs : {C0, C1} × 2 durées.

### Phase 5 — Évaluation

- **5.1** — Harness d'inférence : N tirages par cellule × items de test × températures, avec
  journalisation du **N effectif**.
- **5.2** — Métriques et bootstraps (§5).
- **5.3** — Baselines : marginale triviale, marginale par cellule, roleplay à gabarit identique.
- **5.4** — Figures : courbe distance-performance, température × condition, aplatissement par
  sous-groupe.

### Phase 6 — Note de recherche

Note autonome décrivant notre approche et nos résultats, à transmettre à Justin ; mise en commun
éventuelle ensuite.

---

## 9. Refactor du dépôt

### À supprimer

| Élément | Motif |
|---|---|
| `scripts/00_*` → `08_*` | pipeline spécifique CES 2021, source de données remplacée |
| `data/raw/ces_2021/` (315 Mo) | CES 2021 vient désormais du corpus commun |
| `data/processed/*.jsonl` (1,7 Go) | datasets SFT de l'ancien design |
| `data/results/condition1_samples.parquet` | résultats de l'ancien design |
| `jobs/sft_runner.sh`, `scripts/setup_narval.sh`, `scripts/submit_sft.sh`, `doc_calculcan/` | infra Narval/Slurm, remplacée par Azure (et le runner contient des marqueurs de conflit non résolus) |
| `notebooks/finetune_colab.ipynb`, `scripts/generate_colab_notebook.py`, `scripts/modal_finetune.py` | pistes de compute abandonnées |
| `.venv/` commité | ~500 Mo dans git ; à remplacer par `uv` + lockfile |
| `email.md`, `finetuning_infos_dump.md` | notes périmées |

### À conserver et adapter

| Élément | Devenir |
|---|---|
| `scripts/finetune.py` | moteur SFT/LoRA générique → cible Azure |
| `scripts/generate_sft_data.py` | logique leave-one-out → réécrite pour C0/C1 |
| `src/.../rag_similarity.py` | → récupération sur l'index AI Search |
| `finetuning_decisions.md` | archivé comme trace historique |
| `AGENTS.md` | mis à jour (venv → `uv`, politique de données, workflow) |

### Structure visée

```
src/     accès corpus (DuckDB/Blob, AI Search), personas, prompts, métriques
scripts/ pipeline numérotée du nouveau design
data/    crosswalks SES, heldout_items.json, résultats — aucune microdonnée dupliquée
paper/   note de recherche (Quarto)
docs/    ce plan + décisions
```

---

## 10. Décisions ouvertes

1. **Modèle de base** — après l'inventaire Azure. Arbitrage entre comparabilité avec Justin
   (Qwen3-4B) et le « plus gros puisqu'on a du cloud ».
2. ~~**k**~~ — fixé à 6 (voir §0).
3. **Langue des CES** — vérifier si le catalogue a conservé les deux wordings (§3.3).
4. **N tirages par cellule** — 100 par défaut, à valider par simulation contre l'erreur
   d'échantillonnage des cellules observées.
