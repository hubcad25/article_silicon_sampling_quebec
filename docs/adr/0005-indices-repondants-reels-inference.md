# ADR 0005 - Indices présentés comme des répondants réels à l'inférence

- **Statut** : accepté (27 septembre 2026), avant le diagnostic et avant toute nouvelle inférence
- **Portée** : présentation des indices au modèle à indices (C1) ; aucun changement d'entraînement
- **Liens** : ADR 0001 (bras), ADR 0003 (analyse), `docs/note_recherche_2026-09-27.md` (section 3)

## Contexte

Sur les 12 questions du premier test (ADR 0003, température 1,0), les indices n'améliorent pas la
distance aux vrais répondants :

- BS − B0 (effet des indices, même modèle C1) : +0,003 [−0,038 ; 0,040] ;
- BS − A (meilleur montage contre modèle standard) : +0,026 [−0,005 ; 0,057], et significativement
  défavorable à 0,7 ;
- B − BS (fuite) : −0,005 [−0,024 ; 0,012].

Ils ont pourtant un effet : ils rétablissent les écarts entre cellules que A et B0 compressent de
moitié (rapport de variance 1,09 contre 0,57 ; 3 questions seulement). Question par question, BS
gagne 4 fois sur 12, B0 5 fois et A 3 fois ; BS perd parfois beaucoup (jusqu'à 0,11 de la meilleure
condition).

Le modèle C1 est entraîné comme un **répondant réel** : persona du répondant, **ses propres**
réponses à au plus 6 questions voisines du même sondage (similarité < 0,95 avec la cible), cible =
sa réponse. À l'inférence, BS et B lui présentent autre chose : les **distributions** de la cellule
sur ces questions voisines.

| | Entraînement (C1) | Inférence actuelle (B, BS) |
|---|---|---|
| En-tête | « Tes réponses à d'autres questions du sondage : » | « Réponses observées dans ton groupe à d'autres questions : » |
| Ligne | `question → réponse` | `question : option 26 %, option 67 %, … (n=45)` |
| Unité | une personne | un groupe résumé |

Hypothèses retenues pour l'échec :

1. **Format hors distribution.** En-tête, pourcentages et effectif n'ont jamais été vus à
   l'entraînement.
2. **Signal appris inapplicable.** Le modèle a appris qu'une réponse individuelle à une question
   voisine prédit la réponse de la même personne à la cible ; un résumé de groupe n'offre pas ce
   signal.
3. **Lecture du groupe comme une seule personne.** Le modèle répondrait comme un membre typique de
   la majorité des indices, ce qui accentue les écarts entre cellules mais écrase les minorités dans
   chaque cellule (exemple : item 43, cellule 45-54 ans × homme, indépendance 9 % prédit contre
   34 % observé, alors que les indices montrent environ 30 % d'appui).

## Décision

1. **L'entraînement ne change pas.** Le LLM reste un répondant : il voit ses propres réponses et
   donne la sienne. Les modèles C0 et C1 (8k, et 20k s'ils sont retenus) sont réutilisés tels quels.

2. **Diagnostic d'abord, sur les tirages existants** (aucun nouvel appel). Pour chaque paire
   item × cellule à T = 1,0, avec `obs` la distribution observée de la cellule, `moy` la moyenne
   non pondérée de `obs` sur les cellules de l'item, et `p_BS`, `p_B0` les distributions prédites :
   - **direction** : part des couples (paire, option) où `p_BS − p_B0` a le même signe que
     `obs − moy` (au-dessus de 0,5 : les indices poussent dans la bonne direction) ;
   - **amplification** : pente de `p_BS − moy_BS` sur `obs − moy` (au-dessus de 1 : les indices
     exagèrent l'écart de la cellule), comparée à la même pente pour B0 ;
   - **concentration** : entropie de `p_BS`, de `p_B0` et de `obs` par paire.

   L'hypothèse 3 est confirmée si la direction est bonne, l'amplification supérieure à 1 et
   l'entropie de BS inférieure à celle des vrais répondants. Le diagnostic informe l'interprétation ;
   il ne conditionne pas la décision 3.

3. **Nouveau bras BR (« indices réels »).** Même modèle C1, même persona de cellule, même question
   et mêmes options que BS. Seul le bloc d'indices change :
   - chaque tirage reçoit les réponses **individuelles d'un seul répondant de contexte** de la
     cellule (moitié `context` du second partage, ADR 0003), rendues dans le **format exact de
     l'entraînement** (« Tes réponses à d'autres questions du sondage : » puis `question → réponse`) ;
   - questions voisines : la règle d'entraînement (même sondage, au plus 6, similarité < 0,95,
     questions de test exclues), restreinte à celles auxquelles ce répondant a une réponse valide ;
   - un répondant sans aucune réponse voisine valide est écarté ;
   - attribution des tirages : les répondants admissibles de la cellule sont ordonnés par une graine
     fixe (20260927) et parcourus en boucle, le tirage `i` recevant le répondant `i mod n` ; chaque
     répondant reçoit ainsi 100 / n tirages à une unité près ;
   - aucun répondant d'évaluation n'entre dans le prompt.

   La distribution de la cellule émerge alors de vraies personnes ; le modèle fait exactement la
   tâche apprise.

4. **Évaluation** selon l'ADR 0003, à T = 1,0 seulement (la température est tranchée), sur les
   mêmes 275 paires. Contrastes appariés principaux : **BR − B0** (effet des indices réels) et
   **BR − A** (meilleur montage contre modèle standard) ; secondaire : **BR − BS** (format réel contre
   résumé). Diagnostic d'aplatissement (annexe D de la note) recalculé pour BR.

5. **Options écartées**, parce qu'elles abandonnent le LLM comme répondant :
   - entraîner C1 sur des distributions de groupe (ancien C2, déjà abandonné le 24 septembre) :
     aucun vrai répondant ne dispose de telles statistiques ;
   - faire prédire au modèle la distribution de la cellule en une seule réponse : il devient un
     estimateur, plus un répondant.

## Conséquences

- Coût : une campagne d'inférence (275 paires × 100 tirages = 27 500 appels sur le déploiement C1),
  plus le diagnostic, local.
- La persona reste celle de la cellule, comme dans les autres bras, pour isoler l'effet du format
  des indices. Donner à chaque tirage la persona complète du répondant de contexte serait encore plus
  proche de l'entraînement ; c'est une variante possible, pas l'analyse principale.
- Si BR bat B0 et A, les indices deviennent l'angle principal ; sinon, la conclusion « l'entraînement
  direct suffit, les indices n'ajoutent rien » est renforcée, avec les indices présentés dans leur
  meilleur format.
- Correspondance avec la note : A = Entraîné, B0 = Indices retirés, BS = Entraîné + indices,
  B = Fuite, BR = nouvelle condition (nom à fixer dans la note, par exemple « Entraîné + répondants »).
