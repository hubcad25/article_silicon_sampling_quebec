# Plan — prochaine note : fixer la recette de l'approche C

*19 octobre 2026. But : trouver la meilleure façon de faire C (dossier, prompt, entraînement),
mesurée sur une sortie qui ressemble au produit. Pas de calibration des marges ni de contexte
d'actualité dans cette note : ça vient une fois la recette fixée.*

## Ce qui change dans le test

1. **Grille plus riche.** En plus de région × âge (15 cases) : sexe, scolarité, et deux
   croisements (âge × sexe, région × sexe). C prédit les écarts par variable et par croisement ;
   la grille complète se reconstruit par calage sur le recensement (repondération en cascade).
2. **Plus de questions jamais posées.** 8 questions, c'est trop peu (les 2 questions Trump
   dominent). Passer à environ 25-30 items CES 2025 tirés des 119 candidats du script 39
   (n suffisant, hors filtre de contenu). Rapporter avec et sans les items liés à l'actualité 2025.
3. **Trois mesures par variante**, toutes converties en « vaut un sondage représentatif de n »
   et en marge d'erreur par groupe :
   - **C seul** (le client n'a aucune donnée) ;
   - **qualité des écarts** : C recalé sur le vrai niveau national (ce que l'hybride utilise) ;
   - **hybride** : 200 répondants biaisés + C, simulation fixe du script 44.

## Variantes comparées (toutes sur Llama 3.3 70B, deux entraînements chacune)

| | Variante | Question posée |
|---|---|---|
| V0 | C par l'exemple actuel | Référence (0,152 / 0,214) |
| V1 | Dossier enrichi : 15 questions semblables au lieu de 8, tailles d'échantillon par groupe, écarts par sexe et scolarité | Plus de preuves aide-t-il ? |
| V2 | Plus d'exercices : 10 variantes par question au lieu de 3, plus les études québécoises du corpus (EEQ, CECD), soit environ 5-10 fois plus d'exemples | Combien rapporte la quantité ? |
| V3 | Exemple + récompense : V0 produit 8 tableaux par exercice, on réentraîne sur les meilleurs | La récompense aide-t-elle une fois partie d'un bon modèle ? |
| V4 | Sortie en deux temps : le modèle écrit d'abord le national, puis seulement les écarts | Séparer niveau et écarts améliore-t-il les écarts ? |

Plus deux références sans entraînement : C Llama de base et gpt-5.5 (signalé, contamination
possible).

## Avant de commencer

- Vérifier la courbe « échantillon seul » du script 44 (équivalence « 1 200 » jugée trop optimiste).
- Les questions semblables du dossier ne doivent jamais inclure une question à cosinus ≥ 0,80 d'un
  item testé (règle actuelle), y compris dans les nouveaux exercices de V2.
- Tout en ACI (inférence) ou serverless (entraînement) ; rien de long en local.

## Coût et durée estimés

- Construction des dossiers et exercices : 1-2 jours.
- 8 entraînements (V1-V4 × 2) : environ 1 jour en parallèle ; V3 demande un tour d'échantillonnage
  de plus (environ 15k requêtes, ACI).
- Inférence : environ 30 items × 3 tirages × 10 modèles, quelques milliers de requêtes ; négligeable.

## Ce que la note dira

Un tableau : variantes × (C seul, qualité des écarts, hybride), en TV et en « vaut n répondants ».
Une recommandation de recette pour la suite (calibration des marges, contexte d'actualité, agent).
