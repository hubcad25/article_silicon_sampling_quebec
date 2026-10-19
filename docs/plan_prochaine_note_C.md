# Plan — prochaine note : fixer la recette de l'approche C

*19 octobre 2026. But : trouver la meilleure façon de faire C (dossier, prompt, entraînement),
mesurée sur une sortie qui ressemble au produit. Pas de calibration des marges dans cette note :
elle vient une fois la recette fixée.*

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
| V2 | Plus d'entraînement : plus de questions (études québécoises du corpus : EEQ, CECD) et 10 variantes de dossier par question au lieu de 3, soit environ 5-10 fois plus d'exemples | Plafonne-t-on déjà, ou la quantité rapporte-t-elle ? |
| V3 | Exemple + récompense, en deux étapes : (1) V0 écrit 8 tableaux par exercice, notés contre la vérité ; (2) on réentraîne V0 sur ses meilleurs tableaux | La récompense aide-t-elle une fois partie d'un bon modèle ? |
| V5 | Contexte d'actualité : un court résumé des faits de l'année, daté d'avant la collecte, ajouté au dossier, à l'entraînement (un résumé par année 2019-2024) comme au test (2025) | Corrige-t-il les questions liées à l'actualité sans nuire aux autres ? |

Plus deux références sans entraînement : C Llama de base et gpt-5.5 (signalé, contamination
possible).

V4 (sortie en deux temps, national puis écarts) écartée : dans le produit hybride, le niveau vient
de l'échantillon du client, et la qualité des écarts se mesure déjà directement.

## Avant de commencer

- Vérifier la courbe « échantillon seul » du script 44 (équivalence « 1 200 » jugée trop optimiste).
- Les questions semblables du dossier ne doivent jamais inclure une question à cosinus ≥ 0,80 d'un
  item testé (règle actuelle), y compris dans les nouveaux exercices de V2.
- Tout en ACI (inférence) ou serverless (entraînement) ; rien de long en local.

## Coût et durée estimés

- Construction des dossiers et exercices : 1-2 jours ; résumés d'actualité (V5) : environ 1 jour,
  à faire valider par Hubert (faits datés seulement, jamais de résultats d'opinion).
- 8 entraînements (V1, V2, V3, V5 × 2) : environ 1 jour en parallèle ; V3 demande un tour d'échantillonnage
  de plus (environ 15k requêtes, ACI).
- Inférence : environ 30 items × 3 tirages × 10 modèles, quelques milliers de requêtes ; négligeable.

## Ce que la note dira

Un tableau : variantes × (C seul, qualité des écarts, hybride), en TV et en « vaut n répondants ».
Une recommandation de recette pour la suite (calibration des marges, contexte d'actualité, agent).
