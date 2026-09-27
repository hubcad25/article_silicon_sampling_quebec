# ADR 0004 - Benchmark S par logit conditionnel

- **Statut** : accepte (27 septembre 2026), avant production de toute sortie S
- **Portee** : specification et integration du benchmark statistique S au pilote

## Contexte

L'ADR 0003 a gele l'analyse des cinq bras LLM avant que S existe et lui interdit
donc, a juste titre, d'inventer des lignes S. La note de protocole annonce toutefois
S comme un logit conditionnel regularise, entraine sur les memes 8 000 exemples et
produisant directement une distribution. Cette decision fixe son implementation
avant l'examen de ses predictions ou de ses metriques.

## Decision

1. **Information disponible.** S recoit le prompt C0 : profil sociodemographique,
   annee et population, libelle de la question et ensemble complet des options. Il
   ne recoit aucune reponse de contexte et aucune reponse aux questions test.
2. **Representation des alternatives.** Chaque alternative est encodee sous la
   forme `profil + question et options + alternative selectionnee` par le deploiement
   Azure `text-embedding-3-large` utilise par le projet. Les vecteurs de 3 072
   dimensions sont figes; seul le logit est ajuste. Le manifeste conserve le nom du
   deploiement, la version API, la dimension et le SHA-256 de la matrice complete.
3. **Estimateur.** Une seule fonction d'utilite lineaire est partagee entre toutes
   les questions et alternatives. Les utilites d'un ensemble de choix sont
   normalisees par softmax. L'objectif minimise est la log-vraisemblance negative
   conditionnelle moyenne plus `lambda / 2 * ||beta||^2`, sans constante, sans
   standardisation et sans normalisation prealable des embeddings.
4. **Donnees.** L'ajustement utilise exactement les 8 000 lignes de
   `c0_train_8000.jsonl`. Les 500 lignes de `c0_validation.jsonl` servent seulement
   a choisir la penalite et ne sont pas ajoutees au modele final. Cette validation
   partage certains items et repondants avec l'entrainement; elle selectionne une
   regularisation pour le meme regime que les fine-tunings et ne constitue pas une
   estimation de generalisation aux items test.
5. **Regularisation.** La grille L2 est `0.0001, 0.001, 0.01, 0.1, 1, 10`. La valeur
   minimisant la log-vraisemblance negative moyenne des 500 choix de validation est
   retenue. Une egalite est tranchee en faveur de la plus petite penalite. Chaque
   candidat est ajuste par L-BFGS-B, avec au plus 500 iterations, `ftol = 1e-10` et
   `gtol = 1e-6`. Un ajustement non converge est inadmissible a la selection;
   l'execution echoue si aucun candidat ne converge.
6. **Prediction.** S produit les probabilites exactes des options pour les memes 275
   couples item x cellule. `temperature = 1.0` est uniquement une cle de jointure
   avec la table principale; S n'a pas de temperature de decodage. `n` est nul afin
   de distinguer une probabilite exacte d'une frequence sur 100 tirages.
7. **Metriques.** La variation totale est calculee directement sur les probabilites
   de S. La KL est `KL(observee || S)` sans pseudo-compte, le softmax donnant une
   probabilite strictement positive. Pour les bras generatifs, la convention gelee
   de l'ADR 0003 demeure le pseudo-compte 0,5 sur leurs comptes empiriques. Cette
   difference de mesure est rapportee : S est une distribution predictive exacte,
   les bras LLM une approximation Monte-Carlo a 100 tirages.
8. **Incertitude.** Pour S, le bootstrap reechantillonne les repondants observes et
   les items, conditionnellement au modele S ajuste. Il ne reechantillonne pas des
   pseudo-tirages et ne propage pas l'incertitude d'entrainement. Les intervalles LLM
   conservent en plus leur incertitude de tirage gelee dans l'ADR 0003.
9. **Comparaisons.** Les resumes de S suivent la meme moyenne non ponderee des
   cellules dans chaque item, puis le meme poids aux 12 items. Les contrastes avec A,
   B0, B et BS sont tous publies; aucun bras n'est choisi apres observation comme
   « meilleur ». Aucun contraste R-S n'est produit, R n'ayant pas le meme budget
   d'entrainement comparable. Ces contrastes sont une extension pre-specifiee par le
   present ADR, distincte des six contrastes LLM geles dans l'ADR 0003.
10. **Integrite.** Le manifeste est ecrit en dernier, contient les empreintes des
    entrees et sorties, et l'analyse refuse des sorties absentes, partielles ou dont
    l'empreinte ne correspond pas. La couverture exige exactement 275 couples.
11. **Execution distante.** La generation complete des embeddings et l'ajustement de
    S sont des traitements longs qui doivent pouvoir continuer lorsque le poste de
    travail est ferme, en veille ou eteint. L'execution de production se fait donc sur
    Azure Container Instances (ACI),
    avec cache et sorties sur un stockage persistant. Un processus local en arriere-plan
    ne constitue pas une execution de production : il peut servir a un test court, mais
    ne doit pas etre utilise pour la campagne complete. Le runner distant doit reprendre
    les embeddings deja valides, limiter explicitement ses ressources CPU et memoire,
    journaliser sa progression et ecrire le manifeste seulement apres validation de
    toutes les sorties.

## Interpretation

S est un benchmark discriminatif a budget d'exemples comparable, pas un estimateur
d'enquete pondere a l'entrainement. Comme les fine-tunings, il donne le meme poids a
chaque exemple du fichier equilibre; seules les distributions humaines servant de
verite terrain sont ponderees. Un avantage de S peut inclure l'absence de bruit lie
aux 100 generations. Cette difference operationnelle fait partie des methodes telles
qu'elles seraient utilisees, mais doit accompagner toute comparaison quantitative.
