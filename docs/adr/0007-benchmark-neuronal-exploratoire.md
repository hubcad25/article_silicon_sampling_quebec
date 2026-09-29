# ADR 0007 - Benchmark neuronal exploratoire N

- **Statut** : accepte (29 septembre 2026), avant production de toute sortie N
- **Portee** : specification d'un benchmark neuronal exploratoire distinct du benchmark S

## Contexte

Le benchmark S applique une utilite lineaire aux embeddings geles de chaque
alternative. Il obtient une variation totale moyenne de 0,281 sur le pilote,
contre 0,219 pour le LLM entraine sur 8 000 exemples. Un reseau leger permet de
tester si une partie de cet ecart vient simplement de relations non lineaires
que S ne peut representer.

N demeure une analyse exploratoire. Il ne remplace ni le benchmark S
prespecifie, ni les contrastes confirmatoires deja annonces.

## Decision

1. **Donnees et information.** N reutilise exactement les embeddings
   `text-embedding-3-large`, les 8 000 choix d'entrainement, les 500 choix de
   validation et les 275 couples item x cellule de S. Il ne recoit aucune
   reponse de contexte ni aucune reponse aux questions test.
2. **Point de depart lineaire.** Les utilites du logit S ajuste sont gelees. N
   apprend un residu ajoute a chaque utilite; un residu nul reproduit donc S
   exactement.
3. **Projection.** Une ACP randomisee de 128 composantes est ajustee uniquement
   sur les alternatives d'entrainement (`random_state = 20260929`, trois
   iterations de puissance). Chaque score est divise par son ecart-type dans
   l'entrainement, avec un plancher de `1e-6`. Validation et test sont seulement
   transformes par cette projection gelee.
4. **Architecture.** Le residu est un perceptron partage entre toutes les
   questions et alternatives : 128 entrees, une couche cachee de 64 unites
   `tanh`, puis une sortie scalaire sans constante. Le softmax est calcule a
   l'interieur de chaque ensemble de choix. Le reseau compte 8 256 parametres.
5. **Initialisation.** La matrice cachee utilise l'initialisation de Glorot avec
   la graine 20260929; les biais et les poids de sortie commencent a zero. Le
   modele a l'epoque zero est ainsi identique a S.
6. **Ajustement.** La perte est la log-vraisemblance negative conditionnelle
   moyenne, avec penalite L2 de `1e-4` sur les deux matrices de poids. Adam est
   utilise avec un taux de `1e-3`, des lots de 256 choix et au plus 200 epoques.
   L'ordre des choix est melange de facon deterministe a chaque epoque.
7. **Arret.** La log-vraisemblance negative des 500 choix de validation est
   mesuree apres chaque epoque. L'etat est conserve seulement s'il ameliore le
   meilleur resultat d'au moins `1e-5`. L'ajustement s'arrete apres 20 epoques
   sans amelioration. L'epoque zero est admissible; N peut donc etre exactement
   egal a S si aucun residu n'aide la validation.
8. **Prediction et metriques.** N produit des probabilites exactes (`arm = N`,
   `temperature = 1.0`, `n` nul) pour les 275 couples. L'analyse exploratoire
   rapporte N-S et N-A avec la meme variation totale, la meme KL et le meme
   bootstrap apparie que l'analyse existante.
9. **Integrite.** Les sorties sont ecrites sous `neural_benchmark/`, jamais dans
   `statistical_benchmark/`. Le manifeste conserve les hyperparametres, les
   empreintes des sources et sorties, l'epoque retenue et les pertes de
   validation avant et apres correction.
10. **Execution.** La production se fait sur Azure Container Instances, avec le
    cache d'embeddings existant sur Azure Files, 4 CPU et 16 Gio de memoire.
    Aucun GPU et aucun nouvel appel d'embedding ne sont requis si le cache est
    complet.

## Interpretation

Un gain de N sur S indiquerait que la non-linearite d'une petite tete
discriminative explique une partie de l'avantage du LLM. L'absence de gain ne
prouverait pas que toute architecture neuronale est inutile : elle ne porterait
que sur cette projection, cette taille de donnees et cette architecture fixee.
Dans les deux cas, N reste exploratoire parce que son ajout a ete decide apres
observation des premiers resultats de S et du LLM.
