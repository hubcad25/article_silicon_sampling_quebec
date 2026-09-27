# ADR 0003 - Analyse comparative des bras du pilote

- **Statut** : accepte (25 septembre 2026), avant examen des metriques des campagnes
- **Portee** : comparaison des bras R, A, B0, B et BS sur les 12 items pilotes

## Contexte

Le second partage gele des 30 144 repondants tenus a l'ecart laisse 15 197 personnes
dans la moitie `eval`. Parmi les 275 paires item x cellule du pilote, 144 conservent au
moins 30 reponses valides dans cette moitie; les autres restent utiles aux contrastes
apparies, mais leur distribution observee prise isolement est imprecise.

Le protocole doit etre fixe avant de consulter les resultats des bras. Le benchmark
statistique S n'est pas implemente et ne fait donc partie d'aucun calcul.

## Decision

1. **Verite terrain unique.** Tous les bras sont evalues exclusivement contre les
   distributions ponderees des repondants de la moitie `eval`. B conserve son contexte
   construit sur les 30 144 repondants tenus a l'ecart; BS conserve son contexte
   construit sur la moitie `context`. Les sorties d'inference ne sont pas melangees
   entre temperatures.
2. **Petites cellules.** L'analyse principale conserve les 275 paires afin de preserver
   le plan apparie et le meme ensemble de cellules entre bras. Une cellule dont
   `observed_n < 30` est explicitement marquee `small_cell`; elle ne sert jamais a une
   interpretation de precision absolue. Une analyse de sensibilite refait les resumes
   et contrastes sur les seules cellules dont `observed_n >= 30`. Si ce filtre laisse
   un item sans cellule admissible, la sensibilite reste presentee par item et aucun
   resume global a moins de 12 items n'est substitue au resultat preregistre. Il n'y a
   pas d'agregation post-hoc des cellules.
3. **Metrique principale.** La variation totale,
   `TV(p, q) = 0,5 * somme_k |p_k - q_k|`, est calculee par item x cellule. Elle ne
   requiert aucun lissage. Une condition dont les 100 sorties sont invalides conserve
   une metrique manquante et n'est pas transformee en distribution uniforme.
4. **Robustesse.** La divergence `KL(observee || modele)` est calculee avec un
   pseudo-compte additif de 0,5 sur les comptes du modele seulement, puis
   renormalisation. Ce choix stabilise les probabilites modeles nulles sans modifier la
   distribution observee.
5. **Agregation.** Pour chaque bras et temperature, les cellules admissibles sont
   moyennees sans ponderation dans chaque item, puis les moyennes des 12 items recoivent
   le meme poids. Une estimation n'est produite que si les 12 items sont representes.
   La meme regle s'applique aux contrastes apparies.
6. **Analyse principale.** Seule `T = 1.0` figure dans les tables principales. Les
   contrastes signes sont `B-A`, `B-B0`, `BS-A`, `BS-B0`, `B-BS` et `A-R`; une valeur
   negative indique que le premier bras a une distance plus faible. Les 12 items sont
   conserves dans chaque contraste. Pour `meilpm`, B et B0 n'ont aucun contexte par
   construction; sa contribution a `B-B0` est donc rapportee comme un controle nul et
   non comme une estimation de l'effet du contexte. Un contraste secondaire B-B0 sur
   les 11 items avec contexte peut etre presente en annexe, clairement etiquete.
7. **Incertitude.** Les intervalles percentiles a 95 % utilisent 2 000 repetitions et
   la graine 20260925. Chaque repetition reechantillonne avec remise les repondants de
   chaque cellule en conservant leur poids, les tirages valides de chaque bras, puis les
   12 items avec remise. Le reechantillonnage des repondants et des items est commun aux
   bras d'un contraste; les tirages modeles sont reechantillonnes dans leur propre bras.
8. **Annexes.** Les taux d'invalidite, les trois autres temperatures et
   l'aplatissement sont reserves aux annexes. L'aplatissement est le ratio, calcule par
   item puis option, de la variance non ponderee entre cellules des parts modeles a la
   variance correspondante observee; les ratios dont le denominateur est nul sont
   manquants. Conformement au plan initial, ce diagnostic descriptif est limite aux
   cellules dont `observed_n >= 100` et n'est pas interprete lorsqu'un item a moins de
   deux cellules admissibles.

## Sorties attendues

- metriques par bras x item x cellule x temperature, avec `observed_n` et `small_cell`;
- resumes principaux a `T = 1.0`, toutes cellules puis sensibilite `n >= 30`;
- six contrastes apparies pour la variation totale et KL;
- annexes d'invalidite, de temperature et d'aplatissement;
- manifeste des parametres d'analyse et des fichiers sources.

Les resultats ne doivent contenir aucune ligne pour S tant que ce benchmark n'existe
pas effectivement.
