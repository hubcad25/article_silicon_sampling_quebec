<!-- Généré par scripts/30_diagnose_old_indices.py; ne pas modifier à la main. -->

## Diagnostic des anciens indices en pourcentages

À température 1,0, l’ajout des indices déplace les probabilités dans la direction des écarts
humains pour **0,471 [0,434 ; 0,507]** des unités question × cellule × option admissibles. La pente
d’amplification est de **0,144 [0,059 ; 0,215]** pour BS, contre **0,087 [0,033 ; 0,127]** pour B0; leur
différence est de **0,057 [-0,025 ; 0,118]**.

L’entropie moyenne, en nats, vaut **1,188 [0,929 ; 1,460]** chez les répondants,
**1,284 [1,056 ; 1,528]** pour B0 et **1,268 [1,027 ; 1,506]** pour BS. La différence BS − observée
est de **0,080 [0,021 ; 0,142]**; une valeur négative indique des réponses synthétiques plus
concentrées.

**En clair**, les pourcentages donnés en indices font bouger le modèle, mais ils ne l'aident pas à
repérer de façon fiable ce qui distingue un sous-groupe. Ses déplacements ne vont pas plus souvent
dans la bonne direction que dans la mauvaise et demeurent beaucoup trop faibles. Contrairement à ce
qu'on soupçonnait, le modèle ne semble pas non plus se rabattre excessivement sur la réponse
majoritaire : il disperse plutôt ses réponses davantage que les vrais répondants.

**Conclusion.** Les trois critères ponctuels de l’ADR 0005 ne sont pas tous satisfaits. L’incertitude bootstrap ne permet pas d’affirmer les trois conclusions simultanément.

Les estimations regroupent les unités prévues dans l’ADR 0005. Les intervalles à 95 % proviennent
d’un bootstrap apparié par question (12 questions). Les écarts humains nuls
sont exclus du dénominateur de direction; une absence de déplacement de BS par rapport à B0 compte
comme un désaccord. Les pentes sont des régressions par l’origine après centrage par question et
option. L’entropie est calculée pour chaque paire question × cellule avant agrégation.

### Résultats par question

| Question | Direction | Pente BS | Pente B0 | Entropie BS − observée |
|---:|---:|---:|---:|---:|
| 1 | 0,397 | -0,013 | 0,063 | 0,098 |
| 4 | 0,488 | 0,093 | 0,196 | 0,180 |
| 7 | 0,593 | -0,091 | -0,162 | 0,019 |
| 20 | 0,491 | -0,080 | -0,067 | -0,144 |
| 21 | 0,469 | 0,018 | 0,282 | 0,076 |
| 25 | 0,410 | 0,149 | 0,101 | 0,254 |
| 33 | 0,500 | 0,405 | -0,007 | -0,073 |
| 40 | 0,414 | 0,303 | 0,142 | 0,190 |
| 43 | 0,545 | 0,063 | 0,030 | 0,174 |
| 48 | 0,512 | 0,165 | 0,115 | -0,022 |
| 53 | 0,527 | 0,143 | 0,007 | 0,046 |
| 56 | 0,483 | 0,079 | -0,000 | 0,027 |
