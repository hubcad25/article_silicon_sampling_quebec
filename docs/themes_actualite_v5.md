# Thèmes d'actualité pour V5 — brouillon à valider

Un résumé par thème × année (2019-2024 entraînement, 2025 test), faits datés d'avant la collecte,
jamais de résultats d'opinion. Chaque question est classée dans un thème (ou « aucun ») par gpt-5.5,
puis vérifiée. Exemples tirés des exercices (script 41) et des candidats CES 2025 (script 39).

| # | Thème | Exemples entraînement | Exemples test 2025 |
|---|---|---|---|
| 1 | Relations Canada–É.-U. et commerce | liens avec les É.-U., ACEUM, libre-échange | `tarrifs_1/2` |
| 2 | Politique étrangère et conflits | Russie-Ukraine, liens avec la Chine | `foreign_pol_*` |
| 3a | Inflation et coût de la vie | finances perso, perspectives financières | — |
| 3b | Conjoncture et finances publiques | économie rétrospective, impact des politiques fédérales, impôts | — |
| 4 | Logement | — | `decent_housing`, `rent_security` |
| 5 | Système de santé (hors COVID) | soins privés payants | — |
| 6 | COVID-19 et mesures sanitaires | vaccins, PCU, restrictions | — |
| 7 | Climat et énergie | tarif carbone, réglementation env. | — |
| 8 | Immigration et intégration | priorité aux nés au Canada, langue requise | (items exclus : filtre) |
| 9 | Québec, langue et fédéralisme | souveraineté, menace au français, bilinguisme | — |
| 10 | Autochtones et réconciliation | pensionnats, « moins que mérité » | — |
| 11 | Égalité, genre et discrimination | droits égaux, femmes au Parlement, AMM, avortement | (sexisme/racisme exclus) |
| 12 | Libertés et contestation | convoi, droits en crise, discours haineux | `student_protes` |
| 13 | Élections et démocratie | ingérence étrangère, vote par la poste | `emb_*`, `ai`, `howvote1` |
| 14 | Gouvernement fédéral et partis | satisfaction Trudeau, SNC-Lavalin | — |
| P | Panorama de l'année (items multi-enjeux) | enjeu le plus important | `big_issue`, `big_issue_gov` |
| 15 | Criminalité, drogues et sécurité | cannabis | `safe_injection` |
| 0 | Aucun | socio-démo, santé perso, médias, valeurs stables, vote passé | — |

## Décisions (22 oct.)

- Économie et logement séparés ; économie coupée en 3a / 3b.
- Une question peut recevoir jusqu'à 2 thèmes (les deux résumés sont joints).
- Les questions de valeurs stables (rôle de l'État, égalitarisme) restent « aucun », même si le sujet est économique.
- `big_issue` (« le plus grand enjeu des prochaines années ») couvre tous les thèmes : il reçoit un résumé « panorama » de l'année (P).
