---
title: "Fine-tuner un LLM sur des sondages québécois et canadiens"
subtitle: "Note d'étape — silicon sampling par strate"
author: "Hubert Cadieux"
date: "Septembre 2026"
lang: fr
---

## En une phrase

Un modèle de langue entraîné sur les réponses de vrais répondants à des sondages peut-il
reproduire la distribution d'opinion d'un groupe sociodémographique **sur une question qu'il
n'a jamais vue** ? Et est-ce que ça aide de lui montrer d'autres réponses du même répondant ?

## Pourquoi

- Sonder coûte cher et prend du temps. Si un modèle peut estimer « ce que penseraient les femmes
  de 55 ans et plus au Québec » sur une nouvelle question, c'est utile pour préparer ou compléter
  un sondage, sans le remplacer.
- Ton travail a montré que ça se joue beaucoup dans le décodage (température) et que le
  fine-tuning change la nature du problème. Je reprends ces leçons : décodage échantillonné,
  température balayée, même prompt pour le modèle fine-tuné et le modèle de base.
- Ce que j'ajoute : un corpus beaucoup plus large (17 sondages, 28 ans), une évaluation **par
  groupe sociodémographique** plutôt que sur l'ensemble, et un test de ce qui aide le modèle
  quand la question est loin de tout ce qu'il a vu.

## Les données

| | |
|---|---|
| Sondages | 17, de 1998 à 2025 : Études électorales canadiennes (CES), Études électorales québécoises, sondages CECD |
| Répondants | 108 199 |
| Questions retenues | 1 633 questions d'opinion (les questions factuelles, les variables dérivées et le comportement déclaré comme la participation sont exclues comme cibles) |
| Langues | français et anglais, questions gardées dans leur langue d'origine |

Les caractéristiques sociodémographiques (âge, genre, scolarité, région, revenu, langue
maternelle) sont harmonisées entre les 17 sondages.

## Ce que j'ai fait

### 1. Mettre de côté le test avant tout entraînement

Avant d'entraîner quoi que ce soit, j'ai gelé **60 questions de test** (30 en français, 30 en
anglais) et **30 144 répondants** que le modèle ne verra jamais. Les 60 questions sont réparties
selon leur **distance aux questions d'entraînement** : des questions presque identiques à
d'autres jusqu'à des questions isolées. Ça permet de mesurer si le modèle généralise ou s'il
reconnaît juste des questions déjà vues.

### 2. Un exemple d'entraînement = un répondant, une question

Chaque exemple montre au modèle un profil de répondant et une question, et lui demande le numéro
de la réponse. Exemple (profil raccourci) :

```
Système : Tu es un répondant à un sondage d'opinion mené en 2018.
          Population : Québec
          Âge : 55-64 ans
          Genre : Femme

Question : Et pour quel parti diriez-vous que vous auriez tendance à voter?
Options :
1) Le Parti libéral du Québec (PLQ)
2) Le Parti Québécois (PQ)
3) La Coalition Avenir Québec (CAQ)
...
Réponds uniquement par le numéro de l'option choisie.

Cible : le numéro choisi par le vrai répondant
```

Les champs du profil sont retirés au hasard d'un exemple à l'autre, pour que le modèle sache
répondre avec un profil détaillé comme avec un profil minimal.

### 3. Deux conditions, deux tailles

| Condition | Ce que le modèle voit |
|---|---|
| **C0** | le profil + la question |
| **C1** | le profil + les vraies réponses du même répondant à 6 questions voisines du même sondage + la question |

Chaque condition est entraînée à 8 000 et à 20 000 exemples, ce qui donne **4 modèles**.
Modèle de base : **Llama 3.3 70B Instruct**, fine-tuné sur Azure AI Foundry (une époque).

### 4. Comment on évalue

- Le modèle répond toujours comme **un** répondant. La distribution vient de la répétition : on
  interroge le même profil de groupe plusieurs fois, à une température donnée, et on compte les
  réponses.
- On compare cette distribution à la vraie distribution observée dans ce groupe (répondants
  tenus à l'écart), avec la divergence KL et des intervalles par bootstrap.
- Références : tirer au hasard dans la distribution observée de la question (le plancher), et le
  même prompt envoyé au modèle **non fine-tuné** (le jeu de rôle classique).

## Où j'en suis

- [x] Corpus, harmonisation et split de test gelé
- [x] Jeux d'entraînement générés et vérifiés
- [x] Modèle C0, 8 000 exemples : entraîné
- [ ] Modèle C0, 20 000 exemples : en cours
- [ ] Modèles C1, 8 000 et 20 000 exemples
- [ ] Évaluation sur les 60 questions de test

## Premières observations

*Test rapide sur quelques questions de validation, pas sur le test.*

- Le modèle respecte le format dans 100 % des cas : il répond un numéro d'option valide, rien
  d'autre.
- Ses réponses dépendent du profil et ne s'effondrent pas sur une seule option à température 1.
- Deux défauts visibles : des distributions trop plates (des options marginales qui reçoivent
  10 à 15 %) et « Ne sais pas » presque jamais choisi, alors que les vrais répondants le
  choisissent souvent.

## Résultats

*À venir.*

## Questions pour toi

1. *À compléter.*
