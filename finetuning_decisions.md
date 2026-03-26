# Décisions fine-tuning (condition 4)

**Modèle** : Llama-3.1-8B-Instruct pour le SFT (conditions 4-5) ; le 70B reste pour les conditions 1-3. Un 8B fine-tuné sur des données locales suffit pour apprendre des distributions de réponses — le 70B est le plafond généraliste contre lequel on compare (silicon sampling simple).

**Méthode** : LoRA si A100+, QLoRA (NF4) si GPU consumer. Pas de QLoRA par défaut — seulement si le modèle ne rentre pas en mémoire.

**Stratégie de données** : un sample = un répondant, avec ses réponses aux 75 questions train comme contexte et une question train retirée comme cible (leave-one-out). Le loss est calculé uniquement sur la réponse cible. ~6317 répondants × 76 questions = ~480k samples.

**Format** : single-turn instruction (pas multi-turn). Toutes les réponses contexte sont injectées dans un seul message `user`. La distribution train correspond exactement à la distribution d'inférence.

```json
{"messages": [
  {"role": "system", "content": "Tu es un répondant québécois : 34 ans, femme, université, francophone. A voté en 2019 : oui."},
  {"role": "user", "content": "Voici tes réponses au sondage :\nQ: Satisfaction envers le gouvernement fédéral ? R: Plutôt insatisfait\nQ: Immigration au Canada ? R: En admettre environ le même nombre\n[...73 autres questions...]\n\nQuestion : Appuyez-vous la taxation carbone ?\nOptions: 1) Fortement en faveur  2) Plutôt en faveur  3) Plutôt contre  4) Fortement contre"},
  {"role": "assistant", "content": "2"}
]}
```
