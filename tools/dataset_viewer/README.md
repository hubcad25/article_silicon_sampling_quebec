# Dataset viewer

Petite app React (Vite) pour parcourir les jeux de fine-tuning de `data/datasets/`.

```bash
cd tools/dataset_viewer
npm install      # une seule fois
npm run dev      # http://localhost:5173
```

Le serveur de dev lit directement `../../data/datasets/` (lecture seule, voir `vite.config.js`) :
`/api/files` liste les `.jsonl`, `/data/datasets/<fichier>` les sert. Un fichier `.jsonl`
quelconque peut aussi être chargé avec le sélecteur de fichier.

Affiche chaque exemple comme une conversation (persona, question, réponse), avec filtres par
langue, année du sondage, présence de contexte (C1) et recherche libre, et un résumé du
sous-ensemble filtré (FR/EN, années, réponses les plus fréquentes, lignes de contexte).
