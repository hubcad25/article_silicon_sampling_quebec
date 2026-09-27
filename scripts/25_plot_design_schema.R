#!/usr/bin/env Rscript

# Schéma autonome du protocole expérimental.
# Usage : Rscript scripts/25_plot_design_schema.R [output_dir]

suppressPackageStartupMessages({
  library(ggplot2)
  library(showtext)
  library(sysfonts)
})

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[[1]] else
  "data/analysis/inference/figures"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Privilégier l'installation locale pour que le script fonctionne hors ligne.
local_nunito <- sysfonts::font_files()
local_nunito <- local_nunito[
  grepl("^Nunito Sans$", local_nunito$family, ignore.case = TRUE),
]

if (nrow(local_nunito) > 0) {
  regular <- local_nunito[grepl("regular", local_nunito$style, ignore.case = TRUE),]
  bold <- local_nunito[grepl("bold", local_nunito$style, ignore.case = TRUE),]
  if (nrow(regular) == 0) regular <- local_nunito[1,]
  if (nrow(bold) == 0) bold <- regular[1,]
  sysfonts::font_add(
    "nunito",
    regular = file.path(regular$path[1], regular$file[1]),
    bold = file.path(bold$path[1], bold$file[1])
  )
} else {
  sysfonts::font_add_google("Nunito Sans", "nunito")
}
showtext::showtext_auto()
showtext::showtext_opts(dpi = 300)

dashboard_colors <- list(
  green = "#00A087",
  red = "#f0695a",
  blue = "#0072B2",
  yellow = "#E69F00",
  opubliq_lightblue = "#0d8491",
  opubliq_darkblue = "#012326"
)

ink <- "#243238"
muted <- "#657278"
rule <- "#C9D1D4"
panel_fill <- "#F5F7F7"
header_fill <- "#E8EDEE"
blue_fill <- "#E8F2F8"
teal_fill <- "#E7F5F4"
green_dark <- "#287A69"

# Panneau A : un arbre, sans suggérer une suite d'étapes analytiques.
tree_nodes <- data.frame(
  id = c("total", "train", "test", "c0", "c1", "context", "evaluation"),
  xmin = c(5.70, 0.65, 7.75, 0.65, 3.85, 7.75, 11.05),
  xmax = c(9.30, 7.15, 14.25, 3.45, 7.15, 10.75, 14.25),
  ymin = c(9.27, 8.12, 8.12, 6.50, 6.50, 6.50, 6.50),
  ymax = c(10.05, 8.90, 8.90, 7.60, 7.60, 10.75 - 3.15, 7.60),
  fill = c("white", teal_fill, blue_fill, "white", "white", "white", "white"),
  colour = c(rule, dashboard_colors$opubliq_lightblue,
             dashboard_colors$blue, dashboard_colors$blue,
             dashboard_colors$green, dashboard_colors$opubliq_lightblue,
             dashboard_colors$blue),
  label = c(
    "108 199 répondants au total",
    "BRANCHE ENTRAÎNEMENT  ·  78 055 répondants\nMêmes 8 000 couples répondant–question pour C0 et C1",
    "TEST GELÉ  ·  30 144 répondants\nJamais vus au fine-tuning",
    "C0\nProfil SES + question cible\nProduit une réponse individuelle",
    "C1\nProfil SES + jusqu’à 6 réponses individuelles\nà des questions voisines + question cible\nProduit une réponse individuelle",
    "CONTEXTE\n14 947 répondants",
    "ÉVALUATION\n15 197 répondants"
  ),
  size = c(3.75, 3.28, 3.28, 3.18, 2.90, 3.10, 3.10),
  stringsAsFactors = FALSE
)

tree_lines <- data.frame(
  x = c(7.50, 7.50, 3.90, 3.90, 11.00, 11.00),
  y = c(9.27, 9.27, 8.12, 8.12, 8.12, 8.12),
  xend = c(3.90, 11.00, 2.05, 5.50, 9.25, 12.65),
  yend = c(8.90, 8.90, 7.60, 7.60, 7.60, 7.60)
)

# Panneau B : les cinq conditions LLM restent le cœur visuel.
conditions <- data.frame(
  condition = c("R", "A-C0", "B0-C1", "B-C1", "BS-C1"),
  model = c("base", "C0", "C1", "C1", "C1"),
  common = rep("Profil SES +\nquestion cible", 5),
  context = c(
    "Aucun",
    "Aucun",
    "Aucun",
    "Distributions voisines des 30 144\nrépondants gelés  ·  chevauchement",
    "Distributions voisines des 14 947\nrépondants de contexte  ·  répondants distincts"
  ),
  reference = rep("15 197 répondants\nd’évaluation", 5),
  colour = c(
    dashboard_colors$red,
    dashboard_colors$blue,
    dashboard_colors$yellow,
    dashboard_colors$green,
    green_dark
  ),
  y = seq(4.77, 2.45, length.out = 5),
  stringsAsFactors = FALSE
)

table_rows <- transform(
  conditions,
  ymin = y - 0.285,
  ymax = y + 0.285,
  fill = rep(c("white", "#F9FAFA"), length.out = 5)
)

# Référence secondaire : benchmark statistique, visuellement séparé des bras LLM.
benchmark <- data.frame(
  condition = "S",
  model = "Logit conditionnel\nrégularisé",
  common = "Profil SES + question cible\n+ options",
  context = "Aucun contexte",
  reference = "15 197 répondants\nd’évaluation",
  colour = muted,
  y = 1.84,
  ymin = 1.61,
  ymax = 2.07,
  fill = "#EEF1F2",
  stringsAsFactors = FALSE
)

headers <- data.frame(
  x = c(1.18, 2.55, 4.62, 8.70, 12.85),
  label = c(
    "CONDITION", "MODÈLE", "ENTRÉE COMMUNE",
    "CONTEXTE AJOUTÉ", "RÉFÉRENCE OBSERVÉE"
  )
)

flow_boxes <- data.frame(
  xmin = c(0.65, 5.55, 9.25),
  xmax = c(4.60, 8.30, 14.25),
  ymin = c(0.51, 0.51, 0.51),
  ymax = c(1.11, 1.11, 1.11),
  label = c(
    "Prédiction par condition",
    "Distribution\nprédite",
    "Comparaison à la distribution observée chez\nles 15 197 répondants d’évaluation"
  )
)

p <- ggplot() +
  # Fonds et titres des deux panneaux.
  annotate("rect", xmin = 0.25, xmax = 14.65, ymin = 6.18, ymax = 10.72,
           fill = panel_fill, colour = NA) +
  annotate("rect", xmin = 0.25, xmax = 14.65, ymin = 0.18, ymax = 5.95,
           fill = panel_fill, colour = NA) +
  annotate(
    "text", x = 0.60, y = 10.42, hjust = 0,
    label = "A   CONSTRUCTION DE L’EXPÉRIENCE",
    family = "nunito", fontface = "bold", size = 4.15, colour = ink
  ) +
  annotate(
    "text", x = 0.60, y = 5.65, hjust = 0,
    label = "B   CONDITIONS ET RÉFÉRENCES",
    family = "nunito", fontface = "bold", size = 4.15, colour = ink
  ) +
  # Arbre du panneau A.
  geom_segment(
    data = tree_lines,
    aes(x = x, y = y, xend = xend, yend = yend),
    linewidth = 0.62, colour = rule
  ) +
  geom_rect(
    data = tree_nodes,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
        fill = fill, colour = colour),
    linewidth = 0.75, show.legend = FALSE
  ) +
  scale_fill_identity() +
  scale_colour_identity() +
  geom_text(
    data = tree_nodes,
    aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2,
        label = label, size = size),
    family = "nunito", colour = ink, lineheight = 0.88,
    show.legend = FALSE
  ) +
  scale_size_identity() +
  annotate(
    "text", x = 11.00, y = 7.83,
    label = "SPLIT PAR RÉPONDANT DANS CHAQUE SONDAGE × CELLULE",
    family = "nunito", fontface = "bold", size = 2.45, colour = muted
  ) +
  # En-tête et lignes du tableau.
  annotate("rect", xmin = 0.60, xmax = 14.30, ymin = 5.10, ymax = 5.48,
           fill = header_fill, colour = NA) +
  geom_text(
    data = headers,
    aes(x = x, y = 5.29, label = label),
    family = "nunito", fontface = "bold", size = 2.45, colour = muted
  ) +
  geom_rect(
    data = table_rows,
    aes(xmin = 0.60, xmax = 14.30, ymin = ymin, ymax = ymax, fill = fill),
    colour = NA, show.legend = FALSE
  ) +
  geom_segment(
    data = conditions,
    aes(x = 0.72, xend = 0.72, y = y - 0.19, yend = y + 0.19,
        colour = colour),
    linewidth = 2.0, lineend = "round", show.legend = FALSE
  ) +
  geom_text(
    data = conditions,
    aes(x = 1.18, y = y, label = condition, colour = colour),
    family = "nunito", fontface = "bold", size = 3.20,
    show.legend = FALSE
  ) +
  geom_text(
    data = conditions,
    aes(x = 2.55, y = y, label = model),
    family = "nunito", fontface = "bold", size = 3.05, colour = ink
  ) +
  geom_text(
    data = conditions,
    aes(x = 4.62, y = y, label = common),
    family = "nunito", size = 2.82, lineheight = 0.88, colour = ink
  ) +
  geom_text(
    data = conditions,
    aes(x = 8.70, y = y, label = context),
    family = "nunito", size = 2.72, lineheight = 0.88, colour = ink
  ) +
  geom_text(
    data = conditions,
    aes(x = 12.85, y = y, label = reference),
    family = "nunito", size = 2.76, lineheight = 0.88, colour = ink
  ) +
  # Benchmark statistique, en retrait sous les cinq conditions LLM.
  geom_segment(
    aes(x = 0.60, xend = 14.30, y = 2.12, yend = 2.12),
    linewidth = 0.55, colour = rule
  ) +
  geom_rect(
    data = benchmark,
    aes(xmin = 0.60, xmax = 14.30, ymin = ymin, ymax = ymax),
    fill = benchmark$fill, colour = NA
  ) +
  geom_segment(
    data = benchmark,
    aes(x = 0.72, xend = 0.72, y = y - 0.15, yend = y + 0.15),
    linewidth = 1.5, lineend = "round", colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 1.18, y = y + 0.07, label = condition),
    family = "nunito", fontface = "bold", size = 3.05, colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 1.18, y = y - 0.11, label = "Benchmark statistique"),
    family = "nunito", size = 1.85, colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 2.55, y = y, label = model),
    family = "nunito", fontface = "bold", size = 2.55,
    lineheight = 0.88, colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 4.62, y = y, label = common),
    family = "nunito", size = 2.48, lineheight = 0.88, colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 8.70, y = y, label = context),
    family = "nunito", size = 2.48, colour = muted
  ) +
  geom_text(
    data = benchmark,
    aes(x = 12.85, y = y, label = reference),
    family = "nunito", size = 2.48, lineheight = 0.88, colour = muted
  ) +
  annotate(
    "text", x = 0.68, y = 1.43, hjust = 0,
    label = "S — Embeddings textuels figés; mêmes 8 000 exemples; aucune réponse cible.",
    family = "nunito", size = 2.20, colour = muted
  ) +
  # Repère humain–humain associé à l'évaluation, et non aux conditions.
  annotate(
    "rect", xmin = 9.25, xmax = 14.25, ymin = 1.16, ymax = 1.57,
    fill = "#EEF1F2", colour = rule, linewidth = 0.55
  ) +
  annotate(
    "text", x = 11.75, y = 1.385,
    label = paste0(
      "HUMAIN–HUMAIN  ·  REPÈRE DE BRUIT D’ÉCHANTILLONNAGE\n",
      "Question cible : 14 947 contexte ↔ 15 197 évaluation\n",
      "Non applicable à une question nouvelle"
    ),
    family = "nunito", fontface = "bold", size = 2.05,
    lineheight = 0.84, colour = muted
  ) +
  geom_segment(
    aes(x = 11.75, y = 1.16, xend = 11.75, yend = 1.11),
    linewidth = 0.50, linetype = "dashed", colour = muted,
    arrow = grid::arrow(length = grid::unit(0.05, "inches"), type = "closed")
  ) +
  # Un seul flux commun d'évaluation.
  geom_segment(
    aes(x = 4.60, y = 0.81, xend = 5.55, yend = 0.81),
    linewidth = 0.65, colour = muted,
    arrow = grid::arrow(length = grid::unit(0.07, "inches"), type = "closed")
  ) +
  geom_segment(
    aes(x = 8.30, y = 0.81, xend = 9.25, yend = 0.81),
    linewidth = 0.65, colour = muted,
    arrow = grid::arrow(length = grid::unit(0.07, "inches"), type = "closed")
  ) +
  geom_rect(
    data = flow_boxes,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "white", colour = rule, linewidth = 0.70
  ) +
  geom_text(
    data = flow_boxes,
    aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = label),
    family = "nunito", size = 2.95, lineheight = 0.88, colour = ink
  ) +
  annotate(
    "text", x = 2.625, y = 0.37,
    label = "Bras LLM : 100 tirages par question × cellule × température",
    family = "nunito", size = 2.15, colour = muted
  ) +
  annotate(
    "text", x = 7.45, y = 0.23,
    label = "Note — Les contextes contiennent seulement des questions voisines, jamais la question cible ni sa distribution.",
    family = "nunito", size = 2.45, colour = muted
  ) +
  coord_cartesian(xlim = c(0, 14.90), ylim = c(0, 10.85), clip = "off") +
  labs(title = "Protocole expérimental") +
  guides(colour = "none", fill = "none") +
  theme_void(base_size = 12, base_family = "nunito") +
  theme(
    text = element_text(colour = ink, lineheight = 0.3),
    plot.background = element_rect(fill = "white", colour = NA),
    plot.title = element_text(
      face = "bold", size = 23, hjust = 0, lineheight = 0.45,
      margin = margin(b = 10)
    ),
    plot.margin = margin(15, 15, 15, 15)
  )

ggsave(
  file.path(output_dir, "schema_protocole.png"),
  p,
  width = 14,
  height = 10,
  dpi = 300,
  bg = "white"
)

message("Figure written to ", file.path(output_dir, "schema_protocole.png"))
