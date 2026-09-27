#!/usr/bin/env Rscript

# Figures for the short methodological note to Justin.
# Usage:
#   Rscript scripts/24_plot_note_justin.R [input_dir] [output_dir] [temperature]

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(showtext)
  library(stringr)
  library(sysfonts)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "data/analysis/fake"
output_dir <- if (length(args) >= 2) args[[2]] else file.path(input_dir, "figures")
selected_temperature <- if (length(args) >= 3) as.numeric(args[[3]]) else 1.0

if (is.na(selected_temperature)) {
  stop("The temperature argument must be numeric.", call. = FALSE)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Use a local Nunito Sans installation when available; otherwise sysfonts
# downloads and registers the Google Fonts version.
local_nunito <- sysfonts::font_files() |>
  filter(str_detect(family, regex("^Nunito Sans$", ignore_case = TRUE)))

if (nrow(local_nunito) > 0) {
  regular_font <- local_nunito |>
    filter(str_detect(style, regex("regular", ignore_case = TRUE))) |>
    slice_head(n = 1)
  bold_font <- local_nunito |>
    filter(str_detect(style, regex("bold", ignore_case = TRUE))) |>
    slice_head(n = 1)

  if (nrow(regular_font) == 0) regular_font <- slice_head(local_nunito, n = 1)
  if (nrow(bold_font) == 0) bold_font <- regular_font

  sysfonts::font_add(
    family = "nunito",
    regular = file.path(regular_font$path, regular_font$file),
    bold = file.path(bold_font$path, bold_font$file)
  )
} else {
  sysfonts::font_add_google("Nunito Sans", "nunito")
}
showtext::showtext_auto()
# Match showtext's rasterization to ggsave(). Without this, glyphs are rendered
# at showtext's low default DPI and become much too small in the PDF.
showtext::showtext_opts(dpi = 300)

dashboard_colors <- list(
  green = "#00A087",
  red = "#f0695a",
  blue = "#0072B2",
  yellow = "#E69F00"
)

arm_order <- c("A", "B", "B0", "R")
arm_labels <- c(
  A = "A — méthode simple (fine-tunée)",
  B = "B — stratégie enrichie (avec contexte)",
  B0 = "B0 — stratégie enrichie sans contexte",
  R = "R — modèle de base"
)
arm_colors <- c(
  A = dashboard_colors$blue,
  B = dashboard_colors$green,
  B0 = dashboard_colors$yellow,
  R = dashboard_colors$red
)

theme_dashboard_light <- function(base_size = 20, base_family = "nunito") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      text = element_text(color = "grey20", lineheight = 0.3),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      axis.title = element_text(size = 20, lineheight = 0.3),
      axis.text = element_text(size = 19, color = "grey20", lineheight = 0.3),
      plot.title = element_text(face = "bold", hjust = 0, size = 26, lineheight = 0.45),
      plot.subtitle = element_text(hjust = 0, size = 20, lineheight = 0.8),
      plot.caption = element_text(hjust = 1, size = 18, lineheight = 0.75),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.text = element_text(size = 19, lineheight = 0.45),
      strip.text = element_text(face = "bold", size = 20, lineheight = 0.45),
      panel.spacing = grid::unit(1.5, "lines"),
      plot.title.position = "plot",
      plot.margin = margin(15, 15, 15, 15)
    )
}

read_required_csv <- function(filename, required_columns) {
  path <- file.path(input_dir, filename)
  if (!file.exists(path)) stop("Missing input file: ", path, call. = FALSE)
  data <- readr::read_csv(path, show_col_types = FALSE)
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0) {
    stop(
      filename, " is missing required columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  data
}

metric_summary <- read_required_csv(
  "metric_summary.csv",
  c("arm", "temperature", "mean_kl")
) |>
  filter(arm %in% arm_order) |>
  mutate(arm = factor(arm, levels = arm_order))

arm_contrasts <- read_required_csv(
  "arm_contrasts.csv",
  c("contrast", "temperature", "mean_kl_difference", "ci_low", "ci_high")
) |>
  filter(dplyr::near(temperature, selected_temperature))

if (nrow(metric_summary) == 0) {
  stop("No recognized arms in metric_summary.csv.", call. = FALSE)
}
if (nrow(arm_contrasts) == 0) {
  stop("No contrasts found at temperature ", selected_temperature, ".", call. = FALSE)
}

temperature_breaks <- sort(unique(metric_summary$temperature))

p_kl <- ggplot(
  metric_summary,
  aes(x = temperature, y = mean_kl, color = arm, group = arm)
) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.8) +
  scale_color_manual(values = arm_colors, labels = arm_labels, drop = FALSE) +
  scale_x_continuous(
    breaks = temperature_breaks,
    expand = expansion(mult = c(0.03, 0.03))
  ) +
  scale_y_continuous(
    labels = label_number(accuracy = 0.01, decimal.mark = ","),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Quelle méthode reproduit le mieux les distributions observées?",
    subtitle = str_wrap(
      "Plus une courbe est basse, plus la distribution produite est fidèle aux réponses observées.",
      width = 95
    ),
    x = "Température",
    y = "KL moyenne\n",
    color = NULL,
    caption = str_wrap(
      "Résultats simulés — répétition générale du pipeline; aucune performance réelle de modèle.",
      width = 70
    )
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  theme_dashboard_light()

contrast_order <- c("B - A", "B - B0", "A - R")
contrast_labels <- c(
  "B - A" = "B − A : stratégie enrichie vs simple",
  "B - B0" = "B − B0 : apport du contexte",
  "A - R" = "A − R : apport du fine-tuning"
)
arm_contrasts <- arm_contrasts |>
  mutate(contrast = factor(contrast, levels = rev(contrast_order)))

p_contrasts <- ggplot(
  arm_contrasts,
  aes(x = mean_kl_difference, y = contrast)
) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.7) +
  geom_errorbarh(
    aes(xmin = ci_low, xmax = ci_high),
    height = 0.16,
    linewidth = 0.8,
    color = "grey35"
  ) +
  geom_point(size = 3.2, color = dashboard_colors$blue) +
  annotate(
    "text",
    x = -Inf,
    y = Inf,
    label = "À gauche : premier bras meilleur",
    hjust = 0,
    vjust = 1,
    size = 5.2,
    family = "nunito",
    color = "grey35"
  ) +
  annotate(
    "text",
    x = Inf,
    y = Inf,
    label = "À droite : second bras meilleur",
    hjust = 1,
    vjust = 1,
    size = 5.2,
    family = "nunito",
    color = "grey35"
  ) +
  scale_x_continuous(
    labels = label_number(accuracy = 0.01, decimal.mark = ",", style_negative = "minus"),
    expand = expansion(mult = c(0.08, 0.08))
  ) +
  scale_y_discrete(
    labels = contrast_labels,
    expand = expansion(add = c(0.5, 1.15))
  ) +
  labs(
    title = "Les trois comparaisons qui guident la décision",
    subtitle = paste0(
      "À température ",
      format(selected_temperature, nsmall = 1, decimal.mark = ","),
      " : différence moyenne de KL et intervalle à 95 %.\n",
      "Pour B − A, un intervalle entièrement sous zéro favorise B."
    ),
    x = "Différence de KL (premier bras − second bras)",
    y = "",
    caption = str_wrap(
      "Résultats simulés — répétition générale du pipeline; aucune performance réelle de modèle.",
      width = 70
    )
  ) +
  theme_dashboard_light()

ggsave(
  file.path(output_dir, "kl_temperature.png"),
  p_kl,
  width = 12,
  height = 6.2,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "contrastes_kl.png"),
  p_contrasts,
  width = 12,
  height = 5.6,
  dpi = 300,
  bg = "white"
)

message("Figures written to ", output_dir)
