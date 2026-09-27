#!/usr/bin/env Rscript

# Figures for the results note to Justin (pilot, 12 items, 5 arms + human reference).
# Usage:
#   Rscript scripts/24_plot_note_justin.R [input_dir] [output_dir]

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(showtext)
  library(stringr)
  library(sysfonts)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "data/analysis"
output_dir <- if (length(args) >= 2) args[[2]] else file.path(input_dir, "inference", "figures")
primary_temperature <- 1.0

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
  green_dark = "#287A69",
  red = "#f0695a",
  blue = "#0072B2",
  yellow = "#E69F00",
  grey = "#657278"
)

# Codes in the analysis files -> names used in the note.
arm_names <- c(
  R = "Non entraîné", A = "Entraîné", BS = "Entraîné + indices",
  B0 = "Indices retirés", B = "Fuite"
)
arm_order <- c("Non entraîné", "Entraîné", "Entraîné + indices", "Indices retirés", "Fuite")
arm_colors <- c(
  `Non entraîné` = dashboard_colors$red,
  Entraîné = dashboard_colors$blue,
  `Entraîné + indices` = dashboard_colors$green_dark,
  `Indices retirés` = dashboard_colors$yellow,
  Fuite = "grey70",
  Humain = "grey20"
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

read_input <- function(filename) {
  path <- file.path(input_dir, filename)
  if (!file.exists(path)) stop("Missing input file: ", path, call. = FALSE)
  readr::read_csv(path, show_col_types = FALSE)
}

name_arm <- function(code) factor(arm_names[code], levels = arm_order)
comma_number <- label_number(accuracy = 0.01, decimal.mark = ",", style_negative = "minus")

human <- read_input("human_reference_summary.csv") |>
  filter(scope == "all", arm == "human")

# 1. Distance by temperature, with the human floor as a band.
by_temperature <- read_input("annex_temperature.csv") |>
  filter(scope == "all", arm != "B") |>
  mutate(arm = name_arm(arm))

p_temperature <- ggplot(by_temperature, aes(temperature, mean_tv, color = arm, group = arm)) +
  annotate(
    "rect", xmin = -Inf, xmax = Inf,
    ymin = human$mean_tv_ci_low, ymax = human$mean_tv_ci_high,
    fill = "grey88"
  ) +
  geom_hline(yintercept = human$mean_tv, color = "grey35", linetype = "dashed", linewidth = 0.7) +
  annotate(
    "text", x = 1.3, y = human$mean_tv_ci_low - 0.012, hjust = 1, vjust = 1,
    label = "Humain : deux moitiés de vrais répondants", family = "nunito",
    size = 5.5, color = "grey35"
  ) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.8) +
  scale_color_manual(values = arm_colors, drop = TRUE) +
  scale_x_continuous(breaks = c(0.3, 0.7, 1.0, 1.3), labels = label_number(accuracy = 0.1, decimal.mark = ",")) +
  scale_y_continuous(
    labels = comma_number, limits = c(0, NA), expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Distance aux vrais répondants selon la température",
    subtitle = "Variation totale moyenne, 12 items, 275 cellules. Plus bas = plus fidèle.",
    x = "Température", y = "Variation totale\n", color = NULL
  ) +
  theme_dashboard_light()

# 2. Paired contrasts at T = 1.0.
contrast_labels <- c(
  "A - R" = "Entraîné − Non entraîné",
  "BS - A" = "Entraîné + indices − Entraîné",
  "BS - B0" = "Entraîné + indices − Indices retirés",
  "B - BS" = "Fuite − Entraîné + indices"
)
contrasts <- read_input("main_arm_contrasts.csv") |>
  filter(contrast %in% names(contrast_labels))
contrasts <- contrasts |>
  mutate(label = factor(contrast_labels[contrast], levels = rev(contrast_labels))) |>
  filter(!is.na(label))

p_contrasts <- ggplot(contrasts, aes(mean_tv_difference, label)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.7) +
  geom_errorbarh(aes(xmin = tv_ci_low, xmax = tv_ci_high), height = 0.16,
                 linewidth = 0.8, color = "grey35") +
  geom_point(size = 3.2, color = dashboard_colors$blue) +
  scale_x_continuous(labels = comma_number, expand = expansion(mult = c(0.08, 0.08))) +
  labs(
    title = "Comparaisons appariées, température 1,0",
    subtitle = "Différence de variation totale et IC 95 % (bootstrap apparié). Négatif = premier bras meilleur.",
    x = "Différence de variation totale", y = NULL
  ) +
  theme_dashboard_light()

# 3. Per item: human floor, base, best fine-tuned arm, context arm.
items <- read_input("test_blocks.csv") |>
  filter(pilot) |>
  select(item_idx, block, short_label)
item_metrics <- read_input("item_metrics.csv") |>
  filter(scope == "all", near(temperature, primary_temperature), arm %in% c("R", "A", "BS")) |>
  transmute(item_idx, arm = as.character(name_arm(arm)), mean_tv)
item_human <- read_input("human_reference_cells.csv") |>
  group_by(item_idx) |>
  summarise(mean_tv = mean(tv, na.rm = TRUE), .groups = "drop") |>
  mutate(arm = "Humain")
block_names <- c(
  identite_qc_federalisme = "Identité et fédéralisme",
  partis_vote = "Partis et vote",
  valeurs_sociales = "Valeurs sociales"
)
per_item <- bind_rows(item_metrics, item_human) |>
  left_join(items, by = "item_idx") |>
  mutate(
    label = str_wrap(short_label, 42),
    block = block_names[block],
    arm = factor(arm, levels = c("Humain", "Entraîné", "Entraîné + indices", "Non entraîné"))
  )
item_order <- per_item |>
  filter(arm == "Entraîné") |>
  arrange(mean_tv) |>
  pull(label)
per_item <- mutate(per_item, label = factor(label, levels = rev(item_order)))

p_items <- ggplot(per_item, aes(mean_tv, label, color = arm, shape = arm)) +
  geom_line(aes(group = label), color = "grey85", linewidth = 0.6) +
  geom_point(size = 3.4) +
  scale_color_manual(values = arm_colors) +
  scale_shape_manual(values = c(Humain = 4, Entraîné = 16, `Entraîné + indices` = 17, `Non entraîné` = 15)) +
  scale_x_continuous(labels = comma_number, limits = c(0, 0.8), breaks = seq(0, 0.8, 0.2)) +
  facet_wrap(~block, ncol = 1, scales = "free_y") +
  labs(
    title = "Résultats par item, température 1,0",
    subtitle = "Variation totale moyenne sur les cellules de chaque item.",
    x = "Variation totale", y = NULL, color = NULL, shape = NULL
  ) +
  theme_dashboard_light() +
  theme(
    strip.text = element_text(hjust = 0),
    axis.text.y = element_text(size = 15, lineheight = 0.8),
    panel.grid.major.x = element_line(color = "grey92")
  )

save_plot <- function(plot, filename, height) {
  ggsave(file.path(output_dir, filename), plot, width = 12, height = height,
         dpi = 300, bg = "white")
}
save_plot(p_temperature, "tv_temperature.png", 6.4)
save_plot(p_contrasts, "contrastes_tv.png", 5.2)
save_plot(p_items, "items_tv.png", 11)

message("Figures written to ", output_dir)
