#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[[1]] else "data/raw/ces_2021"

if (!requireNamespace("ces", quietly = TRUE)) {
  stop("Package 'ces' is required. Install with: install.packages('ces')")
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

message("Downloading CES 2021 codebook...")
invisible(capture.output(ces::download_pdf_codebook(
  year = 2021,
  path = output_dir,
  overwrite = TRUE,
  verbose = FALSE
)))

message("Downloading CES 2021 dataset...")
invisible(capture.output(ces::download_ces_dataset(
  year = 2021,
  path = output_dir,
  overwrite = TRUE,
  verbose = FALSE
)))

normalize_name <- function(old_name, new_name) {
  old_path <- file.path(output_dir, old_name)
  new_path <- file.path(output_dir, new_name)
  if (file.exists(old_path) && !file.exists(new_path)) {
    ok <- file.rename(old_path, new_path)
    if (!ok) {
      warning("Could not rename ", old_name, " to ", new_name)
    }
  }
}

normalize_name("CES_2021_codebook.pdf", "ces_2021_codebook.pdf")
normalize_name("CES_2021.rds", "ces_2021.rds")
normalize_name("CES_2021.dta", "ces_2021.dta")
normalize_name("CES_2021.sav", "ces_2021.sav")

message("Done. Files available in: ", output_dir)
