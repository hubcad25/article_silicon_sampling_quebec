"""Thematic blocks over the 60 frozen test items (analysis grouping only; split untouched).
Assigned from display labels before any result. Output: data/analysis/test_blocks.csv"""
import polars as pl

BLOCKS = {
    "sante": [0, 9, 12, 13, 26, 55],
    "etat_economie": [10, 11, 27, 32, 38, 58],
    "economie_percue": [14, 17, 28, 49, 50],
    "identite_qc_federalisme": [1, 20, 42, 43, 45, 52, 53, 54, 59],
    "valeurs_sociales": [4, 15, 16, 18, 21, 23, 25, 29, 34, 41, 56, 57],
    "democratie_engagement": [2, 3, 5, 19, 22, 36, 37, 39, 51],
    "partis_vote": [6, 7, 8, 24, 30, 31, 33, 35, 40, 44, 46, 47, 48],
}
# Pilot subset (ADR 0001): 4 items per pilot block, hand-picked, spread over distance bins
PILOT = [1, 20, 43, 53,       # identite_qc_federalisme
         4, 21, 25, 56,       # valeurs_sociales
         7, 33, 40, 48]       # partis_vote
d = pl.read_csv("data/split/split_diagnostics.csv").filter(pl.col("kind") == "test_item")
d = d.with_row_index("item_idx")
m = {i: b for b, ix in BLOCKS.items() for i in ix}
assert sorted(m) == list(range(60)), "every item in exactly one block"
d = d.with_columns(pl.col("item_idx").replace_strict(m).alias("block"))
d = d.with_columns(pl.col("item_idx").is_in(PILOT).alias("pilot"))
d = d.select("item_idx", "block", "pilot", "survey_id", "variable", "language", "distance_bin",
             pl.col("n_heldout_cells_ge30").alias("cells_ge30"),
             pl.col("n_heldout_cells_ge100").alias("cells_ge100"),
             pl.col("label").str.split(" — ").list.first().alias("short_label"))
d.write_csv("data/analysis/test_blocks.csv")
print(d.group_by("block").agg(
    pl.len().alias("items"), (pl.col("language") == "fr").sum().alias("fr"),
    (pl.col("language") == "en").sum().alias("en"),
    pl.col("cells_ge30").sum().alias("pairs_ge30"), pl.col("cells_ge100").sum().alias("pairs_ge100"),
    pl.col("distance_bin").n_unique().alias("bins")).sort("items", descending=True))
p = d.filter(pl.col("pilot"))
print(p.select("block", "survey_id", "variable", "language", "distance_bin", "cells_ge30", "cells_ge100", "short_label"))
print("pilot pairs >=30:", p["cells_ge30"].sum(), "| >=100:", p["cells_ge100"].sum())
