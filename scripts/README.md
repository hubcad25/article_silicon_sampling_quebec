# Scripts

Project scripts (download, cleaning, orchestration).

## Download CES 2021

Script: `scripts/download_ces_2021.R`

Install R dependency once:

```r
install.packages("ces")
```

Run from project root:

```bash
Rscript scripts/download_ces_2021.R
```

Optional custom output directory:

```bash
Rscript scripts/download_ces_2021.R data/raw/ces_2021
```
