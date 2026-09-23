"""Microdata access — one Parquet per survey in the Azure Blob container.

Contract of the Parquet files: see
``../mvp_moteur_recherche_sondages/docs/DECISION_microdata_parquet.md``.
One row = one respondent, columns = RAW variable names, plus ``__respondent_id``,
``__survey_id`` and ``__weight``.

Files are downloaded once into ``data/cache/`` (git-ignored) and queried locally
with DuckDB. No microdata is duplicated anywhere else in this repo.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MANIFEST_BLOB = "_manifest.json"


def _env() -> tuple[str, str, str]:
    """Return (account, key, container). Never log or print these."""
    load_dotenv(REPO_ROOT / ".env")
    try:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        key = os.environ["AZURE_STORAGE_KEY"]
    except KeyError as exc:  # pragma: no cover - config error
        raise RuntimeError(f"Missing {exc.args[0]} in .env") from exc
    container = os.environ.get("AZURE_STORAGE_CONTAINER", "survey-responses")
    return account, key, container


@lru_cache(maxsize=1)
def _container_client():
    account, key, container = _env()
    service = BlobServiceClient(
        f"https://{account}.blob.core.windows.net", credential=key
    )
    return service.get_container_client(container)


@lru_cache(maxsize=1)
def manifest() -> dict[str, dict[str, Any]]:
    """Survey-level microdata metadata, keyed by ``survey_id``.

    Fields per survey: ``n_respondents``, ``n_vars``, ``weight_var``,
    ``weight_source``, ``respondent_id_var``, ``updated_at``.
    """
    blob = _container_client().get_blob_client(MANIFEST_BLOB)
    payload = json.loads(blob.download_blob().readall())
    return {s["survey_id"]: s for s in payload["surveys"]}


def available_surveys() -> list[str]:
    """Survey ids that have microdata in the Blob, per the manifest."""
    return sorted(manifest())


def parquet_path(survey_id: str, refresh: bool = False) -> Path:
    """Local path of the survey Parquet, downloading it to the cache if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{survey_id}.parquet"
    if refresh or not path.exists():
        blob = _container_client().get_blob_client(f"{survey_id}.parquet")
        tmp = path.with_suffix(".parquet.part")
        with tmp.open("wb") as fh:
            blob.download_blob().readinto(fh)
        tmp.replace(path)
    return path


def read_survey(survey_id: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one survey's microdata as a Polars DataFrame."""
    return pl.read_parquet(parquet_path(survey_id), columns=columns)


def survey_columns(survey_id: str) -> list[str]:
    """Column names of a survey Parquet, without reading the data."""
    return list(pl.read_parquet_schema(parquet_path(survey_id)))


def connect(survey_ids: list[str] | None = None) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with one view per survey, named after ``survey_id``.

    Views point at the locally cached Parquet files (downloaded on demand), so
    queries are plain SQL over stable table names::

        con = connect(["eeq_2014"])
        con.execute('SELECT "QSEXE", SUM("__weight") FROM eeq_2014 GROUP BY 1')
    """
    con = duckdb.connect()
    for survey_id in survey_ids if survey_ids is not None else available_surveys():
        path = parquet_path(survey_id)
        literal = str(path).replace("'", "''")
        con.execute(
            f'CREATE OR REPLACE VIEW "{survey_id}" AS '
            f"SELECT * FROM read_parquet('{literal}')"
        )
    return con
