import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import polars as pl
from deltalake import DeltaTable, write_deltalake

from src.common.config import COLUMNS_TO_KEEP
from src.common.utils import safe_run

_LOGGER = logging.getLogger(__name__)

DeltaWriteMode = Literal["append", "overwrite"]

BTS_COLUMN_MAP = {
    "FlightDate": "FL_DATE",
    "IATA_Code_Operating_Airline": "OP_UNIQUE_CARRIER",
    "Operating_Airline": "OP_UNIQUE_CARRIER",
    "IATA_Code_Marketing_Airline": "OP_UNIQUE_CARRIER",
    "Flight_Number_Operating_Airline": "OP_CARRIER_FL_NUM",
    "Flight_Number_Marketing_Airline": "OP_CARRIER_FL_NUM",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "Distance": "DISTANCE",
}


class BronzeLayer:

    def __init__(self, bronze_path: Path):
        self._bronze_path = bronze_path
        self._bronze_path.mkdir(parents=True, exist_ok=True)

        _LOGGER.info(f"Initialized Bronze layer at {self._bronze_path}")

    def load_csv(self, csv_path: Path, year: int) -> pl.DataFrame:
        csv_path = Path(csv_path)
        _LOGGER.info(f"Loading CSV from {csv_path} (year {year})")
        try:
            df = pl.read_csv(csv_path, infer_schema_length=1000)
            df = self.normalize_source_schema(df)
            df = self.filter_year(df, year)
            _LOGGER.info(f"Loaded {len(df)} rows from {csv_path}")
            return df
        except Exception as e:
            _LOGGER.error(f"Error loading CSV: {e}")
            raise

    def normalize_source_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        source_by_target: dict[str, str] = {}
        for target in COLUMNS_TO_KEEP:
            if target in df.columns:
                source_by_target[target] = target

        for source, target in BTS_COLUMN_MAP.items():
            if source in df.columns and target not in source_by_target:
                source_by_target[target] = source

        expressions = []
        for column in ["Year", "year"]:
            if column in df.columns:
                expressions.append(pl.col(column))

        expressions.extend(
            pl.col(source).alias(target) for target, source in source_by_target.items()
        )
        return df.select(expressions)

    def filter_year(self, df: pl.DataFrame, year: int) -> pl.DataFrame:
        if "Year" in df.columns:
            return df.filter(pl.col("Year") == year).drop("Year")
        if "year" in df.columns:
            return df.filter(pl.col("year") == year).drop("year")
        if "FL_DATE" in df.columns:
            return df.filter(pl.col("FL_DATE").str.starts_with(str(year)))
        return df

    def validate_schema(self, df: pl.DataFrame) -> bool:
        missing_cols = set(COLUMNS_TO_KEEP) - set(df.columns)
        if missing_cols:
            _LOGGER.warning(f"Missing columns: {missing_cols}")
            return False
        return True

    def add_metadata(self, df: pl.DataFrame, year: int) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.lit(year).alias("year"),
                pl.lit(datetime.now().isoformat()).alias("loaded_at"),
                pl.lit("csv").alias("source_format"),
            ]
        )

    @safe_run(default_return_value=None)
    def write_to_delta(
        self, df: pl.DataFrame, year: int, mode: DeltaWriteMode = "append"
    ) -> None:
        _LOGGER.info(
            f"Writing {len(df)} rows to Delta Lake ({self._bronze_path}) in {mode} mode"
        )
        pdf = df.to_pandas()
        if mode == "append":
            write_deltalake(
                table_or_uri=self._bronze_path,
                data=pdf,
                mode="append",
                engine="rust",
                schema_mode="merge",
            )
        else:
            write_deltalake(
                table_or_uri=self._bronze_path,
                data=pdf,
                mode="overwrite",
                engine="rust",
                schema_mode="overwrite",
            )
        _LOGGER.info(f"Successfully wrote year {year} data to Bronze")

    def ingest_year(
        self, csv_path: Path, year: int, mode: DeltaWriteMode = "append"
    ) -> None:
        _LOGGER.info(f"Starting ingestion for year {year}")
        df = self.load_csv(csv_path, year)
        if not self.validate_schema(df):
            _LOGGER.warning(
                f"Schema validation failed for year {year}, proceeding with available columns"
            )
        existing_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
        df = df.select(existing_cols)
        df = self.add_metadata(df, year)
        df = df.with_columns(pl.lit(csv_path.name).alias("source_file"))
        self.write_to_delta(df, year, mode=mode)
        _LOGGER.info(f"Completed ingestion for year {year}")

    def ingest_batches(self, batches: Iterable[tuple[Path, int]]) -> None:
        for csv_path, year in batches:
            self.ingest_year(csv_path=csv_path, year=year, mode="append")

    @safe_run(default_return_value={})
    def get_table_info(self) -> dict:
        dt = DeltaTable(str(self._bronze_path))
        version = dt.version()
        files = dt.files()
        _LOGGER.info(f"Bronze table version: {version}, files: {len(files)}")
        return {"version": version, "num_files": len(files)}
