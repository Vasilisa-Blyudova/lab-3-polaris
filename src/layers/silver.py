import logging
from pathlib import Path
from typing import List, Literal, Optional

import polars as pl
from deltalake import DeltaTable, write_deltalake

from src.common.config import DELAY_THRESHOLD, NA_THRESHOLDS, PARTITION_COLUMNS
from src.common.utils import safe_run

_LOGGER = logging.getLogger(__name__)

DeltaWriteMode = Literal["append", "overwrite"]


class SilverLayer:

    def __init__(self, bronze_path: Path, silver_path: Path):
        self._bronze_path = bronze_path
        self._silver_path = silver_path

    @safe_run(default_return_value=None, raise_anyway=True)
    def read_bronze(self, year: Optional[int] = None) -> pl.DataFrame:
        df = pl.scan_delta(str(self._bronze_path))
        if year:
            df = df.filter(pl.col("year") == year)
        _LOGGER.info("Read Bronze data successfully")
        return df.collect()

    def remove_cancelled_diverted(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        initial_rows = len(df)
        df = df.filter((pl.col("CANCELLED") == 0) & (pl.col("DIVERTED") == 0))
        removed = initial_rows - len(df)
        _LOGGER.info(f"Removed {removed} cancelled/diverted flights")
        return df

    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        initial_rows = len(df)
        df = df.filter(pl.col("ARR_DELAY").is_not_null())
        removed = initial_rows - len(df)
        _LOGGER.info(f"Removed {removed} rows with null ARR_DELAY")
        for col, threshold in NA_THRESHOLDS.items():
            if col in df.columns:
                null_pct = df.select(
                    (pl.col(col).is_null().sum() / len(df)).alias("null_pct")
                ).to_dicts()[0]["null_pct"]
                if null_pct > threshold:
                    _LOGGER.warning(
                        f"Column {col} has {null_pct:.2%} nulls, exceeds threshold {threshold:.2%}"
                    )
        return df

    def remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        initial_rows = len(df)
        q1 = df.select(pl.col("ARR_DELAY").quantile(0.25))[0, 0]
        q3 = df.select(pl.col("ARR_DELAY").quantile(0.75))[0, 0]
        if q1 is None or q3 is None:
            return df
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        df = df.filter(
            (pl.col("ARR_DELAY") >= lower_bound) & (pl.col("ARR_DELAY") <= upper_bound)
        )
        removed = initial_rows - len(df)
        _LOGGER.info(f"Removed {removed} outlier rows (IQR-based)")
        return df

    def normalize_categories(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        string_cols = []
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                string_cols.append(col)
        for col in string_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).str.strip().str.to_uppercase().alias(col)
                )
        _LOGGER.info(f"Normalized {len(string_cols)} categorical columns")
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        df = df.with_columns(
            pl.col("FL_DATE")
            .str.to_datetime("%Y-%m-%d", time_unit="us")
            .alias("flight_date")
        )
        df = df.with_columns(
            [
                (pl.col("CRS_DEP_TIME").cast(pl.Int64) // 100)
                .clip(0, 23)
                .alias("hour"),
                pl.col("flight_date").dt.weekday().alias("day_of_week"),
                pl.col("flight_date").dt.month().alias("month"),
            ]
        )
        df = df.with_columns(
            pl.when(pl.col("month").is_in([12, 1, 2]))
            .then(pl.lit("winter"))
            .when(pl.col("month").is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("month").is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .otherwise(pl.lit("autumn"))
            .alias("season")
        )
        df = df.with_columns((pl.col("ORIGIN") + "_" + pl.col("DEST")).alias("route"))
        df = df.with_columns(
            (pl.col("ARR_DELAY") > DELAY_THRESHOLD).cast(pl.Int32).alias("is_delayed")
        )
        df = df.with_columns(
            pl.concat_str(
                [
                    pl.col("year").cast(pl.Utf8),
                    pl.col("FL_DATE"),
                    pl.col("OP_UNIQUE_CARRIER"),
                    pl.col("OP_CARRIER_FL_NUM").cast(pl.Utf8),
                    pl.col("ORIGIN"),
                    pl.col("DEST"),
                    pl.col("CRS_DEP_TIME").cast(pl.Utf8),
                    pl.col("CRS_ARR_TIME").cast(pl.Utf8),
                ],
                separator="|",
            ).alias("flight_id")
        )
        _LOGGER.info("Created derived features")
        return df

    def select_final_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        final_cols = [
            "flight_date",
            "year",
            "month",
            "hour",
            "day_of_week",
            "season",
            "route",
            "flight_id",
            "ORIGIN",
            "DEST",
            "OP_UNIQUE_CARRIER",
            "OP_CARRIER_FL_NUM",
            "DISTANCE",
            "CRS_DEP_TIME",
            "DEP_TIME",
            "DEP_DELAY",
            "CRS_ARR_TIME",
            "ARR_TIME",
            "ARR_DELAY",
            "is_delayed",
            "loaded_at",
        ]
        available_cols = [c for c in final_cols if c in df.columns]
        df = df.select(available_cols)
        _LOGGER.info(f"Selected {len(available_cols)} final columns")
        return df

    @safe_run(default_return_value=None, raise_anyway=True)
    def merge_update(self, df: pl.DataFrame, year: int) -> None:
        _LOGGER.info(f"Starting Silver MERGE for year {year}")
        if df.is_empty():
            _LOGGER.warning("No Silver rows to merge for year %s", year)
            return
        df = df.unique(subset=["flight_id"], keep="last")
        _LOGGER.info("Source rows after flight_id deduplication: %s", len(df))
        if not self._silver_path.exists() or not any(self._silver_path.iterdir()):
            _LOGGER.info(
                "Silver table is not present; writing new data instead of merging"
            )
            self.write_to_delta(df, mode="append")
            return

        dt = DeltaTable(str(self._silver_path))
        target_columns = {field.name for field in dt.schema().fields}
        if "flight_id" not in target_columns:
            _LOGGER.warning(
                "Silver table has old schema without flight_id; rewriting once before future MERGE runs"
            )
            self.write_to_delta(df, mode="overwrite", partition_by=PARTITION_COLUMNS)
            return

        existing = pl.scan_delta(str(self._silver_path)).collect()
        if len(existing) != existing.select(pl.col("flight_id").n_unique())[0, 0]:
            _LOGGER.warning(
                "Silver table contains duplicate flight_id values; rewriting deduplicated table once"
            )
            merged = pl.concat([existing, df], how="diagonal").unique(
                subset=["flight_id"],
                keep="last",
            )
            self.write_to_delta(
                merged, mode="overwrite", partition_by=PARTITION_COLUMNS
            )
            return

        result = (
            dt.merge(
                df.to_arrow(),
                predicate="target.flight_id = source.flight_id",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        _LOGGER.info(f"Completed Silver MERGE for year {year}: {result}")

    def write_to_delta(
        self,
        df: pl.DataFrame,
        mode: DeltaWriteMode = "append",
        partition_by: Optional[List[str]] = None,
    ) -> None:
        if partition_by is None:
            partition_by = PARTITION_COLUMNS
        try:
            _LOGGER.info(
                f"Writing {len(df)} rows to Silver (partition_by={partition_by}, mode={mode})"
            )
            pdf = df.to_pandas()
            write_deltalake(
                table_or_uri=self._silver_path,
                data=pdf,
                mode=mode,
                partition_by=partition_by,
                engine="rust",
                schema_mode="merge" if mode == "append" else "overwrite",
            )
            _LOGGER.info("Successfully wrote to Silver")
        except Exception as e:
            _LOGGER.error(f"Error writing to Delta: {e}")
            raise

    def transform(self, year: Optional[int] = None, use_merge: bool = True) -> None:
        _LOGGER.info(f"Starting Silver transformation (year={year})")
        df = self.read_bronze(year=year)
        if df.is_empty():
            _LOGGER.warning(
                "No Bronze rows found for year %s; skipping Silver transformation", year
            )
            return
        df = self.remove_cancelled_diverted(df)
        df = self.handle_missing_values(df)
        if df.is_empty():
            _LOGGER.warning("No usable rows left for year %s after cleaning", year)
            return
        df = self.remove_outliers(df)
        df = self.normalize_categories(df)
        df = self.engineer_features(df)
        df = self.select_final_columns(df)
        if use_merge and year:
            self.merge_update(df, year)
        else:
            self.write_to_delta(df, mode="append")
        _LOGGER.info(f"Completed Silver transformation for year {year}")
