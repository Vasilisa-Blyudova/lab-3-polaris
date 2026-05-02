import logging
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from deltalake import DeltaTable, write_deltalake

from src.common.utils import safe_run

_LOGGER = logging.getLogger(__name__)

DeltaWriteMode = Literal["append", "overwrite"]


class GoldLayer:
    def __init__(
        self,
        silver_path: Path,
        analytics_paths: dict[str, Path],
        features_path: Path,
    ):
        self._silver_path = silver_path
        self._analytics_paths = analytics_paths
        self._features_path = features_path
        for path in self._analytics_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._features_path.parent.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Initialized Gold layer")

    @safe_run(default_return_value=None, raise_anyway=True)
    def read_silver(self, lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
        df = pl.scan_delta(str(self._silver_path))
        _LOGGER.info("Read Silver data successfully")
        return df if lazy else df.collect()

    def create_analytics_by_airport(self) -> pl.DataFrame:
        _LOGGER.info("Creating analytics: delays by airport")
        df = (
            pl.scan_delta(str(self._silver_path))
            .filter(pl.col("ARR_DELAY").is_not_null())
            .group_by("ORIGIN")
            .agg(
                [
                    pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
                    pl.col("ARR_DELAY").median().alias("median_arr_delay"),
                    pl.col("ARR_DELAY").std().alias("std_arr_delay"),
                    pl.col("is_delayed").mean().alias("delay_rate"),
                    pl.len().alias("num_flights"),
                ]
            )
            .sort("avg_arr_delay", descending=True)
            .collect()
        )
        _LOGGER.info(f"Created analytics for {len(df)} airports")
        return df

    def create_analytics_by_carrier(self) -> pl.DataFrame:
        _LOGGER.info("Creating analytics: delays by carrier")
        df = (
            pl.scan_delta(str(self._silver_path))
            .filter(pl.col("ARR_DELAY").is_not_null())
            .group_by("OP_UNIQUE_CARRIER")
            .agg(
                [
                    pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
                    pl.col("ARR_DELAY").median().alias("median_arr_delay"),
                    pl.col("ARR_DELAY").std().alias("std_arr_delay"),
                    pl.col("is_delayed").mean().alias("delay_rate"),
                    pl.len().alias("num_flights"),
                ]
            )
            .sort("avg_arr_delay", descending=True)
            .collect()
        )
        _LOGGER.info(f"Created analytics for {len(df)} carriers")
        return df

    def create_analytics_by_hour(self) -> pl.DataFrame:
        _LOGGER.info("Creating analytics: delays by hour")
        df = (
            pl.scan_delta(str(self._silver_path))
            .filter(pl.col("ARR_DELAY").is_not_null())
            .group_by("hour")
            .agg(
                [
                    pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
                    pl.col("ARR_DELAY").median().alias("median_arr_delay"),
                    pl.col("ARR_DELAY").std().alias("std_arr_delay"),
                    pl.col("is_delayed").mean().alias("delay_rate"),
                    pl.len().alias("num_flights"),
                ]
            )
            .sort("hour")
            .collect()
        )
        _LOGGER.info(f"Created analytics for {len(df)} hours")
        return df

    def create_analytics_by_season(self) -> pl.DataFrame:
        _LOGGER.info("Creating analytics: delays by season")
        df = (
            pl.scan_delta(str(self._silver_path))
            .filter(pl.col("ARR_DELAY").is_not_null())
            .group_by("season")
            .agg(
                [
                    pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
                    pl.col("ARR_DELAY").median().alias("median_arr_delay"),
                    pl.col("ARR_DELAY").std().alias("std_arr_delay"),
                    pl.col("is_delayed").mean().alias("delay_rate"),
                    pl.len().alias("num_flights"),
                ]
            )
            .collect()
        )
        _LOGGER.info(f"Created analytics for {len(df)} seasons")
        return df

    def create_ml_features(
        self, sample_fraction: Optional[float] = None
    ) -> pl.DataFrame:
        _LOGGER.info("Creating ML feature table")
        df = pl.scan_delta(str(self._silver_path)).filter(
            pl.col("ARR_DELAY").is_not_null()
        )
        columns = [
            "flight_id",
            "flight_date",
            "hour",
            "day_of_week",
            "month",
            "season",
            "DISTANCE",
            "DEP_DELAY",
            "OP_UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
            "ARR_DELAY",
            "is_delayed",
        ]
        available = set(pl.scan_delta(str(self._silver_path)).schema.keys())
        df = df.select([pl.col(column) for column in columns if column in available])
        collected = df.collect()
        if sample_fraction:
            collected = collected.sample(fraction=sample_fraction)
        _LOGGER.info(
            f"Created ML features: {len(collected)} rows, {len(collected.columns)} columns"
        )
        return collected

    def create_analytics(self) -> dict[str, pl.DataFrame]:
        return {
            "by_airport": self.create_analytics_by_airport(),
            "by_carrier": self.create_analytics_by_carrier(),
            "by_hour": self.create_analytics_by_hour(),
            "by_season": self.create_analytics_by_season(),
        }

    def write_delta(
        self, df: pl.DataFrame, path: Path, mode: DeltaWriteMode = "overwrite"
    ) -> None:
        pdf = df.to_pandas()
        write_deltalake(
            table_or_uri=path,
            data=pdf,
            mode=mode,
            engine="rust",
            schema_mode="overwrite" if mode == "overwrite" else "merge",
        )

    @safe_run(default_return_value=None, raise_anyway=True)
    def write_analytics(
        self,
        analytics: dict[str, pl.DataFrame],
        mode: DeltaWriteMode = "overwrite",
    ) -> None:
        for name, df in analytics.items():
            path = self._analytics_paths[name]
            _LOGGER.info(f"Writing analytics {name} to {path}")
            self.write_delta(df, path, mode=mode)
        _LOGGER.info("Successfully wrote analytics marts")

    @safe_run(default_return_value=None, raise_anyway=True)
    def write_features(
        self, df: pl.DataFrame, mode: DeltaWriteMode = "overwrite"
    ) -> None:
        _LOGGER.info(f"Writing ML features to {self._features_path}")
        self.write_delta(df, self._features_path, mode=mode)
        _LOGGER.info("Successfully wrote ML features")

    def features_version(self) -> int:
        return DeltaTable(str(self._features_path)).version()

    def build_gold(self) -> None:
        _LOGGER.info("Starting Gold layer build")
        analytics = self.create_analytics()
        self.write_analytics(analytics)
        features = self.create_ml_features()
        self.write_features(features)
        _LOGGER.info("Completed Gold layer build")
