import argparse
import logging
from pathlib import Path

import polars as pl

from src.common.config import COLUMNS_TO_KEEP, RAW_DATA_DIR, YEARS

LOGGER = logging.getLogger(__name__)

BTS_TO_CANONICAL = {
    "FlightDate": "FL_DATE",
    "IATA_Code_Operating_Airline": "OP_UNIQUE_CARRIER",
    "Flight_Number_Operating_Airline": "OP_CARRIER_FL_NUM",
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_source(raw_dir: Path) -> pl.DataFrame:
    parquet_path = raw_dir / "flight_data.parquet"
    csv_path = raw_dir / "flight_data_2018_2024.csv"

    if parquet_path.exists():
        LOGGER.info("Reading %s", parquet_path)
        scan = pl.scan_parquet(str(parquet_path))
    elif csv_path.exists():
        LOGGER.info("Reading %s", csv_path)
        scan = pl.scan_csv(str(csv_path), infer_schema_length=1000)
    else:
        raise FileNotFoundError(
            "Expected data/raw/flight_data.parquet or data/raw/flight_data_2018_2024.csv"
        )

    expressions = [
        pl.col(source).alias(target)
        for source, target in BTS_TO_CANONICAL.items()
        if source in scan.schema
    ]
    df = scan.select(expressions).collect()
    missing = set(COLUMNS_TO_KEEP) - set(df.columns)
    if missing:
        raise ValueError(
            f"Source dataset is missing required columns: {sorted(missing)}"
        )
    return df.select(COLUMNS_TO_KEEP)


def write_year_batches(df: pl.DataFrame, raw_dir: Path, years: list[int]) -> None:
    rows_per_batch = len(df) // len(years)
    LOGGER.info("Splitting %s rows into %s yearly batches", len(df), len(years))

    for index, year in enumerate(years):
        offset = index * rows_per_batch
        length = len(df) - offset if index == len(years) - 1 else rows_per_batch
        batch = df.slice(offset, length).with_columns(
            pl.col("FL_DATE").str.replace(r"^\d{4}", str(year)).alias("FL_DATE")
        )

        output_path = raw_dir / f"flights_{year}.csv"
        batch.write_csv(output_path)
        LOGGER.info("Wrote %s rows to %s", len(batch), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_source(args.raw_dir)
    write_year_batches(df, args.raw_dir, args.years)


if __name__ == "__main__":
    main()
