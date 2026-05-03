import argparse
import logging
import shutil
from pathlib import Path

import kagglehub

from src.common.config import (
    BRONZE_PATH,
    DATA_DIR,
    GOLD_ANALYTICS_PATHS,
    GOLD_FEATURES_PATH,
    LOGS_DIR,
    ML_SAMPLE_FRACTION,
    RAW_DATA_DIR,
    SILVER_PATH,
    YEARS,
)
from src.delta.ops import DeltaMaintenance
from src.layers.bronze import BronzeLayer
from src.layers.gold import GoldLayer
from src.layers.silver import SilverLayer
from src.ml.models import MLModelTrainer
from src.scripts.simulate_year_batches import load_source, write_year_batches

LOGGER = logging.getLogger(__name__)
KAGGLE_DATASET = "shubhamsingh42/flight-delay-dataset-2018-2024"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "pipeline.log"),
        ],
    )


def yearly_batch_path(data_dir: Path, year: int) -> Path | None:
    candidates = [
        data_dir / f"{year}.csv",
        data_dir / f"flights_{year}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_yearly_batches(data_dir: Path, years: list[int]) -> list[tuple[Path, int]]:
    batches: list[tuple[Path, int]] = []
    for year in years:
        batch_path = yearly_batch_path(data_dir, year)
        if batch_path:
            batches.append((batch_path, year))
    return batches


def has_kaggle_source(data_dir: Path) -> bool:
    return (data_dir / "flight_data.parquet").exists() or (
        data_dir / "flight_data_2018_2024.csv"
    ).exists()


def download_kaggle_source(data_dir: Path) -> None:
    if has_kaggle_source(data_dir):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Kaggle dataset %s", KAGGLE_DATASET)
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    copied_files = []

    for source_path in download_path.rglob("*"):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() not in {".csv", ".parquet", ".html"}:
            continue
        target_path = data_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path.name)

    if not has_kaggle_source(data_dir):
        raise FileNotFoundError(
            "Downloaded Kaggle dataset does not contain flight_data.parquet or flight_data_2018_2024.csv"
        )

    LOGGER.info("Copied Kaggle source files to %s: %s", data_dir, copied_files)


def ensure_yearly_batches(
    data_dir: Path,
    years: list[int],
    auto_simulate: bool = True,
    auto_download: bool = True,
) -> None:
    existing_years = {year for _, year in discover_yearly_batches(data_dir, years)}
    missing_years = [year for year in years if year not in existing_years]
    if not missing_years:
        return

    if auto_download and not has_kaggle_source(data_dir):
        download_kaggle_source(data_dir)

    if auto_simulate and has_kaggle_source(data_dir):
        LOGGER.info(
            "Missing yearly raw batches for %s; simulating them from Kaggle source",
            missing_years,
        )
        df = load_source(data_dir)
        write_year_batches(df, data_dir, years)
        return

    LOGGER.warning("Missing yearly raw batches for %s", missing_years)


def discover_batches(
    data_dir: Path,
    years: list[int],
    auto_simulate: bool = True,
    auto_download: bool = True,
) -> list[tuple[Path, int]]:
    ensure_yearly_batches(
        data_dir,
        years,
        auto_simulate=auto_simulate,
        auto_download=auto_download,
    )
    batches = discover_yearly_batches(data_dir, years)

    if not batches and (DATA_DIR / "test_data.csv").exists():
        LOGGER.warning(
            "No yearly raw files found; using data/test_data.csv as demo batch"
        )
        batches.append((DATA_DIR / "test_data.csv", 2024))

    if not batches:
        raise FileNotFoundError(
            f"No CSV batches found in {data_dir}. Expected files like flights_2018.csv."
        )

    return batches


def run_pipeline(
    years: list[int],
    raw_dir: Path,
    skip_ingest: bool = False,
    run_maintenance: bool = True,
    run_ml: bool = True,
    auto_simulate_batches: bool = True,
    auto_download_dataset: bool = True,
) -> None:
    LOGGER.info("Starting flight delay lakehouse pipeline")
    batches = discover_batches(
        raw_dir,
        years,
        auto_simulate=auto_simulate_batches,
        auto_download=auto_download_dataset,
    )
    LOGGER.info(
        "Using yearly raw batches: %s",
        [(year, path.name) for path, year in batches],
    )

    if not skip_ingest:
        bronze = BronzeLayer(bronze_path=BRONZE_PATH)
        bronze.ingest_batches(batches)
        LOGGER.info("Bronze table info: %s", bronze.get_table_info())

    silver = SilverLayer(bronze_path=BRONZE_PATH, silver_path=SILVER_PATH)
    for _, year in batches:
        silver.transform(year=year, use_merge=True)

    gold = GoldLayer(
        silver_path=SILVER_PATH,
        analytics_paths=GOLD_ANALYTICS_PATHS,
        features_path=GOLD_FEATURES_PATH,
    )
    gold.build_gold()
    LOGGER.info("Gold feature table version: %s", gold.features_version())

    if run_maintenance:
        for table_path in [
            SILVER_PATH,
            *GOLD_ANALYTICS_PATHS.values(),
            GOLD_FEATURES_PATH,
        ]:
            maintenance = DeltaMaintenance(table_path)
            LOGGER.info("%s history: %s", table_path, maintenance.history()[:3])
            maintenance.optimize()
            maintenance.z_order()
            maintenance.vacuum(dry_run=True)
        DeltaMaintenance(GOLD_FEATURES_PATH).time_travel(version=0)

    if run_ml:
        trainer = MLModelTrainer(
            features_path=GOLD_FEATURES_PATH,
            sample_fraction=ML_SAMPLE_FRACTION,
        )
        ml_results = trainer.run()
        LOGGER.info("ML training results: %s", ml_results)

    LOGGER.info("Pipeline finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-maintenance", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--no-auto-simulate-batches", action="store_true")
    parser.add_argument("--no-auto-download-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_pipeline(
        years=args.years,
        raw_dir=args.raw_dir,
        skip_ingest=args.skip_ingest,
        run_maintenance=not args.skip_maintenance,
        run_ml=not args.skip_ml,
        auto_simulate_batches=not args.no_auto_simulate_batches,
        auto_download_dataset=not args.no_auto_download_dataset,
    )


if __name__ == "__main__":
    main()