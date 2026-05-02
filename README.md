# Lab 3: Lakehouse with Polars, Delta Lake, and MLflow

This project implements a local lakehouse pipeline for US flight delay prediction:

```text
raw yearly CSV batches -> Bronze Delta -> Silver Delta -> Gold Delta -> MLflow
```

The pipeline is written in Python under `src/`, uses Polars and Delta Lake (`delta-rs`), trains regression and classification models, and logs model metrics and artifacts to an MLflow server started by Docker Compose.

## Quick Start

Run the whole project:

```bash
docker compose up --build
```

Expected successful ending:

```text
Pipeline finished
app-1 exited with code 0
```

MLflow remains available while the `mlflow` container is running:

```text
http://localhost:5050
```

Stop containers:

```bash
docker compose down
```

## Clean Fresh Run

Use this when you want to remove generated lakehouse tables and MLflow history, but keep the raw Kaggle files:

```bash
docker compose down
rm -rf data/bronze data/silver data/gold mlruns logs/pipeline.log
docker compose up --build
```

The pipeline reads `data/raw`, writes Delta tables under `data/`, writes logs to `logs/`, and writes MLflow metadata/artifacts to `mlruns/`.

## How To See The Results

### 1. MLflow UI

Open:

```text
http://localhost:5050
```

Open the experiment:

```text
flight-delay-lakehouse
```

Open the latest run named like:

```text
flight-delay-gold-v...
```

In the run page:

- **Parameters**: `test_size`, `random_state`, `delay_threshold`, `sample_fraction`, `gold_features_version`;
- **Metrics**: regression `r2`, `mae`, `rmse`; classification `accuracy`, `f1`, `roc_auc`;
- **Artifacts**: sklearn model folders and `feature_importance/feature_importance.csv`;
- **Tags**: `gold_features_path`.

If the Artifacts tab is empty, rebuild after checking that `docker-compose.yml` mounts `./mlruns:/mlflow` for both `mlflow` and `app`.

### 2. Delta Lake Tables

After a successful run, these Delta tables should exist:

```text
data/bronze/flights
data/silver/flights
data/gold/analytics/by_airport
data/gold/analytics/by_carrier
data/gold/analytics/by_hour
data/gold/analytics/by_season
data/gold/ml_features
```

Quick check:

```bash
find data -maxdepth 4 -type d -name _delta_log | sort
```

### 3. Pipeline Logs

Open:

```text
logs/pipeline.log
```

Useful log lines to look for:

- Bronze append writes for years `2018..2024`;
- Silver `MERGE` operations;
- Gold analytics and feature table writes;
- `OPTIMIZE`, `Z-ORDER`, `VACUUM`, and time travel;
- ML metrics and `Pipeline finished`.

### 4. EDA Notebook

The notebook is:

```text
notebooks/lab3_eda.ipynb
```

Run the pipeline first, then open the notebook and execute cells to inspect raw batches, Delta table versions, Silver distributions, Gold marts, and ML feature data.

## Dataset

The project uses this Kaggle dataset:

```text
shubhamsingh42/flight-delay-dataset-2018-2024
```

Expected source files:

```text
data/raw/flight_data_2018_2024.csv
data/raw/flight_data.parquet
data/raw/readme.html
```

Important caveat: the downloaded Kaggle file used in this project is named as a 2018-2024 dataset, but its actual rows contain January 2024 only:

```text
Date range: 2024-01-01..2024-01-31
Rows: 582425
```

To satisfy the lab requirement that data arrives incrementally by year, the project simulates yearly arrival batches from that source file. The simulation splits rows into seven batches and rewrites `FL_DATE` years:

```bash
python -m src.scripts.simulate_year_batches
```

Generated batch files:

```text
data/raw/flights_2018.csv
data/raw/flights_2019.csv
data/raw/flights_2020.csv
data/raw/flights_2021.csv
data/raw/flights_2022.csv
data/raw/flights_2023.csv
data/raw/flights_2024.csv
```

The main pipeline automatically creates missing `flights_*.csv` batches if `flight_data.parquet` or `flight_data_2018_2024.csv` exists. Disable automatic simulation with:

```bash
python -m src.pipeline --no-auto-simulate-batches
```

## Project Structure

```text
src/
  common/
    config.py                 paths, constants, thresholds
    utils.py                  shared helper decorator
  delta/
    ops.py                    OPTIMIZE, Z-ORDER, VACUUM, time travel
  layers/
    bronze.py                 raw CSV ingestion into Bronze Delta
    silver.py                 cleaning, feature engineering, Delta MERGE
    gold.py                   analytics marts and ML feature table
  ml/
    models.py                 model training, comparison, MLflow logging
  scripts/
    simulate_year_batches.py  yearly batch simulation
  pipeline.py                 end-to-end pipeline entrypoint
notebooks/
  lab3_eda.ipynb              exploratory analysis notebook
logs/
  pipeline.log                runtime log
  polars_explain.txt          saved Polars explain output
data/
  raw/                        source files and yearly CSV batches
docker-compose.yml
Dockerfile
requirements.txt
requirements_qa.txt
pyproject.toml
```

All Python source files are inside `src/`.

## Local Run Without Docker

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python -m src.pipeline
```

Useful options:

```bash
python -m src.pipeline --skip-ingest
python -m src.pipeline --skip-maintenance
python -m src.pipeline --skip-ml
python -m src.pipeline --years 2018 2019 2020 2021 2022 2023 2024
python -m src.pipeline --no-auto-simulate-batches
```

For local MLflow without Docker, `MLFLOW_TRACKING_URI` defaults to the local `mlruns` directory.

## Docker Compose Details

Compose starts two services:

- `mlflow`: MLflow tracking server on container port `5000`, exposed as `http://localhost:5050`;
- `app`: runs `python -m src.pipeline`.

Important environment variables:

```yaml
MLFLOW_TRACKING_URI: http://mlflow:5000
ML_SAMPLE_FRACTION: "0.1"
```

`ML_SAMPLE_FRACTION=0.1` keeps model training fast enough for local Docker while still logging real metrics, models, feature importance, and the Gold table version.

Important volumes:

```yaml
./data:/app/data
./logs:/app/logs
./mlruns:/mlflow
```

The `./mlruns:/mlflow` mount is intentionally shared by both services so the app can write model artifacts to the same artifact root served by MLflow.

Health check:

```bash
docker compose up -d mlflow
curl http://localhost:5050/health
```

Expected response:

```text
OK
```

## Bronze Layer

Bronze ingests yearly CSV batches into:

```text
data/bronze/flights
```

Each year is written in append mode:

```python
write_deltalake(..., mode="append", schema_mode="merge")
```

This creates a real Delta version history and imitates incremental arrival by year. Bronze stores canonical flight columns and metadata:

```text
year
loaded_at
source_format
source_file
```

`schema_mode="merge"` demonstrates Delta schema evolution.

## Silver Layer

Silver reads Bronze using Polars:

```python
pl.scan_delta(...)
```

Silver performs:

- cancelled and diverted flight removal;
- `ARR_DELAY` null removal;
- IQR-based outlier filtering;
- categorical normalization to uppercase;
- canonical column selection;
- feature engineering: `hour`, `day_of_week`, `month`, `season`, `route`, `flight_id`, `is_delayed`.

Silver is written to:

```text
data/silver/flights
```

Partitioning:

```python
partition_by=["year", "month"]
```

Repeat runs update Silver through Delta `MERGE` instead of duplicating rows:

```text
target.flight_id = source.flight_id
```

The source batch is deduplicated by `flight_id` before merge because Bronze intentionally keeps append history.

## Gold Layer

Gold writes analytical aggregate Delta marts:

```text
data/gold/analytics/by_airport
data/gold/analytics/by_carrier
data/gold/analytics/by_hour
data/gold/analytics/by_season
```

Each mart contains:

```text
avg_arr_delay
median_arr_delay
std_arr_delay
delay_rate
num_flights
```

Each mart also keeps its natural dimension column:

- `ORIGIN` for airport;
- `OP_UNIQUE_CARRIER` for carrier;
- `hour` for hour;
- `season` for season.

Gold also writes the ML feature table:

```text
data/gold/ml_features
```

Targets:

- regression target: `ARR_DELAY`;
- classification target: `is_delayed = ARR_DELAY > 15`.

## Machine Learning

[src/ml/models.py](src/ml/models.py) trains and compares:

Regression:

- `LinearRegression`;
- `RandomForestRegressor`.

Classification:

- `LogisticRegression`;
- `RandomForestClassifier`.

MLflow logs:

- parameters;
- regression metrics: `r2`, `mae`, `rmse`;
- classification metrics: `accuracy`, `f1`, `roc_auc`;
- sklearn model artifacts with signatures and input examples;
- Random Forest feature importance as `feature_importance/feature_importance.csv`;
- Gold feature table version as `gold_features_version`;
- Gold feature table path as `gold_features_path`.

## Delta Lake Features

The project uses required `MERGE` plus these Delta Lake features:

- `OPTIMIZE` compaction via `DeltaTable(...).optimize.compact()`;
- `Z-ORDER` via `DeltaTable(...).optimize.z_order(...)`;
- `VACUUM` in dry-run mode;
- time travel via `pl.scan_delta(..., version=0)`;
- schema evolution in Bronze via `schema_mode="merge"`.

Maintenance code is in:

```text
src/delta/ops.py
```

## Partitioning Rationale

Silver is partitioned by `year` and `month` because flight data naturally arrives and is queried by time. These columns are common filters for incremental processing and analytical queries. Cardinality is controlled: for the lab range this is at most `7 * 12` partitions, avoiding excessive small partition fragmentation.

## Polars Lazy Query And Explain Output

The README requirement asks for a real `.explain()` with visible projection and selection pushdowns. The saved output is:

```text
logs/polars_explain.txt
```

Query:

```python
query = (
    pl.scan_delta("data/silver/flights")
    .filter((pl.col("year") == 2024) & (pl.col("ARR_DELAY").is_not_null()))
    .select(["year", "month", "ORIGIN", "ARR_DELAY", "is_delayed"])
    .group_by(["year", "month", "ORIGIN"])
    .agg(
        [
            pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
            pl.col("is_delayed").mean().alias("delay_rate"),
            pl.len().alias("num_flights"),
        ]
    )
    .sort("avg_arr_delay", descending=True)
)
print(query.explain())
```

Actual output:

```text
SORT BY [col("avg_arr_delay")]
  AGGREGATE
  	[col("ARR_DELAY").mean().alias("avg_arr_delay"), col("is_delayed").mean().alias("delay_rate"), len().alias("num_flights")] BY [col("year"), col("month"), col("ORIGIN")] FROM

      PYTHON SCAN
      PROJECT 5/21 COLUMNS
      SELECTION: [([(col("year")) == (2024)]) & (col("ARR_DELAY").is_not_null())]
```

Important lines:

- `PROJECT 5/21 COLUMNS`: projection pushdown;
- `SELECTION`: filter pushdown.

## QA And Codestyle

Install QA dependencies:

```bash
pip install -r requirements_qa.txt
```

Run the checks:

```bash
python -m black --check src
python -m isort --check-only src
sort-requirements --check requirements.txt requirements_qa.txt
python -m mypy src
python -m pylint src
```

QA configuration is stored in:

```text
pyproject.toml
```
