import logging
from pathlib import Path
from typing import Any

import polars as pl
from deltalake import DeltaTable

from src.common.config import Z_ORDER_COLUMNS

_LOGGER = logging.getLogger(__name__)


class DeltaMaintenance:
    def __init__(self, table_path: Path):
        self.table_path = table_path

    def table(self, version: int | None = None) -> DeltaTable:
        if version is None:
            return DeltaTable(str(self.table_path))
        return DeltaTable(str(self.table_path), version=version)

    def optimize(self) -> dict[str, Any]:
        result = self.table().optimize.compact()
        _LOGGER.info("OPTIMIZE result for %s: %s", self.table_path, result)
        return result

    def z_order(self, columns: list[str] | None = None) -> dict[str, Any]:
        columns = columns or Z_ORDER_COLUMNS
        available = {field.name for field in self.table().schema().fields}
        z_columns = [col for col in columns if col in available]
        if not z_columns:
            _LOGGER.warning("No Z-ORDER columns are present in %s", self.table_path)
            return {}
        result = self.table().optimize.z_order(z_columns)
        _LOGGER.info("Z-ORDER result for %s: %s", self.table_path, result)
        return result

    def vacuum(self, retention_hours: int = 168, dry_run: bool = True) -> list[str]:
        result = self.table().vacuum(retention_hours=retention_hours, dry_run=dry_run)
        _LOGGER.info("VACUUM result for %s: %s", self.table_path, result)
        return result

    def time_travel(self, version: int) -> pl.DataFrame:
        df = pl.scan_delta(str(self.table_path), version=version).collect()
        _LOGGER.info(
            "Read %s rows from %s at version %s",
            len(df),
            self.table_path,
            version,
        )
        return df

    def history(self) -> list[dict[str, Any]]:
        return self.table().history()
