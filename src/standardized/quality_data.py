from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


class QualityRepository:
    """標準化済み品質データの参照窓口。"""

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path.resolve()
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        self.connection = duckdb.connect(database=":memory:")

    def lots(self) -> list[tuple[str, datetime]]:
        query = """
            SELECT lot_number, min(lot_start_time) AS lot_start_time
            FROM read_parquet(?)
            GROUP BY lot_number
            ORDER BY lot_start_time
        """
        return self.connection.execute(
            query,
            [str(self.parquet_path)],
        ).fetchall()

    def close(self) -> None:
        """データベース接続の終了。"""
        self.connection.close()

    def ng_rate_by_frame(
        self,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        lot_filter = ""
        parameters: list[object] = [str(self.parquet_path)]
        if lot_numbers:
            placeholders = ", ".join("?" for _ in lot_numbers)
            lot_filter = f"AND lot_number IN ({placeholders})"
            parameters.extend(lot_numbers)

        query = f"""
            SELECT
                lot_number,
                FrameNo,
                vision,
                colname,
                meta_category,
                count(*) AS total_count,
                sum(
                    CASE
                        WHEN (limmin IS NOT NULL AND value < limmin)
                          OR (limmax IS NOT NULL AND value > limmax)
                        THEN 1
                        ELSE 0
                    END
                ) AS ng_count,
                100.0 * ng_count / total_count AS ng_rate,
                avg(
                    coalesce(spec_position, spec_usage)
                ) AS normalized_mean,
                stddev_pop(
                    coalesce(spec_position, spec_usage)
                ) AS normalized_std
            FROM read_parquet(?)
            WHERE meta_type = 'spec'
              AND NOT meta_ignore
              {lot_filter}
            GROUP BY
                lot_number,
                FrameNo,
                vision,
                colname,
                meta_category
        """
        return self.connection.execute(query, parameters).df()

    def metrics_by_colname_position(
        self,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・検査項目・製品座標別の品質集計。"""
        lot_filter = ""
        parameters: list[object] = [str(self.parquet_path)]
        if lot_numbers:
            placeholders = ", ".join("?" for _ in lot_numbers)
            lot_filter = f"AND lot_number IN ({placeholders})"
            parameters.extend(lot_numbers)

        query = f"""
            SELECT
                lot_number,
                colname,
                PositionX,
                PositionY,
                count(*) AS total_count,
                sum(
                    CASE
                        WHEN (limmin IS NOT NULL AND value < limmin)
                          OR (limmax IS NOT NULL AND value > limmax)
                        THEN 1
                        ELSE 0
                    END
                ) AS ng_count,
                100.0 * ng_count / total_count AS ng_rate,
                avg(
                    coalesce(spec_position, spec_usage)
                ) AS normalized_mean,
                stddev_pop(
                    coalesce(spec_position, spec_usage)
                ) AS normalized_std
            FROM read_parquet(?)
            WHERE meta_type = 'spec'
              AND NOT meta_ignore
              {lot_filter}
            GROUP BY
                lot_number,
                colname,
                PositionX,
                PositionY
        """
        return self.connection.execute(query, parameters).df()
