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

    def kde_bins_by_colname_lot(
        self,
        bins: int,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・検査項目別のKDE用bin集計。"""
        lot_filter = ""
        lot_parameters: list[object] = []
        if lot_numbers:
            placeholders = ", ".join("?" for _ in lot_numbers)
            lot_filter = f"AND lot_number IN ({placeholders})"
            lot_parameters.extend(lot_numbers)
        parameters = [
            str(self.parquet_path),
            *lot_parameters,
            str(self.parquet_path),
            *lot_parameters,
        ]

        query = f"""
            WITH stats AS (
                SELECT
                    colname,
                    min(value) AS value_min,
                    max(value) AS value_max,
                    min(limmin) AS spec_lower,
                    max(limmax) AS spec_upper
                FROM read_parquet(?)
                WHERE meta_type = 'spec'
                  AND NOT meta_ignore
                  {lot_filter}
                GROUP BY colname
            ),
            ranges AS (
                SELECT
                    *,
                    least(
                        value_min,
                        coalesce(spec_lower, value_min),
                        coalesce(spec_upper, value_min)
                    ) AS raw_min,
                    greatest(
                        value_max,
                        coalesce(spec_lower, value_max),
                        coalesce(spec_upper, value_max)
                    ) AS raw_max
                FROM stats
            ),
            bounds AS (
                SELECT
                    *,
                    raw_min - greatest(
                        raw_max - raw_min,
                        abs(raw_max) * 0.01,
                        1e-9
                    ) * 0.04 AS plot_min,
                    raw_max + greatest(
                        raw_max - raw_min,
                        abs(raw_max) * 0.01,
                        1e-9
                    ) * 0.04 AS plot_max
                FROM ranges
            ),
            binned AS (
                SELECT
                    quality.lot_number,
                    quality.colname,
                    least(
                        {bins - 1},
                        greatest(
                            0,
                            cast(
                                floor(
                                    (quality.value - bounds.plot_min)
                                    / (
                                        bounds.plot_max
                                        - bounds.plot_min
                                    )
                                    * {bins}
                                )
                                AS INTEGER
                            )
                        )
                    ) AS bin_index,
                    bounds.plot_min,
                    bounds.plot_max,
                    bounds.spec_lower,
                    bounds.spec_upper
                FROM read_parquet(?) AS quality
                JOIN bounds ON quality.colname = bounds.colname
                WHERE quality.meta_type = 'spec'
                  AND NOT quality.meta_ignore
                  {lot_filter}
            )
            SELECT
                lot_number,
                colname,
                bin_index,
                any_value(plot_min) AS plot_min,
                any_value(plot_max) AS plot_max,
                any_value(spec_lower) AS spec_lower,
                any_value(spec_upper) AS spec_upper,
                count(*) AS count
            FROM binned
            GROUP BY lot_number, colname, bin_index
            ORDER BY colname, lot_number, bin_index
        """
        return self.connection.execute(query, parameters).df()
