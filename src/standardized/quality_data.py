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

    def density_bins_by_colname_frame(
        self,
        bins: int,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・FrameNo・検査項目別の密度bin集計。"""
        if lot_numbers is None:
            lot_numbers = tuple(record[0] for record in self.lots())
        placeholders = ", ".join("?" for _ in lot_numbers)
        lot_filter = f"AND lot_number IN ({placeholders})"
        bounds_query = f"""
            CREATE OR REPLACE TEMP TABLE quality_density_bounds AS
            WITH frame_stats AS (
                SELECT
                    colname,
                    lot_number,
                    FrameNo,
                    min(value) AS value_min,
                    max(value) AS value_max,
                    avg(value) AS value_mean,
                    stddev_pop(value) AS value_std,
                    min(limmin) AS spec_lower,
                    max(limmax) AS spec_upper,
                    min(meta_best) AS spec_best,
                    any_value(meta_unit) AS meta_unit
                FROM read_parquet(?)
                WHERE meta_type = 'spec'
                  AND NOT meta_ignore
                  {lot_filter}
                GROUP BY colname, lot_number, FrameNo
            ),
            colname_stats AS (
                SELECT
                    colname,
                    greatest(
                        min(value_min),
                        min(value_mean - 3.5 * value_std)
                    ) AS core_min,
                    least(
                        max(value_max),
                        max(value_mean + 3.5 * value_std)
                    ) AS core_max,
                    min(spec_lower) AS spec_lower,
                    max(spec_upper) AS spec_upper,
                    min(spec_best) AS spec_best,
                    any_value(meta_unit) AS meta_unit
                FROM frame_stats
                GROUP BY colname
            ),
            raw_bounds AS (
                SELECT
                    *,
                    least(
                        core_min,
                        coalesce(spec_lower, core_min),
                        coalesce(spec_upper, core_min),
                        coalesce(spec_best, core_min)
                    ) AS raw_min,
                    greatest(
                        core_max,
                        coalesce(spec_lower, core_max),
                        coalesce(spec_upper, core_max),
                        coalesce(spec_best, core_max)
                    ) AS raw_max
                FROM colname_stats
            )
            SELECT
                colname,
                CASE
                    WHEN spec_lower IS NULL
                     AND spec_best IS NOT NULL
                    THEN least(spec_best, raw_min)
                    ELSE raw_min - greatest(
                        raw_max - raw_min,
                        abs(raw_max) * 0.01,
                        1e-9
                    ) * 0.04
                END AS plot_min,
                raw_max + greatest(
                    raw_max - raw_min,
                    abs(raw_max) * 0.01,
                    1e-9
                ) * 0.04 AS plot_max,
                spec_lower,
                spec_upper,
                meta_unit
            FROM raw_bounds
        """
        self.connection.execute(
            bounds_query,
            [str(self.parquet_path), *lot_numbers],
        )

        frames = []
        for start in range(0, len(lot_numbers), 10):
            batch = lot_numbers[start : start + 10]
            batch_placeholders = ", ".join("?" for _ in batch)
            query = f"""
                WITH binned AS (
                    SELECT
                        quality.lot_number,
                        quality.FrameNo,
                        quality.colname,
                        CASE
                            WHEN quality.value IS NULL
                            THEN NULL
                            WHEN quality.value < bounds.plot_min
                              OR quality.value > bounds.plot_max
                            THEN NULL
                            ELSE least(
                                {bins - 1},
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
                        END AS bin_index,
                        bounds.plot_min,
                        bounds.plot_max,
                        bounds.spec_lower,
                        bounds.spec_upper,
                        bounds.meta_unit,
                        CASE
                            WHEN (
                                quality.limmin IS NOT NULL
                                AND quality.value < quality.limmin
                            )
                              OR (
                                quality.limmax IS NOT NULL
                                AND quality.value > quality.limmax
                            )
                            THEN 1
                            ELSE 0
                        END AS is_ng,
                        quality.value
                    FROM read_parquet(?) AS quality
                    JOIN quality_density_bounds AS bounds
                      ON quality.colname = bounds.colname
                    WHERE quality.meta_type = 'spec'
                      AND NOT quality.meta_ignore
                      AND quality.lot_number IN ({batch_placeholders})
                ),
                bin_counts AS (
                    SELECT
                        lot_number,
                        FrameNo,
                        colname,
                        bin_index,
                        any_value(plot_min) AS plot_min,
                        any_value(plot_max) AS plot_max,
                        any_value(spec_lower) AS spec_lower,
                        any_value(spec_upper) AS spec_upper,
                        any_value(meta_unit) AS meta_unit,
                        count(value) AS sample_count,
                        sum(is_ng) AS ng_count
                    FROM binned
                    GROUP BY lot_number, FrameNo, colname, bin_index
                )
                SELECT
                    lot_number,
                    FrameNo,
                    colname,
                    coalesce(
                        list(bin_index ORDER BY bin_index)
                            FILTER (WHERE bin_index IS NOT NULL),
                        []::INTEGER[]
                    ) AS bin_indices,
                    coalesce(
                        list(sample_count ORDER BY bin_index)
                            FILTER (WHERE bin_index IS NOT NULL),
                        []::BIGINT[]
                    ) AS bin_counts,
                    sum(sample_count) AS sample_count,
                    coalesce(
                        sum(sample_count)
                            FILTER (WHERE bin_index IS NOT NULL),
                        0
                    ) AS in_range_count,
                    sum(ng_count) AS ng_count,
                    any_value(plot_min) AS plot_min,
                    any_value(plot_max) AS plot_max,
                    any_value(spec_lower) AS spec_lower,
                    any_value(spec_upper) AS spec_upper,
                    any_value(meta_unit) AS meta_unit
                FROM bin_counts
                GROUP BY lot_number, FrameNo, colname
            """
            frames.append(
                self.connection.execute(
                    query,
                    [str(self.parquet_path), *batch],
                ).df()
            )
        return pd.concat(frames, ignore_index=True)

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
                    max(limmax) AS spec_upper,
                    min(meta_best) AS spec_best
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
                    CASE
                        WHEN spec_lower IS NOT NULL
                         AND spec_upper IS NOT NULL
                        THEN spec_lower
                             - (spec_upper - spec_lower) / 4.0
                        WHEN spec_upper IS NOT NULL
                         AND spec_best IS NOT NULL
                        THEN spec_best
                        WHEN spec_lower IS NOT NULL
                         AND spec_best IS NOT NULL
                        THEN spec_lower
                             - (spec_best - spec_lower) / 5.0
                        ELSE raw_min - greatest(
                            raw_max - raw_min,
                            abs(raw_max) * 0.01,
                            1e-9
                        ) * 0.04
                    END AS plot_min,
                    CASE
                        WHEN spec_lower IS NOT NULL
                         AND spec_upper IS NOT NULL
                        THEN spec_upper
                             + (spec_upper - spec_lower) / 4.0
                        WHEN spec_upper IS NOT NULL
                         AND spec_best IS NOT NULL
                        THEN spec_best
                             + (spec_upper - spec_best) * 1.2
                        WHEN spec_lower IS NOT NULL
                         AND spec_best IS NOT NULL
                        THEN spec_best
                        ELSE raw_max + greatest(
                            raw_max - raw_min,
                            abs(raw_max) * 0.01,
                            1e-9
                        ) * 0.04
                    END AS plot_max
                FROM ranges
            ),
            binned AS (
                SELECT
                    quality.lot_number,
                    quality.colname,
                    CASE
                        WHEN quality.value < bounds.plot_min
                          OR quality.value > bounds.plot_max
                        THEN NULL
                        ELSE least(
                            {bins - 1},
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
                    END AS bin_index,
                    bounds.plot_min,
                    bounds.plot_max,
                    bounds.spec_lower,
                    bounds.spec_upper,
                    bounds.spec_best
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
                any_value(spec_best) AS spec_best,
                count(*) AS count
            FROM binned
            GROUP BY lot_number, colname, bin_index
            ORDER BY colname, lot_number, bin_index
        """
        return self.connection.execute(query, parameters).df()
