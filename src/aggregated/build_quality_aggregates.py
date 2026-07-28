#!/usr/bin/env python3
"""判定済み品質データからのダッシュボード用事前集計。"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.analysis.quality_columns import SPEC_ORDER
from src.dashboard_config import DASHBOARD_CONFIG

ANALYSIS_VERSION = "1.0"
FILE_NAMES = {
    "lots": "lots.parquet",
    "frame_item_stats": "frame_item_stats.parquet",
    "position_item_stats": "position_item_stats.parquet",
    "lot_item_histogram": "lot_item_histogram.parquet",
    "piece_ng": "piece_ng.parquet",
}


def _sql_string(value: str) -> str:
    """SQL文字列リテラル。"""
    return "'" + value.replace("'", "''") + "'"


def _write_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: Sequence[object],
    output_path: Path,
) -> int:
    """SQL結果のParquet保存。"""
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)
    copy_query = f"""
        COPY ({query})
        TO {_sql_string(str(temporary_path))}
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 122880
        )
    """
    try:
        connection.execute(copy_query, parameters)
        row_count = pq.ParquetFile(temporary_path).metadata.num_rows
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count


def _lots_query() -> str:
    """lot一覧集計SQL。"""
    return """
        SELECT
            lot_number,
            min(lot_start_time) AS lot_start_time
        FROM read_parquet(?)
        GROUP BY lot_number
        ORDER BY lot_start_time
    """


def _frame_item_stats_query() -> str:
    """lot・Frame・検査項目別集計SQL。"""
    return """
        SELECT
            lot_number,
            min(lot_start_time) AS lot_start_time,
            FrameNo,
            vision,
            colname,
            min(meta_category) AS meta_category,
            min(meta_unit) AS meta_unit,
            min(limmin) AS spec_lower,
            max(limmax) AS spec_upper,
            min(meta_best) AS spec_best,
            count(*) AS total_count,
            count(value) AS sample_count,
            count_if(is_ng) AS ng_count,
            100.0 * count_if(is_ng) / count(*) AS ng_rate,
            avg(normalized_deviation) AS normalized_mean,
            stddev_pop(normalized_value) AS normalized_std,
            min(value) AS minimum,
            quantile_cont(value, 0.05) AS p05,
            quantile_cont(value, 0.25) AS p25,
            quantile_cont(value, 0.50) AS p50,
            quantile_cont(value, 0.75) AS p75,
            quantile_cont(value, 0.95) AS p95,
            max(value) AS maximum
        FROM read_parquet(?)
        WHERE is_judgement_target
        GROUP BY lot_number, FrameNo, vision, colname
        ORDER BY lot_start_time, FrameNo, colname
    """


def _position_item_stats_query() -> str:
    """lot・検査項目・製品座標別集計SQL。"""
    return """
        SELECT
            lot_number,
            min(lot_start_time) AS lot_start_time,
            vision,
            colname,
            min(meta_category) AS meta_category,
            min(meta_unit) AS meta_unit,
            PositionX,
            PositionY,
            count(*) AS total_count,
            count(value) AS sample_count,
            count_if(is_ng) AS ng_count,
            100.0 * count_if(is_ng) / count(*) AS ng_rate,
            avg(normalized_deviation) AS normalized_mean,
            stddev_pop(normalized_value) AS normalized_std
        FROM read_parquet(?)
        WHERE is_judgement_target
        GROUP BY
            lot_number,
            vision,
            colname,
            PositionX,
            PositionY
        ORDER BY lot_start_time, colname, PositionY, PositionX
    """


def _lot_item_histogram_query(bins: int) -> str:
    """lot・検査項目・bin別集計SQL。"""
    return f"""
        WITH stats AS (
            SELECT
                colname,
                min(value) AS value_min,
                max(value) AS value_max,
                min(limmin) AS spec_lower,
                max(limmax) AS spec_upper,
                min(meta_best) AS spec_best
            FROM read_parquet(?)
            WHERE is_judgement_target
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
                    THEN spec_lower - (spec_upper - spec_lower) / 4.0
                    WHEN spec_upper IS NOT NULL
                     AND spec_best IS NOT NULL
                    THEN spec_best
                    WHEN spec_lower IS NOT NULL
                     AND spec_best IS NOT NULL
                    THEN spec_lower - (spec_best - spec_lower) / 5.0
                    ELSE raw_min - greatest(
                        raw_max - raw_min,
                        abs(raw_max) * 0.01,
                        1e-9
                    ) * 0.04
                END AS plot_min,
                CASE
                    WHEN spec_lower IS NOT NULL
                     AND spec_upper IS NOT NULL
                    THEN spec_upper + (spec_upper - spec_lower) / 4.0
                    WHEN spec_upper IS NOT NULL
                     AND spec_best IS NOT NULL
                    THEN spec_best + (spec_upper - spec_best) * 1.2
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
        indexed AS (
            SELECT
                quality.lot_number,
                quality.lot_start_time,
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
                                / (bounds.plot_max - bounds.plot_min)
                                * {bins}
                            ) AS INTEGER
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
            WHERE quality.is_judgement_target
        ),
        grouped AS (
            SELECT
                lot_number,
                min(lot_start_time) AS lot_start_time,
                colname,
                bin_index,
                min(plot_min) AS plot_min,
                max(plot_max) AS plot_max,
                min(spec_lower) AS spec_lower,
                max(spec_upper) AS spec_upper,
                min(spec_best) AS spec_best,
                count(*) AS count
            FROM indexed
            GROUP BY lot_number, colname, bin_index
        )
        SELECT
            *,
            CASE
                WHEN bin_index IS NOT NULL
                THEN plot_min
                     + (plot_max - plot_min) * bin_index / {bins}
            END AS bin_left,
            CASE
                WHEN bin_index IS NOT NULL
                THEN plot_min
                     + (plot_max - plot_min) * (bin_index + 1) / {bins}
            END AS bin_right,
            CASE
                WHEN bin_index IS NOT NULL
                THEN plot_min
                     + (plot_max - plot_min) * (bin_index + 0.5) / {bins}
            END AS bin_center
        FROM grouped
        ORDER BY lot_start_time, colname, bin_index NULLS LAST
    """


def _piece_ng_query() -> str:
    """個片別NGビットマスク集計SQL。"""
    bit_cases = " ".join(
        f"WHEN {_sql_string(colname)} THEN (1::UBIGINT << {bit_index})"
        for bit_index, colname in enumerate(SPEC_ORDER)
    )
    return f"""
        WITH pieces AS (
            SELECT
                lot_number,
                min(lot_start_time) AS lot_start_time,
                FrameNo,
                PositionX,
                PositionY,
                count(*) AS total_item_count,
                count_if(is_ng) AS ng_item_count,
                bit_or(
                    CASE
                        WHEN is_ng THEN CASE colname
                            {bit_cases}
                            ELSE 0::UBIGINT
                        END
                        ELSE 0::UBIGINT
                    END
                ) AS ng_mask
            FROM read_parquet(?)
            WHERE is_judgement_target
            GROUP BY lot_number, FrameNo, PositionX, PositionY
        )
        SELECT
            *,
            ng_mask != 0 AS is_ng
        FROM pieces
        ORDER BY lot_start_time, FrameNo, PositionY, PositionX
    """


def _write_manifest(
    input_path: Path,
    output_dir: Path,
    bins: int,
    source_row_count: int,
    row_counts: dict[str, int],
) -> Path:
    """分析データ構成のマニフェスト保存。"""
    grains = {
        "lots": ["lot_number"],
        "frame_item_stats": ["lot_number", "FrameNo", "colname"],
        "position_item_stats": [
            "lot_number",
            "colname",
            "PositionX",
            "PositionY",
        ],
        "lot_item_histogram": [
            "lot_number",
            "colname",
            "bin_index",
        ],
        "piece_ng": [
            "lot_number",
            "FrameNo",
            "PositionX",
            "PositionY",
        ],
    }
    manifest = {
        "dataset": "quality_dashboard_analysis",
        "dataset_stage": "aggregated",
        "dataset_version": ANALYSIS_VERSION,
        "source_detail": os.path.relpath(input_path, output_dir),
        "source_row_count": source_row_count,
        "aggregation_filter": "is_judgement_target",
        "kde_bins": bins,
        "files": {
            name: {
                "file": FILE_NAMES[name],
                "grain": grains[name],
                "row_count": row_counts[name],
            }
            for name in FILE_NAMES
        },
        "ng_bit_mapping": [
            {
                "colname": colname,
                "bit_index": bit_index,
                "bit_value": 1 << bit_index,
            }
            for bit_index, colname in enumerate(SPEC_ORDER)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_dir / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def aggregate(
    input_path: Path,
    output_dir: Path,
    bins: int,
) -> Path:
    """判定済みParquetの粒度別事前集計。"""
    if bins < 2:
        raise ValueError("--kde-bins must be at least 2")

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = pq.ParquetFile(input_path)
    metadata = source_file.schema_arrow.metadata or {}
    if metadata.get(b"dataset_stage") != b"judged":
        raise ValueError("input dataset_stage must be judged")

    queries = {
        "lots": (_lots_query(), [str(input_path)]),
        "frame_item_stats": (
            _frame_item_stats_query(),
            [str(input_path)],
        ),
        "position_item_stats": (
            _position_item_stats_query(),
            [str(input_path)],
        ),
        "lot_item_histogram": (
            _lot_item_histogram_query(bins),
            [str(input_path), str(input_path)],
        ),
        "piece_ng": (_piece_ng_query(), [str(input_path)]),
    }

    row_counts: dict[str, int] = {}
    with duckdb.connect(database=":memory:") as connection:
        for name, (query, parameters) in queries.items():
            output_path = output_dir / FILE_NAMES[name]
            row_counts[name] = _write_query(
                connection,
                query,
                parameters,
                output_path,
            )
            print(f"wrote {row_counts[name]:,} rows to {output_path}")

    manifest_path = _write_manifest(
        input_path,
        output_dir,
        bins,
        source_file.metadata.num_rows,
        row_counts,
    )
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """コマンドライン引数の定義。"""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(project_root / "data" / "judged" / "quality_data_100lots.parquet"),
        help="判定済みParquetの入力パス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(project_root / "data" / "analysis" / "quality_data_100lots"),
        help="事前集計Parquet群の出力ディレクトリ",
    )
    parser.add_argument(
        "--kde-bins",
        type=int,
        default=DASHBOARD_CONFIG.kde_bins,
        help="lot・検査項目別ヒストグラムのbin数",
    )
    return parser.parse_args()


def main() -> None:
    """事前集計処理の実行。"""
    args = parse_args()
    aggregate(args.input, args.output_dir, args.kde_bins)


if __name__ == "__main__":
    main()
