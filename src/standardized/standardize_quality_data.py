#!/usr/bin/env python3
"""raw品質データへの規格正規化列付与。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SPEC_POSITION_FORMULA = (
    "(value - (limmin + limmax) / 2) "
    "/ ((limmax - limmin) / 2)"
)
SPEC_USAGE_FORMULA = (
    "(value - meta_best) / (limmax - meta_best)"
)


def _standardized_schema(
    raw_schema: pa.Schema,
    input_path: Path,
) -> pa.Schema:
    """規格正規化列を含む標準化スキーマ。"""
    fields = list(raw_schema)
    insert_at = raw_schema.get_field_index("value") + 1
    fields[insert_at:insert_at] = [
        pa.field("spec_position", pa.float64(), nullable=True),
        pa.field("spec_usage", pa.float64(), nullable=True),
    ]

    metadata = dict(raw_schema.metadata or {})
    metadata.update(
        {
            b"dataset_stage": b"standardized",
            b"dataset_version": b"3.1",
            b"source_file": input_path.name.encode(),
            b"spec_position_formula": SPEC_POSITION_FORMULA.encode(),
            b"spec_usage_formula": SPEC_USAGE_FORMULA.encode(),
        }
    )
    return pa.schema(fields, metadata=metadata)


def _float_values(table: pa.Table, column: str) -> np.ndarray:
    """nullable float列のNumPy配列化。"""
    return np.asarray(
        table.column(column)
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def _add_spec_columns(
    raw_table: pa.Table,
    schema: pa.Schema,
) -> pa.Table:
    """1 row groupへの規格正規化列付与。"""
    value = _float_values(raw_table, "value")
    limmin = _float_values(raw_table, "limmin")
    limmax = _float_values(raw_table, "limmax")
    meta_best = _float_values(raw_table, "meta_best")

    spec_position = np.full(value.shape, np.nan)
    two_sided = np.isfinite(limmin) & np.isfinite(limmax)
    spec_position[two_sided] = (
        value[two_sided]
        - (limmin[two_sided] + limmax[two_sided]) / 2.0
    ) / (
        (limmax[two_sided] - limmin[two_sided]) / 2.0
    )

    spec_usage = np.full(value.shape, np.nan)
    usage_target = (
        ~np.isfinite(limmin)
        & np.isfinite(meta_best)
        & np.isfinite(limmax)
    )
    spec_usage[usage_target] = (
        value[usage_target] - meta_best[usage_target]
    ) / (
        limmax[usage_target] - meta_best[usage_target]
    )

    derived = {
        "spec_position": pa.array(
            spec_position,
            type=pa.float64(),
            from_pandas=True,
        ),
        "spec_usage": pa.array(
            spec_usage,
            type=pa.float64(),
            from_pandas=True,
        ),
    }
    arrays = [
        derived[field.name]
        if field.name in derived
        else raw_table.column(field.name).combine_chunks()
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_manifest(
    input_path: Path,
    output_path: Path,
    parquet_file: pq.ParquetFile,
) -> Path:
    """標準化条件のマニフェスト保存。"""
    manifest = {
        "file": output_path.name,
        "source_file": input_path.name,
        "row_count": parquet_file.metadata.num_rows,
        "row_group_count": parquet_file.metadata.num_row_groups,
        "derived_columns": {
            "spec_position": SPEC_POSITION_FORMULA,
            "spec_usage": SPEC_USAGE_FORMULA,
        },
    }
    manifest_path = output_path.with_name(
        f"{output_path.stem}_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def standardize(
    input_path: Path,
    output_path: Path,
) -> tuple[int, Path]:
    """raw Parquetの標準化。"""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    raw_file = pq.ParquetFile(input_path)
    schema = _standardized_schema(raw_file.schema_arrow, input_path)
    writer = pq.ParquetWriter(
        temporary_path,
        schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=[
            "vision",
            "lot_number",
            "colname",
            "meta_type",
            "meta_category",
        ],
        write_statistics=True,
        version="2.6",
    )
    try:
        for row_group in range(raw_file.metadata.num_row_groups):
            raw_table = raw_file.read_row_group(row_group)
            table = _add_spec_columns(raw_table, schema)
            writer.write_table(table, row_group_size=table.num_rows)
            if (
                (row_group + 1) % 10 == 0
                or row_group + 1 == raw_file.metadata.num_row_groups
            ):
                print(
                    "standardized "
                    f"{row_group + 1:>3}/"
                    f"{raw_file.metadata.num_row_groups} row groups"
                )
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    standardized_file = pq.ParquetFile(temporary_path)
    if (
        standardized_file.metadata.num_rows
        != raw_file.metadata.num_rows
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("row count mismatch")
    if (
        standardized_file.metadata.num_row_groups
        != raw_file.metadata.num_row_groups
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("row-group count mismatch")

    temporary_path.replace(output_path)
    manifest_path = _write_manifest(
        input_path,
        output_path,
        standardized_file,
    )
    return standardized_file.metadata.num_rows, manifest_path


def parse_args() -> argparse.Namespace:
    """コマンドライン引数の定義。"""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            project_root
            / "data"
            / "raw"
            / "quality_data_100lots.parquet"
        ),
        help="raw Parquetの入力パス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "data"
            / "standardized"
            / "quality_data_100lots.parquet"
        ),
        help="標準化済みParquetの出力パス",
    )
    return parser.parse_args()


def main() -> None:
    """標準化処理の実行。"""
    args = parse_args()
    row_count, manifest_path = standardize(
        args.input,
        args.output,
    )
    print(f"wrote {row_count:,} rows to {args.output.resolve()}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
