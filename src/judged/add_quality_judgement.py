#!/usr/bin/env python3
"""標準化済み品質データへの判定列付与。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NORMALIZED_VALUE_FORMULA = "coalesce(spec_position, spec_usage)"
NORMALIZED_DEVIATION_FORMULA = (
    "case when spec_position is not null then abs(spec_position) "
    "when spec_usage is not null then greatest(spec_usage, 0) end"
)
IS_NG_FORMULA = (
    "is_judgement_target and "
    "((limmin is not null and value < limmin) or "
    "(limmax is not null and value > limmax))"
)


def _judged_schema(
    standardized_schema: pa.Schema,
    input_path: Path,
) -> pa.Schema:
    """判定列を含む品質データスキーマ。"""
    fields = list(standardized_schema)
    insert_at = standardized_schema.get_field_index("spec_usage") + 1
    fields[insert_at:insert_at] = [
        pa.field("normalized_value", pa.float64(), nullable=True),
        pa.field("normalized_deviation", pa.float64(), nullable=True),
        pa.field("is_judgement_target", pa.bool_(), nullable=False),
        pa.field("is_ng", pa.bool_(), nullable=False),
        pa.field("ng_direction", pa.int8(), nullable=False),
    ]

    metadata = dict(standardized_schema.metadata or {})
    metadata.update(
        {
            b"dataset_stage": b"judged",
            b"dataset_version": b"4.2",
            b"source_file": input_path.name.encode(),
            b"normalized_value_formula": NORMALIZED_VALUE_FORMULA.encode(),
            b"normalized_deviation_formula": (NORMALIZED_DEVIATION_FORMULA.encode()),
            b"is_ng_formula": IS_NG_FORMULA.encode(),
            b"ng_direction_values": b"-1:lower,0:ok_or_not_target,1:upper",
        }
    )
    return pa.schema(fields, metadata=metadata)


def _numpy_values(
    table: pa.Table,
    column: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    """Arrow列のNumPy配列化。"""
    return np.asarray(
        table.column(column).combine_chunks().to_numpy(zero_copy_only=False),
        dtype=dtype,
    )


def _add_judgement_columns(
    standardized_table: pa.Table,
    schema: pa.Schema,
) -> pa.Table:
    """1 row groupへの判定列付与。"""
    value = _numpy_values(standardized_table, "value", np.float64)
    spec_position = _numpy_values(
        standardized_table,
        "spec_position",
        np.float64,
    )
    spec_usage = _numpy_values(
        standardized_table,
        "spec_usage",
        np.float64,
    )
    limmin = _numpy_values(standardized_table, "limmin", np.float64)
    limmax = _numpy_values(standardized_table, "limmax", np.float64)
    meta_type = _numpy_values(standardized_table, "meta_type", object)
    meta_ignore = _numpy_values(standardized_table, "meta_ignore", bool)

    has_position = np.isfinite(spec_position)
    has_usage = np.isfinite(spec_usage)
    normalized_value = np.where(
        has_position,
        spec_position,
        spec_usage,
    )
    normalized_deviation = np.where(
        has_position,
        np.abs(spec_position),
        np.where(has_usage, np.maximum(spec_usage, 0.0), np.nan),
    )

    is_judgement_target = (meta_type == "spec") & ~meta_ignore
    lower_ng = is_judgement_target & np.isfinite(limmin) & (value < limmin)
    upper_ng = is_judgement_target & np.isfinite(limmax) & (value > limmax)
    is_ng = lower_ng | upper_ng
    ng_direction = np.where(lower_ng, -1, np.where(upper_ng, 1, 0)).astype(np.int8)

    derived = {
        "normalized_value": pa.array(
            normalized_value,
            type=pa.float64(),
            from_pandas=True,
        ),
        "normalized_deviation": pa.array(
            normalized_deviation,
            type=pa.float64(),
            from_pandas=True,
        ),
        "is_judgement_target": pa.array(
            is_judgement_target,
            type=pa.bool_(),
        ),
        "is_ng": pa.array(is_ng, type=pa.bool_()),
        "ng_direction": pa.array(ng_direction, type=pa.int8()),
    }
    arrays = [
        derived[field.name]
        if field.name in derived
        else standardized_table.column(field.name).combine_chunks()
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_manifest(
    input_path: Path,
    output_path: Path,
    parquet_file: pq.ParquetFile,
) -> Path:
    """判定列付与条件のマニフェスト保存。"""
    manifest = {
        "file": output_path.name,
        "source_file": input_path.name,
        "row_count": parquet_file.metadata.num_rows,
        "row_group_count": parquet_file.metadata.num_row_groups,
        "derived_columns": {
            "normalized_value": NORMALIZED_VALUE_FORMULA,
            "normalized_deviation": NORMALIZED_DEVIATION_FORMULA,
            "is_judgement_target": ("meta_type = 'spec' and not meta_ignore"),
            "is_ng": IS_NG_FORMULA,
            "ng_direction": "-1=lower NG, 0=OK or not target, 1=upper NG",
        },
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def add_judgement(
    input_path: Path,
    output_path: Path,
) -> tuple[int, Path]:
    """標準化済みParquetへの判定列付与。"""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    standardized_file = pq.ParquetFile(input_path)
    schema = _judged_schema(standardized_file.schema_arrow, input_path)
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
            "meta_unit",
        ],
        write_statistics=True,
        version="2.6",
    )
    try:
        for row_group in range(standardized_file.metadata.num_row_groups):
            standardized_table = standardized_file.read_row_group(row_group)
            table = _add_judgement_columns(standardized_table, schema)
            writer.write_table(table, row_group_size=table.num_rows)
            if (
                row_group + 1
            ) % 10 == 0 or row_group + 1 == standardized_file.metadata.num_row_groups:
                print(
                    "judged "
                    f"{row_group + 1:>3}/"
                    f"{standardized_file.metadata.num_row_groups} row groups"
                )
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    judged_file = pq.ParquetFile(temporary_path)
    if judged_file.metadata.num_rows != standardized_file.metadata.num_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("row count mismatch")
    if judged_file.metadata.num_row_groups != standardized_file.metadata.num_row_groups:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("row-group count mismatch")

    temporary_path.replace(output_path)
    manifest_path = _write_manifest(input_path, output_path, judged_file)
    return judged_file.metadata.num_rows, manifest_path


def parse_args() -> argparse.Namespace:
    """コマンドライン引数の定義。"""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            project_root / "data" / "standardized" / "quality_data_100lots.parquet"
        ),
        help="標準化済みParquetの入力パス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(project_root / "data" / "judged" / "quality_data_100lots.parquet"),
        help="判定済みParquetの出力パス",
    )
    return parser.parse_args()


def main() -> None:
    """判定列付与処理の実行。"""
    args = parse_args()
    row_count, manifest_path = add_judgement(args.input, args.output)
    print(f"wrote {row_count:,} rows to {args.output.resolve()}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
