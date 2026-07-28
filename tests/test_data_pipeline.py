from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.aggregated.build_quality_aggregates import aggregate
from src.judged.add_quality_judgement import add_judgement
from src.quality_repository import QualityRepository
from src.standardized.standardize_quality_data import standardize


def _write_raw_fixture(path: Path) -> None:
    """2 lotの小規模raw fixture保存。"""
    work_values = {
        ("LOT_A", 1): (-1.0, 5.0),
        ("LOT_A", 2): (10.0, 11.0),
        ("LOT_B", 1): (2.0, 8.0),
        ("LOT_B", 2): (4.0, 6.0),
    }
    foreign_values = {
        ("LOT_A", 1): (-1.0, 4.0),
        ("LOT_A", 2): (0.0, 5.0),
        ("LOT_B", 1): (1.0, 2.0),
        ("LOT_B", 2): (3.0, 4.0),
    }
    rows: list[dict[str, object]] = []
    for lot_index, lot_number in enumerate(("LOT_A", "LOT_B")):
        start = datetime(2026, 1, 1) + timedelta(  # noqa: DTZ001
            days=lot_index
        )
        for frame_no in (1, 2):
            for position_x in (1, 2):
                value_index = position_x - 1
                rows.extend(
                    [
                        {
                            "vision": "vision_1",
                            "lot_number": lot_number,
                            "lot_start_time": start,
                            "FrameNo": frame_no,
                            "PositionX": position_x,
                            "PositionY": 1,
                            "value": work_values[(lot_number, frame_no)][value_index],
                            "colname": "Work_Xw_v1",
                            "limmin": 0.0,
                            "limmax": 10.0,
                            "meta_type": "spec",
                            "meta_ignore": False,
                            "meta_best": None,
                            "meta_category": "PKGサイズ",
                            "meta_unit": "mm",
                        },
                        {
                            "vision": "vision_1",
                            "lot_number": lot_number,
                            "lot_start_time": start,
                            "FrameNo": frame_no,
                            "PositionX": position_x,
                            "PositionY": 1,
                            "value": foreign_values[(lot_number, frame_no)][
                                value_index
                            ],
                            "colname": "Foreign_Length_Long_v1",
                            "limmin": None,
                            "limmax": 4.0,
                            "meta_type": "spec",
                            "meta_ignore": False,
                            "meta_best": 0.0,
                            "meta_category": "異物",
                            "meta_unit": "mm",
                        },
                    ]
                )

    schema = pa.schema(
        [
            pa.field("vision", pa.string(), nullable=False),
            pa.field("lot_number", pa.string(), nullable=False),
            pa.field("lot_start_time", pa.timestamp("ns"), nullable=False),
            pa.field("FrameNo", pa.int16(), nullable=False),
            pa.field("PositionX", pa.int8(), nullable=False),
            pa.field("PositionY", pa.int8(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("colname", pa.string(), nullable=False),
            pa.field("limmin", pa.float64(), nullable=True),
            pa.field("limmax", pa.float64(), nullable=True),
            pa.field("meta_type", pa.string(), nullable=False),
            pa.field("meta_ignore", pa.bool_(), nullable=False),
            pa.field("meta_best", pa.float64(), nullable=True),
            pa.field("meta_category", pa.string(), nullable=False),
            pa.field("meta_unit", pa.string(), nullable=False),
        ],
        metadata={b"dataset_stage": b"raw", b"dataset_version": b"test"},
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(table.slice(0, 8))
        writer.write_table(table.slice(8, 8))


@pytest.fixture()
def pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    """全データ加工段階を通した小規模fixture。"""
    paths = {
        "raw": tmp_path / "raw.parquet",
        "standardized": tmp_path / "standardized.parquet",
        "judged": tmp_path / "judged.parquet",
        "analysis": tmp_path / "analysis",
    }
    _write_raw_fixture(paths["raw"])
    standardize(paths["raw"], paths["standardized"])
    add_judgement(paths["standardized"], paths["judged"])
    aggregate(paths["judged"], paths["analysis"], bins=4)
    return paths


def test_standardized_and_judged_columns(
    pipeline_paths: dict[str, Path],
) -> None:
    standardized = pq.read_table(pipeline_paths["standardized"]).to_pandas()
    judged_file = pq.ParquetFile(pipeline_paths["judged"])
    judged = judged_file.read().to_pandas()

    assert len(standardized) == len(judged) == 16
    assert judged_file.metadata.num_row_groups == 2
    assert judged_file.schema_arrow.metadata[b"dataset_stage"] == b"judged"

    work_lower = judged[
        (judged["lot_number"] == "LOT_A")
        & (judged["FrameNo"] == 1)
        & (judged["PositionX"] == 1)
        & (judged["colname"] == "Work_Xw_v1")
    ].iloc[0]
    assert work_lower["spec_position"] == pytest.approx(-1.2)
    assert work_lower["normalized_value"] == pytest.approx(-1.2)
    assert work_lower["normalized_deviation"] == pytest.approx(1.2)
    assert bool(work_lower["is_judgement_target"])
    assert bool(work_lower["is_ng"])
    assert work_lower["ng_direction"] == -1

    foreign_best_side = judged[
        (judged["lot_number"] == "LOT_A")
        & (judged["FrameNo"] == 1)
        & (judged["PositionX"] == 1)
        & (judged["colname"] == "Foreign_Length_Long_v1")
    ].iloc[0]
    assert foreign_best_side["spec_usage"] == pytest.approx(-0.25)
    assert foreign_best_side["normalized_deviation"] == pytest.approx(0.0)
    assert not bool(foreign_best_side["is_ng"])

    boundary = judged[
        (judged["lot_number"] == "LOT_A")
        & (judged["FrameNo"] == 2)
        & (judged["PositionX"] == 1)
        & (judged["colname"] == "Work_Xw_v1")
    ].iloc[0]
    assert boundary["value"] == 10.0
    assert not bool(boundary["is_ng"])
    assert boundary["ng_direction"] == 0


def test_grain_specific_aggregates(
    pipeline_paths: dict[str, Path],
) -> None:
    analysis_dir = pipeline_paths["analysis"]
    manifest = json.loads((analysis_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_row_count"] == 16
    assert manifest["kde_bins"] == 4
    assert manifest["files"]["lots"]["row_count"] == 2
    assert manifest["files"]["frame_item_stats"]["row_count"] == 8
    assert manifest["files"]["position_item_stats"]["row_count"] == 8
    assert manifest["files"]["piece_ng"]["row_count"] == 8

    frame = pq.read_table(analysis_dir / "frame_item_stats.parquet").to_pandas()
    work = frame[
        (frame["lot_number"] == "LOT_A")
        & (frame["FrameNo"] == 1)
        & (frame["colname"] == "Work_Xw_v1")
    ].iloc[0]
    assert work["total_count"] == 2
    assert work["sample_count"] == 2
    assert work["ng_count"] == 1
    assert work["ng_rate"] == pytest.approx(50.0)
    assert work["normalized_mean"] == pytest.approx(0.6)
    assert work["normalized_std"] == pytest.approx(0.6)
    assert work["minimum"] == -1.0
    assert work["p50"] == pytest.approx(2.0)
    assert work["maximum"] == 5.0

    foreign = frame[
        (frame["lot_number"] == "LOT_A")
        & (frame["FrameNo"] == 1)
        & (frame["colname"] == "Foreign_Length_Long_v1")
    ].iloc[0]
    assert foreign["ng_count"] == 0
    assert foreign["normalized_mean"] == pytest.approx(0.5)
    assert foreign["normalized_std"] == pytest.approx(0.625)

    histogram = pq.read_table(analysis_dir / "lot_item_histogram.parquet").to_pandas()
    assert histogram["count"].sum() == 16
    assert histogram.loc[histogram["bin_index"].isna(), "count"].sum() == 2

    piece = pq.read_table(analysis_dir / "piece_ng.parquet").to_pandas()
    both_ng = piece[
        (piece["lot_number"] == "LOT_A")
        & (piece["FrameNo"] == 2)
        & (piece["PositionX"] == 2)
    ].iloc[0]
    mapping = {
        item["colname"]: item["bit_value"] for item in manifest["ng_bit_mapping"]
    }
    expected_mask = mapping["Work_Xw_v1"] | mapping["Foreign_Length_Long_v1"]
    assert both_ng["ng_item_count"] == 2
    assert both_ng["ng_mask"] == expected_mask
    assert bool(both_ng["is_ng"])


def test_repository_reads_precomputed_data(
    pipeline_paths: dict[str, Path],
) -> None:
    repository = QualityRepository(pipeline_paths["analysis"])
    try:
        assert [lot[0] for lot in repository.lots()] == ["LOT_A", "LOT_B"]
        assert len(repository.ng_rate_by_frame()) == 8
        assert len(repository.metrics_by_colname_position()) == 8
        assert repository.piece_ng_masks(
            ("Foreign_Length_Long_v1", "Work_Xw_v1")
        ).shape == (8,)
        assert len(repository.kde_bins_by_colname_lot(4)) > 0

        detail = repository.values_by_colname_frame(
            "LOT_A",
            1,
            "Work_Xw_v1",
        )
        assert len(detail) == 2
        assert detail["ng_direction"].tolist() == [-1, 0]
    finally:
        repository.close()
