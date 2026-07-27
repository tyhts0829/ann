from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from dashboard import STYLESHEET, DashboardWindow
from src.analysis.fq_map import (
    DETAIL_DIVIDER_HEIGHT,
    FMAP_SECTION_HEIGHT,
    FQ_MAP_SECTION_HEIGHT,
    FQ_MAP_SEPARATOR_HEIGHT,
    FQ_MAP_TOP_AXIS_HEIGHT,
    KDE_SECTION_HEIGHT,
    QUALITY_TREND_SECTION_HEIGHT,
    FqMapPlotWidget,
    FqMapWidget,
    build_fq_map_data,
)
from src.analysis.frame_map import (
    DEFAULT_ROW_HEIGHT,
    LOWER_NG_RGBA,
    UPPER_NG_RGBA,
    VISIBLE_LOTS,
    build_frame_map_data,
    build_ng_overlay,
    build_single_frame_map_data,
)
from src.analysis.kde import build_kde_data
from src.analysis.map_palettes import BLUES, MAP_DEFINITIONS
from src.analysis.plot_style import (
    LOT_SEPARATOR_COLOR,
    LOT_SEPARATOR_WIDTH,
)
from src.analysis.processing_path import (
    EQUIPMENT_PATHS_PATH,
    EquipmentPaths,
    ProcessingPath,
    build_processing_path_series,
    load_equipment_path_catalog,
)
from src.analysis.single_frame_kde import (
    build_single_frame_kde_data,
)
from src.dashboard_config import (
    CONFIG_PATH,
    DASHBOARD_CONFIG,
    load_dashboard_config,
)
from src.raw.generate_quality_data import (
    DEFAULT_SEED,
    MEASUREMENTS,
    _base_grid,
    _generate_values,
    _long_metadata,
    _perlin_lot_factors,
    _schema,
    _to_table,
)
from src.raw.generate_quality_data import (
    _write_manifest as write_raw_manifest,
)
from src.standardized.quality_data import QualityRepository
from src.standardized.standardize_quality_data import standardize

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "standardized"
    / "quality_data_100lots.parquet"
)
RAW_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "quality_data_100lots.parquet"
)


@pytest.fixture(scope="module")
def repository() -> QualityRepository:
    return QualityRepository(DATA_PATH)


def test_all_lot_fq_map(repository: QualityRepository) -> None:
    data = build_fq_map_data(repository)

    assert data.ng_rates.shape == (45, 2_400)
    assert data.normalized_mean.shape == (45, 2_400)
    assert data.normalized_std.shape == (45, 2_400)
    assert data.lot_count == 100
    assert data.lot_start_times[0] == datetime(2026, 6, 1, 8, 11)
    assert data.lot_start_times[-1] == datetime(
        2026,
        7,
        20,
        20,
        20,
        3,
    )
    assert data.total_measurements == 31_104_000
    assert data.total_ng == 78_126
    assert data.total_pieces == 691_200
    assert data.ng_pieces == 37_989
    assert data.ok_pieces == 653_211
    assert data.piece_yield == pytest.approx(94.50390625)
    assert data.frame_numbers.tolist()[:25] == [*range(1, 25), 1]
    assert len(set(data.colnames)) == 45
    assert all(
        colname.endswith(("_v1", "_v2", "_v3")) for colname in data.colnames
    )
    assert set(data.visions) == {"vision_1", "vision_2", "vision_3"}
    assert data.filter_vision("vision_2").ng_rates.shape == (15, 2_400)
    assert (
        data.filter_vision("vision_2").lot_start_times == data.lot_start_times
    )
    assert data.filter_rows("異物", "vision_2").ng_rates.shape == (
        3,
        2_400,
    )
    assert np.isfinite(data.ng_rates).all()
    assert set(data.categories) == {
        "異物",
        "リード",
        "PKGサイズ",
        "標印",
        "欠陥",
    }
    work_x = data.colnames.index("Work_Xw_v1")
    foreign = data.colnames.index("Foreign_Length_Long_v1")
    assert data.normalized_mean[work_x, 0] == pytest.approx(
        0.10415277777777814
    )
    assert data.normalized_mean[foreign, 0] == pytest.approx(
        0.06663888888888889
    )
    assert np.nanmax(data.normalized_std) > 0.0


def test_raw_and_standardized_spec_columns() -> None:
    raw_file = pq.ParquetFile(RAW_DATA_PATH)
    standardized_file = pq.ParquetFile(DATA_PATH)

    assert raw_file.metadata.num_rows == 31_104_000
    assert standardized_file.metadata.num_rows == 31_104_000
    assert "spec_position" not in raw_file.schema_arrow.names
    assert "spec_usage" not in raw_file.schema_arrow.names
    assert "spec_position" in standardized_file.schema_arrow.names
    assert "spec_usage" in standardized_file.schema_arrow.names
    assert raw_file.schema_arrow.metadata[b"dataset_stage"] == b"raw"
    assert (
        raw_file.schema_arrow.metadata[b"baseline_noise"]
        == b"fractal_perlin_3d"
    )
    assert raw_file.schema_arrow.metadata[b"perlin_period"] == b"4096"
    assert raw_file.schema_arrow.metadata[b"visions_per_product"] == b"3"
    assert (
        standardized_file.schema_arrow.metadata[b"dataset_stage"]
        == b"standardized"
    )

    frame = standardized_file.read_row_group(
        0,
        columns=[
            "value",
            "limmin",
            "limmax",
            "meta_best",
            "spec_position",
            "spec_usage",
        ],
    ).to_pandas()
    two_sided = frame["limmin"].notna() & frame["limmax"].notna()
    one_sided = frame["meta_best"].notna() & frame["limmax"].notna()

    expected_position = (
        frame.loc[two_sided, "value"]
        - (frame.loc[two_sided, "limmin"] + frame.loc[two_sided, "limmax"])
        / 2.0
    ) / (
        (frame.loc[two_sided, "limmax"] - frame.loc[two_sided, "limmin"]) / 2.0
    )
    expected_usage = (
        frame.loc[one_sided, "value"] - frame.loc[one_sided, "meta_best"]
    ) / (frame.loc[one_sided, "limmax"] - frame.loc[one_sided, "meta_best"])

    assert np.allclose(
        frame.loc[two_sided, "spec_position"],
        expected_position,
    )
    assert np.allclose(
        frame.loc[one_sided, "spec_usage"],
        expected_usage,
    )
    assert frame.loc[two_sided, "spec_usage"].isna().all()
    assert frame.loc[one_sided, "spec_position"].isna().all()


def test_meta_unit_generation_and_standardization(
    tmp_path: Path,
) -> None:
    expected_units = {
        "Foreign_Length_Long": "mm",
        "Foreign_Length_Short": "mm",
        "Foreign_Size": "mm²",
        "Lead_Length_L": "mm",
        "Lead_Length_R": "mm",
        "Lead_Pitch": "mm",
        "Work_Xw": "mm",
        "Work_Yw": "mm",
        "Work_Center_X": "mm",
        "Work_Center_Y": "mm",
        "Mark_Center_X": "mm",
        "Mark_Center_Y": "mm",
        "Defect_Length_Long": "mm",
        "Defect_Length_Short": "mm",
        "Defect_Size": "mm²",
    }
    assert {
        measurement["colname"]: measurement["meta_unit"]
        for measurement in MEASUREMENTS
    } == expected_units

    schema = _schema(DEFAULT_SEED, 1)
    assert schema.field("meta_unit").type == pa.string()
    assert not schema.field("meta_unit").nullable
    assert schema.metadata[b"dataset_version"] == b"4.0"

    grid = {name: values[:1] for name, values in _base_grid().items()}
    long_metadata = _long_metadata(grid)
    raw_table = _to_table(
        "LOT_TEST",
        datetime(2026, 1, 1),
        grid,
        np.zeros((1, len(MEASUREMENTS))),
        long_metadata,
        schema,
    )
    assert raw_table.column("meta_unit").to_pylist() == list(
        expected_units.values()
    )

    raw_path = tmp_path / "raw.parquet"
    standardized_path = tmp_path / "standardized.parquet"
    pq.write_table(
        raw_table,
        raw_path,
        use_dictionary=["meta_unit"],
    )
    raw_manifest_path = write_raw_manifest(
        raw_path,
        DEFAULT_SEED,
        1,
        raw_table.num_rows,
    )
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    assert raw_manifest["meta_units"] == expected_units

    _, standardized_manifest_path = standardize(
        raw_path,
        standardized_path,
    )
    standardized_file = pq.ParquetFile(standardized_path)
    standardized_table = standardized_file.read()
    assert (
        standardized_table.column("meta_unit").to_pylist()
        == raw_table.column("meta_unit").to_pylist()
    )
    assert (
        standardized_file.schema_arrow.metadata[b"dataset_version"] == b"4.1"
    )
    meta_unit_index = standardized_file.schema_arrow.get_field_index(
        "meta_unit"
    )
    encodings = (
        standardized_file.metadata.row_group(0)
        .column(meta_unit_index)
        .encodings
    )
    assert "RLE_DICTIONARY" in encodings
    standardized_manifest = json.loads(
        standardized_manifest_path.read_text(encoding="utf-8")
    )
    assert standardized_manifest["metadata_columns"] == ["meta_unit"]


@pytest.mark.parametrize("path", [RAW_DATA_PATH, DATA_PATH])
def test_foreign_and_defect_values_are_nonnegative(path: Path) -> None:
    with duckdb.connect() as connection:
        result = connection.execute(
            """
                SELECT min(value), count_if(value < 0.0)
                FROM read_parquet(?)
                WHERE meta_category IN ('異物', '欠陥')
            """,
            [str(path)],
        ).fetchone()
        assert result is not None
        minimum, negative_count = result

    assert minimum == 0.0
    assert negative_count == 0


def test_generator_keeps_foreign_and_defect_nonnegative() -> None:
    grid = _base_grid()
    lot_factors = _perlin_lot_factors(100, DEFAULT_SEED)
    columns = [
        index
        for index, measurement in enumerate(MEASUREMENTS)
        if measurement["meta_category"] in {"異物", "欠陥"}
    ]

    for lot_index in (0, 58, 99):
        values = _generate_values(
            lot_index,
            grid,
            lot_factors[lot_index],
            DEFAULT_SEED,
        )
        assert np.all(values[:, columns] >= 0.0)


def test_data_pipeline_script_layout() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "raw" / "generate_quality_data.py").is_file()
    assert (
        root / "src" / "standardized" / "standardize_quality_data.py"
    ).is_file()
    assert not (root / "scripts" / "generate_quality_data.py").exists()
    assert (root / "src" / "analysis" / "fq_map.py").is_file()
    assert (root / "src" / "analysis" / "frame_map.py").is_file()
    assert not (root / "src" / "analysis" / "ng_rate_heatmap.py").exists()


def test_dashboard_config() -> None:
    config = load_dashboard_config(CONFIG_PATH)

    assert config == DASHBOARD_CONFIG
    assert VISIBLE_LOTS == config.visible_lots
    assert FQ_MAP_SECTION_HEIGHT == config.fqmap_height
    assert FMAP_SECTION_HEIGHT == config.fmap_height
    assert KDE_SECTION_HEIGHT == config.kde_height
    assert QUALITY_TREND_SECTION_HEIGHT == (config.quality_trend_height)
    assert config.fqmap_min_cell_height == 14


def test_fq_map_deviation_palette() -> None:
    assert BLUES == (
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    )
    assert MAP_DEFINITIONS[1] == (
        "normalized_mean",
        "規格逸脱度 平均",
        BLUES,
    )


def test_deviation_color_range_uses_nice_upper() -> None:
    assert FqMapWidget._nice_upper(0.278, minimum=0.1) == pytest.approx(0.3)
    assert FqMapWidget._nice_upper(1.2, minimum=0.1) == pytest.approx(2.0)


def test_latest_lot_fq_map(repository: QualityRepository) -> None:
    data = build_fq_map_data(repository, ("LOT_20260720_B",))

    assert data.ng_rates.shape == (45, 24)
    assert data.lot_count == 1
    assert data.lot_start_times == (datetime(2026, 7, 20, 20, 20, 3),)
    assert data.total_measurements == 311_040
    assert data.total_ng == 3_574
    assert data.total_pieces == 6_912


def test_frame_ticks_keep_lot_boundaries_readable() -> None:
    assert FqMapWidget._frame_ticks(0, 2) == [
        (0.0, "1"),
        (5.0, "6"),
        (11.0, "12"),
        (17.0, "18"),
        (23.5, "24/1"),
        (29.0, "6"),
        (35.0, "12"),
        (41.0, "18"),
        (47.0, "24"),
    ]


def test_fq_map_wheel_scrolls_by_rows(qtbot: QtBot) -> None:
    class WheelEvent:
        def __init__(self, angle_y: int = 0, pixel_y: int = 0) -> None:
            self._angle_delta = QtCore.QPoint(0, angle_y)
            self._pixel_delta = QtCore.QPoint(0, pixel_y)
            self.accepted = False

        def angleDelta(self) -> QtCore.QPoint:
            return self._angle_delta

        def pixelDelta(self) -> QtCore.QPoint:
            return self._pixel_delta

        def accept(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.accepted = False

    plot = FqMapPlotWidget()
    scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    qtbot.addWidget(plot)
    qtbot.addWidget(scrollbar)
    scrollbar.setRange(0, 3)
    plot.set_vertical_scrollbar(scrollbar)

    def send_wheel(*, angle_y: int = 0, pixel_y: int = 0) -> WheelEvent:
        event = WheelEvent(angle_y, pixel_y)
        plot.wheelEvent(cast(QtGui.QWheelEvent, event))
        return event

    first_half_step = send_wheel(angle_y=-20)
    assert first_half_step.accepted
    assert scrollbar.value() == 0

    second_half_step = send_wheel(angle_y=-20)
    assert second_half_step.accepted
    assert scrollbar.value() == 1

    scrollbar.setValue(scrollbar.minimum())
    beyond_top = send_wheel(angle_y=40)
    assert not beyond_top.accepted
    assert scrollbar.value() == scrollbar.minimum()

    scrollbar.setValue(scrollbar.maximum())
    beyond_bottom = send_wheel(angle_y=-40)
    assert not beyond_bottom.accepted
    assert scrollbar.value() == scrollbar.maximum()

    scrollbar.setRange(0, 0)
    fixed_range = send_wheel(angle_y=-40)
    assert not fixed_range.accepted
    assert scrollbar.value() == 0

    scrollbar.setRange(0, 3)
    pixel_step = send_wheel(angle_y=120, pixel_y=-14)
    assert pixel_step.accepted
    assert scrollbar.value() == 1


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (0, (0, 1)),
        (23, (0, 24)),
        (24, (1, 1)),
        (191, (7, 24)),
        (192, (8, 1)),
        (2_399, (99, 24)),
    ],
)
def test_fq_column_to_lot_and_frame(
    column: int,
    expected: tuple[int, int],
) -> None:
    assert FqMapWidget._lot_frame_from_column(column) == expected


def test_frame_map_data(repository: QualityRepository) -> None:
    data = build_frame_map_data(repository)

    assert data.ng_rates.shape == (45, 100, 12, 24)
    assert data.normalized_mean.shape == (45, 100, 12, 24)
    assert data.normalized_std.shape == (45, 100, 12, 24)
    assert data.colnames[:6] == (
        "Foreign_Length_Long_v1",
        "Foreign_Length_Long_v2",
        "Foreign_Length_Long_v3",
        "Foreign_Length_Short_v1",
        "Foreign_Length_Short_v2",
        "Foreign_Length_Short_v3",
    )
    assert data.colnames[-1] == "Defect_Size_v3"
    assert np.isfinite(data.ng_rates).all()
    assert np.std(data.ng_rates[-1, -1]) > 1.0
    work_x = data.colname_index("Work_Xw_v1")
    foreign = data.colname_index("Foreign_Length_Long_v1")
    assert data.normalized_mean[work_x, 0, 0, 0] == pytest.approx(
        0.19395833333333432
    )
    assert data.normalized_mean[foreign, 0, 0, 0] == pytest.approx(
        0.037125
    )


def test_single_frame_map_data(repository: QualityRepository) -> None:
    lot_number = "LOT_20260601_A"
    frame_no = 1
    colname = "Foreign_Length_Long_v1"
    data = build_single_frame_map_data(
        repository,
        lot_number,
        frame_no,
        colname,
    )
    source = repository.values_by_colname_frame(
        lot_number,
        frame_no,
        colname,
    )

    assert len(source) == 288
    assert len(source[["PositionX", "PositionY"]].drop_duplicates()) == 288
    assert (source["PositionX"].min(), source["PositionX"].max()) == (1, 24)
    assert (source["PositionY"].min(), source["PositionY"].max()) == (1, 12)
    assert data.sample_count == 288
    assert data.raw_values.shape == (12, 24)
    assert data.normalized_values.shape == (12, 24)
    assert data.ng_flags.shape == (12, 24)

    expected_raw = source["value"].to_numpy(dtype=float).reshape(12, 24)
    expected_normalized = (
        source["normalized_value"].to_numpy(dtype=float).reshape(12, 24)
    )
    expected_ng = source["is_ng"].to_numpy(dtype=bool).reshape(12, 24)
    assert np.allclose(data.raw_values, expected_raw)
    assert np.allclose(data.normalized_values, expected_normalized)
    assert np.array_equal(data.ng_flags, expected_ng)

    assert int(data.ng_flags.sum()) == 4
    assert np.argwhere(data.ng_flags).tolist() == [
        [0, 20],
        [2, 2],
        [3, 0],
        [5, 1],
    ]
    assert data.raw_values[0, 20] == pytest.approx(0.3868)
    assert data.normalized_values[0, 20] == pytest.approx(1.2893333333333332)
    assert bool(data.ng_flags[0, 20])


def test_single_frame_ng_overlay_encodes_ng_direction(
    repository: QualityRepository,
) -> None:
    data = build_single_frame_map_data(
        repository,
        "LOT_20260707_A",
        1,
        "Lead_Pitch_v1",
    )
    overlay = build_ng_overlay(data)

    assert data.spec_lower == pytest.approx(2.49)
    assert data.spec_upper == pytest.approx(2.59)
    assert int(data.lower_ng_flags.sum()) == 22
    assert int(data.upper_ng_flags.sum()) == 3
    assert data.raw_values[0, 0] == pytest.approx(2.4727)
    assert data.raw_values[0, 22] == pytest.approx(2.5920)
    assert tuple(overlay[0, 0]) == LOWER_NG_RGBA
    assert tuple(overlay[0, 22]) == UPPER_NG_RGBA
    assert np.all(overlay[data.lower_ng_flags] == LOWER_NG_RGBA)
    assert np.all(overlay[data.upper_ng_flags] == UPPER_NG_RGBA)
    assert np.all(overlay[~data.ng_flags, 3] == 0)


def test_single_frame_kde_uses_selected_frame(
    repository: QualityRepository,
) -> None:
    frame_data = build_single_frame_map_data(
        repository,
        "LOT_20260601_A",
        6,
        "Work_Xw_v2",
    )
    data = build_single_frame_kde_data(frame_data)

    assert data.colname == "Work_Xw_v2"
    assert data.frame_no == 6
    assert data.sample_count == 288
    assert data.x_values.shape == (DASHBOARD_CONFIG.kde_bins,)
    assert data.density.shape == (DASHBOARD_CONFIG.kde_bins,)
    bin_width = data.x_values[1] - data.x_values[0]
    assert data.density.sum() * bin_width == pytest.approx(
        data.in_range_count / data.sample_count
    )


def test_processing_path_catalog_and_xy_series() -> None:
    catalog = load_equipment_path_catalog()

    assert EQUIPMENT_PATHS_PATH.suffix == ".toml"
    assert EQUIPMENT_PATHS_PATH.is_file()
    assert len(catalog.equipments) == 2
    assert catalog.default_equipment.id == "row_serpentine"
    mold = catalog.equipment("mold_4flow")
    assert [path.id for path in mold.paths] == [
        "flow_1",
        "flow_2",
        "flow_3",
        "flow_4",
    ]
    assert all(len(path.positions) == 72 for path in mold.paths)

    raw_values = np.arange(12 * 24, dtype=float).reshape(12, 24)
    ng_directions = np.zeros((12, 24), dtype=np.int8)
    ng_directions[0, 0] = -1
    ng_directions[1, 1] = 1
    equipment = EquipmentPaths(
        id="xy_test",
        label="XYテスト",
        paths=(
            ProcessingPath(
                id="path",
                label="パス",
                positions=((1, 1), (2, 1), (2, 2)),
            ),
        ),
    )
    (series,) = build_processing_path_series(
        raw_values,
        ng_directions,
        equipment,
    )
    assert series.steps.tolist() == [1, 2, 3]
    assert series.position_x.tolist() == [1, 2, 2]
    assert series.position_y.tolist() == [1, 1, 2]
    assert series.values.tolist() == [0.0, 1.0, 25.0]
    assert series.ng_directions.tolist() == [-1, 0, 1]

    mold_series = build_processing_path_series(
        raw_values,
        ng_directions,
        mold,
    )
    assert [series.path_id for series in mold_series] == [
        "flow_1",
        "flow_2",
        "flow_3",
        "flow_4",
    ]
    assert len(mold_series) == 4


def test_kde_data(repository: QualityRepository) -> None:
    data = build_kde_data(repository)

    assert data.x_values.shape == (
        45,
        DASHBOARD_CONFIG.kde_bins,
    )
    assert data.densities.shape == (
        45,
        100,
        DASHBOARD_CONFIG.kde_bins,
    )
    assert data.sample_counts.shape == (45, 100)
    assert data.in_range_counts.shape == (45, 100)
    assert np.all(data.sample_counts == 6_912)
    assert np.all(data.in_range_counts <= data.sample_counts)
    assert int(data.sample_counts.sum()) == 31_104_000
    assert np.all(np.diff(data.x_values, axis=1) > 0)
    assert np.isfinite(data.densities).all()
    assert np.all(data.densities >= 0.0)

    bin_widths = data.x_values[:, 1] - data.x_values[:, 0]
    integrals = data.densities.sum(axis=2) * bin_widths[:, None]
    coverage = data.in_range_counts / data.sample_counts
    assert np.allclose(integrals, coverage)

    work_x = data.colname_index("Work_Xw_v2")
    assert data.spec_lower[work_x] == pytest.approx(3.9)
    assert data.spec_upper[work_x] == pytest.approx(4.1)
    assert data.display_min[work_x] == pytest.approx(3.85)
    assert data.display_max[work_x] == pytest.approx(4.15)

    foreign = data.colname_index("Foreign_Length_Long_v1")
    assert np.isnan(data.spec_lower[foreign])
    assert data.spec_best[foreign] == pytest.approx(0.0)
    assert data.spec_upper[foreign] == pytest.approx(0.3)
    assert data.display_min[foreign] == pytest.approx(0.0)
    assert data.display_max[foreign] == pytest.approx(0.36)

    two_sided = np.isfinite(data.spec_lower)
    widths = data.display_max - data.display_min
    centers = (data.spec_lower + data.spec_upper) / 2.0
    assert np.allclose(
        (data.spec_lower[two_sided] - data.display_min[two_sided])
        / widths[two_sided],
        1 / 6,
    )
    assert np.allclose(
        (centers[two_sided] - data.display_min[two_sided]) / widths[two_sided],
        1 / 2,
    )
    assert np.allclose(
        (data.spec_upper[two_sided] - data.display_min[two_sided])
        / widths[two_sided],
        5 / 6,
    )

    upper_only = ~two_sided & np.isfinite(data.spec_upper)
    assert np.all(data.display_min[upper_only] >= 0.0)
    assert np.allclose(
        (data.spec_upper[upper_only] - data.display_min[upper_only])
        / widths[upper_only],
        5 / 6,
    )


def test_dashboard_window(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    repository: QualityRepository,
) -> None:
    window = DashboardWindow(repository)
    qtbot.addWidget(window)
    assert window.fq_map is None
    assert window.loading_progress.minimum() == 0
    assert window.loading_progress.maximum() == 0
    with qtbot.waitSignal(window.data_loaded, timeout=15_000):
        window.show()
    qtbot.wait(50)
    fq_map = cast(FqMapWidget, window.fq_map)
    assert window.dashboard_scroll_area is not None
    assert window.dashboard_scroll_area.widget() is fq_map

    assert window.windowTitle() == "Quality Dashboard"
    section_texts = {
        label.text()
        for label in fq_map.findChildren(QtWidgets.QLabel)
        if label.objectName() == "mapSectionTitle"
    }
    assert section_texts == {
        "FQmap",
        "Fmap",
        "KDE",
        "F推移",
        "加工パス推移",
    }
    assert len(fq_map.views) == 3
    assert len(fq_map.frame_map.rows) == 3
    plots_layout = fq_map.plots_container.layout()
    assert plots_layout is not None
    assert plots_layout.spacing() == 0
    assert len(fq_map.fq_map_separators) == 2
    assert all(
        separator.height() == FQ_MAP_SEPARATOR_HEIGHT
        for separator in fq_map.fq_map_separators
    )
    fq_map_parts = [
        fq_map.views[0].card,
        fq_map.fq_map_separators[0],
        fq_map.views[1].card,
        fq_map.fq_map_separators[1],
        fq_map.views[2].card,
    ]
    assert all(
        current.geometry().top() == previous.geometry().bottom() + 1
        for previous, current in zip(fq_map_parts, fq_map_parts[1:])
    )
    assert fq_map.fq_map_section.height() == FQ_MAP_SECTION_HEIGHT
    assert fq_map.fmap_section.height() == FMAP_SECTION_HEIGHT
    assert fq_map.fq_map_section.isAncestorOf(fq_map.toolbar)
    assert (
        len(
            [
                frame
                for frame in fq_map.findChildren(QtWidgets.QFrame)
                if frame.objectName() == "mapCard"
            ]
        )
        == 1
    )
    detail_groups = [
        frame
        for frame in fq_map.findChildren(QtWidgets.QFrame)
        if frame.objectName() == "detailGroup"
    ]
    assert detail_groups == [fq_map.detail_section]
    assert fq_map.detail_mode == "lot"
    assert fq_map.single_frame_data is None
    assert fq_map.detail_stack.currentWidget() is fq_map.lot_detail_page
    lot_detail_layout = fq_map.lot_detail_page.layout()
    assert isinstance(lot_detail_layout, QtWidgets.QVBoxLayout)
    lot_detail_widgets = []
    for index in (0, 2, 4):
        layout_item = lot_detail_layout.itemAt(index)
        assert layout_item is not None
        widget = layout_item.widget()
        assert widget is not None
        lot_detail_widgets.append(widget)
    assert lot_detail_widgets == [
        fq_map.quality_trend_section,
        fq_map.fmap_section,
        fq_map.kde_section,
    ]
    assert fq_map.detail_section.isAncestorOf(fq_map.fmap_section)
    assert fq_map.detail_section.isAncestorOf(fq_map.kde_section)
    assert fq_map.detail_section.isAncestorOf(fq_map.quality_trend_section)
    assert not fq_map.fq_map_section.isAncestorOf(fq_map.detail_section)
    assert fq_map.detail_divider.height() == DETAIL_DIVIDER_HEIGHT
    assert fq_map.quality_trend_divider.height() == DETAIL_DIVIDER_HEIGHT
    assert fq_map.lot_detail_page.contentsRect().contains(
        fq_map.fmap_section.geometry()
    )
    assert fq_map.lot_detail_page.contentsRect().contains(
        fq_map.kde_section.geometry()
    )
    assert fq_map.lot_detail_page.contentsRect().contains(
        fq_map.quality_trend_section.geometry()
    )
    assert fq_map.kde_section.height() == KDE_SECTION_HEIGHT
    assert (
        fq_map.quality_trend_section.height() == QUALITY_TREND_SECTION_HEIGHT
    )
    assert fq_map.kde.current_colname == "Foreign_Length_Long_v1"
    assert fq_map.quality_trend.current_colname == ("Foreign_Length_Long_v1")
    overview_badge = fq_map.findChild(
        QtWidgets.QLabel,
        "overviewBadge",
    )
    assert overview_badge is not None
    assert overview_badge.text() == "全検査項目の俯瞰"
    detail_badge = fq_map.findChild(
        QtWidgets.QLabel,
        "detailBadge",
    )
    assert detail_badge is not None
    assert detail_badge.text() == "選択項目の詳細"
    assert (
        len(
            fq_map.detail_section.findChildren(
                QtWidgets.QLabel,
                "frameMapSelectionLabel",
            )
        )
        == 1
    )
    assert len(fq_map.kde.plot_widgets) == VISIBLE_LOTS
    assert (
        fq_map.kde.current_lot_numbers
        == (fq_map.full_data.lot_numbers[-VISIBLE_LOTS:])
    )
    assert (
        fq_map.quality_trend.current_lot_numbers
        == (fq_map.full_data.lot_numbers[-VISIBLE_LOTS:])
    )
    initial_row = fq_map.kde.data.colname_index("Foreign_Length_Long_v1")
    initial_first_lot = len(fq_map.kde.data.lot_numbers) - VISIBLE_LOTS
    for offset, curve in enumerate(fq_map.kde.curve_items):
        x_values, densities = curve.getData()
        assert np.allclose(
            x_values,
            fq_map.kde.data.x_values[initial_row],
        )
        assert np.allclose(
            densities,
            fq_map.kde.data.densities[
                initial_row,
                initial_first_lot + offset,
            ],
        )
    assert all(not line.isVisible() for line in fq_map.kde.lower_lines)
    assert all(not line.isVisible() for line in fq_map.kde.center_lines)
    assert all(line.isVisible() for line in fq_map.kde.upper_lines)
    assert all(
        line.value() == pytest.approx(0.3) for line in fq_map.kde.upper_lines
    )
    assert not any(
        isinstance(item, pg.BarGraphItem)
        for plot_item in fq_map.kde.plot_items
        for item in plot_item.items
    )
    assert all(
        not plot_item.titleLabel.isVisible()
        for plot_item in fq_map.kde.plot_items
    )
    assert "各 6,912測定" in fq_map.kde.summary_label.text()
    assert "最良値 0" in fq_map.kde.summary_label.text()
    assert all(
        plot_item.viewRange()[0] == pytest.approx([0.0, 0.36])
        for plot_item in fq_map.kde.plot_items
    )
    assert (0.3 - 0.0) / (0.36 - 0.0) == pytest.approx(5 / 6)
    assert fq_map.vertical_scroll_area.parentWidget() is fq_map.fq_map_section
    assert not fq_map.plots_container.isAncestorOf(fq_map.frame_map)
    assert fq_map.findChildren(QtWidgets.QScrollArea) == [
        fq_map.vertical_scroll_area
    ]
    assert all(
        view.image_item.image.shape == (45, 2_400) for view in fq_map.views
    )
    expected_blues = np.asarray(
        [QtGui.QColor(color).getRgb() for color in BLUES],
        dtype=np.ubyte,
    )
    deviation_view = fq_map.views[1]
    assert deviation_view.metric == "normalized_mean"
    assert deviation_view.label == "規格逸脱度 平均"
    assert deviation_view.color_bar.levels() == pytest.approx((0.0, 0.3))
    assert np.array_equal(
        deviation_view.image_item.getColorMap().getColors(
            pg.ColorMap.BYTE
        ),
        expected_blues,
    )
    deviation_row = fq_map.frame_map.rows[1]
    assert deviation_row.metric == "normalized_mean"
    assert deviation_row.label == "規格逸脱度 平均"
    assert deviation_row.levels == pytest.approx((0.0, 0.2))
    assert np.array_equal(
        deviation_row.image_items[0].getColorMap().getColors(
            pg.ColorMap.BYTE
        ),
        expected_blues,
    )
    assert "逸脱度 0–0.3" in fq_map.scale_label.text()
    assert fq_map.scale_label.toolTip() == (
        "規格逸脱度: 個々の測定では"
        "0=最良・規格中心 / 1=規格限界 / 1超=NG。"
        "表示値はセル内の平均"
    )
    assert all(
        len(row.image_items) == VISIBLE_LOTS
        and all(image.image.shape == (12, 24) for image in row.image_items)
        for row in fq_map.frame_map.rows
    )
    assert all(
        len(row.lot_separators) == VISIBLE_LOTS - 1
        for row in fq_map.frame_map.rows
    )
    assert len(fq_map.kde.lot_separators) == VISIBLE_LOTS - 1
    assert len(fq_map.quality_trend.lot_separators) == 99
    plot_separators = [
        *fq_map.views[0].separators,
        *fq_map.quality_trend.lot_separators,
    ]
    assert {
        separator.pen.color().name()
        for separator in plot_separators
    } == {LOT_SEPARATOR_COLOR}
    assert {
        separator.pen.widthF()
        for separator in plot_separators
    } == {LOT_SEPARATOR_WIDTH}
    widget_separators = [
        *fq_map.frame_map.rows[0].lot_separators,
        *fq_map.kde.lot_separators,
    ]
    assert all(
        separator.objectName() == "lotSeparator"
        and separator.width() == round(LOT_SEPARATOR_WIDTH)
        and separator.line_color == LOT_SEPARATOR_COLOR
        and separator.line_width == LOT_SEPARATOR_WIDTH
        for separator in widget_separators
    )
    assert all(
        separator.bottom_margin == 0
        for separator in fq_map.frame_map.rows[0].lot_separators
    )
    assert all(
        separator.bottom_margin == 28
        for separator in fq_map.kde.lot_separators
    )
    assert not fq_map.frame_map.frame_mode_button.isEnabled()
    fq_lot_width = (
        fq_map.views[0].plot_item.getViewBox().sceneBoundingRect().width()
        / VISIBLE_LOTS
    )
    fmap_lot_width = (
        fq_map.frame_map.rows[0]
        .plot_widgets[0]
        .getPlotItem()
        .getViewBox()
        .sceneBoundingRect()
        .width()
    )
    kde_lot_width = (
        fq_map.kde.plot_items[0].getViewBox().sceneBoundingRect().width()
    )
    trend_lot_width = (
        fq_map.quality_trend.plot_item.getViewBox().sceneBoundingRect().width()
        / VISIBLE_LOTS
    )
    assert fmap_lot_width == pytest.approx(fq_lot_width, rel=0.01)
    assert kde_lot_width == pytest.approx(fmap_lot_width, rel=0.01)
    assert trend_lot_width == pytest.approx(fq_lot_width, rel=0.01)
    fq_plot_x = (
        fq_map.views[0].plot_widget.mapToGlobal(QtCore.QPoint(0, 0)).x()
        + fq_map.views[0].plot_item.getViewBox().sceneBoundingRect().left()
    )
    fmap_plot_x = (
        fq_map.frame_map.rows[0]
        .plot_widgets[0]
        .mapToGlobal(QtCore.QPoint(0, 0))
        .x()
        + fq_map.frame_map.rows[0]
        .plot_widgets[0]
        .getPlotItem()
        .getViewBox()
        .sceneBoundingRect()
        .left()
    )
    kde_plot_x = (
        fq_map.kde.plot_widgets[0].mapToGlobal(QtCore.QPoint(0, 0)).x()
        + fq_map.kde.plot_items[0].getViewBox().sceneBoundingRect().left()
    )
    trend_plot_x = (
        fq_map.quality_trend.plot_widget.mapToGlobal(QtCore.QPoint(0, 0)).x()
        + fq_map.quality_trend.plot_item.getViewBox()
        .sceneBoundingRect()
        .left()
    )
    assert fmap_plot_x == pytest.approx(fq_plot_x, abs=2.0)
    assert kde_plot_x == pytest.approx(fmap_plot_x, abs=2.0)
    assert trend_plot_x == pytest.approx(fq_plot_x, abs=2.0)
    grid_lines = [
        item
        for item in fq_map.frame_map.rows[0]
        .plot_widgets[0]
        .getPlotItem()
        .items
        if isinstance(item, pg.InfiniteLine)
    ]
    assert len(grid_lines) == 34
    assert {line.pen.color().name() for line in grid_lines} == {"#ffffff"}
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[-1].image,
        fq_map.frame_map.data.ng_rates[0, -1],
    )
    assert "border-radius" not in STYLESHEET
    assert "background: #f3f5f7" in STYLESHEET
    assert (
        "QFrame#chartCard {\n    background: #ffffff;\n    border: none;"
        in (STYLESHEET)
    )
    for view in fq_map.views:
        left_axis = view.plot_item.getAxis("left")
        assert not left_axis.label.isVisible()
        assert left_axis.grid is False
    top_axis = fq_map.views[0].plot_item.getAxis("top")
    assert top_axis.isVisible()
    lot_axis_font = top_axis.style["tickFont"]
    assert isinstance(lot_axis_font, QtGui.QFont)
    assert lot_axis_font.pointSize() == 9
    assert len(top_axis._tickLevels) == 1
    assert all(
        label.startswith("LOT_") for _, label in top_axis._tickLevels[0]
    )
    assert top_axis._tickLevels[0][-1][1] == ("LOT_20260720_B  20:20:03")
    frame_axis = fq_map.views[0].frame_axis
    assert frame_axis is not None
    assert frame_axis.linkedView() is top_axis.linkedView()
    assert len(frame_axis._tickLevels) == 1
    initial_first_lot = fq_map.current_data.lot_count - VISIBLE_LOTS
    assert frame_axis._tickLevels[0] == FqMapWidget._frame_ticks(
        initial_first_lot,
        VISIBLE_LOTS,
    )
    assert all(
        not view.plot_item.getAxis("bottom").isVisible()
        for view in fq_map.views
    )
    assert all(
        not view.plot_item.getAxis("top").isVisible()
        for view in fq_map.views[1:]
    )
    assert [view.card.height() for view in fq_map.views] == [
        DASHBOARD_CONFIG.fqmap_plot_height + FQ_MAP_TOP_AXIS_HEIGHT,
        DASHBOARD_CONFIG.fqmap_plot_height,
        DASHBOARD_CONFIG.fqmap_plot_height,
    ]
    assert fq_map.current_data.lot_count == 100
    last_page = fq_map.current_data.lot_count - VISIBLE_LOTS
    assert fq_map.horizontal_scrollbar.maximum() == last_page
    assert fq_map.horizontal_scrollbar.value() == last_page
    assert fq_map.all_lots_label.text() == "全100 lot"
    assert fq_map.piece_yield_label.text() == ("個片歩留まり 94.504%")
    assert "検査項目が1つでもNG" in (fq_map.piece_yield_label.toolTip())
    assert [
        fq_map.vision_combo.itemData(index)
        for index in range(fq_map.vision_combo.count())
    ] == [None, "vision_1", "vision_2", "vision_3"]
    assert fq_map.fmap_selection_label.text() == ("Foreign_Length_Long_v1")
    assert all(
        view.selection_region.getRegion() == pytest.approx([-0.5, 0.5])
        for view in fq_map.views
    )
    assert (
        fq_map.vertical_scroll_area.verticalScrollBarPolicy()
        == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    fmap_position = fq_map.fmap_section.pos()
    fq_scrollbar = fq_map.vertical_scroll_area.verticalScrollBar()
    assert fq_scrollbar.maximum() == 0
    assert fq_scrollbar.value() == 0
    assert fq_map.fmap_section.pos() == fmap_position

    row_scrollbars = [view.vertical_scrollbar for view in fq_map.views]
    available_height = min(
        view.plot_item.getViewBox().sceneBoundingRect().height()
        for view in fq_map.views
    )
    expected_page_step = min(
        len(fq_map.current_data.colnames),
        max(
            1,
            int(
                available_height
                // DASHBOARD_CONFIG.fqmap_min_cell_height
            ),
        ),
    )
    assert expected_page_step == 12
    assert all(
        scrollbar.parentWidget() is view.plot_widget
        for view, scrollbar in zip(fq_map.views, row_scrollbars)
    )
    assert all(scrollbar.isVisible() for scrollbar in row_scrollbars)
    assert all(
        scrollbar.pageStep() == expected_page_step
        for scrollbar in row_scrollbars
    )
    assert all(
        scrollbar.maximum()
        == len(fq_map.current_data.colnames) - expected_page_step
        for scrollbar in row_scrollbars
    )
    assert all(scrollbar.value() == 0 for scrollbar in row_scrollbars)
    assert all(
        view.plot_item.viewRange()[1]
        == pytest.approx([-0.5, expected_page_step - 0.5])
        for view in fq_map.views
    )
    for view in fq_map.views:
        view_box = view.plot_item.getViewBox()
        row_top = view_box.mapViewToScene(QtCore.QPointF(0.0, -0.5))
        row_bottom = view_box.mapViewToScene(QtCore.QPointF(0.0, 0.5))
        assert abs(row_bottom.y() - row_top.y()) >= (
            DASHBOARD_CONFIG.fqmap_min_cell_height - 0.01
        )

    row_scrollbars[1].setValue(row_scrollbars[1].maximum())
    maximum_first_row = len(fq_map.current_data.colnames) - expected_page_step
    assert all(
        scrollbar.value() == maximum_first_row
        for scrollbar in row_scrollbars
    )
    assert all(
        view.plot_item.viewRange()[1]
        == pytest.approx(
            [maximum_first_row - 0.5, 44.5]
        )
        for view in fq_map.views
    )

    selected_first_row = 12
    row_scrollbars[0].setValue(selected_first_row)
    assert all(
        scrollbar.value() == selected_first_row
        for scrollbar in row_scrollbars
    )
    assert all(
        view.plot_item.viewRange()[1]
        == pytest.approx(
            [
                selected_first_row - 0.5,
                selected_first_row + expected_page_step - 0.5,
            ]
        )
        for view in fq_map.views
    )
    assert all(
        view.plot_item.viewRange()[0]
        == pytest.approx(
            [
                last_page * 24 - 0.5,
                fq_map.current_data.lot_count * 24 - 0.5,
            ]
        )
        for view in fq_map.views
    )
    fq_map.horizontal_scrollbar.setValue(0)
    assert frame_axis._tickLevels[0] == FqMapWidget._frame_ticks(
        0,
        VISIBLE_LOTS,
    )
    assert all(
        view.plot_item.viewRange()[0]
        == pytest.approx([-0.5, VISIBLE_LOTS * 24 - 0.5])
        for view in fq_map.views
    )
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[0].image,
        fq_map.frame_map.data.ng_rates[0, 0],
    )
    assert (
        fq_map.kde.current_lot_numbers
        == (fq_map.full_data.lot_numbers[:VISIBLE_LOTS])
    )
    assert (
        fq_map.quality_trend.current_lot_numbers
        == (fq_map.full_data.lot_numbers[:VISIBLE_LOTS])
    )
    assert fq_map.quality_trend.plot_item.viewRange()[0] == pytest.approx(
        [-0.5, VISIBLE_LOTS * 24 - 0.5]
    )
    assert np.allclose(
        fq_map.kde.curve_items[0].getData()[1],
        fq_map.kde.data.densities[initial_row, 0],
    )

    selected_view = fq_map.views[0]
    scene_position = selected_view.plot_item.getViewBox().mapViewToScene(
        QtCore.QPointF(5.0, 19.0)
    )

    class ClickEvent:
        def scenePos(self) -> QtCore.QPointF:
            return scene_position

    fq_map._select_fq_row(selected_view, ClickEvent())
    assert fq_map.selected_colname == "Work_Xw_v2"
    assert fq_map.selected_lot_index == 0
    assert fq_map.selected_frame_no == 6
    assert all(
        view.selection_region.getRegion() == pytest.approx([18.5, 19.5])
        for view in fq_map.views
    )
    for view in fq_map.views:
        assert view.cell_selection_rect.isVisible()
        rectangle = view.cell_selection_rect.rect()
        assert (
            rectangle.x(),
            rectangle.y(),
            rectangle.width(),
            rectangle.height(),
        ) == pytest.approx((4.5, 18.5, 1.0, 1.0))

    frame_map = fq_map.frame_map
    assert fq_map.detail_mode == "lot"
    assert fq_map.single_frame_data is None
    assert fq_map.detail_stack.currentWidget() is fq_map.lot_detail_page
    assert frame_map.frame_mode_button.isEnabled()
    work_x_index = frame_map.data.colname_index("Work_Xw_v2")
    for frame_row in frame_map.rows:
        expected = getattr(frame_map.data, frame_row.metric)[
            work_x_index,
            :VISIBLE_LOTS,
        ]
        for image_item, matrix in zip(frame_row.image_items, expected):
            assert np.allclose(image_item.image, matrix)

    frame_map.frame_mode_button.click()
    single_frame_data = fq_map.single_frame_data
    assert fq_map.detail_mode == "frame"
    assert fq_map.detail_stack.currentWidget() is fq_map.frame_detail_page
    assert single_frame_data is not None
    assert single_frame_data.lot_number == "LOT_20260601_A"
    assert single_frame_data.frame_no == 6
    assert single_frame_data.colname == "Work_Xw_v2"
    assert single_frame_data.sample_count == 288
    frame_detail_layout = fq_map.frame_detail_page.layout()
    assert isinstance(frame_detail_layout, QtWidgets.QHBoxLayout)
    frame_detail_widgets = []
    for index in (0, 2, 4):
        layout_item = frame_detail_layout.itemAt(index)
        assert layout_item is not None
        widget = layout_item.widget()
        assert widget is not None
        frame_detail_widgets.append(widget)
    assert frame_detail_widgets == [
        fq_map.processing_path_trend,
        fq_map.single_frame_map,
        fq_map.single_frame_kde,
    ]
    assert np.allclose(
        fq_map.single_frame_map.view.image_item.image,
        single_frame_data.raw_values,
    )
    assert np.array_equal(
        fq_map.single_frame_map.view.ng_overlay_item.image,
        build_ng_overlay(single_frame_data),
    )
    assert fq_map.single_frame_map.selection_label.text() == "Work_Xw_v2"
    assert fq_map.single_frame_map.context_label.text() == (
        "FrameNo 06  |  288個片"
    )
    frame_detail_text = "\n".join(
        label.text()
        for label in fq_map.frame_detail_page.findChildren(QtWidgets.QLabel)
    )
    assert "LOT_20260601_A" not in frame_detail_text
    assert "08:11:00" not in frame_detail_text

    frame_kde_data = fq_map.single_frame_kde.data
    assert frame_kde_data is not None
    assert frame_kde_data.colname == "Work_Xw_v2"
    assert frame_kde_data.frame_no == 6
    assert frame_kde_data.sample_count == 288
    kde_x, kde_density = fq_map.single_frame_kde.curve_item.getData()
    assert np.allclose(kde_x, frame_kde_data.x_values)
    assert np.allclose(kde_density, frame_kde_data.density)
    assert "288測定" in fq_map.single_frame_kde.caption_label.text()

    processing_path = fq_map.processing_path_trend
    assert processing_path.current_data is single_frame_data
    assert processing_path.equipment_combo.currentData() == ("row_serpentine")
    assert len(processing_path.current_series) == 1
    assert len(processing_path.path_curves) == 1

    next_scene_position = (
        selected_view.plot_item.getViewBox().mapViewToScene(
            QtCore.QPointF(6.0, 19.0)
        )
    )

    class NextClickEvent:
        def scenePos(self) -> QtCore.QPointF:
            return next_scene_position

    fq_map._select_fq_row(selected_view, NextClickEvent())
    updated_frame_data = fq_map.single_frame_data
    assert fq_map.detail_mode == "frame"
    assert fq_map.detail_stack.currentWidget() is fq_map.frame_detail_page
    assert fq_map.selected_lot_index == 0
    assert fq_map.selected_frame_no == 7
    assert updated_frame_data is not None
    assert updated_frame_data is not single_frame_data
    assert updated_frame_data.lot_number == "LOT_20260601_A"
    assert updated_frame_data.frame_no == 7
    assert updated_frame_data.colname == "Work_Xw_v2"
    assert fq_map.processing_path_trend.current_data is updated_frame_data
    assert fq_map.single_frame_kde.data is not None
    assert fq_map.single_frame_kde.data.frame_no == 7
    assert np.allclose(
        fq_map.single_frame_map.view.image_item.image,
        updated_frame_data.raw_values,
    )
    single_frame_data = updated_frame_data

    def unexpected_frame_query(*_: object) -> None:
        raise AssertionError("設備切替でFrame再取得が発生しました")

    monkeypatch.setattr(
        repository,
        "values_by_colname_frame",
        unexpected_frame_query,
    )
    mold_index = processing_path.equipment_combo.findData("mold_4flow")
    assert mold_index >= 0
    processing_path.equipment_combo.setCurrentIndex(mold_index)
    assert len(processing_path.current_series) == 4
    assert len(processing_path.path_curves) == 4
    assert len(processing_path.legend.items) == 4
    for series, curve in zip(
        processing_path.current_series,
        processing_path.path_curves,
    ):
        curve_steps, curve_values = curve.getData()
        expected_values = single_frame_data.raw_values[
            series.position_y - 1,
            series.position_x - 1,
        ]
        assert np.array_equal(curve_steps, series.steps)
        assert np.allclose(curve_values, series.values)
        assert np.allclose(series.values, expected_values)

    processing_path.lot_mode_button.click()
    assert fq_map.detail_mode == "lot"
    assert fq_map.detail_stack.currentWidget() is fq_map.lot_detail_page
    assert fq_map.single_frame_data is single_frame_data
    assert frame_map.frame_mode_button.isEnabled()
    assert all(view.cell_selection_rect.isVisible() for view in fq_map.views)
    for frame_row in frame_map.rows:
        expected = getattr(frame_map.data, frame_row.metric)[
            work_x_index,
            :VISIBLE_LOTS,
        ]
        for image_item, matrix in zip(frame_row.image_items, expected):
            assert np.allclose(image_item.image, matrix)

    fq_map.horizontal_scrollbar.setValue(1)
    assert fq_map.selected_lot_index is None
    assert fq_map.selected_frame_no is None
    assert fq_map.single_frame_data is None
    assert fq_map.detail_mode == "lot"
    assert not frame_map.frame_mode_button.isEnabled()
    assert all(
        not view.cell_selection_rect.isVisible() for view in fq_map.views
    )
    fq_map.horizontal_scrollbar.setValue(0)

    assert fq_map.fmap_selection_label.text().endswith("Work_Xw_v2")
    assert fq_map.kde.current_colname == "Work_Xw_v2"
    assert fq_map.quality_trend.current_colname == "Work_Xw_v2"
    assert fq_map.quality_trend.lower_line.isVisible()
    assert fq_map.quality_trend.upper_line.isVisible()
    assert all(line.isVisible() for line in fq_map.kde.lower_lines)
    assert all(line.isVisible() for line in fq_map.kde.center_lines)
    assert all(line.isVisible() for line in fq_map.kde.upper_lines)
    assert all(
        line.value() == pytest.approx(3.9) for line in fq_map.kde.lower_lines
    )
    assert all(
        line.value() == pytest.approx(4.1) for line in fq_map.kde.upper_lines
    )
    assert all(
        line.value() == pytest.approx(4.0) for line in fq_map.kde.center_lines
    )
    assert all(
        plot_item.viewRange()[0] == pytest.approx([3.85, 4.15])
        for plot_item in fq_map.kde.plot_items
    )
    assert (3.9 - 3.85) / (4.15 - 3.85) == pytest.approx(1 / 6)
    assert (4.0 - 3.85) / (4.15 - 3.85) == pytest.approx(1 / 2)
    assert (4.1 - 3.85) / (4.15 - 3.85) == pytest.approx(5 / 6)
    kde_work_x_index = fq_map.kde.data.colname_index("Work_Xw_v2")
    assert np.allclose(
        fq_map.kde.curve_items[0].getData()[1],
        fq_map.kde.data.densities[kde_work_x_index, 0],
    )
    work_x_index = fq_map.frame_map.data.colname_index("Work_Xw_v2")
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[0].image,
        fq_map.frame_map.data.ng_rates[work_x_index, 0],
    )

    fq_map.category_combo.setCurrentIndex(
        fq_map.category_combo.findData("異物")
    )
    assert fq_map.current_data.ng_rates.shape == (9, 2_400)
    assert all(
        view.image_item.image.shape == (9, 2_400) for view in fq_map.views
    )
    assert fq_map.selected_colname == "Foreign_Length_Long_v1"
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[0].image,
        fq_map.frame_map.data.ng_rates[0, last_page],
    )
    assert all(
        scrollbar.maximum() == 0
        and scrollbar.pageStep() == 9
        and scrollbar.value() == 0
        and not scrollbar.isVisible()
        for scrollbar in row_scrollbars
    )
    assert all(
        view.plot_item.viewRange()[1] == pytest.approx([-0.5, 8.5])
        for view in fq_map.views
    )

    fq_map.vision_combo.setCurrentIndex(
        fq_map.vision_combo.findData("vision_2")
    )
    assert fq_map.current_data.ng_rates.shape == (3, 2_400)
    assert all(
        colname.endswith("_v2") for colname in fq_map.current_data.colnames
    )
    assert fq_map.selected_colname == "Foreign_Length_Long_v2"
    assert "異物 / vision_2" in fq_map.scope_label.toolTip()
    assert fq_map.fmap_selection_label.text().endswith(
        "Foreign_Length_Long_v2"
    )
    assert fq_map.kde.current_colname == ("Foreign_Length_Long_v2")
    assert fq_map.quality_trend.current_colname == ("Foreign_Length_Long_v2")
    assert not fq_map.quality_trend.lower_line.isVisible()
    assert fq_map.quality_trend.upper_line.isVisible()
    assert all(not line.isVisible() for line in fq_map.kde.center_lines)
    assert all(
        plot_item.viewRange()[0] == pytest.approx([0.0, 0.36])
        for plot_item in fq_map.kde.plot_items
    )
    qtbot.wait(50)
    assert [view.card.height() for view in fq_map.views] == [
        DASHBOARD_CONFIG.fqmap_plot_height + FQ_MAP_TOP_AXIS_HEIGHT,
        DASHBOARD_CONFIG.fqmap_plot_height,
        DASHBOARD_CONFIG.fqmap_plot_height,
    ]
    window.resize(2_000, 1_650)
    qtbot.wait(100)
    row_rects = [row.widget.geometry() for row in fq_map.frame_map.rows]
    assert row_rects[0].top() == 0
    assert all(
        current.top() == previous.bottom() + 1
        for previous, current in zip(row_rects, row_rects[1:])
    )
    assert fq_map.frame_map.height() == sum(
        rectangle.height() for rectangle in row_rects
    )
    assert min(rectangle.height() for rectangle in row_rects) == (
        DEFAULT_ROW_HEIGHT
    )
    assert max(rectangle.height() for rectangle in row_rects) <= (
        DEFAULT_ROW_HEIGHT + 1
    )
    assert fq_map.fq_map_section.height() == FQ_MAP_SECTION_HEIGHT
    assert fq_map.fmap_section.height() == FMAP_SECTION_HEIGHT
    assert fq_map.kde_section.height() == KDE_SECTION_HEIGHT
    assert (
        fq_map.quality_trend_section.height() == QUALITY_TREND_SECTION_HEIGHT
    )

    window.resize(1_600, 1_550)
    qtbot.wait(100)
    assert (
        sum(row.widget.height() for row in fq_map.frame_map.rows)
        == FMAP_SECTION_HEIGHT
    )
    assert fq_map.fq_map_section.height() == FQ_MAP_SECTION_HEIGHT
    assert fq_map.fmap_section.height() == FMAP_SECTION_HEIGHT
    window.resize(1_100, 760)
    qtbot.wait(100)
    narrow_lot_axis_font = top_axis.style["tickFont"]
    assert isinstance(narrow_lot_axis_font, QtGui.QFont)
    assert narrow_lot_axis_font.pointSize() == 9
    assert 0 < narrow_lot_axis_font.stretch() < 100
    scope_width = QtGui.QFontMetrics(
        fq_map.scope_label.font()
    ).horizontalAdvance(fq_map.scope_label.text())
    assert scope_width <= fq_map.scope_label.width()
    piece_yield_width = QtGui.QFontMetrics(
        fq_map.piece_yield_label.font()
    ).horizontalAdvance(fq_map.piece_yield_label.text())
    assert piece_yield_width <= (
        fq_map.piece_yield_label.contentsRect().width()
    )
    assert (
        fq_map.piece_yield_label.sizeHint().width()
        <= fq_map.piece_yield_label.width()
    )
    assert not (fq_map.quality_trend.plot_item.getAxis("top").isVisible())
    dashboard_scrollbar = window.dashboard_scroll_area.verticalScrollBar()
    assert dashboard_scrollbar.maximum() > 0
    dashboard_scrollbar.setValue(dashboard_scrollbar.maximum())
    assert dashboard_scrollbar.value() == dashboard_scrollbar.maximum()
    qtbot.keyClick(window, QtCore.Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not window.isVisible())
