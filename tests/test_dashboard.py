from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from dashboard import DashboardWindow, STYLESHEET
from src.analysis.fq_map import (
    FMAP_SECTION_HEIGHT,
    FQ_MAP_SECTION_HEIGHT,
    KDE_SECTION_HEIGHT,
    build_fq_map_data,
)
from src.analysis.frame_map import (
    DEFAULT_ROW_HEIGHT,
    VISIBLE_LOTS,
    build_frame_map_data,
)
from src.analysis.kde import build_kde_data
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
    _perlin_lot_factors,
)
from src.standardized.quality_data import QualityRepository


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
    assert data.total_measurements == 31_104_000
    assert data.total_ng == 78_126
    assert data.frame_numbers.tolist()[:25] == [*range(1, 25), 1]
    assert len(set(data.colnames)) == 45
    assert all(colname.endswith(("_v1", "_v2", "_v3")) for colname in data.colnames)
    assert set(data.visions) == {"vision_1", "vision_2", "vision_3"}
    assert data.filter_vision("vision_2").ng_rates.shape == (15, 2_400)
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

    frame = (
        standardized_file
        .read_row_group(
            0,
            columns=[
                "value",
                "limmin",
                "limmax",
                "meta_best",
                "spec_position",
                "spec_usage",
            ],
        )
        .to_pandas()
    )
    two_sided = frame["limmin"].notna() & frame["limmax"].notna()
    one_sided = frame["meta_best"].notna() & frame["limmax"].notna()

    expected_position = (
        frame.loc[two_sided, "value"]
        - (
            frame.loc[two_sided, "limmin"]
            + frame.loc[two_sided, "limmax"]
        )
        / 2.0
    ) / (
        (
            frame.loc[two_sided, "limmax"]
            - frame.loc[two_sided, "limmin"]
        )
        / 2.0
    )
    expected_usage = (
        frame.loc[one_sided, "value"]
        - frame.loc[one_sided, "meta_best"]
    ) / (
        frame.loc[one_sided, "limmax"]
        - frame.loc[one_sided, "meta_best"]
    )

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


@pytest.mark.parametrize("path", [RAW_DATA_PATH, DATA_PATH])
def test_foreign_and_defect_values_are_nonnegative(path: Path) -> None:
    with duckdb.connect() as connection:
        minimum, negative_count = connection.execute(
            """
                SELECT min(value), count_if(value < 0.0)
                FROM read_parquet(?)
                WHERE meta_category IN ('異物', '欠陥')
            """,
            [str(path)],
        ).fetchone()

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
        root
        / "src"
        / "standardized"
        / "standardize_quality_data.py"
    ).is_file()
    assert not (root / "scripts" / "generate_quality_data.py").exists()
    assert (root / "src" / "analysis" / "fq_map.py").is_file()
    assert (root / "src" / "analysis" / "frame_map.py").is_file()
    assert not (
        root / "src" / "analysis" / "ng_rate_heatmap.py"
    ).exists()


def test_dashboard_config() -> None:
    config = load_dashboard_config(CONFIG_PATH)

    assert config == DASHBOARD_CONFIG
    assert VISIBLE_LOTS == config.visible_lots
    assert FQ_MAP_SECTION_HEIGHT == config.fqmap_height
    assert FMAP_SECTION_HEIGHT == config.fmap_height
    assert KDE_SECTION_HEIGHT == config.kde_height


def test_latest_lot_fq_map(repository: QualityRepository) -> None:
    data = build_fq_map_data(repository, ("LOT_20260720_B",))

    assert data.ng_rates.shape == (45, 24)
    assert data.lot_count == 1
    assert data.total_measurements == 311_040
    assert data.total_ng == 3_574


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
        (centers[two_sided] - data.display_min[two_sided])
        / widths[two_sided],
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


def test_dashboard_window(qtbot, repository: QualityRepository) -> None:
    window = DashboardWindow(repository)
    qtbot.addWidget(window)
    assert window.fq_map is None
    assert window.loading_progress.minimum() == 0
    assert window.loading_progress.maximum() == 0
    with qtbot.waitSignal(window.data_loaded, timeout=15_000):
        window.show()
    qtbot.wait(50)
    fq_map = window.fq_map
    assert fq_map is not None

    assert window.windowTitle() == "Quality Dashboard"
    section_texts = {
        label.text()
        for label in fq_map.findChildren(QtWidgets.QLabel)
        if label.objectName() == "mapSectionTitle"
    }
    assert section_texts == {"FQmap", "Fmap", "KDE"}
    assert len(fq_map.views) == 3
    assert len(fq_map.frame_map.rows) == 3
    assert fq_map.plots_container.layout().spacing() == 0
    assert fq_map.fq_map_section.height() == FQ_MAP_SECTION_HEIGHT
    assert fq_map.fmap_section.height() == FMAP_SECTION_HEIGHT
    assert fq_map.fq_map_section.isAncestorOf(fq_map.toolbar)
    assert len(
        [
            frame
            for frame in fq_map.findChildren(QtWidgets.QFrame)
            if frame.objectName() == "mapCard"
        ]
    ) == 3
    assert fq_map.kde_section.height() == KDE_SECTION_HEIGHT
    assert fq_map.kde.current_colname == "Foreign_Length_Long_v1"
    assert fq_map.kde.selection_label.text() == (
        "Foreign_Length_Long_v1"
    )
    assert len(fq_map.kde.plot_widgets) == VISIBLE_LOTS
    assert fq_map.kde.current_lot_numbers == (
        fq_map.full_data.lot_numbers[-VISIBLE_LOTS:]
    )
    initial_row = fq_map.kde.data.colname_index(
        "Foreign_Length_Long_v1"
    )
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
        line.value() == pytest.approx(0.3)
        for line in fq_map.kde.upper_lines
    )
    assert not any(
        isinstance(item, pg.BarGraphItem)
        for plot_item in fq_map.kde.plot_items
        for item in plot_item.items
    )
    assert (
        f"表示中 {VISIBLE_LOTS} lot"
        in fq_map.kde.summary_label.text()
    )
    assert "各 6,912測定" in fq_map.kde.summary_label.text()
    assert "最良値 0" in fq_map.kde.summary_label.text()
    assert all(
        plot_item.viewRange()[0] == pytest.approx([0.0, 0.36])
        for plot_item in fq_map.kde.plot_items
    )
    assert (0.3 - 0.0) / (0.36 - 0.0) == pytest.approx(5 / 6)
    assert (
        fq_map.vertical_scroll_area.parentWidget()
        is fq_map.fq_map_section
    )
    assert not fq_map.plots_container.isAncestorOf(fq_map.frame_map)
    assert fq_map.findChildren(QtWidgets.QScrollArea) == [
        fq_map.vertical_scroll_area
    ]
    assert all(
        view.image_item.image.shape == (45, 2_400)
        for view in fq_map.views
    )
    assert all(
        len(row.image_items) == VISIBLE_LOTS
        and all(image.image.shape == (12, 24) for image in row.image_items)
        for row in fq_map.frame_map.rows
    )
    fq_lot_width = (
        fq_map.views[0]
        .plot_item.getViewBox()
        .sceneBoundingRect()
        .width()
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
        fq_map.kde.plot_items[0]
        .getViewBox()
        .sceneBoundingRect()
        .width()
    )
    assert fmap_lot_width == pytest.approx(fq_lot_width, rel=0.01)
    assert kde_lot_width == pytest.approx(fmap_lot_width, rel=0.01)
    grid_lines = [
        item
        for item in fq_map.frame_map.rows[0]
        .plot_widgets[0]
        .getPlotItem()
        .items
        if isinstance(item, pg.InfiniteLine)
    ]
    assert len(grid_lines) == 34
    assert {line.pen.color().name() for line in grid_lines} == {
        "#ffffff"
    }
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[-1].image,
        fq_map.frame_map.data.ng_rates[0, -1],
    )
    assert "border-radius" not in STYLESHEET
    assert "background: #f3f5f7" in STYLESHEET
    assert "QFrame#chartCard {\n    background: #ffffff;\n    border: none;" in (
        STYLESHEET
    )
    for view in fq_map.views:
        left_axis = view.plot_item.getAxis("left")
        assert not left_axis.label.isVisible()
        assert left_axis.grid is False
    top_axis = fq_map.views[0].plot_item.getAxis("top")
    assert top_axis.isVisible()
    assert len(top_axis._tickLevels) == 1
    assert all(
        label.startswith("LOT_")
        for _, label in top_axis._tickLevels[0]
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
        DASHBOARD_CONFIG.fqmap_plot_height + 28,
        DASHBOARD_CONFIG.fqmap_plot_height,
        DASHBOARD_CONFIG.fqmap_plot_height,
    ]
    assert fq_map.current_data.lot_count == 100
    last_page = fq_map.current_data.lot_count - VISIBLE_LOTS
    assert fq_map.horizontal_scrollbar.maximum() == last_page
    assert fq_map.horizontal_scrollbar.value() == last_page
    assert fq_map.all_lots_label.text() == "全100 lot"
    assert "総合NG率" in fq_map.ng_rate_label.text()
    assert [
        fq_map.vision_combo.itemData(index)
        for index in range(fq_map.vision_combo.count())
    ] == [None, "vision_1", "vision_2", "vision_3"]
    assert fq_map.findChild(
        QtWidgets.QLabel,
        "frameMapContextCaption",
    ).text() == "選択中の検査項目"
    assert fq_map.fmap_selection_label.text() == (
        "Foreign_Length_Long_v1"
    )
    assert all(
        view.selection_region.getRegion()
        == pytest.approx([-0.5, 0.5])
        for view in fq_map.views
    )
    assert (
        fq_map.vertical_scroll_area.verticalScrollBarPolicy()
        == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    fmap_position = fq_map.fmap_section.pos()
    fq_scrollbar = fq_map.vertical_scroll_area.verticalScrollBar()
    fq_scrollbar.setValue(fq_scrollbar.maximum())
    assert fq_scrollbar.value() == fq_scrollbar.maximum()
    assert fq_map.fmap_section.pos() == fmap_position
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
    assert all(
        view.plot_item.viewRange()[0]
        == pytest.approx([-0.5, VISIBLE_LOTS * 24 - 0.5])
        for view in fq_map.views
    )
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[0].image,
        fq_map.frame_map.data.ng_rates[0, 0],
    )
    assert fq_map.kde.current_lot_numbers == (
        fq_map.full_data.lot_numbers[:VISIBLE_LOTS]
    )
    assert np.allclose(
        fq_map.kde.curve_items[0].getData()[1],
        fq_map.kde.data.densities[initial_row, 0],
    )

    selected_view = fq_map.views[0]
    scene_position = (
        selected_view.plot_item.getViewBox().mapViewToScene(
            QtCore.QPointF(5.0, 19.0)
        )
    )

    class ClickEvent:
        def scenePos(self) -> QtCore.QPointF:
            return scene_position

    fq_map._select_fq_row(selected_view, ClickEvent())
    assert fq_map.selected_colname == "Work_Xw_v2"
    assert all(
        view.selection_region.getRegion()
        == pytest.approx([18.5, 19.5])
        for view in fq_map.views
    )
    assert fq_map.fmap_selection_label.text().endswith("Work_Xw_v2")
    assert fq_map.kde.current_colname == "Work_Xw_v2"
    assert fq_map.kde.selection_label.text() == "Work_Xw_v2"
    assert all(line.isVisible() for line in fq_map.kde.lower_lines)
    assert all(line.isVisible() for line in fq_map.kde.center_lines)
    assert all(line.isVisible() for line in fq_map.kde.upper_lines)
    assert all(
        line.value() == pytest.approx(3.9)
        for line in fq_map.kde.lower_lines
    )
    assert all(
        line.value() == pytest.approx(4.1)
        for line in fq_map.kde.upper_lines
    )
    assert all(
        line.value() == pytest.approx(4.0)
        for line in fq_map.kde.center_lines
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
        view.image_item.image.shape == (9, 2_400)
        for view in fq_map.views
    )
    assert fq_map.selected_colname == "Foreign_Length_Long_v1"
    assert np.allclose(
        fq_map.frame_map.rows[0].image_items[0].image,
        fq_map.frame_map.data.ng_rates[0, last_page],
    )

    fq_map.vision_combo.setCurrentIndex(
        fq_map.vision_combo.findData("vision_2")
    )
    assert fq_map.current_data.ng_rates.shape == (3, 2_400)
    assert all(
        colname.endswith("_v2")
        for colname in fq_map.current_data.colnames
    )
    assert fq_map.selected_colname == "Foreign_Length_Long_v2"
    assert "異物 / vision_2" in fq_map.scope_label.text()
    assert fq_map.fmap_selection_label.text().endswith(
        "Foreign_Length_Long_v2"
    )
    assert fq_map.kde.current_colname == (
        "Foreign_Length_Long_v2"
    )
    assert all(not line.isVisible() for line in fq_map.kde.center_lines)
    assert all(
        plot_item.viewRange()[0] == pytest.approx([0.0, 0.36])
        for plot_item in fq_map.kde.plot_items
    )
    qtbot.wait(50)
    assert [view.card.height() for view in fq_map.views] == [
        DASHBOARD_CONFIG.fqmap_plot_height + 28,
        DASHBOARD_CONFIG.fqmap_plot_height,
        DASHBOARD_CONFIG.fqmap_plot_height,
    ]
    window.resize(2_000, 1_650)
    qtbot.wait(100)
    row_rects = [
        row.widget.geometry()
        for row in fq_map.frame_map.rows
    ]
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

    window.resize(1_600, 1_550)
    qtbot.wait(100)
    assert sum(
        row.widget.height() for row in fq_map.frame_map.rows
    ) == FMAP_SECTION_HEIGHT
    assert fq_map.fq_map_section.height() == FQ_MAP_SECTION_HEIGHT
    assert fq_map.fmap_section.height() == FMAP_SECTION_HEIGHT
    qtbot.keyClick(window, QtCore.Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not window.isVisible())
