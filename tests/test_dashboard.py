from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyqtgraph as pg
import pytest
from PySide6 import QtCore

from dashboard import DashboardWindow, STYLESHEET
from src.analysis.ng_rate_heatmap import (
    build_frame_map_data,
    build_heatmap_data,
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


def test_all_lot_heatmap(repository: QualityRepository) -> None:
    data = build_heatmap_data(repository)

    assert data.ng_rates.shape == (15, 2_400)
    assert data.normalized_mean.shape == (15, 2_400)
    assert data.normalized_std.shape == (15, 2_400)
    assert data.lot_count == 100
    assert data.total_measurements == 10_368_000
    assert data.total_ng == 26_232
    assert data.frame_numbers.tolist()[:25] == [*range(1, 25), 1]
    assert len(set(data.colnames)) == 15
    assert all("_v" not in colname for colname in data.colnames)
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

    assert raw_file.metadata.num_rows == 10_368_000
    assert standardized_file.metadata.num_rows == 10_368_000
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


def test_latest_lot_heatmap(repository: QualityRepository) -> None:
    data = build_heatmap_data(repository, ("LOT_20260720_B",))

    assert data.ng_rates.shape == (15, 24)
    assert data.lot_count == 1
    assert data.total_measurements == 103_680
    assert data.total_ng == 1_207


def test_frame_map_data(repository: QualityRepository) -> None:
    data = build_frame_map_data(repository)

    assert data.ng_rates.shape == (15, 100, 12, 24)
    assert data.normalized_mean.shape == (15, 100, 12, 24)
    assert data.normalized_std.shape == (15, 100, 12, 24)
    assert data.colnames == (
        "Foreign_Length_Long",
        "Foreign_Length_Short",
        "Foreign_Size",
        "Lead_Length_L",
        "Lead_Length_R",
        "Lead_Pitch",
        "Work_Xw",
        "Work_Yw",
        "Work_Center_X",
        "Work_Center_Y",
        "Mark_Center_X",
        "Mark_Center_Y",
        "Defect_Length_Long",
        "Defect_Length_Short",
        "Defect_Size",
    )
    assert np.isfinite(data.ng_rates).all()
    assert np.std(data.ng_rates[-1, -1]) > 1.0


def test_dashboard_window(qtbot, repository: QualityRepository) -> None:
    window = DashboardWindow(repository)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)
    heatmap = window.heatmap

    assert window.windowTitle() == "Quality Dashboard"
    assert len(heatmap.heatmap_views) == 3
    assert len(heatmap.frame_map_rows) == 3
    assert heatmap.plots_container.layout().spacing() == 0
    assert all(
        view.image_item.image.shape == (15, 2_400)
        for view in heatmap.heatmap_views
    )
    assert all(
        len(row.image_items) == 5
        and all(image.image.shape == (12, 24) for image in row.image_items)
        for row in heatmap.frame_map_rows
    )
    grid_lines = [
        item
        for item in heatmap.frame_map_rows[0]
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
        heatmap.frame_map_rows[0].image_items[-1].image,
        heatmap.frame_map_data.ng_rates[0, -1],
    )
    assert "border-radius" not in STYLESHEET
    assert "background: #f3f5f7" in STYLESHEET
    assert "QFrame#chartCard {\n    background: #ffffff;\n    border: none;" in (
        STYLESHEET
    )
    for view in heatmap.heatmap_views:
        bottom_axis = view.plot_item.getAxis("bottom")
        left_axis = view.plot_item.getAxis("left")
        assert not bottom_axis.label.isVisible()
        assert not left_axis.label.isVisible()
        assert len(bottom_axis._tickLevels) == 1
        assert all(
            label.startswith("LOT_")
            for _, label in bottom_axis._tickLevels[0]
        )
    assert heatmap.current_heatmap is not None
    assert heatmap.current_heatmap.lot_count == 100
    assert heatmap.horizontal_scrollbar.maximum() == 95
    assert heatmap.horizontal_scrollbar.value() == 95
    assert heatmap.all_lots_label.text() == "全100 lot"
    assert "総合NG率" in heatmap.ng_rate_label.text()
    assert heatmap.vertical_scroll_area.verticalScrollBar().maximum() > 0
    assert all(
        view.plot_item.viewRange()[0]
        == pytest.approx([2_279.5, 2_399.5])
        for view in heatmap.heatmap_views
    )

    heatmap.horizontal_scrollbar.setValue(0)
    assert all(
        view.plot_item.viewRange()[0]
        == pytest.approx([-0.5, 119.5])
        for view in heatmap.heatmap_views
    )
    assert np.allclose(
        heatmap.frame_map_rows[0].image_items[0].image,
        heatmap.frame_map_data.ng_rates[0, 0],
    )

    selected_view = heatmap.heatmap_views[0]
    scene_position = (
        selected_view.plot_item.getViewBox().mapViewToScene(
            QtCore.QPointF(5.0, 6.0)
        )
    )

    class ClickEvent:
        def scenePos(self) -> QtCore.QPointF:
            return scene_position

    heatmap._select_heatmap_cell(selected_view, ClickEvent())
    assert heatmap.selected_colname == "Work_Xw"
    assert heatmap.selected_column == 5
    work_x_index = heatmap.frame_map_data.colname_index("Work_Xw")
    assert np.allclose(
        heatmap.frame_map_rows[0].image_items[0].image,
        heatmap.frame_map_data.ng_rates[work_x_index, 0],
    )

    heatmap.category_combo.setCurrentIndex(
        heatmap.category_combo.findData("異物")
    )
    assert heatmap.current_heatmap is not None
    assert heatmap.current_heatmap.ng_rates.shape == (3, 2_400)
    assert all(
        view.image_item.image.shape == (3, 2_400)
        for view in heatmap.heatmap_views
    )
    assert heatmap.selected_colname == "Foreign_Length_Long"
    assert np.allclose(
        heatmap.frame_map_rows[0].image_items[0].image,
        heatmap.frame_map_data.ng_rates[0, 95],
    )
