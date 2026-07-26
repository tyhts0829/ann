from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets
from scipy.ndimage import gaussian_filter1d

from src.analysis.map_palettes import make_density_color_map
from src.analysis.plot_style import (
    LOT_SEPARATOR_COLOR,
    LOT_SEPARATOR_WIDTH,
)
from src.analysis.quality_trend import (
    QualityTrendData,
    QualityTrendWidget,
    build_quality_trend_data,
)
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "standardized"
    / "quality_data_100lots.parquet"
)


@pytest.fixture(scope="module")
def quality_trend_data() -> QualityTrendData:
    repository = QualityRepository(DATA_PATH)
    try:
        return build_quality_trend_data(repository)
    finally:
        repository.close()


def test_quality_trend_data_contract(
    quality_trend_data: QualityTrendData,
) -> None:
    data = quality_trend_data

    assert data.densities.shape == (45, 60, 2_400)
    assert data.densities.dtype == np.float32
    assert data.bin_values.shape == (45, 60)
    assert data.sample_counts.shape == (45, 2_400)
    assert np.all(data.sample_counts == 288)
    assert int(data.sample_counts.sum()) == 31_104_000
    assert int(data.ng_counts.sum()) == 78_126
    assert data.frame_numbers.tolist()[:25] == [*range(1, 25), 1]
    assert data.column_lots[:24] == (
        "LOT_20260601_A",
    ) * 24
    assert data.lot_numbers[-1] == "LOT_20260720_B"
    assert np.isfinite(data.densities).all()
    assert np.all(data.densities >= 0.0)
    assert np.all(data.density_scales > 0.0)
    assert set(data.units) == {"mm", "mm²"}
    assert all(data.units)

    bin_widths = (
        data.display_max - data.display_min
    ) / data.densities.shape[1]
    integrated_density = (
        data.densities.sum(axis=1) * bin_widths[:, None]
    )
    coverage = data.in_range_counts / data.sample_counts
    assert np.allclose(integrated_density, coverage, atol=1e-5)
    assert float(np.max(1.0 - coverage)) < 0.025


def test_quality_trend_matches_source_histogram(
    quality_trend_data: QualityTrendData,
) -> None:
    lot_number = "LOT_20260601_A"
    frame_number = 1
    colname = "Work_Xw_v2"
    with duckdb.connect() as connection:
        source = connection.execute(
            """
                SELECT value, limmin, limmax
                FROM read_parquet(?)
                WHERE lot_number = ?
                  AND FrameNo = ?
                  AND colname = ?
                  AND meta_type = 'spec'
                  AND NOT meta_ignore
            """,
            [
                str(DATA_PATH),
                lot_number,
                frame_number,
                colname,
            ],
        ).df()

    row = quality_trend_data.colname_index(colname)
    bin_count = quality_trend_data.densities.shape[1]
    edges = np.linspace(
        quality_trend_data.display_min[row],
        quality_trend_data.display_max[row],
        bin_count + 1,
    )
    expected_counts = np.histogram(
        source["value"].to_numpy(),
        bins=edges,
    )[0].astype(float)
    expected_density = gaussian_filter1d(
        expected_counts,
        DASHBOARD_CONFIG.kde_bandwidth_bins,
        mode="constant",
        truncate=4.0,
    )
    expected_density /= expected_density.sum()
    expected_density *= (
        expected_counts.sum()
        / len(source)
        / (edges[1] - edges[0])
    )
    ng_count = (
        (source["value"] < source["limmin"])
        | (source["value"] > source["limmax"])
    ).sum()

    assert len(source) == 288
    assert np.allclose(
        quality_trend_data.densities[row, :, 0],
        expected_density,
        rtol=1e-5,
        atol=1e-6,
    )
    assert quality_trend_data.in_range_counts[row, 0] == (
        expected_counts.sum()
    )
    assert quality_trend_data.ng_counts[row, 0] == ng_count


def test_quality_trend_density_scale_is_all_lot_common(
    quality_trend_data: QualityTrendData,
) -> None:
    row = quality_trend_data.colname_index("Work_Xw_v2")
    positive = quality_trend_data.densities[row][
        quality_trend_data.densities[row] > 0.0
    ]

    assert quality_trend_data.density_scales[row] == pytest.approx(
        np.percentile(positive, 99.5)
    )


def test_quality_trend_source_units() -> None:
    with duckdb.connect() as connection:
        result = connection.execute(
            """
                SELECT
                    sum(missing_count),
                    max(unit_count)
                FROM (
                    SELECT
                        colname,
                        count_if(meta_unit IS NULL) AS missing_count,
                        count(DISTINCT meta_unit) AS unit_count
                    FROM read_parquet(?)
                    WHERE meta_type = 'spec'
                      AND NOT meta_ignore
                    GROUP BY colname
                )
            """,
            [str(DATA_PATH)],
        ).fetchone()
        assert result is not None
        missing_count, maximum_unit_count = result

    assert missing_count == 0
    assert maximum_unit_count == 1


def test_quality_trend_excludes_missing_values_from_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_value.parquet"
    pq.write_table(
        pa.table(
            {
                "lot_number": ["LOT_A"] * 3,
                "FrameNo": [1] * 3,
                "colname": ["Work_Xw_v1"] * 3,
                "value": pa.array(
                    [1.0, 2.0, None],
                    type=pa.float64(),
                ),
                "limmin": [0.0] * 3,
                "limmax": [3.0] * 3,
                "meta_best": pa.array(
                    [None] * 3,
                    type=pa.float64(),
                ),
                "meta_unit": ["mm"] * 3,
                "meta_type": ["spec"] * 3,
                "meta_ignore": [False] * 3,
            }
        ),
        path,
    )
    repository = QualityRepository(path)
    try:
        frame = repository.density_bins_by_colname_frame(
            DASHBOARD_CONFIG.kde_bins,
            ("LOT_A",),
        )
    finally:
        repository.close()

    assert frame.loc[0, "sample_count"] == 2
    assert frame.loc[0, "in_range_count"] == 2
    assert len(frame.loc[0, "bin_indices"]) == 2
    assert sum(frame.loc[0, "bin_counts"]) == 2


def test_quality_trend_widget_context(
    qtbot,
    quality_trend_data: QualityTrendData,
) -> None:
    widget = QualityTrendWidget(quality_trend_data)
    qtbot.addWidget(widget)
    widget.resize(1_600, DASHBOARD_CONFIG.quality_trend_height)
    widget.show()
    widget.set_context("Work_Xw_v2", 92)
    qtbot.wait(20)

    row = quality_trend_data.colname_index("Work_Xw_v2")
    assert isinstance(widget.image_item, pg.ImageItem)
    assert widget.image_item.image.shape == (60, 2_400)
    assert np.allclose(
        widget.image_item.image,
        quality_trend_data.densities[row],
    )
    assert widget.image_item.getLevels() == pytest.approx(
        [0.0, quality_trend_data.density_scales[row]]
    )
    assert not hasattr(widget, "median_curve")
    assert widget.lower_line.isVisible()
    assert widget.upper_line.isVisible()
    assert widget.lower_line.value() == pytest.approx(3.9)
    assert widget.upper_line.value() == pytest.approx(4.1)
    assert widget.current_lot_numbers == (
        quality_trend_data.lot_numbers[-8:]
    )
    assert widget.plot_item.viewRange()[0] == pytest.approx(
        [92 * 24 - 0.5, 100 * 24 - 0.5]
    )
    assert not widget.plot_item.getAxis("top").isVisible()
    assert all(
        "LOT_" not in label.text()
        for label in widget.findChildren(QtWidgets.QLabel)
    )

    y_range = widget.plot_item.viewRange()[1]
    levels = widget.image_item.getLevels()
    widget.set_context("Work_Xw_v2", 0)
    assert widget.plot_item.viewRange()[1] == pytest.approx(y_range)
    assert widget.image_item.getLevels() == pytest.approx(levels)
    assert widget.plot_item.viewRange()[0] == pytest.approx(
        [-0.5, 8 * 24 - 0.5]
    )
    assert widget.plot_item.getAxis("bottom")._tickLevels[0] == (
        QualityTrendWidget._frame_ticks(0, 8)
    )

    widget.set_context("Foreign_Length_Long_v1", 0)
    assert not widget.lower_line.isVisible()
    assert widget.upper_line.isVisible()
    assert widget.upper_line.value() == pytest.approx(0.3)


def test_quality_trend_uses_shared_separator_design(
    qtbot,
    quality_trend_data: QualityTrendData,
) -> None:
    widget = QualityTrendWidget(quality_trend_data)
    qtbot.addWidget(widget)

    assert len(widget.lot_separators) == 99
    assert [
        separator.value() for separator in widget.lot_separators[:2]
    ] == pytest.approx([23.5, 47.5])
    assert {
        separator.pen.color().name()
        for separator in widget.lot_separators
    } == {LOT_SEPARATOR_COLOR}
    assert {
        separator.pen.widthF()
        for separator in widget.lot_separators
    } == {LOT_SEPARATOR_WIDTH}


def test_density_palette_has_light_low_and_warm_high() -> None:
    lookup = make_density_color_map().getLookupTable(
        0.0,
        1.0,
        256,
    )

    assert tuple(lookup[0]) == (255, 255, 255)
    assert tuple(lookup[-1]) == (201, 52, 47)


def test_quality_trend_hover(
    qtbot,
    quality_trend_data: QualityTrendData,
) -> None:
    widget = QualityTrendWidget(quality_trend_data)
    qtbot.addWidget(widget)
    widget.resize(1_600, DASHBOARD_CONFIG.quality_trend_height)
    widget.show()
    widget.set_context("Work_Xw_v2", 0)
    qtbot.wait(20)

    row = quality_trend_data.colname_index("Work_Xw_v2")
    scene_position = widget.plot_item.getViewBox().mapViewToScene(
        QtCore.QPointF(5.0, quality_trend_data.bin_values[row, 30])
    )
    hover_texts: list[str] = []
    widget.hover_text_changed.connect(hover_texts.append)
    widget._show_hover((scene_position,))

    assert hover_texts
    assert "LOT_20260601_A" in hover_texts[-1]
    assert "FrameNo 6" in hover_texts[-1]
    assert "生値" in hover_texts[-1]
    assert "密度" in hover_texts[-1]
    assert "P05" not in hover_texts[-1]
    assert "中央値" not in hover_texts[-1]
    assert "N 288" in hover_texts[-1]
    assert "NG" in hover_texts[-1]
    assert widget.hover_line.isVisible()

    widget.eventFilter(
        widget.plot_widget.viewport(),
        QtCore.QEvent(QtCore.QEvent.Type.Leave),
    )
    assert hover_texts[-1] == ""
    assert not widget.hover_line.isVisible()

    widget._show_hover((scene_position,))
    widget.set_context("Work_Xw_v2", 1)
    assert hover_texts[-1] == ""
    assert not widget.hover_line.isVisible()
