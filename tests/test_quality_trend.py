from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyqtgraph as pg
import pytest
from PySide6 import QtCore

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

    assert data.p50.shape == (45, 2_400)
    assert data.sample_counts.shape == (45, 2_400)
    assert np.all(data.sample_counts == 288)
    assert int(data.ng_counts.sum()) == 78_126
    assert data.frame_numbers.tolist()[:25] == [*range(1, 25), 1]
    assert data.column_lots[:24] == (
        "LOT_20260601_A",
    ) * 24
    assert data.lot_numbers[-1] == "LOT_20260720_B"
    assert np.all(data.minimum <= data.p05)
    assert np.all(data.p05 <= data.p25)
    assert np.all(data.p25 <= data.p50)
    assert np.all(data.p50 <= data.p75)
    assert np.all(data.p75 <= data.p95)
    assert np.all(data.p95 <= data.maximum)
    assert set(data.units) == {"mm", "mm²"}
    assert all(data.units)


def test_quality_trend_matches_source_values(
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
    expected = np.quantile(
        source["value"].to_numpy(),
        [0.05, 0.25, 0.50, 0.75, 0.95],
        method="linear",
    )
    actual = np.asarray(
        [
            quality_trend_data.p05[row, 0],
            quality_trend_data.p25[row, 0],
            quality_trend_data.p50[row, 0],
            quality_trend_data.p75[row, 0],
            quality_trend_data.p95[row, 0],
        ]
    )
    ng_count = (
        (source["value"] < source["limmin"])
        | (source["value"] > source["limmax"])
    ).sum()

    assert len(source) == 288
    assert quality_trend_data.minimum[row, 0] == pytest.approx(
        source["value"].min()
    )
    assert np.allclose(actual, expected)
    assert quality_trend_data.maximum[row, 0] == pytest.approx(
        source["value"].max()
    )
    assert quality_trend_data.ng_counts[row, 0] == ng_count


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
        frame = repository.quantiles_by_colname_frame()
    finally:
        repository.close()

    assert frame.loc[0, "sample_count"] == 2
    assert frame.loc[0, "minimum"] == pytest.approx(1.0)
    assert frame.loc[0, "p50"] == pytest.approx(1.5)
    assert frame.loc[0, "maximum"] == pytest.approx(2.0)


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
    _, median = widget.median_curve.getData()
    _, minimum = widget.minimum_curve.getData()
    _, maximum = widget.maximum_curve.getData()
    assert np.allclose(median, quality_trend_data.p50[row])
    assert np.allclose(minimum, quality_trend_data.minimum[row])
    assert np.allclose(maximum, quality_trend_data.maximum[row])
    assert widget.minimum_curve.opts["pen"].color().name() == "#526f79"
    assert widget.maximum_curve.opts["pen"].color().name() == "#526f79"
    assert isinstance(widget.outer_band, pg.FillBetweenItem)
    assert isinstance(widget.inner_band, pg.FillBetweenItem)
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
    y_min, y_max = widget.plot_item.viewRange()[1]
    assert y_min < float(np.nanmin(quality_trend_data.minimum[row]))
    assert y_max > float(np.nanmax(quality_trend_data.maximum[row]))
    assert (
        widget.plot_item.getAxis("right").label.toPlainText().strip()
        == "生値 (mm)"
    )
    assert not widget.plot_item.getAxis("top").isVisible()

    y_range = widget.plot_item.viewRange()[1]
    widget.set_context("Work_Xw_v2", 0)
    assert widget.plot_item.viewRange()[1] == pytest.approx(y_range)
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
        QtCore.QPointF(5.0, quality_trend_data.p50[row, 5])
    )
    hover_texts: list[str] = []
    widget.hover_text_changed.connect(hover_texts.append)
    widget._show_hover((scene_position,))

    assert hover_texts
    assert "LOT_20260601_A" in hover_texts[-1]
    assert "FrameNo 6" in hover_texts[-1]
    assert "P05" in hover_texts[-1]
    assert "中央値" in hover_texts[-1]
    assert "min" in hover_texts[-1]
    assert "max" in hover_texts[-1]
    assert "N 288" in hover_texts[-1]
    assert "NG" in hover_texts[-1]
    assert widget.hover_line.isVisible()
    assert widget.hover_point.isVisible()

    widget.eventFilter(
        widget.plot_widget.viewport(),
        QtCore.QEvent(QtCore.QEvent.Type.Leave),
    )
    assert hover_texts[-1] == ""
    assert not widget.hover_line.isVisible()
    assert not widget.hover_point.isVisible()

    widget._show_hover((scene_position,))
    widget.set_context("Work_Xw_v2", 1)
    assert hover_texts[-1] == ""
    assert not widget.hover_line.isVisible()
    assert not widget.hover_point.isVisible()
