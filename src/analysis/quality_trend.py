from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.plot_style import make_lot_separator
from src.analysis.quality_columns import SPEC_ORDER
from src.dashboard_config import DASHBOARD_CONFIG
from src.quality_repository import QualityRepository

FRAME_NUMBERS = np.arange(1, 25)
FRAME_TICK_NUMBERS = (1, 6, 12, 18, 24)
VISIBLE_LOTS = DASHBOARD_CONFIG.visible_lots
QUALITY_TREND_HEIGHT = DASHBOARD_CONFIG.quality_trend_height


@dataclass(frozen=True)
class QualityTrendData:
    """lot・FrameNo別の測定値統計データ。"""

    colnames: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    column_lots: tuple[str, ...]
    frame_numbers: np.ndarray
    sample_counts: np.ndarray
    ng_counts: np.ndarray
    minimum: np.ndarray
    p05: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p95: np.ndarray
    maximum: np.ndarray
    spec_lower: np.ndarray
    spec_upper: np.ndarray
    units: tuple[str, ...]

    def colname_index(self, colname: str) -> int:
        """検査項目の配列位置。"""
        return self.colnames.index(colname)


def build_quality_trend_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> QualityTrendData:
    """lot・FrameNo別の測定値統計データ生成。"""
    lot_records = repository.lots()
    all_lot_numbers = tuple(record[0] for record in lot_records)
    if lot_numbers is None:
        lot_numbers = all_lot_numbers
    query_lot_numbers = (
        None if lot_numbers == all_lot_numbers else lot_numbers
    )

    frame = repository.quantiles_by_colname_frame(query_lot_numbers)
    available = set(frame["colname"])
    colnames = tuple(
        colname for colname in SPEC_ORDER if colname in available
    )
    index = pd.MultiIndex.from_product(
        [colnames, lot_numbers, FRAME_NUMBERS],
        names=["colname", "lot_number", "FrameNo"],
    )
    ordered = frame.set_index(
        ["colname", "lot_number", "FrameNo"]
    ).reindex(index)
    shape = (len(colnames), len(lot_numbers) * len(FRAME_NUMBERS))

    metadata = (
        frame.drop_duplicates("colname")
        .set_index("colname")
        .reindex(colnames)
    )
    return QualityTrendData(
        colnames=colnames,
        lot_numbers=lot_numbers,
        column_lots=tuple(
            lot_number
            for lot_number in lot_numbers
            for _ in FRAME_NUMBERS
        ),
        frame_numbers=np.tile(FRAME_NUMBERS, len(lot_numbers)),
        sample_counts=(
            ordered["sample_count"]
            .fillna(0)
            .to_numpy(dtype=np.int64)
            .reshape(shape)
        ),
        ng_counts=(
            ordered["ng_count"]
            .fillna(0)
            .to_numpy(dtype=np.int64)
            .reshape(shape)
        ),
        minimum=ordered["minimum"].to_numpy(dtype=float).reshape(shape),
        p05=ordered["p05"].to_numpy(dtype=float).reshape(shape),
        p25=ordered["p25"].to_numpy(dtype=float).reshape(shape),
        p50=ordered["p50"].to_numpy(dtype=float).reshape(shape),
        p75=ordered["p75"].to_numpy(dtype=float).reshape(shape),
        p95=ordered["p95"].to_numpy(dtype=float).reshape(shape),
        maximum=ordered["maximum"].to_numpy(dtype=float).reshape(shape),
        spec_lower=metadata["spec_lower"].to_numpy(dtype=float),
        spec_upper=metadata["spec_upper"].to_numpy(dtype=float),
        units=tuple(metadata["meta_unit"]),
    )


class QualityTrendWidget(QtWidgets.QWidget):
    """選択項目のFrame別測定値トレンド。"""

    hover_text_changed = QtCore.Signal(str)

    def __init__(self, data: QualityTrendData) -> None:
        super().__init__()
        self.data = data
        self.current_colname: str | None = None
        self.current_lot_numbers: tuple[str, ...] = ()
        self.first_lot = 0
        self.current_y_range = (0.0, 1.0)
        self.lot_separators: list[pg.InfiniteLine] = []
        self.mouse_proxy: pg.SignalProxy | None = None
        self._build_ui()
        self.setFixedHeight(QUALITY_TREND_HEIGHT)

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_context_panel())
        layout.addWidget(self._build_plot(), stretch=1)

    def _build_context_panel(self) -> QtWidgets.QWidget:
        """測定値範囲と規格の説明表示。"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("qualityTrendLabelPanel")
        panel.setFixedWidth(DASHBOARD_CONFIG.left_label_width)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 7, 10, 7)
        layout.setSpacing(2)

        title = QtWidgets.QLabel("F推移")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        caption = QtWidgets.QLabel("Frame別の生値範囲")
        caption.setObjectName("qualityTrendCaption")
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("qualityTrendSelectionLabel")
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("qualityTrendSummaryLabel")
        self.summary_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(caption)
        layout.addWidget(self.selection_label)
        layout.addWidget(self.summary_label)
        layout.addStretch()
        return panel

    def _build_plot(self) -> pg.PlotWidget:
        """測定値トレンドプロット生成。"""
        self.plot_widget = pg.PlotWidget(background="#ffffff")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.hideButtons()
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.hideAxis("left")
        self.plot_item.showAxis("right")
        self.plot_item.showGrid(y=True, alpha=0.18)

        bottom_axis = self.plot_item.getAxis("bottom")
        bottom_font = QtGui.QFont()
        bottom_font.setPointSize(6)
        bottom_axis.setHeight(24)
        bottom_axis.setPen(pg.mkPen("#7a8492"))
        bottom_axis.setTextPen(pg.mkPen("#4b5563"))
        bottom_axis.setStyle(
            tickFont=bottom_font,
            tickTextOffset=2,
            tickLength=3,
            hideOverlappingLabels=False,
        )

        right_axis = self.plot_item.getAxis("right")
        right_font = QtGui.QFont()
        right_font.setPointSize(7)
        right_axis.setWidth(DASHBOARD_CONFIG.color_bar_width)
        right_axis.setPen(pg.mkPen("#7a8492"))
        right_axis.setTextPen(pg.mkPen("#303846"))
        right_axis.setStyle(tickFont=right_font, tickTextOffset=3)
        right_axis.enableAutoSIPrefix(False)

        hidden_pen = pg.mkPen(None)
        self.p05_curve = pg.PlotDataItem(pen=hidden_pen)
        self.p95_curve = pg.PlotDataItem(pen=hidden_pen)
        self.p25_curve = pg.PlotDataItem(pen=hidden_pen)
        self.p75_curve = pg.PlotDataItem(pen=hidden_pen)
        self.minimum_curve = pg.PlotDataItem(
            pen=pg.mkPen("#526f79", width=1.0),
        )
        self.maximum_curve = pg.PlotDataItem(
            pen=pg.mkPen("#526f79", width=1.0),
        )
        self.outer_band = pg.FillBetweenItem(
            self.p05_curve,
            self.p95_curve,
            brush=pg.mkBrush(74, 144, 164, 55),
        )
        self.inner_band = pg.FillBetweenItem(
            self.p25_curve,
            self.p75_curve,
            brush=pg.mkBrush(49, 112, 139, 105),
        )
        self.median_curve = pg.PlotDataItem(
            pen=pg.mkPen("#0f6f78", width=1.8),
        )
        self.lower_line = self._build_spec_line("#3157a4")
        self.upper_line = self._build_spec_line("#c43d3d")
        self.hover_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#667080",
                width=1,
                style=QtCore.Qt.PenStyle.DotLine,
            ),
        )
        self.hover_point = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen("#ffffff", width=1),
            brush=pg.mkBrush("#0f6f78"),
        )
        self.hover_line.hide()
        self.hover_point.hide()

        for item, z_value in (
            (self.p05_curve, 1),
            (self.p95_curve, 1),
            (self.outer_band, 1),
            (self.p25_curve, 2),
            (self.p75_curve, 2),
            (self.inner_band, 2),
            (self.minimum_curve, 3),
            (self.maximum_curve, 3),
            (self.median_curve, 4),
            (self.lower_line, 5),
            (self.upper_line, 5),
            (self.hover_line, 8),
            (self.hover_point, 9),
        ):
            item.setZValue(z_value)
            self.plot_item.addItem(item)

        for lot_index in range(1, len(self.data.lot_numbers)):
            separator = make_lot_separator(
                lot_index * len(FRAME_NUMBERS) - 0.5
            )
            self.plot_item.addItem(separator)
            self.lot_separators.append(separator)

        self.mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._show_hover,
        )
        self.plot_widget.viewport().installEventFilter(self)
        return self.plot_widget

    @staticmethod
    def _build_spec_line(color: str) -> pg.InfiniteLine:
        """規格線の生成。"""
        line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(
                color,
                width=1.5,
                style=QtCore.Qt.PenStyle.DashLine,
            ),
        )
        line.hide()
        return line

    def set_context(self, colname: str, first_lot: int) -> None:
        """選択項目と表示lot範囲の反映。"""
        self._clear_hover()
        if self.current_colname != colname:
            self.current_colname = colname
            self._render_colname(colname)
        self._set_lot_range(first_lot)

    def eventFilter(
        self,
        watched: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        """プロット退出時のホバー表示解除。"""
        if (
            watched is self.plot_widget.viewport()
            and event.type() == QtCore.QEvent.Type.Leave
        ):
            self._clear_hover()
        return super().eventFilter(watched, event)

    def _render_colname(self, colname: str) -> None:
        """選択項目の全lot統計値描画。"""
        row = self.data.colname_index(colname)
        x_values = np.arange(len(self.data.frame_numbers), dtype=float)
        self.minimum_curve.setData(x_values, self.data.minimum[row])
        self.p05_curve.setData(x_values, self.data.p05[row])
        self.p25_curve.setData(x_values, self.data.p25[row])
        self.median_curve.setData(x_values, self.data.p50[row])
        self.p75_curve.setData(x_values, self.data.p75[row])
        self.p95_curve.setData(x_values, self.data.p95[row])
        self.maximum_curve.setData(x_values, self.data.maximum[row])

        lower = self.data.spec_lower[row]
        upper = self.data.spec_upper[row]
        self._set_spec_line(self.lower_line, lower)
        self._set_spec_line(self.upper_line, upper)
        self.current_y_range = self._y_range(row)
        self.plot_item.setYRange(
            *self.current_y_range,
            padding=0.0,
        )

        unit = self.data.units[row]
        self.plot_item.getAxis("right").setLabel(
            text=f"生値 ({unit})",
            color="#4b5563",
        )
        self.selection_label.setText(colname)
        spec_lines = []
        if np.isfinite(lower):
            spec_lines.append(f"下限 {lower:g} {unit}")
        if np.isfinite(upper):
            spec_lines.append(f"上限 {upper:g} {unit}")
        self.summary_label.setText(
            "min / max（灰細線）/ 中央値（緑線）\n"
            "P25–P75（濃帯）/ P05–P95（淡帯）\n"
            + "  ".join(spec_lines)
        )

    def _set_lot_range(self, first_lot: int) -> None:
        """表示lot範囲とFrameNo目盛の反映。"""
        last_lot = min(
            first_lot + VISIBLE_LOTS,
            len(self.data.lot_numbers),
        )
        self.first_lot = first_lot
        self.current_lot_numbers = self.data.lot_numbers[
            first_lot:last_lot
        ]
        self.plot_item.setXRange(
            first_lot * len(FRAME_NUMBERS) - 0.5,
            last_lot * len(FRAME_NUMBERS) - 0.5,
            padding=0.0,
        )
        self.plot_item.getAxis("bottom").setTicks(
            [
                self._frame_ticks(
                    first_lot,
                    last_lot - first_lot,
                )
            ]
        )

    def _y_range(self, row: int) -> tuple[float, float]:
        """全lotで共通の生値表示範囲。"""
        values = [self.data.minimum[row], self.data.maximum[row]]
        for spec in (
            self.data.spec_lower[row],
            self.data.spec_upper[row],
        ):
            if np.isfinite(spec):
                values.append(np.asarray([spec]))
        finite_values = np.concatenate(values)
        minimum = float(np.nanmin(finite_values))
        maximum = float(np.nanmax(finite_values))
        span = maximum - minimum
        padding = (
            span * 0.08
            if span > 0.0
            else max(abs(maximum) * 0.05, 1.0)
        )
        return minimum - padding, maximum + padding

    @staticmethod
    def _set_spec_line(
        line: pg.InfiniteLine,
        value: float,
    ) -> None:
        """規格線位置の反映。"""
        line.setVisible(bool(np.isfinite(value)))
        if np.isfinite(value):
            line.setPos(float(value))

    @staticmethod
    def _frame_ticks(
        first_lot: int,
        visible_lot_count: int,
    ) -> list[tuple[float, str]]:
        """表示中lotのFrameNo補助目盛。"""
        ticks: list[tuple[float, str]] = []
        for offset in range(visible_lot_count):
            lot_start = (first_lot + offset) * len(FRAME_NUMBERS)
            if offset == 0:
                ticks.append((float(lot_start), "1"))
            ticks.extend(
                (
                    float(lot_start + frame_number - 1),
                    str(frame_number),
                )
                for frame_number in FRAME_TICK_NUMBERS[1:-1]
            )
            if offset < visible_lot_count - 1:
                ticks.append(
                    (
                        float(lot_start + len(FRAME_NUMBERS) - 0.5),
                        "24/1",
                    )
                )
            else:
                ticks.append(
                    (
                        float(lot_start + len(FRAME_NUMBERS) - 1),
                        "24",
                    )
                )
        return ticks

    def _show_hover(
        self,
        event: tuple[QtCore.QPointF],
    ) -> None:
        """カーソル位置の分位点詳細表示。"""
        scene_position = event[0]
        if not self.plot_widget.sceneBoundingRect().contains(
            scene_position
        ):
            self._clear_hover()
            return
        point = self.plot_item.getViewBox().mapSceneToView(scene_position)
        column = round(point.x())
        if self.current_colname is None:
            self._clear_hover()
            return
        first_column = self.first_lot * len(FRAME_NUMBERS)
        last_column = first_column + (
            len(self.current_lot_numbers) * len(FRAME_NUMBERS)
        )
        if not first_column <= column < last_column:
            self._clear_hover()
            return

        row = self.data.colname_index(self.current_colname)
        median = self.data.p50[row, column]
        self.hover_line.setPos(float(column))
        self.hover_point.setData([float(column)], [float(median)])
        self.hover_line.show()
        self.hover_point.show()
        unit = self.data.units[row]
        self.hover_text_changed.emit(
            f"{self.data.column_lots[column]}  |  "
            f"FrameNo {int(self.data.frame_numbers[column])}  |  "
            f"{self.current_colname}  |  "
            f"min {self.data.minimum[row, column]:.5g}  "
            f"P05 {self.data.p05[row, column]:.5g}  "
            f"P25 {self.data.p25[row, column]:.5g}  "
            f"中央値 {median:.5g}  "
            f"P75 {self.data.p75[row, column]:.5g}  "
            f"P95 {self.data.p95[row, column]:.5g}  "
            f"max {self.data.maximum[row, column]:.5g} {unit}  |  "
            f"N {int(self.data.sample_counts[row, column]):,}  "
            f"NG {int(self.data.ng_counts[row, column]):,}"
        )

    def _clear_hover(self) -> None:
        """ホバー表示の解除。"""
        self.hover_line.hide()
        self.hover_point.hide()
        self.hover_text_changed.emit("")
