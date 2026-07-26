from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

from src.analysis.map_palettes import (
    DENSITY_STOPS,
    make_density_color_map,
)
from src.analysis.plot_style import make_lot_separator
from src.analysis.quality_columns import SPEC_ORDER
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository

FRAME_NUMBERS = np.arange(1, 25)
FRAME_TICK_NUMBERS = (1, 6, 12, 18, 24)
VISIBLE_LOTS = DASHBOARD_CONFIG.visible_lots
QUALITY_TREND_HEIGHT = DASHBOARD_CONFIG.quality_trend_height
VALUE_AXIS_WIDTH = 50
DENSITY_LEGEND_WIDTH = (
    DASHBOARD_CONFIG.color_bar_width - VALUE_AXIS_WIDTH
)


@dataclass(frozen=True)
class QualityTrendData:
    """lot・FrameNo別の測定値密度データ。"""

    colnames: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    column_lots: tuple[str, ...]
    frame_numbers: np.ndarray
    bin_values: np.ndarray
    densities: np.ndarray
    density_scales: np.ndarray
    sample_counts: np.ndarray
    in_range_counts: np.ndarray
    ng_counts: np.ndarray
    display_min: np.ndarray
    display_max: np.ndarray
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
    """lot・FrameNo別の測定値密度データ生成。"""
    if lot_numbers is None:
        lot_numbers = tuple(
            record[0] for record in repository.lots()
        )

    bin_count = DASHBOARD_CONFIG.kde_bins
    frame = repository.density_bins_by_colname_frame(
        bin_count,
        lot_numbers,
    )
    available = set(frame["colname"])
    colnames = tuple(
        colname for colname in SPEC_ORDER if colname in available
    )
    colname_indices = {
        colname: index for index, colname in enumerate(colnames)
    }
    lot_indices = {
        lot_number: index
        for index, lot_number in enumerate(lot_numbers)
    }
    column_count = len(lot_numbers) * len(FRAME_NUMBERS)
    shape = (len(colnames), column_count)

    counts = np.zeros(
        (*shape, bin_count),
        dtype=np.float32,
    )
    sample_counts = np.zeros(shape, dtype=np.int64)
    in_range_counts = np.zeros(shape, dtype=np.int64)
    ng_counts = np.zeros(shape, dtype=np.int64)
    for record in frame.itertuples(index=False):
        row = colname_indices[record.colname]
        column = (
            lot_indices[record.lot_number] * len(FRAME_NUMBERS)
            + int(record.FrameNo)
            - 1
        )
        bin_indices = np.asarray(record.bin_indices, dtype=int)
        counts[row, column, bin_indices] = np.asarray(
            record.bin_counts,
            dtype=np.float32,
        )
        sample_counts[row, column] = int(record.sample_count)
        in_range_counts[row, column] = int(record.in_range_count)
        ng_counts[row, column] = int(record.ng_count)

    metadata = (
        frame.drop_duplicates("colname")
        .set_index("colname")
        .reindex(colnames)
    )
    display_min = metadata["plot_min"].to_numpy(dtype=float)
    display_max = metadata["plot_max"].to_numpy(dtype=float)
    bin_widths = (display_max - display_min) / bin_count
    bin_values = (
        display_min[:, None]
        + (
            np.arange(bin_count, dtype=float)[None, :] + 0.5
        )
        * bin_widths[:, None]
    )

    gaussian_filter1d(
        counts,
        DASHBOARD_CONFIG.kde_bandwidth_bins,
        axis=2,
        output=counts,
        mode="constant",
        truncate=4.0,
    )
    totals = counts.sum(axis=2, keepdims=True)
    np.divide(
        counts,
        totals,
        out=counts,
        where=totals > 0.0,
    )
    coverage = np.divide(
        in_range_counts,
        sample_counts,
        out=np.zeros(shape, dtype=float),
        where=sample_counts > 0,
    )
    counts *= coverage[:, :, None]
    counts /= bin_widths[:, None, None]
    densities = counts.transpose(0, 2, 1).copy()
    density_scales = np.asarray(
        [_density_scale(density) for density in densities],
        dtype=float,
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
        bin_values=bin_values,
        densities=densities,
        density_scales=density_scales,
        sample_counts=sample_counts,
        in_range_counts=in_range_counts,
        ng_counts=ng_counts,
        display_min=display_min,
        display_max=display_max,
        spec_lower=metadata["spec_lower"].to_numpy(dtype=float),
        spec_upper=metadata["spec_upper"].to_numpy(dtype=float),
        units=tuple(metadata["meta_unit"]),
    )


def _density_scale(density: np.ndarray) -> float:
    """全lot共通の密度色上限。"""
    positive = density[density > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.percentile(positive, 99.5))


class QualityTrendWidget(QtWidgets.QWidget):
    """選択項目のFrame別測定値密度マップ。"""

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
        layout.addWidget(self._build_density_legend())

    def _build_context_panel(self) -> QtWidgets.QWidget:
        """密度と規格の説明表示。"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("qualityTrendLabelPanel")
        panel.setFixedWidth(DASHBOARD_CONFIG.left_label_width)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 7, 10, 7)
        layout.setSpacing(2)

        title = QtWidgets.QLabel("F推移")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        caption = QtWidgets.QLabel("Frame別の生値密度")
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
        """測定値密度マップ生成。"""
        self.plot_widget = pg.PlotWidget(background="#ffffff")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.hideButtons()
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("top")
        self.plot_item.showAxis("right")
        self.plot_item.showGrid(y=True, alpha=0.16)

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
        right_axis.setWidth(VALUE_AXIS_WIDTH)
        right_axis.setPen(pg.mkPen("#7a8492"))
        right_axis.setTextPen(pg.mkPen("#303846"))
        right_axis.setStyle(tickFont=right_font, tickTextOffset=3)
        right_axis.enableAutoSIPrefix(False)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setColorMap(make_density_color_map())
        self.image_item.setZValue(0)
        self.plot_item.addItem(self.image_item)

        self.lower_line = self._build_spec_line("#3157a4")
        self.upper_line = self._build_spec_line("#c43d3d")
        self.hover_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#4f5968",
                width=1,
                style=QtCore.Qt.PenStyle.DotLine,
            ),
        )
        self.hover_line.setZValue(30)
        self.hover_line.hide()
        self.plot_item.addItem(self.lower_line)
        self.plot_item.addItem(self.upper_line)
        self.plot_item.addItem(self.hover_line)

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

    def _build_density_legend(self) -> QtWidgets.QWidget:
        """密度カラーマップ凡例の生成。"""
        legend = QtWidgets.QWidget()
        legend.setObjectName("densityLegend")
        legend.setFixedWidth(DENSITY_LEGEND_WIDTH)
        layout = QtWidgets.QVBoxLayout(legend)
        layout.setContentsMargins(4, 4, 4, 24)
        layout.setSpacing(1)

        title = QtWidgets.QLabel("密度")
        title.setObjectName("densityLegendTitle")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        high = QtWidgets.QLabel("高")
        high.setObjectName("densityLegendLabel")
        high.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        gradient = QtWidgets.QFrame()
        gradient.setObjectName("densityGradient")
        gradient.setMinimumWidth(12)
        gradient.setStyleSheet(self._density_gradient_stylesheet())
        low = QtWidgets.QLabel("低")
        low.setObjectName("densityLegendLabel")
        low.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(high)
        layout.addWidget(gradient, stretch=1)
        layout.addWidget(low)
        return legend

    @staticmethod
    def _density_gradient_stylesheet() -> str:
        """密度凡例用グラデーション定義。"""
        stops = ", ".join(
            f"stop:{position:g} {color}"
            for position, color in DENSITY_STOPS
        )
        return (
            "QFrame#densityGradient {"
            "background: qlineargradient("
            f"x1:0, y1:1, x2:0, y2:0, {stops}"
            ");"
            "border: 1px solid #aeb7c4;"
            "}"
        )

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
        line.setZValue(20)
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
        """選択項目の全lot密度描画。"""
        row = self.data.colname_index(colname)
        minimum = self.data.display_min[row]
        maximum = self.data.display_max[row]
        density_scale = self.data.density_scales[row]
        self.image_item.setImage(
            self.data.densities[row],
            autoLevels=False,
            levels=(0.0, density_scale),
        )
        self.image_item.setRect(
            QtCore.QRectF(
                -0.5,
                minimum,
                float(len(self.data.frame_numbers)),
                maximum - minimum,
            )
        )

        lower = self.data.spec_lower[row]
        upper = self.data.spec_upper[row]
        self._set_spec_line(self.lower_line, lower)
        self._set_spec_line(self.upper_line, upper)
        self.current_y_range = (minimum, maximum)
        self.plot_item.setYRange(
            *self.current_y_range,
            padding=0.0,
        )

        unit = self.data.units[row]
        self.selection_label.setText(colname)
        spec_lines = []
        if np.isfinite(lower):
            spec_lines.append(f"下限 {lower:g} {unit}")
        if np.isfinite(upper):
            spec_lines.append(f"上限 {upper:g} {unit}")
        self.summary_label.setText(
            f"生値軸 {minimum:.4g}–{maximum:.4g} {unit}\n"
            "密度色 低→高（全lot共通）\n"
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
        """カーソル位置の密度詳細表示。"""
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
        minimum, maximum = self.current_y_range
        bin_index = int(
            np.clip(
                (point.y() - minimum)
                / (maximum - minimum)
                * self.data.densities.shape[1],
                0,
                self.data.densities.shape[1] - 1,
            )
        )
        density = self.data.densities[row, bin_index, column]
        self.hover_line.setPos(float(column))
        self.hover_line.show()
        unit = self.data.units[row]
        self.hover_text_changed.emit(
            f"{self.data.column_lots[column]}  |  "
            f"FrameNo {int(self.data.frame_numbers[column])}  |  "
            f"{self.current_colname}  |  "
            f"生値 {point.y():.5g} {unit}  "
            f"密度 {density:.5g}  |  "
            f"N {int(self.data.sample_counts[row, column]):,}  "
            f"NG {int(self.data.ng_counts[row, column]):,}"
        )

    def _clear_hover(self) -> None:
        """ホバー表示の解除。"""
        self.hover_line.hide()
        self.hover_text_changed.emit("")
