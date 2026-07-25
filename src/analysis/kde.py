from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.quality_columns import SPEC_ORDER
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository


@dataclass(frozen=True)
class KdeData:
    """lot・検査項目別のKDEデータ。"""

    colnames: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    x_values: np.ndarray
    densities: np.ndarray
    sample_counts: np.ndarray
    in_range_counts: np.ndarray
    display_min: np.ndarray
    display_max: np.ndarray
    spec_lower: np.ndarray
    spec_upper: np.ndarray
    spec_best: np.ndarray

    def colname_index(self, colname: str) -> int:
        """検査項目の配列位置。"""
        return self.colnames.index(colname)


def build_kde_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> KdeData:
    """lot・検査項目別KDEデータ生成。"""
    all_lot_numbers = tuple(
        record[0] for record in repository.lots()
    )
    if lot_numbers is None:
        lot_numbers = all_lot_numbers
    query_lot_numbers = (
        None if lot_numbers == all_lot_numbers else lot_numbers
    )

    bin_count = DASHBOARD_CONFIG.kde_bins
    frame = repository.kde_bins_by_colname_lot(
        bin_count,
        query_lot_numbers,
    )
    available = set(frame["colname"])
    colnames = tuple(
        colname for colname in SPEC_ORDER if colname in available
    )
    frame = frame[frame["colname"].isin(colnames)]
    colname_indices = {
        colname: index for index, colname in enumerate(colnames)
    }
    lot_indices = {
        lot_number: index
        for index, lot_number in enumerate(lot_numbers)
    }

    counts = np.zeros(
        (len(colnames), len(lot_numbers), bin_count),
        dtype=np.int64,
    )
    rows = frame["colname"].map(colname_indices).to_numpy(dtype=int)
    lots = frame["lot_number"].map(lot_indices).to_numpy(dtype=int)
    row_counts = frame["count"].to_numpy(dtype=np.int64)
    sample_counts = np.zeros(
        (len(colnames), len(lot_numbers)),
        dtype=np.int64,
    )
    np.add.at(sample_counts, (rows, lots), row_counts)

    binned = frame[frame["bin_index"].notna()]
    rows = binned["colname"].map(colname_indices).to_numpy(dtype=int)
    lots = binned["lot_number"].map(lot_indices).to_numpy(dtype=int)
    bins = binned["bin_index"].to_numpy(dtype=int)
    counts[rows, lots, bins] = binned["count"].to_numpy(
        dtype=np.int64
    )
    in_range_counts = counts.sum(axis=2)

    x_values = np.empty((len(colnames), bin_count))
    bin_widths = np.empty(len(colnames))
    display_min = np.empty(len(colnames))
    display_max = np.empty(len(colnames))
    spec_lower = np.full(len(colnames), np.nan)
    spec_upper = np.full(len(colnames), np.nan)
    spec_best = np.full(len(colnames), np.nan)
    for row, colname in enumerate(colnames):
        first = frame[frame["colname"] == colname].iloc[0]
        display_min[row] = float(first["plot_min"])
        display_max[row] = float(first["plot_max"])
        edges = np.linspace(
            display_min[row],
            display_max[row],
            bin_count + 1,
        )
        bin_widths[row] = edges[1] - edges[0]
        x_values[row] = edges[:-1] + bin_widths[row] / 2.0
        spec_lower[row] = float(first["spec_lower"])
        spec_upper[row] = float(first["spec_upper"])
        spec_best[row] = float(first["spec_best"])

    kernel = _gaussian_kernel(DASHBOARD_CONFIG.kde_bandwidth_bins)
    radius = len(kernel) // 2
    padded = np.pad(counts, ((0, 0), (0, 0), (radius, radius)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        len(kernel),
        axis=2,
    )
    smoothed = np.sum(windows * kernel, axis=-1)
    smoothed_totals = smoothed.sum(axis=2, keepdims=True)
    normalized = np.divide(
        smoothed,
        smoothed_totals,
        out=np.zeros_like(smoothed),
        where=smoothed_totals > 0.0,
    )
    coverage = in_range_counts / sample_counts
    densities = (
        normalized
        * coverage[:, :, None]
        / bin_widths[:, None, None]
    )

    return KdeData(
        colnames=colnames,
        lot_numbers=lot_numbers,
        x_values=x_values,
        densities=densities,
        sample_counts=sample_counts,
        in_range_counts=in_range_counts,
        display_min=display_min,
        display_max=display_max,
        spec_lower=spec_lower,
        spec_upper=spec_upper,
        spec_best=spec_best,
    )


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """bin空間のGaussianカーネル。"""
    radius = math.ceil(4.0 * sigma)
    positions = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()


class KdeWidget(QtWidgets.QWidget):
    """表示中lot別のKDEプロット。"""

    def __init__(self, data: KdeData) -> None:
        super().__init__()
        self.data = data
        self.current_colname: str | None = None
        self.current_lot_numbers: tuple[str, ...] = ()
        self.plot_widgets: list[pg.PlotWidget] = []
        self.plot_items: list[pg.PlotItem] = []
        self.curve_items: list[pg.PlotDataItem] = []
        self.lower_lines: list[pg.InfiniteLine] = []
        self.center_lines: list[pg.InfiniteLine] = []
        self.upper_lines: list[pg.InfiniteLine] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_context_panel())

        for _ in range(DASHBOARD_CONFIG.visible_lots):
            plot_widget = pg.PlotWidget(background="#ffffff")
            plot_item = plot_widget.getPlotItem()
            plot_item.setMenuEnabled(False)
            plot_item.hideButtons()
            plot_item.hideAxis("left")
            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.getViewBox().setBorder(
                pg.mkPen("#dce2e8", width=0.8)
            )

            axis = plot_item.getAxis("bottom")
            axis_font = QtGui.QFont()
            axis_font.setPointSize(6)
            axis.setHeight(28)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=3)

            curve = pg.PlotDataItem(
                pen=pg.mkPen("#287d78", width=1.5),
                fillLevel=0.0,
                brush=pg.mkBrush(58, 154, 147, 80),
            )
            lower_line = self._build_spec_line("#3157a4")
            center_line = self._build_spec_line(
                "#667080",
                QtCore.Qt.PenStyle.DashDotLine,
            )
            upper_line = self._build_spec_line("#c43d3d")
            plot_item.addItem(curve)
            plot_item.addItem(lower_line)
            plot_item.addItem(center_line)
            plot_item.addItem(upper_line)

            self.plot_widgets.append(plot_widget)
            self.plot_items.append(plot_item)
            self.curve_items.append(curve)
            self.lower_lines.append(lower_line)
            self.center_lines.append(center_line)
            self.upper_lines.append(upper_line)
            layout.addWidget(plot_widget, stretch=1)

        right_spacer = QtWidgets.QWidget()
        right_spacer.setFixedWidth(DASHBOARD_CONFIG.color_bar_width)
        layout.addWidget(right_spacer)

    def _build_context_panel(self) -> QtWidgets.QWidget:
        """分布種別と規格値の表示。"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("kdeLabelPanel")
        panel.setFixedWidth(DASHBOARD_CONFIG.left_label_width)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 10, 8)
        layout.setSpacing(3)

        title = QtWidgets.QLabel("KDE")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        caption = QtWidgets.QLabel("lot別の測定値分布")
        caption.setObjectName("kdeContextCaption")
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("kdeSummaryLabel")
        self.summary_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(caption)
        layout.addSpacing(5)
        layout.addWidget(self.summary_label)
        layout.addStretch()
        return panel

    @staticmethod
    def _build_spec_line(
        color: str,
        style: QtCore.Qt.PenStyle = QtCore.Qt.PenStyle.DashLine,
    ) -> pg.InfiniteLine:
        """規格線の生成。"""
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                color,
                width=1.6,
                style=style,
            ),
        )
        line.setZValue(20)
        line.hide()
        return line

    def set_context(self, colname: str, first_lot: int) -> None:
        """選択項目と表示lot範囲の反映。"""
        self.current_colname = colname
        row = self.data.colname_index(colname)
        last_lot = first_lot + DASHBOARD_CONFIG.visible_lots
        self.current_lot_numbers = self.data.lot_numbers[
            first_lot:last_lot
        ]
        x_values = self.data.x_values[row]
        visible_densities = self.data.densities[
            row,
            first_lot:last_lot,
        ]
        display_min = self.data.display_min[row]
        display_max = self.data.display_max[row]
        y_max = max(
            1.0,
            float(self.data.densities[row].max()) * 1.12,
        )
        lower = self.data.spec_lower[row]
        upper = self.data.spec_upper[row]
        best = self.data.spec_best[row]
        center = (
            (lower + upper) / 2.0
            if np.isfinite(lower) and np.isfinite(upper)
            else np.nan
        )

        for offset, plot_item in enumerate(self.plot_items):
            density = visible_densities[offset]
            self.curve_items[offset].setData(x_values, density)
            plot_item.setXRange(
                display_min,
                display_max,
                padding=0.0,
            )
            plot_item.setYRange(0.0, y_max, padding=0.0)
            self._set_spec_line(self.lower_lines[offset], lower)
            self._set_spec_line(self.center_lines[offset], center)
            self._set_spec_line(self.upper_lines[offset], upper)

        visible_counts = self.data.sample_counts[
            row,
            first_lot:last_lot,
        ]
        sample_count = int(visible_counts.min())
        outside_rates = 1.0 - (
            self.data.in_range_counts[row, first_lot:last_lot]
            / visible_counts
        )
        lines = [f"各 {sample_count:,}測定"]
        if np.isfinite(center):
            lines.extend(
                [
                    f"規格下限（青） {self._format_limit(lower)}",
                    f"規格中心（灰） {self._format_limit(center)}",
                    f"規格上限（赤） {self._format_limit(upper)}",
                ]
            )
        else:
            lines.append(f"最良値 {self._format_limit(best)}")
            if np.isfinite(upper):
                lines.append(
                    f"規格上限（赤） {self._format_limit(upper)}"
                )
            else:
                lines.append(
                    f"規格下限（青） {self._format_limit(lower)}"
                )
        lines.append(
            f"表示範囲外 最大 {outside_rates.max():.1%}"
        )
        self.summary_label.setText("\n".join(lines))

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
    def _format_limit(value: float) -> str:
        """規格値の表示文字列。"""
        return f"{value:g}" if np.isfinite(value) else "—"
