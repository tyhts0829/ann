from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.frame_map import SingleFrameMapData
from src.analysis.kde import _gaussian_kernel
from src.dashboard_config import DASHBOARD_CONFIG


@dataclass(frozen=True)
class SingleFrameKdeData:
    """単一FrameのKDEデータ。"""

    colname: str
    frame_no: int
    x_values: np.ndarray
    density: np.ndarray
    sample_count: int
    in_range_count: int
    display_min: float
    display_max: float
    spec_lower: float
    spec_upper: float
    spec_best: float
    unit: str

    @property
    def outside_rate(self) -> float:
        """KDE表示範囲外の測定割合。"""
        return 1.0 - self.in_range_count / self.sample_count


def build_single_frame_kde_data(
    data: SingleFrameMapData,
) -> SingleFrameKdeData:
    """単一Frameの測定値によるKDEデータ生成。"""
    values = np.asarray(data.raw_values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    display_min, display_max = _display_range(data, values)
    edges = np.linspace(
        display_min,
        display_max,
        DASHBOARD_CONFIG.kde_bins + 1,
    )
    in_range = values[(values >= display_min) & (values <= display_max)]
    counts, _ = np.histogram(in_range, bins=edges)
    kernel = _gaussian_kernel(DASHBOARD_CONFIG.kde_bandwidth_bins)
    radius = len(kernel) // 2
    padded = np.pad(counts, (radius, radius))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        len(kernel),
    )
    smoothed = np.sum(windows * kernel, axis=-1)
    bin_width = edges[1] - edges[0]
    density = np.divide(
        smoothed,
        smoothed.sum(),
        out=np.zeros_like(smoothed),
        where=smoothed.sum() > 0.0,
    )
    density *= len(in_range) / len(values) / bin_width

    return SingleFrameKdeData(
        colname=data.colname,
        frame_no=data.frame_no,
        x_values=edges[:-1] + bin_width / 2.0,
        density=density,
        sample_count=len(values),
        in_range_count=len(in_range),
        display_min=display_min,
        display_max=display_max,
        spec_lower=data.spec_lower,
        spec_upper=data.spec_upper,
        spec_best=data.spec_best,
        unit=data.unit,
    )


def _display_range(
    data: SingleFrameMapData,
    values: np.ndarray,
) -> tuple[float, float]:
    """規格を基準とするKDE表示範囲。"""
    lower = data.spec_lower
    upper = data.spec_upper
    best = data.spec_best
    if np.isfinite(lower) and np.isfinite(upper):
        spec_span = upper - lower
        return lower - spec_span / 4.0, upper + spec_span / 4.0
    if np.isfinite(upper) and np.isfinite(best):
        return best, best + (upper - best) * 1.2
    if np.isfinite(lower) and np.isfinite(best):
        return lower - (best - lower) / 5.0, best

    raw_min = float(values.min())
    raw_max = float(values.max())
    raw_span = max(raw_max - raw_min, abs(raw_max) * 0.01, 1e-9)
    return raw_min - raw_span * 0.04, raw_max + raw_span * 0.04


class SingleFrameKdeWidget(QtWidgets.QWidget):
    """単一Frameの測定値KDE。"""

    def __init__(
        self,
        data: SingleFrameKdeData | None = None,
    ) -> None:
        super().__init__()
        self.data: SingleFrameKdeData | None = None
        self._build_ui()
        if data is not None:
            self.set_data(data)

    def _build_ui(self) -> None:
        """KDE表示部品の生成。"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_plot(), stretch=1)

    def _build_header(self) -> QtWidgets.QWidget:
        """選択Frame情報欄の生成。"""
        header = QtWidgets.QWidget()
        header.setObjectName("singleFrameKdeHeader")
        layout = QtWidgets.QVBoxLayout(header)
        layout.setContentsMargins(12, 9, 12, 7)
        layout.setSpacing(3)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("KDE")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        self.caption_label = QtWidgets.QLabel()
        self.caption_label.setObjectName("kdeContextCaption")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.caption_label)

        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("singleFrameKdeSelectionLabel")
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("kdeSummaryLabel")
        self.summary_label.setWordWrap(True)

        layout.addLayout(title_row)
        layout.addWidget(self.selection_label)
        layout.addWidget(self.summary_label)
        return header

    def _build_plot(self) -> pg.PlotWidget:
        """KDEプロットの生成。"""
        self.plot_widget = pg.PlotWidget(background="#ffffff")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.hideButtons()
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.showGrid(y=True, alpha=0.18)
        self.plot_item.getViewBox().setBorder(pg.mkPen("#dce2e8", width=0.8))

        axis_font = QtGui.QFont()
        axis_font.setPointSize(7)
        for axis_name in ("bottom", "left"):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=3)
        self.plot_item.getAxis("left").setLabel(
            text="密度",
            color="#4b5563",
        )

        self.curve_item = pg.PlotDataItem(
            pen=pg.mkPen("#287d78", width=1.8),
            fillLevel=0.0,
            brush=pg.mkBrush(58, 154, 147, 80),
        )
        self.lower_line = self._build_spec_line("#3157a4")
        self.center_line = self._build_spec_line(
            "#667080",
            QtCore.Qt.PenStyle.DashDotLine,
        )
        self.upper_line = self._build_spec_line("#c43d3d")
        for item in (
            self.curve_item,
            self.lower_line,
            self.center_line,
            self.upper_line,
        ):
            self.plot_item.addItem(item)
        return self.plot_widget

    @staticmethod
    def _build_spec_line(
        color: str,
        style: QtCore.Qt.PenStyle = QtCore.Qt.PenStyle.DashLine,
    ) -> pg.InfiniteLine:
        """規格線の生成。"""
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color, width=1.6, style=style),
        )
        line.setZValue(20)
        line.hide()
        return line

    def set_data(self, data: SingleFrameKdeData) -> None:
        """選択FrameのKDE表示。"""
        self.data = data
        self.curve_item.setData(data.x_values, data.density)
        self.plot_item.setXRange(
            data.display_min,
            data.display_max,
            padding=0.0,
        )
        y_max = max(1.0, float(data.density.max()) * 1.12)
        self.plot_item.setYRange(0.0, y_max, padding=0.0)

        center = (
            (data.spec_lower + data.spec_upper) / 2.0
            if np.isfinite(data.spec_lower) and np.isfinite(data.spec_upper)
            else np.nan
        )
        self._set_spec_line(self.lower_line, data.spec_lower)
        self._set_spec_line(self.center_line, center)
        self._set_spec_line(self.upper_line, data.spec_upper)
        self.plot_item.getAxis("bottom").setLabel(
            text=f"生値 ({data.unit})",
            color="#4b5563",
        )

        self.caption_label.setText(
            f"FrameNo {data.frame_no:02d}  |  {data.sample_count:,}測定"
        )
        self.selection_label.setText(data.colname)
        self.summary_label.setText(self._summary(data, center))

    @staticmethod
    def _set_spec_line(
        line: pg.InfiniteLine,
        value: float,
    ) -> None:
        """規格線位置の反映。"""
        line.setVisible(bool(np.isfinite(value)))
        if np.isfinite(value):
            line.setPos(float(value))

    @classmethod
    def _summary(
        cls,
        data: SingleFrameKdeData,
        center: float,
    ) -> str:
        """規格値と表示範囲外割合の要約。"""
        lines: list[str] = []
        if np.isfinite(center):
            lines.extend(
                (
                    f"規格下限（青） {cls._format_limit(data.spec_lower)}",
                    f"規格中心（灰） {cls._format_limit(center)}",
                    f"規格上限（赤） {cls._format_limit(data.spec_upper)}",
                )
            )
        else:
            lines.append(f"最良値 {cls._format_limit(data.spec_best)}")
            if np.isfinite(data.spec_upper):
                lines.append(f"規格上限（赤） {cls._format_limit(data.spec_upper)}")
            else:
                lines.append(f"規格下限（青） {cls._format_limit(data.spec_lower)}")
        lines.append(f"表示範囲外 {data.outside_rate:.1%}")
        return "\n".join(lines)

    @staticmethod
    def _format_limit(value: float) -> str:
        """規格値の表示文字列。"""
        return f"{value:g}" if np.isfinite(value) else "—"
