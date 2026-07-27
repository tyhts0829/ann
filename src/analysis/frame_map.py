from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.map_palettes import (
    MAP_DEFINITIONS,
    PURPLES,
    make_color_map,
)
from src.analysis.plot_style import (
    LotSeparatorWidget,
    make_lot_separator_widget,
)
from src.analysis.quality_columns import SPEC_ORDER
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository

POSITION_X = np.arange(1, 25)
POSITION_Y = np.arange(1, 13)
VISIBLE_LOTS = DASHBOARD_CONFIG.visible_lots
LEFT_LABEL_WIDTH = DASHBOARD_CONFIG.left_label_width
COLOR_BAR_WIDTH = DASHBOARD_CONFIG.color_bar_width
FRAME_MAP_HEIGHT = DASHBOARD_CONFIG.fmap_height
DEFAULT_ROW_HEIGHT = FRAME_MAP_HEIGHT // len(MAP_DEFINITIONS)
LOWER_NG_RGBA = (23, 63, 138, 255)
UPPER_NG_RGBA = (143, 29, 29, 255)


@dataclass(frozen=True)
class FrameMapData:
    """lot・検査項目別の製品座標マップ集計結果。"""

    colnames: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    ng_rates: np.ndarray
    normalized_mean: np.ndarray
    normalized_std: np.ndarray

    def colname_index(self, colname: str) -> int:
        """検査項目の配列位置。"""
        return self.colnames.index(colname)


@dataclass(frozen=True)
class SingleFrameMapData:
    """単一Frameの製品座標マップ。"""

    lot_number: str
    frame_no: int
    colname: str
    raw_values: np.ndarray
    normalized_values: np.ndarray
    ng_flags: np.ndarray
    lower_ng_flags: np.ndarray
    upper_ng_flags: np.ndarray
    sample_count: int
    spec_lower: float
    spec_upper: float
    spec_best: float
    unit: str


@dataclass
class FrameMapRow:
    """同一指標の表示中lot別フレームマップ行。"""

    metric: str
    label: str
    widget: QtWidgets.QWidget
    plot_widgets: list[pg.PlotWidget]
    image_items: list[pg.ImageItem]
    color_bar: pg.ColorBarItem
    lot_separators: list[LotSeparatorWidget]
    levels: tuple[float, float] = (0.0, 1.0)


@dataclass
class SingleFrameMapView:
    """単一Frameの生値製品座標マップ。"""

    plot_widget: pg.PlotWidget
    plot_item: pg.PlotItem
    image_item: pg.ImageItem
    ng_overlay_item: pg.ImageItem
    color_bar: pg.ColorBarItem
    levels: tuple[float, float] = (0.0, 1.0)


def build_frame_map_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> FrameMapData:
    """PositionX・PositionY別の3種類のマップデータ生成。"""
    if lot_numbers is None:
        lot_numbers = tuple(record[0] for record in repository.lots())

    frame = repository.metrics_by_colname_position(lot_numbers)
    available = set(frame["colname"])
    colnames = tuple(colname for colname in SPEC_ORDER if colname in available)
    return FrameMapData(
        colnames=colnames,
        lot_numbers=lot_numbers,
        ng_rates=_position_matrices(
            frame,
            "ng_rate",
            colnames,
            lot_numbers,
        ),
        normalized_mean=_position_matrices(
            frame,
            "normalized_mean",
            colnames,
            lot_numbers,
        ),
        normalized_std=_position_matrices(
            frame,
            "normalized_std",
            colnames,
            lot_numbers,
        ),
    )


def build_single_frame_map_data(
    repository: QualityRepository,
    lot_number: str,
    frame_no: int,
    colname: str,
) -> SingleFrameMapData:
    """指定Frameの個片マップデータ生成。"""
    frame = repository.values_by_colname_frame(
        lot_number,
        frame_no,
        colname,
    )
    if frame.empty:
        raise ValueError("対象Frameのデータがありません。")

    index = pd.MultiIndex.from_product(
        [POSITION_Y, POSITION_X],
        names=["PositionY", "PositionX"],
    )
    indexed = frame.set_index(["PositionY", "PositionX"]).reindex(index)
    shape = (len(POSITION_Y), len(POSITION_X))
    raw_values = indexed["value"].to_numpy(dtype=float).reshape(shape)
    first = frame.iloc[0]
    spec_lower = float(first["limmin"])
    spec_upper = float(first["limmax"])
    lower_ng_flags = (
        raw_values < spec_lower
        if np.isfinite(spec_lower)
        else np.zeros(shape, dtype=bool)
    )
    upper_ng_flags = (
        raw_values > spec_upper
        if np.isfinite(spec_upper)
        else np.zeros(shape, dtype=bool)
    )
    return SingleFrameMapData(
        lot_number=lot_number,
        frame_no=frame_no,
        colname=colname,
        raw_values=raw_values,
        normalized_values=(
            indexed["normalized_value"].to_numpy(dtype=float).reshape(shape)
        ),
        ng_flags=lower_ng_flags | upper_ng_flags,
        lower_ng_flags=lower_ng_flags,
        upper_ng_flags=upper_ng_flags,
        sample_count=len(frame),
        spec_lower=spec_lower,
        spec_upper=spec_upper,
        spec_best=float(first["meta_best"]),
        unit=str(first["meta_unit"]),
    )


def build_ng_overlay(data: SingleFrameMapData) -> np.ndarray:
    """上下限NG方向を表すRGBA画像生成。"""
    overlay = np.zeros((*data.raw_values.shape, 4), dtype=np.ubyte)
    overlay[data.lower_ng_flags] = LOWER_NG_RGBA
    overlay[data.upper_ng_flags] = UPPER_NG_RGBA
    return overlay


def _position_matrices(
    frame: pd.DataFrame,
    value_column: str,
    colnames: tuple[str, ...],
    lot_numbers: tuple[str, ...],
) -> np.ndarray:
    """検査項目ごとのlot・製品座標行列。"""
    index = pd.MultiIndex.from_product(
        [lot_numbers, POSITION_Y, POSITION_X],
        names=["lot_number", "PositionY", "PositionX"],
    )
    matrices = []
    for colname in colnames:
        subset = frame[frame["colname"] == colname]
        matrix = (
            subset.set_index(["lot_number", "PositionY", "PositionX"])[
                value_column
            ]
            .reindex(index)
            .to_numpy(dtype=float)
            .reshape(
                len(lot_numbers),
                len(POSITION_Y),
                len(POSITION_X),
            )
        )
        matrices.append(matrix)
    return np.stack(matrices)


def _add_white_grid(plot_item: pg.PlotItem) -> None:
    """24×12製品セルの白グリッド。"""
    grid_pen = pg.mkPen("#ffffff", width=0.7)
    for x_position in np.arange(1.5, 24.5, 1.0):
        line = pg.InfiniteLine(
            pos=x_position,
            angle=90,
            movable=False,
            pen=grid_pen,
        )
        line.setZValue(20)
        plot_item.addItem(line)
    for y_position in np.arange(1.5, 12.5, 1.0):
        line = pg.InfiniteLine(
            pos=y_position,
            angle=0,
            movable=False,
            pen=grid_pen,
        )
        line.setZValue(20)
        plot_item.addItem(line)


class FrameMapWidget(QtWidgets.QWidget):
    """lot集約の3段製品座標マップ。"""

    frame_mode_requested = QtCore.Signal()

    def __init__(
        self,
        repository: QualityRepository,
        lot_numbers: tuple[str, ...],
        data: FrameMapData | None = None,
    ) -> None:
        super().__init__()
        self.data = (
            build_frame_map_data(repository, lot_numbers)
            if data is None
            else data
        )
        self.rows: list[FrameMapRow] = []
        self.selected_colname: str | None = None
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("frameMapSelectionLabel")
        self.selection_label.setWordWrap(True)
        self.frame_mode_button = QtWidgets.QToolButton()
        self.frame_mode_button.setObjectName("frameMapModeButton")
        self.frame_mode_button.setText("選択Frameを表示")
        self.frame_mode_button.setEnabled(False)
        self.frame_mode_button.clicked.connect(
            self.frame_mode_requested.emit
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._build_ui()
        self.setFixedHeight(FRAME_MAP_HEIGHT)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        base_height, remainder = divmod(
            FRAME_MAP_HEIGHT,
            len(MAP_DEFINITIONS),
        )
        for index, (metric, label, colors) in enumerate(MAP_DEFINITIONS):
            row = self._build_row(
                metric,
                label,
                make_color_map(colors),
                base_height + (index < remainder),
            )
            self.rows.append(row)
            layout.addWidget(row.widget)

    def _build_row(
        self,
        metric: str,
        label: str,
        color_map: pg.ColorMap,
        row_height: int,
    ) -> FrameMapRow:
        """表示中lotのPositionマップ行生成。"""
        widget = QtWidgets.QWidget()
        widget.setObjectName("frameMapRow")
        widget.setFixedHeight(row_height)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_row_label(metric, label))

        plot_widgets = []
        image_items = []
        lot_separators: list[LotSeparatorWidget] = []
        for lot_offset in range(VISIBLE_LOTS):
            if lot_offset:
                separator = make_lot_separator_widget()
                lot_separators.append(separator)
                layout.addWidget(separator)
            plot_widget = pg.PlotWidget(background="#ffffff")
            plot_item = plot_widget.getPlotItem()
            plot_item.setMenuEnabled(False)
            plot_item.hideButtons()
            plot_item.hideAxis("left")
            plot_item.hideAxis("bottom")
            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.setXRange(0.5, 24.5, padding=0.0)
            plot_item.setYRange(0.5, 12.5, padding=0.0)

            image_item = pg.ImageItem(axisOrder="row-major")
            image_item.setColorMap(color_map)
            plot_item.addItem(image_item)
            _add_white_grid(plot_item)
            plot_widgets.append(plot_widget)
            image_items.append(image_item)
            layout.addWidget(plot_widget, stretch=1)

        legend = pg.GraphicsLayoutWidget()
        legend.setBackground("#ffffff")
        legend.setFixedWidth(COLOR_BAR_WIDTH)
        color_bar = pg.ColorBarItem(
            values=(0.0, 1.0),
            width=18,
            colorMap=color_map,
            label=label,
            interactive=False,
            rounding=0.1,
            pen=pg.mkPen("#5f6875"),
        )
        color_bar.axis.setTextPen(pg.mkPen("#303846"))
        color_bar.axis.setLabel(color="#4b5563")
        color_bar.setImageItem(image_items[0])
        legend.addItem(color_bar)
        layout.addWidget(legend)

        return FrameMapRow(
            metric=metric,
            label=label,
            widget=widget,
            plot_widgets=plot_widgets,
            image_items=image_items,
            color_bar=color_bar,
            lot_separators=lot_separators,
        )

    def _build_row_label(
        self,
        metric: str,
        label: str,
    ) -> QtWidgets.QWidget:
        """Fmapの項目情報表示。"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("frameMapLabelPanel")
        panel.setFixedWidth(LEFT_LABEL_WIDTH)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 7, 10, 7)
        layout.setSpacing(2)

        if metric == "ng_rates":
            title = QtWidgets.QLabel("Fmap")
            title.setObjectName("mapSectionTitle")
            title.setProperty("sectionRole", "detail")
            caption = QtWidgets.QLabel("選択項目の詳細")
            caption.setObjectName("detailBadge")
            layout.addWidget(title)
            layout.addWidget(caption)
            layout.addWidget(self.selection_label)
            layout.addStretch()
            layout.addWidget(self.frame_mode_button)
        else:
            metric_label = QtWidgets.QLabel(label)
            metric_label.setObjectName("frameMapMetricLabel")
            metric_label.setWordWrap(True)
            layout.addWidget(metric_label)
            layout.addStretch()
        return panel

    def set_context(
        self,
        colname: str,
        first_lot: int,
    ) -> None:
        """選択項目と表示lot範囲の反映。"""
        self.selection_label.setText(colname)
        if self.selected_colname != colname:
            self.selected_colname = colname
            self._configure_levels(colname)
        self._render(first_lot)

    def set_frame_available(self, available: bool) -> None:
        """選択Frame表示ボタンの有効状態反映。"""
        self.frame_mode_button.setEnabled(available)

    def _configure_levels(self, colname: str) -> None:
        """選択検査項目の色範囲設定。"""
        colname_index = self.data.colname_index(colname)
        ng_rates = self.data.ng_rates[colname_index]
        normalized_mean = self.data.normalized_mean[colname_index]
        normalized_std = self.data.normalized_std[colname_index]
        ng_max = self._nice_upper(
            float(np.nanpercentile(ng_rates, 99.5)),
            minimum=1.0,
        )
        deviation_max = self._nice_upper(
            float(np.nanpercentile(normalized_mean, 99.5)),
            minimum=0.1,
        )
        std_max = self._nice_upper(
            float(np.nanpercentile(normalized_std, 99.5)),
            minimum=0.1,
        )
        levels = {
            "ng_rates": (0.0, ng_max),
            "normalized_mean": (0.0, deviation_max),
            "normalized_std": (0.0, std_max),
        }
        for row in self.rows:
            row.levels = levels[row.metric]
            row.color_bar.setLevels(row.levels)

    def _render(self, first_lot: int) -> None:
        """横スクロール位置に対応するlotマップ描画。"""
        if self.selected_colname is None:
            return
        colname_index = self.data.colname_index(self.selected_colname)
        for row in self.rows:
            matrices = getattr(self.data, row.metric)[colname_index]
            for offset, (plot_widget, image_item) in enumerate(
                zip(row.plot_widgets, row.image_items)
            ):
                lot_index = first_lot + offset
                image_item.setImage(
                    matrices[lot_index],
                    autoLevels=False,
                    levels=row.levels,
                )
                image_item.setRect(
                    QtCore.QRectF(
                        0.5,
                        0.5,
                        float(len(POSITION_X)),
                        float(len(POSITION_Y)),
                    )
                )
                plot_widget.getPlotItem().getViewBox().setBorder(None)

    @staticmethod
    def _nice_upper(max_value: float, minimum: float) -> float:
        """色範囲上限の切り上げ。"""
        if max_value <= 0.0:
            return minimum
        magnitude = 10.0 ** math.floor(math.log10(max_value))
        normalized = max_value / magnitude
        for step in (1.0, 2.0, 3.0, 5.0, 10.0):
            if normalized <= step:
                return max(minimum, step * magnitude)
        return max(minimum, max_value)


class SingleFrameMapWidget(QtWidgets.QWidget):
    """上下限NG方向を重ねた単一Frame生値マップ。"""

    def __init__(self) -> None:
        super().__init__()
        self.data: SingleFrameMapData | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar())
        self.view = self._build_map_view()
        layout.addWidget(self.view.plot_widget, stretch=1)

    def _build_toolbar(self) -> QtWidgets.QWidget:
        """単一Frame Fmapの見出し生成。"""
        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("frameDetailToolbar")
        layout = QtWidgets.QGridLayout(toolbar)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        title = QtWidgets.QLabel("Fmap")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        badge = QtWidgets.QLabel("選択Frame・生値")
        badge.setObjectName("frameModeBadge")
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("singleFrameSelectionLabel")
        self.context_label = QtWidgets.QLabel()
        self.context_label.setObjectName("frameMapFrameLabel")
        direction = QtWidgets.QLabel("原点 左下　X →　Y ↑")
        direction.setObjectName("frameMapOrientationLabel")
        lower_key = QtWidgets.QLabel("● 下限NG")
        lower_key.setObjectName("lowerNgKey")
        upper_key = QtWidgets.QLabel("● 上限NG")
        upper_key.setObjectName("upperNgKey")

        layout.addWidget(title, 0, 0)
        layout.addWidget(badge, 0, 1)
        layout.addWidget(self.selection_label, 0, 2)
        layout.addWidget(self.context_label, 1, 0, 1, 2)
        layout.addWidget(direction, 1, 2)
        layout.addWidget(lower_key, 2, 0)
        layout.addWidget(upper_key, 2, 1)
        layout.setColumnStretch(2, 1)
        return toolbar

    def _build_map_view(self) -> SingleFrameMapView:
        """単一Frame生値マップ生成。"""
        color_map = make_color_map(PURPLES)
        plot_widget = pg.PlotWidget(background="#ffffff")
        plot_item = plot_widget.getPlotItem()
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.setXRange(0.5, 24.5, padding=0.0)
        plot_item.setYRange(0.5, 12.5, padding=0.0)
        plot_item.getViewBox().setAspectLocked(True, ratio=1.0)
        plot_item.getViewBox().setBorder(pg.mkPen("#c5ccd5", width=1))
        plot_item.getAxis("bottom").setTicks(
            [[(float(value), str(value)) for value in (1, 6, 12, 18, 24)]]
        )
        plot_item.getAxis("left").setTicks(
            [[(float(value), str(value)) for value in (1, 6, 12)]]
        )
        axis_font = QtGui.QFont()
        axis_font.setPointSize(7)
        for axis_name in ("bottom", "left"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#4b5563"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=3)

        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setColorMap(color_map)
        plot_item.addItem(image_item)
        ng_overlay_item = pg.ImageItem(axisOrder="row-major")
        ng_overlay_item.setZValue(10)
        plot_item.addItem(ng_overlay_item)
        _add_white_grid(plot_item)

        color_bar = pg.ColorBarItem(
            values=(0.0, 1.0),
            width=16,
            colorMap=color_map,
            label="生値",
            interactive=False,
            rounding=0.1,
            pen=pg.mkPen("#5f6875"),
        )
        color_bar.setImageItem(image_item, insert_in=plot_item)
        color_bar.axis.setTextPen(pg.mkPen("#303846"))
        color_bar.axis.setLabel(color="#4b5563")
        return SingleFrameMapView(
            plot_widget=plot_widget,
            plot_item=plot_item,
            image_item=image_item,
            ng_overlay_item=ng_overlay_item,
            color_bar=color_bar,
        )

    def set_data(self, data: SingleFrameMapData) -> None:
        """選択Frameの生値・NG方向反映。"""
        self.data = data
        raw_min = float(np.nanmin(data.raw_values))
        raw_max = float(np.nanmax(data.raw_values))
        span = max(raw_max - raw_min, abs(raw_max) * 0.01, 1e-9)
        self.view.levels = (
            raw_min - span * 0.04,
            raw_max + span * 0.04,
        )
        image_rect = QtCore.QRectF(
            0.5,
            0.5,
            float(len(POSITION_X)),
            float(len(POSITION_Y)),
        )
        self.view.image_item.setImage(
            data.raw_values,
            autoLevels=False,
            levels=self.view.levels,
        )
        self.view.image_item.setRect(image_rect)
        self.view.ng_overlay_item.setImage(build_ng_overlay(data))
        self.view.ng_overlay_item.setRect(image_rect)
        self.view.color_bar.setLevels(self.view.levels)
        self.view.color_bar.getAxis("left").setLabel(
            f"生値 ({data.unit})",
            color="#4b5563",
        )
        self.selection_label.setText(data.colname)
        self.context_label.setText(
            f"FrameNo {data.frame_no:02d}  |  "
            f"{data.sample_count:,}個片"
        )
