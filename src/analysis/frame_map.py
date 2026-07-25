from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from src.analysis.map_palettes import MAP_DEFINITIONS, make_color_map
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


@dataclass
class FrameMapRow:
    """同一指標の表示中lot別フレームマップ行。"""

    metric: str
    label: str
    widget: QtWidgets.QWidget
    plot_widgets: list[pg.PlotWidget]
    image_items: list[pg.ImageItem]
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
            subset.set_index(["lot_number", "PositionY", "PositionX"])[value_column]
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


class FrameMapWidget(QtWidgets.QWidget):
    """選択検査項目の3段フレームマップ。"""

    def __init__(
        self,
        repository: QualityRepository,
        lot_numbers: tuple[str, ...],
        data: FrameMapData | None = None,
    ) -> None:
        super().__init__()
        self.data = (
            build_frame_map_data(repository, lot_numbers) if data is None else data
        )
        self.rows: list[FrameMapRow] = []
        self.selected_colname: str | None = None
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("frameMapSelectionLabel")
        self.selection_label.setWordWrap(True)
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
        for _ in range(VISIBLE_LOTS):
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
            self._add_white_grid(plot_item)
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
        else:
            metric_label = QtWidgets.QLabel(label)
            metric_label.setObjectName("frameMapMetricLabel")
            metric_label.setWordWrap(True)
            layout.addWidget(metric_label)
        layout.addStretch()
        return panel

    @staticmethod
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
        mean_abs_max = self._nice_upper(
            float(np.nanpercentile(np.abs(normalized_mean), 99.5)),
            minimum=0.1,
        )
        std_max = self._nice_upper(
            float(np.nanpercentile(normalized_std, 99.5)),
            minimum=0.1,
        )
        levels = {
            "ng_rates": (0.0, ng_max),
            "normalized_mean": (-mean_abs_max, mean_abs_max),
            "normalized_std": (0.0, std_max),
        }
        for row in self.rows:
            row.levels = levels[row.metric]
            row.color_bar.setLevels(row.levels)

    def _render(self, first_lot: int) -> None:
        """横スクロール位置に対応する5 lotのマップ表示。"""
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
