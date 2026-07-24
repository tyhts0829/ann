from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.standardized.quality_data import QualityRepository


BASE_COLNAMES = (
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
SPEC_ORDER = BASE_COLNAMES
CATEGORY_ORDER = ("異物", "リード", "PKGサイズ", "標印", "欠陥")
FRAME_NUMBERS = np.arange(1, 25)
POSITION_X = np.arange(1, 25)
POSITION_Y = np.arange(1, 13)
VISIBLE_LOTS = 5
VISIBLE_COLUMNS = len(FRAME_NUMBERS) * VISIBLE_LOTS

REDS = (
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#99000d",
)
RDBU_R = (
    "#053061",
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#f7f7f7",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
)
PURPLES = (
    "#fcfbfd",
    "#efedf5",
    "#dadaeb",
    "#bcbddc",
    "#9e9ac8",
    "#807dba",
    "#6a51a3",
    "#54278f",
    "#3f007d",
)


@dataclass(frozen=True)
class HeatmapData:
    """3種類の品質ヒートマップ集計結果。"""

    colnames: tuple[str, ...]
    categories: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    column_lots: tuple[str, ...]
    frame_numbers: np.ndarray
    ng_rates: np.ndarray
    normalized_mean: np.ndarray
    normalized_std: np.ndarray
    ng_counts: np.ndarray
    total_counts: np.ndarray

    @property
    def total_ng(self) -> int:
        return int(self.ng_counts.sum())

    @property
    def total_measurements(self) -> int:
        return int(self.total_counts.sum())

    @property
    def overall_ng_rate(self) -> float:
        return 100.0 * self.total_ng / self.total_measurements

    @property
    def lot_count(self) -> int:
        return len(self.lot_numbers)

    def filter_category(self, category: str | None) -> HeatmapData:
        """meta_categoryによる行抽出。"""
        if category is None:
            return self

        row_indices = np.flatnonzero(
            np.asarray(self.categories, dtype=object) == category
        )
        return HeatmapData(
            colnames=tuple(self.colnames[index] for index in row_indices),
            categories=tuple(
                self.categories[index] for index in row_indices
            ),
            lot_numbers=self.lot_numbers,
            column_lots=self.column_lots,
            frame_numbers=self.frame_numbers,
            ng_rates=self.ng_rates[row_indices],
            normalized_mean=self.normalized_mean[row_indices],
            normalized_std=self.normalized_std[row_indices],
            ng_counts=self.ng_counts[row_indices],
            total_counts=self.total_counts[row_indices],
        )


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
class HeatmapView:
    """1種類のヒートマップ表示部品。"""

    metric: str
    label: str
    card: QtWidgets.QFrame
    plot_widget: pg.PlotWidget
    plot_item: pg.PlotItem
    image_item: pg.ImageItem
    color_bar: pg.ColorBarItem
    selection_item: QtWidgets.QGraphicsRectItem
    separators: list[pg.InfiniteLine] = field(default_factory=list)
    mouse_proxy: pg.SignalProxy | None = None


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


def build_heatmap_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> HeatmapData:
    """FrameNo別の3種類のヒートマップデータ生成。"""
    if lot_numbers is None:
        lot_numbers = tuple(record[0] for record in repository.lots())

    frame = repository.ng_rate_by_frame(lot_numbers)
    if frame.empty:
        raise ValueError("対象データがありません。")
    frame["colname"] = frame["colname"].str.replace(
        r"_v[123]$",
        "",
        regex=True,
    )

    available = set(frame["colname"])
    colnames = tuple(name for name in SPEC_ORDER if name in available)
    category_lookup = (
        frame[["colname", "meta_category"]]
        .drop_duplicates("colname")
        .set_index("colname")["meta_category"]
        .to_dict()
    )
    categories = tuple(category_lookup[colname] for colname in colnames)

    return HeatmapData(
        colnames=colnames,
        categories=categories,
        lot_numbers=lot_numbers,
        column_lots=tuple(
            lot_number
            for lot_number in lot_numbers
            for _ in FRAME_NUMBERS
        ),
        frame_numbers=np.tile(FRAME_NUMBERS, len(lot_numbers)),
        ng_rates=_matrix(
            frame,
            "ng_rate",
            colnames,
            lot_numbers,
            fill_value=np.nan,
        ),
        normalized_mean=_matrix(
            frame,
            "normalized_mean",
            colnames,
            lot_numbers,
            fill_value=np.nan,
        ),
        normalized_std=_matrix(
            frame,
            "normalized_std",
            colnames,
            lot_numbers,
            fill_value=np.nan,
        ),
        ng_counts=_matrix(
            frame,
            "ng_count",
            colnames,
            lot_numbers,
            fill_value=0.0,
        ),
        total_counts=_matrix(
            frame,
            "total_count",
            colnames,
            lot_numbers,
            fill_value=0.0,
        ),
    )


def build_frame_map_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> FrameMapData:
    """PositionX・PositionY別の3種類のマップデータ生成。"""
    if lot_numbers is None:
        lot_numbers = tuple(record[0] for record in repository.lots())

    frame = repository.metrics_by_colname_position(lot_numbers)
    available = set(frame["colname"])
    colnames = tuple(
        colname for colname in BASE_COLNAMES if colname in available
    )
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


def _matrix(
    frame: pd.DataFrame,
    value_column: str,
    colnames: tuple[str, ...],
    lot_numbers: tuple[str, ...],
    fill_value: float,
) -> np.ndarray:
    """lotとFrameNoを列に展開した行列。"""
    matrix = frame.pivot(
        index="colname",
        columns=["lot_number", "FrameNo"],
        values=value_column,
    )
    columns = pd.MultiIndex.from_product(
        [lot_numbers, FRAME_NUMBERS],
        names=["lot_number", "FrameNo"],
    )
    return (
        matrix.reindex(index=colnames, columns=columns)
        .fillna(fill_value)
        .to_numpy(dtype=float)
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
            subset.set_index(
                ["lot_number", "PositionY", "PositionX"]
            )[value_column]
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


def _make_color_map(colors: tuple[str, ...]) -> pg.ColorMap:
    """16進色列からの連続カラーマップ。"""
    rgb = np.asarray(
        [pg.mkColor(color).getRgb()[:3] for color in colors],
        dtype=np.ubyte,
    )
    return pg.ColorMap(
        np.linspace(0.0, 1.0, len(colors)),
        rgb,
    )


class NgRateHeatmapWidget(QtWidgets.QWidget):
    """NG率・規格位置平均・標準偏差の3段ヒートマップ。"""

    def __init__(self, repository: QualityRepository) -> None:
        super().__init__()
        self.repository = repository
        self.full_heatmap = build_heatmap_data(repository)
        self.current_heatmap = self.full_heatmap
        self.selected_colname = self.full_heatmap.colnames[0]
        self.selected_column = len(self.full_heatmap.frame_numbers) - 1
        self.frame_map_data = build_frame_map_data(
            repository,
            self.full_heatmap.lot_numbers,
        )
        self.heatmap_views: list[HeatmapView] = []
        self.frame_map_rows: list[FrameMapRow] = []
        self._build_ui()
        self._populate_categories()
        self._render_heatmaps()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_toolbar())
        layout.addWidget(self._build_vertical_scroll_area(), stretch=1)
        layout.addLayout(self._build_horizontal_navigation())
        layout.addWidget(self._build_footer())

    def _build_toolbar(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("toolbarCard")
        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(12)

        lot_label = QtWidgets.QLabel("対象")
        lot_label.setObjectName("fieldLabel")
        self.all_lots_label = QtWidgets.QLabel("全lot")
        self.all_lots_label.setObjectName("allLotsLabel")

        self.scope_label = QtWidgets.QLabel()
        self.scope_label.setObjectName("scopeLabel")

        category_label = QtWidgets.QLabel("カテゴリ")
        category_label.setObjectName("fieldLabel")
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setObjectName("categoryCombo")
        self.category_combo.setMinimumWidth(150)

        self.ng_rate_label = QtWidgets.QLabel()
        self.ng_rate_label.setObjectName("ngRateLabel")

        layout.addWidget(lot_label)
        layout.addWidget(self.all_lots_label)
        layout.addSpacing(8)
        layout.addWidget(self.scope_label)
        layout.addStretch()
        layout.addWidget(category_label)
        layout.addWidget(self.category_combo)
        layout.addSpacing(8)
        layout.addWidget(self.ng_rate_label)
        return card

    def _build_vertical_scroll_area(self) -> QtWidgets.QScrollArea:
        self.vertical_scroll_area = QtWidgets.QScrollArea()
        self.vertical_scroll_area.setObjectName("heatmapScrollArea")
        self.vertical_scroll_area.setWidgetResizable(True)
        self.vertical_scroll_area.setFrameShape(
            QtWidgets.QFrame.Shape.NoFrame
        )
        self.vertical_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.vertical_scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setObjectName("plotsContainer")
        plots_layout = QtWidgets.QVBoxLayout(self.plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(0)

        definitions = (
            ("ng_rates", "NG率 (%)", REDS),
            (
                "normalized_mean",
                "規格位置・使用率 平均",
                RDBU_R,
            ),
            (
                "normalized_std",
                "規格位置・使用率 std",
                PURPLES,
            ),
        )
        for metric, label, colors in definitions:
            view = self._build_heatmap_view(
                metric,
                label,
                _make_color_map(colors),
            )
            self.heatmap_views.append(view)
            plots_layout.addWidget(view.card)

        for metric, label, colors in definitions:
            row = self._build_frame_map_row(
                metric,
                label,
                _make_color_map(colors),
            )
            self.frame_map_rows.append(row)
            plots_layout.addWidget(row.widget)

        self.vertical_scroll_area.setWidget(self.plots_container)
        return self.vertical_scroll_area

    def _build_heatmap_view(
        self,
        metric: str,
        label: str,
        color_map: pg.ColorMap,
    ) -> HeatmapView:
        card = QtWidgets.QFrame()
        card.setObjectName("chartCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_widget = pg.PlotWidget(background="#ffffff")
        plot_item = plot_widget.getPlotItem()
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.getViewBox().invertY(True)
        plot_item.getAxis("left").setWidth(205)
        plot_item.getAxis("bottom").setHeight(28)
        plot_item.showGrid(x=False, y=True, alpha=0.18)

        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis_font = QtGui.QFont()
            axis_font.setPointSize(8)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=5)

        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setColorMap(color_map)
        plot_item.addItem(image_item)

        selection_item = QtWidgets.QGraphicsRectItem()
        selection_item.setPen(pg.mkPen("#00796b", width=2))
        selection_item.setBrush(
            QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        )
        selection_item.setZValue(30)
        plot_item.addItem(selection_item)

        color_bar = pg.ColorBarItem(
            values=(0.0, 1.0),
            width=22,
            colorMap=color_map,
            label=label,
            interactive=False,
            rounding=0.1,
            pen=pg.mkPen("#5f6875"),
        )
        color_bar.setImageItem(image_item, insert_in=plot_item)
        color_bar.axis.setTextPen(pg.mkPen("#303846"))
        color_bar.axis.setLabel(color="#4b5563")

        layout.addWidget(plot_widget)
        view = HeatmapView(
            metric=metric,
            label=label,
            card=card,
            plot_widget=plot_widget,
            plot_item=plot_item,
            image_item=image_item,
            color_bar=color_bar,
            selection_item=selection_item,
        )
        view.mouse_proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=lambda event, heatmap_view=view: (
                self._show_hover_details(heatmap_view, event)
            ),
        )
        plot_widget.scene().sigMouseClicked.connect(
            lambda event, heatmap_view=view: self._select_heatmap_cell(
                heatmap_view,
                event,
            )
        )
        return view

    def _build_frame_map_row(
        self,
        metric: str,
        label: str,
        color_map: pg.ColorMap,
    ) -> FrameMapRow:
        """表示中5 lotのPositionマップ行生成。"""
        widget = QtWidgets.QWidget()
        widget.setObjectName("frameMapRow")
        widget.setFixedHeight(126)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        left_spacer = QtWidgets.QWidget()
        left_spacer.setFixedWidth(205)
        layout.addWidget(left_spacer)

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
            plot_item.getViewBox().setAspectLocked(True)
            plot_item.setXRange(0.5, 24.5, padding=0.0)
            plot_item.setYRange(0.5, 12.5, padding=0.0)

            image_item = pg.ImageItem(axisOrder="row-major")
            image_item.setColorMap(color_map)
            image_item.setRect(
                QtCore.QRectF(
                    0.5,
                    0.5,
                    float(len(POSITION_X)),
                    float(len(POSITION_Y)),
                )
            )
            plot_item.addItem(image_item)
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
            plot_widgets.append(plot_widget)
            image_items.append(image_item)
            layout.addWidget(plot_widget, stretch=1)

        legend = pg.GraphicsLayoutWidget()
        legend.setBackground("#ffffff")
        legend.setFixedWidth(90)
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

    def _build_horizontal_navigation(self) -> QtWidgets.QLayout:
        navigation = QtWidgets.QHBoxLayout()
        navigation.setContentsMargins(0, 2, 0, 2)
        navigation.setSpacing(10)

        self.first_button = QtWidgets.QToolButton()
        self.first_button.setObjectName("navigationButton")
        self.first_button.setText("先頭")
        self.first_button.clicked.connect(
            lambda: self.horizontal_scrollbar.setValue(0)
        )

        self.horizontal_scrollbar = QtWidgets.QScrollBar(
            QtCore.Qt.Orientation.Horizontal
        )
        self.horizontal_scrollbar.setObjectName("heatmapScrollBar")
        self.horizontal_scrollbar.valueChanged.connect(
            self._scroll_heatmaps
        )

        self.latest_button = QtWidgets.QToolButton()
        self.latest_button.setObjectName("navigationButton")
        self.latest_button.setText("最新")
        self.latest_button.clicked.connect(
            lambda: self.horizontal_scrollbar.setValue(
                self.horizontal_scrollbar.maximum()
            )
        )

        navigation.addWidget(self.first_button)
        navigation.addWidget(self.horizontal_scrollbar, stretch=1)
        navigation.addWidget(self.latest_button)
        return navigation

    def _build_footer(self) -> QtWidgets.QWidget:
        footer = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(footer)
        layout.setContentsMargins(4, 0, 4, 0)

        self.hover_label = QtWidgets.QLabel(
            "セルにカーソルを合わせると詳細を表示"
        )
        self.hover_label.setObjectName("footerNote")
        self.scale_label = QtWidgets.QLabel()
        self.scale_label.setObjectName("footerNote")

        layout.addWidget(self.hover_label)
        layout.addStretch()
        layout.addWidget(self.scale_label)
        return footer

    def _populate_categories(self) -> None:
        blocker = QtCore.QSignalBlocker(self.category_combo)
        self.category_combo.addItem("None", None)
        available = set(self.full_heatmap.categories)
        for category in CATEGORY_ORDER:
            if category in available:
                self.category_combo.addItem(category, category)
        del blocker
        self.category_combo.currentIndexChanged.connect(
            self._render_heatmaps
        )

    @QtCore.Slot()
    def _render_heatmaps(self) -> None:
        category = self.category_combo.currentData()
        data = self.full_heatmap.filter_category(category)
        self.current_heatmap = data
        if self.selected_colname not in data.colnames:
            self.selected_colname = data.colnames[0]

        ng_max = self._nice_upper(
            float(np.nanpercentile(data.ng_rates, 99.5)),
            minimum=1.0,
        )
        mean_abs_max = self._nice_upper(
            float(
                np.nanpercentile(
                    np.abs(data.normalized_mean),
                    99.5,
                )
            ),
            minimum=0.1,
        )
        std_max = self._nice_upper(
            float(np.nanpercentile(data.normalized_std, 99.5)),
            minimum=0.1,
        )
        render_specs = {
            "ng_rates": (data.ng_rates, (0.0, ng_max)),
            "normalized_mean": (
                data.normalized_mean,
                (-mean_abs_max, mean_abs_max),
            ),
            "normalized_std": (
                data.normalized_std,
                (0.0, std_max),
            ),
        }

        for view in self.heatmap_views:
            matrix, levels = render_specs[view.metric]
            view.image_item.setImage(
                matrix,
                autoLevels=False,
                levels=levels,
            )
            view.image_item.setRect(
                QtCore.QRectF(
                    -0.5,
                    -0.5,
                    float(len(data.frame_numbers)),
                    float(len(data.colnames)),
                )
            )
            view.color_bar.setLevels(levels)
            view.plot_item.getAxis("left").setTicks(
                [
                    [
                        (row, colname)
                        for row, colname in enumerate(data.colnames)
                    ]
                ]
            )
            view.plot_item.setYRange(
                -0.5,
                len(data.colnames) - 0.5,
                padding=0.0,
            )
            self._update_x_axis(view, data)
            self._draw_lot_separators(view, data)

        self._update_selection_items()
        self._update_plot_heights(len(data.colnames))
        self._update_summary(
            data,
            category,
            ng_max,
            mean_abs_max,
            std_max,
        )
        self._configure_frame_map_rows()
        self._configure_horizontal_scrollbar(data)
        self.vertical_scroll_area.verticalScrollBar().setValue(0)

    def _update_plot_heights(self, row_count: int) -> None:
        height = max(260, 40 + row_count * 16)
        for view in self.heatmap_views:
            view.card.setFixedHeight(height)
        self.plots_container.adjustSize()

    def _configure_horizontal_scrollbar(
        self,
        data: HeatmapData,
    ) -> None:
        maximum = max(0, data.lot_count - VISIBLE_LOTS)
        blocker = QtCore.QSignalBlocker(self.horizontal_scrollbar)
        self.horizontal_scrollbar.setRange(0, maximum)
        self.horizontal_scrollbar.setSingleStep(1)
        self.horizontal_scrollbar.setPageStep(VISIBLE_LOTS)
        self.horizontal_scrollbar.setValue(maximum)
        del blocker
        self._scroll_heatmaps(maximum)

    @QtCore.Slot(int)
    def _scroll_heatmaps(self, first_lot: int) -> None:
        data = self.current_heatmap
        if data is None:
            return

        first_column = first_lot * len(FRAME_NUMBERS)
        last_column = min(
            first_column + VISIBLE_COLUMNS,
            len(data.frame_numbers),
        )
        for view in self.heatmap_views:
            view.plot_item.setXRange(
                first_column - 0.5,
                last_column - 0.5,
                padding=0.0,
            )

        self.first_button.setEnabled(first_lot > 0)
        self.latest_button.setEnabled(
            first_lot < self.horizontal_scrollbar.maximum()
        )
        self._render_frame_maps(first_lot)

    def _configure_frame_map_rows(
        self,
    ) -> None:
        """選択検査項目のフレームマップ色範囲設定。"""
        colname_index = self.frame_map_data.colname_index(
            self.selected_colname
        )
        ng_rates = self.frame_map_data.ng_rates[colname_index]
        normalized_mean = self.frame_map_data.normalized_mean[
            colname_index
        ]
        normalized_std = self.frame_map_data.normalized_std[
            colname_index
        ]
        ng_max = self._nice_upper(
            float(np.nanpercentile(ng_rates, 99.5)),
            minimum=1.0,
        )
        mean_abs_max = self._nice_upper(
            float(
                np.nanpercentile(
                    np.abs(normalized_mean),
                    99.5,
                )
            ),
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
        for row in self.frame_map_rows:
            row.levels = levels[row.metric]
            row.color_bar.setLevels(row.levels)

    def _render_frame_maps(self, first_lot: int) -> None:
        """横スクロール位置に対応する5 lotのマップ表示。"""
        colname_index = self.frame_map_data.colname_index(
            self.selected_colname
        )
        selected_lot = self.selected_column // len(FRAME_NUMBERS)
        for row in self.frame_map_rows:
            matrices = getattr(self.frame_map_data, row.metric)[
                colname_index
            ]
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
                border = (
                    pg.mkPen("#00796b", width=2)
                    if lot_index == selected_lot
                    else None
                )
                plot_widget.getPlotItem().getViewBox().setBorder(border)

    def _select_heatmap_cell(
        self,
        view: HeatmapView,
        event: object,
    ) -> None:
        """クリックしたヒートマップセルの選択。"""
        scene_position = event.scenePos()
        if not view.plot_widget.sceneBoundingRect().contains(
            scene_position
        ):
            return
        point = view.plot_item.getViewBox().mapSceneToView(scene_position)
        column = round(point.x())
        row = round(point.y())
        data = self.current_heatmap
        if not (
            0 <= column < len(data.frame_numbers)
            and 0 <= row < len(data.colnames)
        ):
            return

        self.selected_colname = data.colnames[row]
        self.selected_column = column
        self._update_selection_items()
        self._configure_frame_map_rows()
        self._render_frame_maps(self.horizontal_scrollbar.value())
        self._update_scope_label(
            data,
            self.category_combo.currentData(),
        )

    def _update_selection_items(self) -> None:
        """3段ヒートマップの選択枠同期。"""
        data = self.current_heatmap
        row = data.colnames.index(self.selected_colname)
        rectangle = QtCore.QRectF(
            self.selected_column - 0.5,
            row - 0.5,
            1.0,
            1.0,
        )
        for view in self.heatmap_views:
            view.selection_item.setRect(rectangle)

    def _update_x_axis(
        self,
        view: HeatmapView,
        data: HeatmapData,
    ) -> None:
        lot_ticks = [
            (
                lot_index * len(FRAME_NUMBERS)
                + (len(FRAME_NUMBERS) - 1) / 2,
                lot_number,
            )
            for lot_index, lot_number in enumerate(data.lot_numbers)
        ]
        view.plot_item.getAxis("bottom").setTicks([lot_ticks])

    def _draw_lot_separators(
        self,
        view: HeatmapView,
        data: HeatmapData,
    ) -> None:
        for separator in view.separators:
            view.plot_item.removeItem(separator)
        view.separators.clear()

        for lot_index in range(1, data.lot_count):
            separator = pg.InfiniteLine(
                pos=lot_index * len(FRAME_NUMBERS) - 0.5,
                angle=90,
                movable=False,
                pen=pg.mkPen("#8f99a6", width=1),
            )
            separator.setZValue(20)
            view.plot_item.addItem(separator)
            view.separators.append(separator)

    def _update_summary(
        self,
        data: HeatmapData,
        category: str | None,
        ng_max: float,
        mean_abs_max: float,
        std_max: float,
    ) -> None:
        self.all_lots_label.setText(f"全{data.lot_count} lot")
        self._update_scope_label(data, category)
        self.ng_rate_label.setText(
            f"総合NG率  {data.overall_ng_rate:.3f}%"
            f"   ({data.total_ng:,} NG)"
        )
        self.scale_label.setText(
            f"NG 0–{ng_max:g}%  |  "
            f"平均 ±{mean_abs_max:g}  |  std 0–{std_max:g}"
        )

    def _update_scope_label(
        self,
        data: HeatmapData,
        category: str | None,
    ) -> None:
        """表示範囲と選択検査項目の要約。"""
        category_text = category if category is not None else "全カテゴリ"
        self.scope_label.setText(
            f"{category_text}  |  {len(data.colnames)}項目  |  "
            f"{data.total_measurements:,}測定  |  "
            f"選択 {self.selected_colname}"
        )

    def _show_hover_details(
        self,
        view: HeatmapView,
        event: tuple[QtCore.QPointF],
    ) -> None:
        scene_position = event[0]
        if not view.plot_widget.sceneBoundingRect().contains(
            scene_position
        ):
            self._reset_hover_text()
            return

        point = view.plot_item.getViewBox().mapSceneToView(scene_position)
        column = round(point.x())
        row = round(point.y())
        data = self.current_heatmap
        if not (
            0 <= column < len(data.frame_numbers)
            and 0 <= row < len(data.colnames)
        ):
            self._reset_hover_text()
            return

        matrix = getattr(data, view.metric)
        value = matrix[row, column]
        prefix = (
            f"{data.column_lots[column]}  |  "
            f"FrameNo {int(data.frame_numbers[column])}  |  "
            f"{data.colnames[row]}  |  "
        )
        if np.isnan(value):
            self.hover_label.setText(prefix + "対象外")
            return

        if view.metric == "ng_rates":
            ng_count = int(data.ng_counts[row, column])
            total_count = int(data.total_counts[row, column])
            detail = (
                f"NG率 {value:.3f}%  "
                f"({ng_count:,}/{total_count:,})"
            )
        elif view.metric == "normalized_mean":
            detail = f"規格位置・使用率 平均 {value:.4f}"
        else:
            detail = f"規格位置・使用率 std {value:.4f}"
        self.hover_label.setText(prefix + detail)

    def _reset_hover_text(self) -> None:
        self.hover_label.setText(
            "セルにカーソルを合わせると詳細を表示"
        )

    @staticmethod
    def _nice_upper(
        max_value: float,
        minimum: float,
    ) -> float:
        if max_value <= 0.0:
            return minimum
        magnitude = 10.0 ** math.floor(math.log10(max_value))
        normalized = max_value / magnitude
        for step in (1.0, 2.0, 3.0, 5.0, 10.0):
            if normalized <= step:
                return max(minimum, step * magnitude)
        return max(minimum, max_value)
