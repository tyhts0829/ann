from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.frame_map import (
    VISIBLE_LOTS,
    FrameMapData,
    FrameMapWidget,
    SingleFrameMapData,
    SingleFrameMapWidget,
    build_single_frame_map_data,
)
from src.analysis.kde import (
    KdeData,
    KdeWidget,
    build_kde_data,
)
from src.analysis.map_palettes import MAP_DEFINITIONS, make_color_map
from src.analysis.plot_style import make_lot_separator
from src.analysis.processing_path_widget import ProcessingPathWidget
from src.analysis.quality_columns import (
    CATEGORY_ORDER,
    SPEC_ORDER,
    VISION_ORDER,
)
from src.analysis.quality_trend import (
    QualityTrendData,
    QualityTrendWidget,
    build_quality_trend_data,
)
from src.analysis.single_frame_kde import (
    SingleFrameKdeWidget,
    build_single_frame_kde_data,
)
from src.dashboard_config import DASHBOARD_CONFIG
from src.quality_repository import QualityRepository

FRAME_NUMBERS = np.arange(1, 25)
FRAME_TICK_NUMBERS = (1, 6, 12, 18, 24)
VISIBLE_COLUMNS = len(FRAME_NUMBERS) * VISIBLE_LOTS
FQ_MAP_SECTION_HEIGHT = DASHBOARD_CONFIG.fqmap_height
FMAP_SECTION_HEIGHT = DASHBOARD_CONFIG.fmap_height
KDE_SECTION_HEIGHT = DASHBOARD_CONFIG.kde_height
QUALITY_TREND_SECTION_HEIGHT = DASHBOARD_CONFIG.quality_trend_height
FQ_MAP_PLOT_HEIGHT = DASHBOARD_CONFIG.fqmap_plot_height
FQ_MAP_MIN_CELL_HEIGHT = DASHBOARD_CONFIG.fqmap_min_cell_height
FQ_MAP_LOT_AXIS_HEIGHT = 24
FQ_MAP_FRAME_AXIS_HEIGHT = 20
FQ_MAP_TOP_AXIS_HEIGHT = (
    FQ_MAP_LOT_AXIS_HEIGHT + FQ_MAP_FRAME_AXIS_HEIGHT
)
FQ_MAP_SEPARATOR_HEIGHT = 6
DETAIL_DIVIDER_HEIGHT = 1


class SceneMouseEvent(Protocol):
    """シーン座標を持つマウスイベント。"""

    def scenePos(self) -> QtCore.QPointF:
        """シーン座標。"""
        ...


class FqMapPlotWidget(pg.PlotWidget):
    """行単位の縦スクロールを持つFQmap。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._vertical_scrollbar: QtWidgets.QScrollBar | None = None
        self._wheel_remainder = 0
        self._last_wheel_delta = 0

    def set_vertical_scrollbar(
        self,
        scrollbar: QtWidgets.QScrollBar,
    ) -> None:
        """同期対象の行スクロールバー設定。"""
        self._vertical_scrollbar = scrollbar

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """ホイール操作の行スクロール変換。"""
        pixel_delta = event.pixelDelta().y()
        delta = pixel_delta or event.angleDelta().y()
        scrollbar = self._vertical_scrollbar
        if scrollbar is None or delta == 0:
            event.ignore()
            return

        direction = -1 if delta > 0 else 1
        can_scroll = (
            direction < 0 and scrollbar.value() > scrollbar.minimum()
        ) or (
            direction > 0 and scrollbar.value() < scrollbar.maximum()
        )
        if not can_scroll:
            self._wheel_remainder = 0
            self._last_wheel_delta = 0
            event.ignore()
            return

        if delta * self._last_wheel_delta < 0:
            self._wheel_remainder = 0
        self._last_wheel_delta = delta
        self._wheel_remainder -= delta
        unit = FQ_MAP_MIN_CELL_HEIGHT if pixel_delta else 40
        rows = math.trunc(self._wheel_remainder / unit)
        self._wheel_remainder -= rows * unit
        if rows:
            scrollbar.setValue(scrollbar.value() + rows)
        event.accept()


@dataclass(frozen=True)
class FqMapData:
    """3種類のFQマップ集計結果。"""

    colnames: tuple[str, ...]
    categories: tuple[str, ...]
    visions: tuple[str, ...]
    lot_numbers: tuple[str, ...]
    lot_start_times: tuple[datetime, ...]
    column_lots: tuple[str, ...]
    frame_numbers: np.ndarray
    ng_rates: np.ndarray
    normalized_mean: np.ndarray
    normalized_std: np.ndarray
    ng_counts: np.ndarray
    total_counts: np.ndarray
    piece_ng_masks: np.ndarray
    colname_bits: np.ndarray

    @property
    def total_ng(self) -> int:
        return int(self.ng_counts.sum())

    @property
    def total_measurements(self) -> int:
        return int(self.total_counts.sum())

    @property
    def total_pieces(self) -> int:
        return len(self.piece_ng_masks)

    @property
    def ng_pieces(self) -> int:
        selected_bits = np.bitwise_or.reduce(self.colname_bits)
        return int(
            np.count_nonzero(self.piece_ng_masks & selected_bits)
        )

    @property
    def ok_pieces(self) -> int:
        return self.total_pieces - self.ng_pieces

    @property
    def piece_yield(self) -> float:
        return 100.0 * self.ok_pieces / self.total_pieces

    @property
    def lot_count(self) -> int:
        return len(self.lot_numbers)

    def filter_rows(
        self,
        category: str | None = None,
        vision: str | None = None,
    ) -> FqMapData:
        """カテゴリ・visionによる行抽出。"""
        if category is None and vision is None:
            return self

        row_filter = np.ones(len(self.colnames), dtype=bool)
        if category is not None:
            row_filter &= (
                np.asarray(self.categories, dtype=object) == category
            )
        if vision is not None:
            row_filter &= np.asarray(self.visions, dtype=object) == vision
        row_indices = np.flatnonzero(row_filter)
        return FqMapData(
            colnames=tuple(self.colnames[index] for index in row_indices),
            categories=tuple(
                self.categories[index] for index in row_indices
            ),
            visions=tuple(self.visions[index] for index in row_indices),
            lot_numbers=self.lot_numbers,
            lot_start_times=self.lot_start_times,
            column_lots=self.column_lots,
            frame_numbers=self.frame_numbers,
            ng_rates=self.ng_rates[row_indices],
            normalized_mean=self.normalized_mean[row_indices],
            normalized_std=self.normalized_std[row_indices],
            ng_counts=self.ng_counts[row_indices],
            total_counts=self.total_counts[row_indices],
            piece_ng_masks=self.piece_ng_masks,
            colname_bits=self.colname_bits[row_indices],
        )

    def filter_category(self, category: str | None) -> FqMapData:
        """meta_categoryによる行抽出。"""
        return self.filter_rows(category=category)

    def filter_vision(self, vision: str | None) -> FqMapData:
        """visionによる行抽出。"""
        return self.filter_rows(vision=vision)


@dataclass
class FqMapView:
    """1種類のFQマップ表示部品。"""

    metric: str
    label: str
    card: QtWidgets.QFrame
    plot_widget: pg.PlotWidget
    plot_item: pg.PlotItem
    image_item: pg.ImageItem
    color_bar: pg.ColorBarItem
    vertical_scrollbar: QtWidgets.QScrollBar
    selection_region: pg.LinearRegionItem
    cell_selection_rect: QtWidgets.QGraphicsRectItem
    frame_axis: pg.AxisItem | None = None
    separators: list[pg.InfiniteLine] = field(default_factory=list)
    mouse_proxy: pg.SignalProxy | None = None


def build_fq_map_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> FqMapData:
    """FrameNo別の3種類のFQマップデータ生成。"""
    lot_records = repository.lots()
    if lot_numbers is None:
        lot_numbers = tuple(record[0] for record in lot_records)
    lot_time_by_number = dict(lot_records)

    frame = repository.ng_rate_by_frame(lot_numbers)
    if frame.empty:
        raise ValueError("対象データがありません。")
    available = set(frame["colname"])
    colnames = tuple(name for name in SPEC_ORDER if name in available)
    category_lookup = (
        frame[["colname", "meta_category"]]
        .drop_duplicates("colname")
        .set_index("colname")["meta_category"]
        .to_dict()
    )
    categories = tuple(category_lookup[colname] for colname in colnames)
    vision_lookup = (
        frame[["colname", "vision"]]
        .drop_duplicates("colname")
        .set_index("colname")["vision"]
        .to_dict()
    )
    visions = tuple(vision_lookup[colname] for colname in colnames)

    return FqMapData(
        colnames=colnames,
        categories=categories,
        visions=visions,
        lot_numbers=lot_numbers,
        lot_start_times=tuple(
            lot_time_by_number[lot_number]
            for lot_number in lot_numbers
        ),
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
        piece_ng_masks=repository.piece_ng_masks(
            colnames,
            lot_numbers,
        ),
        colname_bits=np.left_shift(
            np.uint64(1),
            np.arange(len(colnames), dtype=np.uint64),
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


class FqMapWidget(QtWidgets.QWidget):
    """NG率・規格逸脱度・標準偏差の3段FQマップ。"""

    def __init__(
        self,
        repository: QualityRepository,
        full_data: FqMapData | None = None,
        frame_map_data: FrameMapData | None = None,
        kde_data: KdeData | None = None,
        quality_trend_data: QualityTrendData | None = None,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.full_data = (
            build_fq_map_data(repository)
            if full_data is None
            else full_data
        )
        self.frame_map_data = frame_map_data
        self.kde_data = (
            build_kde_data(
                repository,
                self.full_data.lot_numbers,
            )
            if kde_data is None
            else kde_data
        )
        self.quality_trend_data = (
            build_quality_trend_data(
                repository,
                self.full_data.lot_numbers,
            )
            if quality_trend_data is None
            else quality_trend_data
        )
        self.current_data = self.full_data
        self.selected_colname = self.full_data.colnames[0]
        self.selected_lot_index: int | None = None
        self.selected_frame_no: int | None = None
        self.single_frame_data: SingleFrameMapData | None = None
        self.detail_mode = "lot"
        self.views: list[FqMapView] = []
        self.fq_map_separators: list[QtWidgets.QFrame] = []
        self._build_ui()
        self._populate_filters()
        self._render_fq_maps()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.fq_map_section = self._build_fq_map_section()
        self.fmap_section = self._build_fmap_section()
        self.kde_section = self._build_kde_section()
        self.quality_trend_section = (
            self._build_quality_trend_section()
        )
        self.detail_section = self._build_detail_section()
        layout.addWidget(self.fq_map_section)
        layout.addWidget(self.detail_section)
        layout.addLayout(self._build_horizontal_navigation())
        layout.addWidget(self._build_footer())
        layout.addStretch()

    def _build_toolbar(self) -> QtWidgets.QWidget:
        card = QtWidgets.QWidget()
        card.setObjectName("toolbarCard")
        card.setFixedHeight(48)
        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(10)

        fq_map_title = QtWidgets.QLabel("FQmap")
        fq_map_title.setObjectName("mapSectionTitle")
        overview_badge = QtWidgets.QLabel("全検査項目の俯瞰")
        overview_badge.setObjectName("overviewBadge")

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
        self.category_combo.setMinimumWidth(120)
        self.category_combo.setAccessibleName("カテゴリフィルター")

        vision_label = QtWidgets.QLabel("Vision")
        vision_label.setObjectName("fieldLabel")
        self.vision_combo = QtWidgets.QComboBox()
        self.vision_combo.setObjectName("visionCombo")
        self.vision_combo.setMinimumWidth(105)
        self.vision_combo.setAccessibleName("Visionフィルター")

        self.piece_yield_label = QtWidgets.QLabel()
        self.piece_yield_label.setObjectName("pieceYieldLabel")

        layout.addWidget(fq_map_title)
        layout.addWidget(overview_badge)
        layout.addWidget(lot_label)
        layout.addWidget(self.all_lots_label)
        layout.addSpacing(8)
        layout.addWidget(self.scope_label)
        layout.addStretch()
        layout.addWidget(category_label)
        layout.addWidget(self.category_combo)
        layout.addWidget(vision_label)
        layout.addWidget(self.vision_combo)
        layout.addSpacing(8)
        layout.addWidget(self.piece_yield_label)
        return card

    def _build_fq_map_section(self) -> QtWidgets.QWidget:
        """固定高のFQmapセクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("mapCard")
        section.setFixedHeight(FQ_MAP_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar = self._build_toolbar()
        layout.addWidget(self.toolbar)
        layout.addWidget(self._build_vertical_scroll_area(), stretch=1)
        return section

    def _build_vertical_scroll_area(self) -> QtWidgets.QScrollArea:
        """FQmap内容専用の縦スクロール領域。"""
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
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setObjectName("plotsContainer")
        plots_layout = QtWidgets.QVBoxLayout(self.plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(0)
        plots_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        for index, (metric, label, colors) in enumerate(
            MAP_DEFINITIONS
        ):
            view = self._build_fq_map_view(
                metric,
                label,
                make_color_map(colors),
            )
            self.views.append(view)
            plots_layout.addWidget(view.card)
            if index < len(MAP_DEFINITIONS) - 1:
                separator = QtWidgets.QFrame()
                separator.setObjectName("fqMapSeparator")
                separator.setFixedHeight(FQ_MAP_SEPARATOR_HEIGHT)
                self.fq_map_separators.append(separator)
                plots_layout.addWidget(separator)

        self.vertical_scroll_area.setWidget(self.plots_container)
        return self.vertical_scroll_area

    def _build_fmap_section(self) -> QtWidgets.QWidget:
        """固定高のFmapセクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("detailSubsection")
        section.setFixedHeight(FMAP_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.frame_map = FrameMapWidget(
            self.repository,
            self.full_data.lot_numbers,
            self.frame_map_data,
        )
        self.frame_map.frame_mode_requested.connect(
            self.show_selected_frame
        )
        self.fmap_selection_label = self.frame_map.selection_label
        layout.addWidget(self.frame_map)
        return section

    def _build_kde_section(self) -> QtWidgets.QWidget:
        """固定高のKDEセクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("detailSubsection")
        section.setFixedHeight(KDE_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        self.kde = KdeWidget(self.kde_data)
        layout.addWidget(self.kde)
        return section

    def _build_quality_trend_section(self) -> QtWidgets.QWidget:
        """固定高のF推移セクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("detailSubsection")
        section.setFixedHeight(QUALITY_TREND_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        self.quality_trend = QualityTrendWidget(
            self.quality_trend_data
        )
        self.quality_trend.hover_text_changed.connect(
            self._set_quality_trend_hover_text
        )
        layout.addWidget(self.quality_trend)
        return section

    def _build_detail_section(self) -> QtWidgets.QWidget:
        """lot集約と選択Frameの詳細セクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("detailGroup")
        section.setFixedHeight(
            FMAP_SECTION_HEIGHT
            + KDE_SECTION_HEIGHT
            + QUALITY_TREND_SECTION_HEIGHT
            + DETAIL_DIVIDER_HEIGHT
            + DETAIL_DIVIDER_HEIGHT
            + 2
        )
        self.detail_stack = QtWidgets.QStackedLayout(section)
        self.detail_stack.setContentsMargins(0, 0, 0, 0)
        self.lot_detail_page = self._build_lot_detail_page()
        self.frame_detail_page = self._build_frame_detail_page()
        self.detail_stack.addWidget(self.lot_detail_page)
        self.detail_stack.addWidget(self.frame_detail_page)
        self.detail_stack.setCurrentWidget(self.lot_detail_page)
        return section

    def _build_lot_detail_page(self) -> QtWidgets.QWidget:
        """F推移・Fmap・KDEのlot集約縦配置。"""
        page = QtWidgets.QWidget()
        page.setObjectName("lotDetailPage")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.quality_trend_section)

        self.quality_trend_divider = QtWidgets.QFrame()
        self.quality_trend_divider.setObjectName("detailDivider")
        self.quality_trend_divider.setFixedHeight(
            DETAIL_DIVIDER_HEIGHT
        )
        layout.addWidget(self.quality_trend_divider)
        layout.addWidget(self.fmap_section)

        self.detail_divider = QtWidgets.QFrame()
        self.detail_divider.setObjectName("detailDivider")
        self.detail_divider.setFixedHeight(DETAIL_DIVIDER_HEIGHT)
        layout.addWidget(self.detail_divider)
        layout.addWidget(self.kde_section)
        return page

    def _build_frame_detail_page(self) -> QtWidgets.QWidget:
        """加工パス・生値Fmap・KDEの選択Frame横配置。"""
        page = QtWidgets.QWidget()
        page.setObjectName("frameDetailPage")
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.processing_path_trend = ProcessingPathWidget()
        self.single_frame_map = SingleFrameMapWidget()
        self.single_frame_kde = SingleFrameKdeWidget()
        self.processing_path_trend.lot_mode_requested.connect(
            self.show_lot_aggregate
        )
        self.frame_detail_widgets = (
            self.processing_path_trend,
            self.single_frame_map,
            self.single_frame_kde,
        )

        for index, widget in enumerate(self.frame_detail_widgets):
            widget.setObjectName(
                widget.objectName() or "frameDetailPanel"
            )
            layout.addWidget(widget, stretch=1)
            if index < len(self.frame_detail_widgets) - 1:
                divider = QtWidgets.QFrame()
                divider.setObjectName("frameDetailDivider")
                divider.setFixedWidth(DETAIL_DIVIDER_HEIGHT)
                layout.addWidget(divider)
        return page

    def _build_fq_map_view(
        self,
        metric: str,
        label: str,
        color_map: pg.ColorMap,
    ) -> FqMapView:
        card = QtWidgets.QFrame()
        card.setObjectName("chartCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_widget = FqMapPlotWidget(background="#ffffff")
        plot_item = plot_widget.getPlotItem()
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.getViewBox().invertY(True)
        plot_item.getAxis("left").setWidth(205)
        plot_item.hideAxis("bottom")
        frame_axis = None
        if metric == "ng_rates":
            plot_item.showAxis("top")
            lot_axis = plot_item.getAxis("top")
            plot_item.layout.removeItem(plot_item.titleLabel)
            plot_item.titleLabel.hide()
            plot_item.layout.removeItem(lot_axis)
            plot_item.layout.addItem(lot_axis, 0, 1)
            plot_item.axes["top"]["pos"] = (0, 1)
            lot_axis.setHeight(FQ_MAP_LOT_AXIS_HEIGHT)

            frame_axis = pg.AxisItem(
                orientation="top",
                parent=plot_item,
            )
            frame_axis.linkToView(plot_item.getViewBox())
            frame_axis.setHeight(FQ_MAP_FRAME_AXIS_HEIGHT)
            plot_item.layout.addItem(frame_axis, 1, 1)
        else:
            plot_item.hideAxis("top")

        axis_names = (
            ("left", "top")
            if metric == "ng_rates"
            else ("left",)
        )
        for axis_name in axis_names:
            axis = plot_item.getAxis(axis_name)
            axis_font = QtGui.QFont()
            axis_font.setPointSize(9 if axis_name == "top" else 8)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=5)

        if frame_axis is not None:
            frame_axis_font = QtGui.QFont()
            frame_axis_font.setPointSize(6)
            frame_axis.setPen(pg.mkPen("#9aa3af"))
            frame_axis.setTextPen(pg.mkPen("#5f6875"))
            frame_axis.setStyle(
                tickFont=frame_axis_font,
                tickTextOffset=2,
                tickLength=3,
                hideOverlappingLabels=False,
            )

        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setColorMap(color_map)
        plot_item.addItem(image_item)

        selection_region = pg.LinearRegionItem(
            values=(-0.5, 0.5),
            orientation=pg.LinearRegionItem.Horizontal,
            movable=False,
            pen=pg.mkPen("#3157a4", width=1.4),
            brush=pg.mkBrush(49, 87, 164, 38),
            hoverPen=pg.mkPen("#3157a4", width=1.4),
            hoverBrush=pg.mkBrush(49, 87, 164, 38),
        )
        selection_region.setAcceptedMouseButtons(
            QtCore.Qt.MouseButton.NoButton
        )
        selection_region.setZValue(30)
        plot_item.addItem(selection_region)

        cell_selection_rect = QtWidgets.QGraphicsRectItem()
        cell_selection_rect.setPen(pg.mkPen("#173f8a", width=2.4))
        cell_selection_rect.setBrush(
            QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        )
        cell_selection_rect.setAcceptedMouseButtons(
            QtCore.Qt.MouseButton.NoButton
        )
        cell_selection_rect.setZValue(40)
        cell_selection_rect.hide()
        plot_item.addItem(cell_selection_rect)

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

        vertical_scrollbar = QtWidgets.QScrollBar(
            QtCore.Qt.Orientation.Vertical,
            plot_widget,
        )
        vertical_scrollbar.setObjectName("fqMapVerticalScrollBar")
        vertical_scrollbar.setAccessibleName(
            "FQmap検査項目の同期縦スクロール"
        )
        vertical_scrollbar.setToolTip(
            "3段のFQmapを同じ検査項目へ縦スクロール"
        )
        vertical_scrollbar.setSingleStep(1)
        vertical_scrollbar.hide()
        vertical_scrollbar.valueChanged.connect(
            self._scroll_fq_maps_vertically
        )
        plot_widget.set_vertical_scrollbar(vertical_scrollbar)

        layout.addWidget(plot_widget)
        view = FqMapView(
            metric=metric,
            label=label,
            card=card,
            plot_widget=plot_widget,
            plot_item=plot_item,
            image_item=image_item,
            color_bar=color_bar,
            vertical_scrollbar=vertical_scrollbar,
            selection_region=selection_region,
            cell_selection_rect=cell_selection_rect,
            frame_axis=frame_axis,
        )
        view.mouse_proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=lambda event, fq_view=view: (
                self._show_hover_details(fq_view, event)
            ),
        )
        plot_widget.scene().sigMouseClicked.connect(
            lambda event, fq_view=view: self._select_fq_row(
                fq_view,
                event,
            )
        )
        plot_item.getViewBox().sigResized.connect(
            lambda *_, fq_view=view: self._fq_map_view_resized(
                fq_view
            )
        )
        return view

    def _build_horizontal_navigation(self) -> QtWidgets.QLayout:
        navigation = QtWidgets.QHBoxLayout()
        navigation.setContentsMargins(0, 2, 0, 2)
        navigation.setSpacing(10)

        self.horizontal_scrollbar = QtWidgets.QScrollBar(
            QtCore.Qt.Orientation.Horizontal
        )
        self.horizontal_scrollbar.setObjectName("heatmapScrollBar")
        self.horizontal_scrollbar.valueChanged.connect(
            self._scroll_fq_maps
        )

        self.first_button = QtWidgets.QToolButton()
        self.first_button.setObjectName("navigationButton")
        self.first_button.setText("先頭")
        self.first_button.clicked.connect(
            lambda: self.horizontal_scrollbar.setValue(0)
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

    def _populate_filters(self) -> None:
        """カテゴリとvisionの選択肢設定。"""
        category_blocker = QtCore.QSignalBlocker(self.category_combo)
        self.category_combo.addItem("すべて", None)
        available = set(self.full_data.categories)
        for category in CATEGORY_ORDER:
            if category in available:
                self.category_combo.addItem(category, category)
        del category_blocker

        vision_blocker = QtCore.QSignalBlocker(self.vision_combo)
        self.vision_combo.addItem("すべて", None)
        available_visions = set(self.full_data.visions)
        for vision in VISION_ORDER:
            if vision in available_visions:
                self.vision_combo.addItem(vision, vision)
        del vision_blocker

        self.category_combo.currentIndexChanged.connect(
            self._render_fq_maps
        )
        self.vision_combo.currentIndexChanged.connect(
            self._render_fq_maps
        )

    @QtCore.Slot()
    def _render_fq_maps(self) -> None:
        category = self.category_combo.currentData()
        vision = self.vision_combo.currentData()
        data = self.full_data.filter_rows(category, vision)
        self.current_data = data
        if self.selected_colname not in data.colnames:
            self.selected_colname = data.colnames[0]
            self._clear_frame_selection()

        ng_max = self._nice_upper(
            float(np.nanpercentile(data.ng_rates, 99.5)),
            minimum=1.0,
        )
        deviation_max = self._nice_upper(
            float(np.nanpercentile(data.normalized_mean, 99.5)),
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
                (0.0, deviation_max),
            ),
            "normalized_std": (
                data.normalized_std,
                (0.0, std_max),
            ),
        }

        for view in self.views:
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
                [self._colname_ticks(data)]
            )
            view.plot_item.setYRange(
                -0.5,
                len(data.colnames) - 0.5,
                padding=0.0,
            )
            self._update_x_axis(view, data)
            self._draw_lot_separators(view, data)

        self._update_selection_regions()
        self._update_plot_heights()
        self._update_summary(
            data,
            category,
            vision,
            ng_max,
            deviation_max,
            std_max,
        )
        self._configure_horizontal_scrollbar(data)
        self._configure_vertical_scrollbars(reset_position=True)

    def _update_plot_heights(self) -> None:
        for view in self.views:
            top_axis_height = (
                FQ_MAP_TOP_AXIS_HEIGHT
                if view.metric == "ng_rates"
                else 0
            )
            view.card.setFixedHeight(
                FQ_MAP_PLOT_HEIGHT + top_axis_height
            )
        self.plots_container.adjustSize()

    def _fq_map_view_resized(self, view: FqMapView) -> None:
        """FQmap内部領域変更時の表示更新。"""
        self._position_vertical_scrollbar(view)
        if view.metric == "ng_rates":
            self._update_lot_axis_font(view)
        self._configure_vertical_scrollbars()

    def _configure_vertical_scrollbars(
        self,
        reset_position: bool = False,
    ) -> None:
        """セル最小高に基づく3段共通の縦表示範囲。"""
        if not self.views:
            return

        plot_heights = [
            view.plot_item.getViewBox().sceneBoundingRect().height()
            for view in self.views
        ]
        available_height = min(plot_heights)
        if available_height <= 0.0:
            return

        row_count = len(self.current_data.colnames)
        visible_rows = min(
            row_count,
            max(1, int(available_height // FQ_MAP_MIN_CELL_HEIGHT)),
        )
        maximum = max(0, row_count - visible_rows)
        current_position = self.views[0].vertical_scrollbar.value()
        if reset_position:
            selected_row = self.current_data.colnames.index(
                self.selected_colname
            )
            current_position = min(
                maximum,
                max(0, selected_row - visible_rows // 2),
            )
        current_position = min(current_position, maximum)

        for view in self.views:
            blocker = QtCore.QSignalBlocker(view.vertical_scrollbar)
            view.vertical_scrollbar.setRange(0, maximum)
            view.vertical_scrollbar.setPageStep(visible_rows)
            view.vertical_scrollbar.setValue(current_position)
            view.vertical_scrollbar.setVisible(maximum > 0)
            del blocker
            self._position_vertical_scrollbar(view)
        self._set_vertical_view_range(current_position, visible_rows)

    @QtCore.Slot(int)
    def _scroll_fq_maps_vertically(self, first_row: int) -> None:
        """3段FQmapの縦スクロール同期。"""
        visible_rows = self.views[0].vertical_scrollbar.pageStep()
        for view in self.views:
            blocker = QtCore.QSignalBlocker(view.vertical_scrollbar)
            view.vertical_scrollbar.setValue(first_row)
            del blocker
        self._set_vertical_view_range(first_row, visible_rows)

    def _set_vertical_view_range(
        self,
        first_row: int,
        visible_rows: int,
    ) -> None:
        """3段FQmapへの共通Y範囲反映。"""
        lower = first_row - 0.5
        upper = first_row + visible_rows - 0.5
        for view in self.views:
            view.plot_item.setYRange(lower, upper, padding=0.0)

    @staticmethod
    def _position_vertical_scrollbar(view: FqMapView) -> None:
        """PlotWidget右端へのスクロールバー重ね表示。"""
        scrollbar = view.vertical_scrollbar
        extent = scrollbar.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
        )
        top = FQ_MAP_TOP_AXIS_HEIGHT if view.metric == "ng_rates" else 0
        scrollbar.setGeometry(
            max(0, view.plot_widget.width() - extent),
            top,
            extent,
            max(0, view.plot_widget.height() - top),
        )
        scrollbar.raise_()

    @staticmethod
    def _colname_ticks(data: FqMapData) -> list[tuple[float, str]]:
        """検査項目ごとのcolname目盛。"""
        return [
            (float(row), colname)
            for row, colname in enumerate(data.colnames)
        ]

    def _configure_horizontal_scrollbar(
        self,
        data: FqMapData,
    ) -> None:
        maximum = max(0, data.lot_count - VISIBLE_LOTS)
        blocker = QtCore.QSignalBlocker(self.horizontal_scrollbar)
        self.horizontal_scrollbar.setRange(0, maximum)
        self.horizontal_scrollbar.setSingleStep(1)
        self.horizontal_scrollbar.setPageStep(VISIBLE_LOTS)
        self.horizontal_scrollbar.setValue(maximum)
        del blocker
        self._scroll_fq_maps(maximum)

    @QtCore.Slot(int)
    def _scroll_fq_maps(self, first_lot: int) -> None:
        data = self.current_data
        first_column = first_lot * len(FRAME_NUMBERS)
        last_column = min(
            first_column + VISIBLE_COLUMNS,
            len(data.frame_numbers),
        )
        for view in self.views:
            view.plot_item.setXRange(
                first_column - 0.5,
                last_column - 0.5,
                padding=0.0,
            )
            if view.frame_axis is not None:
                view.frame_axis.setTicks(
                    [
                        self._frame_ticks(
                            first_lot,
                            math.ceil(
                                (last_column - first_column)
                                / len(FRAME_NUMBERS)
                            ),
                        )
                    ]
                )

        self.first_button.setEnabled(first_lot > 0)
        self.latest_button.setEnabled(
            first_lot < self.horizontal_scrollbar.maximum()
        )
        if (
            self.selected_lot_index is not None
            and not (
                first_lot
                <= self.selected_lot_index
                < first_lot + VISIBLE_LOTS
            )
        ):
            self._clear_frame_selection()
        self.frame_map.set_context(
            self.selected_colname,
            first_lot,
        )
        self.kde.set_context(
            self.selected_colname,
            first_lot,
        )
        self.quality_trend.set_context(
            self.selected_colname,
            first_lot,
        )

    def _select_fq_row(
        self,
        view: FqMapView,
        event: SceneMouseEvent,
    ) -> None:
        """クリックしたFQmap検査項目の選択。"""
        scene_position = event.scenePos()
        if not view.plot_item.getViewBox().sceneBoundingRect().contains(
            scene_position
        ):
            return
        point = view.plot_item.getViewBox().mapSceneToView(scene_position)
        column = math.floor(point.x() + 0.5)
        row = math.floor(point.y() + 0.5)
        data = self.current_data
        if not (
            0 <= column < len(data.frame_numbers)
            and 0 <= row < len(data.colnames)
        ):
            return

        self.selected_colname = data.colnames[row]
        (
            self.selected_lot_index,
            self.selected_frame_no,
        ) = self._lot_frame_from_column(column)
        frame_available = bool(data.total_counts[row, column] > 0)
        self.single_frame_data = None
        self._update_selection_regions()
        self.frame_map.set_context(
            self.selected_colname,
            self.horizontal_scrollbar.value(),
        )
        self.frame_map.set_frame_available(frame_available)
        if self.detail_mode == "frame" and frame_available:
            self._update_selected_frame_detail()
        self._update_scope_label(
            data,
            self.category_combo.currentData(),
            self.vision_combo.currentData(),
        )

    def _update_selection_regions(self) -> None:
        """3段FQマップの選択行・セル同期。"""
        data = self.current_data
        row = data.colnames.index(self.selected_colname)
        for view in self.views:
            view.selection_region.setRegion(
                (float(row) - 0.5, float(row) + 0.5)
            )
            if (
                self.selected_lot_index is None
                or self.selected_frame_no is None
            ):
                view.cell_selection_rect.hide()
                continue
            column = (
                self.selected_lot_index * len(FRAME_NUMBERS)
                + self.selected_frame_no
                - 1
            )
            view.cell_selection_rect.setRect(
                QtCore.QRectF(
                    float(column) - 0.5,
                    float(row) - 0.5,
                    1.0,
                    1.0,
                )
            )
            view.cell_selection_rect.show()
        self.fmap_selection_label.setText(self.selected_colname)
        self.kde.set_context(
            self.selected_colname,
            self.horizontal_scrollbar.value(),
        )
        self.quality_trend.set_context(
            self.selected_colname,
            self.horizontal_scrollbar.value(),
        )

    def _clear_frame_selection(self) -> None:
        """FQmapセルと単一Frame選択の解除。"""
        self.selected_lot_index = None
        self.selected_frame_no = None
        self.single_frame_data = None
        for view in self.views:
            view.cell_selection_rect.hide()
        if hasattr(self, "frame_map"):
            self.frame_map.set_frame_available(False)

    @QtCore.Slot()
    def show_selected_frame(self) -> None:
        """選択候補の単一Frame詳細表示。"""
        if not self._update_selected_frame_detail():
            return
        self.detail_mode = "frame"
        self.detail_stack.setCurrentWidget(self.frame_detail_page)

    def _update_selected_frame_detail(self) -> bool:
        """選択候補の単一Frame詳細更新。"""
        if (
            self.selected_lot_index is None
            or self.selected_frame_no is None
        ):
            return False
        data = build_single_frame_map_data(
            self.repository,
            self.current_data.lot_numbers[self.selected_lot_index],
            self.selected_frame_no,
            self.selected_colname,
        )
        self.single_frame_data = data
        self.single_frame_map.set_data(data)
        self.single_frame_kde.set_data(
            build_single_frame_kde_data(data)
        )
        self.processing_path_trend.set_data(data)
        return True

    @QtCore.Slot()
    def show_lot_aggregate(self) -> None:
        """lot集約詳細表示への切替。"""
        self.detail_mode = "lot"
        self.detail_stack.setCurrentWidget(self.lot_detail_page)

    @staticmethod
    def _lot_frame_from_column(column: int) -> tuple[int, int]:
        """FQmap列からlot位置とFrameNoへの変換。"""
        lot_index, zero_based_frame = divmod(
            column,
            len(FRAME_NUMBERS),
        )
        return lot_index, zero_based_frame + 1

    def _update_x_axis(
        self,
        view: FqMapView,
        data: FqMapData,
    ) -> None:
        if view.metric != "ng_rates":
            return
        lot_ticks = [
            (
                lot_index * len(FRAME_NUMBERS)
                + (len(FRAME_NUMBERS) - 1) / 2,
                (
                    f"{lot_number}  "
                    f"{data.lot_start_times[lot_index]:%H:%M:%S}"
                ),
            )
            for lot_index, lot_number in enumerate(data.lot_numbers)
        ]
        view.plot_item.getAxis("top").setTicks([lot_ticks])
        self._update_lot_axis_font(view)

    def _update_lot_axis_font(self, view: FqMapView) -> None:
        """lot幅に応じた上側目盛フォント。"""
        visible_lots = min(VISIBLE_LOTS, self.current_data.lot_count)
        lot_width = (
            view.plot_item.getViewBox().sceneBoundingRect().width()
            / visible_lots
        )
        axis_font = QtGui.QFont()
        axis_font.setPointSize(9)
        if lot_width < 110:
            axis_font.setStretch(58)
        elif lot_width < 140:
            axis_font.setStretch(70)
        view.plot_item.getAxis("top").setStyle(tickFont=axis_font)

    @staticmethod
    def _frame_ticks(
        first_lot: int,
        visible_lot_count: int,
    ) -> list[tuple[float, str]]:
        """表示中lotのFrameNo補助目盛。"""
        ticks: list[tuple[float, str]] = []
        for offset in range(visible_lot_count):
            lot_index = first_lot + offset
            lot_start = lot_index * len(FRAME_NUMBERS)
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

    def _draw_lot_separators(
        self,
        view: FqMapView,
        data: FqMapData,
    ) -> None:
        for separator in view.separators:
            view.plot_item.removeItem(separator)
        view.separators.clear()

        for lot_index in range(1, data.lot_count):
            separator = make_lot_separator(
                lot_index * len(FRAME_NUMBERS) - 0.5
            )
            view.plot_item.addItem(separator)
            view.separators.append(separator)

    def _update_summary(
        self,
        data: FqMapData,
        category: str | None,
        vision: str | None,
        ng_max: float,
        deviation_max: float,
        std_max: float,
    ) -> None:
        self.all_lots_label.setText(f"全{data.lot_count} lot")
        self._update_scope_label(data, category, vision)
        self.piece_yield_label.setText(
            f"個片歩留まり {data.piece_yield:.3f}%"
        )
        self.piece_yield_label.setToolTip(
            "検査項目が1つでもNGの個片を除外  |  "
            f"OK {data.ok_pieces:,} / {data.total_pieces:,}個片"
        )
        self.scale_label.setText(
            f"NG 0–{ng_max:g}%  |  "
            f"逸脱度 0–{deviation_max:g}  |  "
            f"std 0–{std_max:g}"
        )
        self.scale_label.setToolTip(
            "規格逸脱度: 個々の測定では"
            "0=最良・規格中心 / 1=規格限界 / 1超=NG。"
            "表示値はセル内の平均"
        )

    def _update_scope_label(
        self,
        data: FqMapData,
        category: str | None,
        vision: str | None,
    ) -> None:
        """表示範囲の要約。"""
        category_text = category if category is not None else "全"
        vision_text = vision if vision is not None else "全"
        measurement_count = data.total_measurements
        self.scope_label.setText(
            f"{len(data.colnames)}項目  |  "
            f"{measurement_count / 1_000_000:.1f}M測定"
        )
        self.scope_label.setToolTip(
            f"{category_text} / {vision_text}  |  "
            f"{len(data.colnames)}項目  |  "
            f"{measurement_count:,}測定"
        )

    def _show_hover_details(
        self,
        view: FqMapView,
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
        data = self.current_data
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
            detail = f"規格逸脱度 平均 {value:.4f}"
        else:
            detail = f"規格位置・使用率 std {value:.4f}"
        self.hover_label.setText(prefix + detail)

    def _reset_hover_text(self) -> None:
        self.hover_label.setText(
            "セルにカーソルを合わせると詳細を表示"
        )

    @QtCore.Slot(str)
    def _set_quality_trend_hover_text(self, text: str) -> None:
        """F推移ホバー詳細のフッター反映。"""
        if text:
            self.hover_label.setText(text)
        else:
            self._reset_hover_text()

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
