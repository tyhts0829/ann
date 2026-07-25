from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.frame_map import (
    FrameMapData,
    FrameMapWidget,
    VISIBLE_LOTS,
)
from src.analysis.kde import (
    KdeData,
    KdeWidget,
    build_kde_data,
)
from src.analysis.map_palettes import MAP_DEFINITIONS, make_color_map
from src.analysis.quality_columns import (
    CATEGORY_ORDER,
    SPEC_ORDER,
    VISION_ORDER,
)
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository


FRAME_NUMBERS = np.arange(1, 25)
VISIBLE_COLUMNS = len(FRAME_NUMBERS) * VISIBLE_LOTS
FQ_MAP_SECTION_HEIGHT = DASHBOARD_CONFIG.fqmap_height
FMAP_SECTION_HEIGHT = DASHBOARD_CONFIG.fmap_height
KDE_SECTION_HEIGHT = DASHBOARD_CONFIG.kde_height
FQ_MAP_PLOT_HEIGHT = DASHBOARD_CONFIG.fqmap_plot_height


@dataclass(frozen=True)
class FqMapData:
    """3種類のFQマップ集計結果。"""

    colnames: tuple[str, ...]
    categories: tuple[str, ...]
    visions: tuple[str, ...]
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
            column_lots=self.column_lots,
            frame_numbers=self.frame_numbers,
            ng_rates=self.ng_rates[row_indices],
            normalized_mean=self.normalized_mean[row_indices],
            normalized_std=self.normalized_std[row_indices],
            ng_counts=self.ng_counts[row_indices],
            total_counts=self.total_counts[row_indices],
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
    selection_region: pg.LinearRegionItem
    separators: list[pg.InfiniteLine] = field(default_factory=list)
    mouse_proxy: pg.SignalProxy | None = None


def build_fq_map_data(
    repository: QualityRepository,
    lot_numbers: tuple[str, ...] | None = None,
) -> FqMapData:
    """FrameNo別の3種類のFQマップデータ生成。"""
    if lot_numbers is None:
        lot_numbers = tuple(record[0] for record in repository.lots())

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
    """NG率・規格位置平均・標準偏差の3段FQマップ。"""

    def __init__(
        self,
        repository: QualityRepository,
        full_data: FqMapData | None = None,
        frame_map_data: FrameMapData | None = None,
        kde_data: KdeData | None = None,
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
        self.current_data = self.full_data
        self.selected_colname = self.full_data.colnames[0]
        self.views: list[FqMapView] = []
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
        layout.addWidget(self.fq_map_section)
        layout.addWidget(self.fmap_section)
        layout.addWidget(self.kde_section)
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
        self.category_combo.setAccessibleName("カテゴリフィルター")

        vision_label = QtWidgets.QLabel("Vision")
        vision_label.setObjectName("fieldLabel")
        self.vision_combo = QtWidgets.QComboBox()
        self.vision_combo.setObjectName("visionCombo")
        self.vision_combo.setMinimumWidth(125)
        self.vision_combo.setAccessibleName("Visionフィルター")

        self.ng_rate_label = QtWidgets.QLabel()
        self.ng_rate_label.setObjectName("ngRateLabel")

        layout.addWidget(fq_map_title)
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
        layout.addWidget(self.ng_rate_label)
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
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setObjectName("plotsContainer")
        plots_layout = QtWidgets.QVBoxLayout(self.plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(0)
        plots_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        for metric, label, colors in MAP_DEFINITIONS:
            view = self._build_fq_map_view(
                metric,
                label,
                make_color_map(colors),
            )
            self.views.append(view)
            plots_layout.addWidget(view.card)

        self.vertical_scroll_area.setWidget(self.plots_container)
        return self.vertical_scroll_area

    def _build_fmap_section(self) -> QtWidgets.QWidget:
        """固定高のFmapセクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("mapCard")
        section.setFixedHeight(FMAP_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.frame_map = FrameMapWidget(
            self.repository,
            self.full_data.lot_numbers,
            self.frame_map_data,
        )
        self.fmap_selection_label = self.frame_map.selection_label
        layout.addWidget(self.frame_map)
        return section

    def _build_kde_section(self) -> QtWidgets.QWidget:
        """固定高のKDEセクション。"""
        section = QtWidgets.QFrame()
        section.setObjectName("mapCard")
        section.setFixedHeight(KDE_SECTION_HEIGHT)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        self.kde = KdeWidget(self.kde_data)
        layout.addWidget(self.kde)
        return section

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

        plot_widget = pg.PlotWidget(background="#ffffff")
        plot_item = plot_widget.getPlotItem()
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.getViewBox().invertY(True)
        plot_item.getAxis("left").setWidth(205)
        plot_item.hideAxis("bottom")
        if metric == "ng_rates":
            plot_item.showAxis("top")
            plot_item.getAxis("top").setHeight(28)
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
            axis_font.setPointSize(7)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=5)

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
        view = FqMapView(
            metric=metric,
            label=label,
            card=card,
            plot_widget=plot_widget,
            plot_item=plot_item,
            image_item=image_item,
            color_bar=color_bar,
            selection_region=selection_region,
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
        return view

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
            self._scroll_fq_maps
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

        ng_max = self._nice_upper(
            float(np.nanpercentile(data.ng_rates, 99.5)),
            minimum=1.0,
        )
        mean_abs_max = self._nice_upper(
            float(np.nanpercentile(np.abs(data.normalized_mean), 99.5)),
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
            mean_abs_max,
            std_max,
        )
        self._configure_horizontal_scrollbar(data)
        self.vertical_scroll_area.verticalScrollBar().setValue(0)

    def _update_plot_heights(self) -> None:
        for view in self.views:
            top_axis_height = 28 if view.metric == "ng_rates" else 0
            view.card.setFixedHeight(
                FQ_MAP_PLOT_HEIGHT + top_axis_height
            )
        self.plots_container.adjustSize()

    @staticmethod
    def _colname_ticks(data: FqMapData) -> list[tuple[float, str]]:
        """表示密度に応じたcolname目盛。"""
        if len(data.colnames) <= 18:
            return [
                (float(row), colname)
                for row, colname in enumerate(data.colnames)
            ]

        grouped_rows: dict[str, list[int]] = {}
        for row, colname in enumerate(data.colnames):
            base_colname = colname.rsplit("_v", maxsplit=1)[0]
            grouped_rows.setdefault(base_colname, []).append(row)
        return [
            (
                float(np.mean(rows)),
                f"{base_colname}  ·  _v1/_v2/_v3",
            )
            for base_colname, rows in grouped_rows.items()
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

        self.first_button.setEnabled(first_lot > 0)
        self.latest_button.setEnabled(
            first_lot < self.horizontal_scrollbar.maximum()
        )
        self.frame_map.set_context(
            self.selected_colname,
            first_lot,
        )
        self.kde.set_context(
            self.selected_colname,
            first_lot,
        )

    def _select_fq_row(
        self,
        view: FqMapView,
        event: object,
    ) -> None:
        """クリックしたFQmap検査項目の選択。"""
        scene_position = event.scenePos()
        if not view.plot_widget.sceneBoundingRect().contains(
            scene_position
        ):
            return
        point = view.plot_item.getViewBox().mapSceneToView(scene_position)
        column = round(point.x())
        row = round(point.y())
        data = self.current_data
        if not (
            0 <= column < len(data.frame_numbers)
            and 0 <= row < len(data.colnames)
        ):
            return

        self.selected_colname = data.colnames[row]
        self._update_selection_regions()
        self.frame_map.set_context(
            self.selected_colname,
            self.horizontal_scrollbar.value(),
        )
        self._update_scope_label(
            data,
            self.category_combo.currentData(),
            self.vision_combo.currentData(),
        )

    def _update_selection_regions(self) -> None:
        """3段FQマップの選択行同期。"""
        data = self.current_data
        row = data.colnames.index(self.selected_colname)
        for view in self.views:
            view.selection_region.setRegion(
                (float(row) - 0.5, float(row) + 0.5)
            )
        self.fmap_selection_label.setText(self.selected_colname)
        self.kde.set_context(
            self.selected_colname,
            self.horizontal_scrollbar.value(),
        )

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
                lot_number,
            )
            for lot_index, lot_number in enumerate(data.lot_numbers)
        ]
        view.plot_item.getAxis("top").setTicks([lot_ticks])

    def _draw_lot_separators(
        self,
        view: FqMapView,
        data: FqMapData,
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
        data: FqMapData,
        category: str | None,
        vision: str | None,
        ng_max: float,
        mean_abs_max: float,
        std_max: float,
    ) -> None:
        self.all_lots_label.setText(f"全{data.lot_count} lot")
        self._update_scope_label(data, category, vision)
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
        data: FqMapData,
        category: str | None,
        vision: str | None,
    ) -> None:
        """表示範囲と選択検査項目の要約。"""
        category_text = category if category is not None else "全カテゴリ"
        vision_text = vision if vision is not None else "全vision"
        self.scope_label.setText(
            f"{category_text} / {vision_text}  |  "
            f"{len(data.colnames)}項目  |  "
            f"{data.total_measurements:,}測定  |  "
            f"選択項目 {self.selected_colname}"
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
            detail = f"規格位置・使用率 平均 {value:.4f}"
        else:
            detail = f"規格位置・使用率 std {value:.4f}"
        self.hover_label.setText(prefix + detail)

    def _reset_hover_text(self) -> None:
        self.hover_label.setText(
            "セルにカーソルを合わせると詳細を表示"
        )

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
