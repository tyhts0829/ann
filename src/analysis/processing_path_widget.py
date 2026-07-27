from __future__ import annotations

import math
import os

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.frame_map import SingleFrameMapData
from src.analysis.processing_path import (
    EquipmentPathCatalog,
    ProcessingPathSeries,
    build_processing_path_series,
    load_equipment_path_catalog,
)

PATH_COLORS = (
    "#176b87",
    "#c65d2e",
    "#4f7c45",
    "#865da0",
    "#b58900",
    "#287d78",
)


class ProcessingPathWidget(QtWidgets.QWidget):
    """選択Frameの加工パス推移。"""

    lot_mode_requested = QtCore.Signal()

    def __init__(
        self,
        catalog: EquipmentPathCatalog | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.catalog = load_equipment_path_catalog() if catalog is None else catalog
        self.current_data: SingleFrameMapData | None = None
        self.current_series: tuple[ProcessingPathSeries, ...] = ()
        self.path_curves: list[pg.PlotDataItem] = []

        self.equipment_combo = QtWidgets.QComboBox()
        self.equipment_combo.setObjectName("equipmentCombo")
        self.equipment_combo.setAccessibleName("加工設備")
        self.lot_mode_button = QtWidgets.QToolButton()
        self.lot_mode_button.setObjectName("frameMapModeButton")
        self.lot_mode_button.setText("lot集約に戻る")

        self._build_ui()
        self._populate_equipment_combo()
        self.equipment_combo.currentIndexChanged.connect(self._change_equipment)
        self.lot_mode_button.clicked.connect(self._request_lot_mode)

    def _build_ui(self) -> None:
        """加工パス推移UIの構築。"""
        self.setObjectName("processingPathWidget")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_plot(), stretch=1)

    def _build_header(self) -> QtWidgets.QWidget:
        """設備選択とFrame情報のヘッダー。"""
        header = QtWidgets.QWidget()
        header.setObjectName("processingPathHeader")
        layout = QtWidgets.QGridLayout(header)
        layout.setContentsMargins(10, 6, 10, 5)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(3)

        title = QtWidgets.QLabel("加工パス推移")
        title.setObjectName("mapSectionTitle")
        title.setProperty("sectionRole", "detail")
        equipment_label = QtWidgets.QLabel("設備")
        equipment_label.setObjectName("fieldLabel")
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setObjectName("processingPathSelectionLabel")
        self.frame_label = QtWidgets.QLabel()
        self.frame_label.setObjectName("processingPathFrameLabel")

        layout.addWidget(title, 0, 0)
        layout.addWidget(equipment_label, 0, 1)
        layout.addWidget(self.equipment_combo, 0, 2)
        layout.setColumnStretch(2, 1)
        layout.addWidget(self.selection_label, 1, 0, 1, 2)
        layout.addWidget(self.frame_label, 1, 2)
        layout.addWidget(self.lot_mode_button, 0, 3, 2, 1)
        return header

    def _build_plot(self) -> pg.PlotWidget:
        """加工順別の生値プロット生成。"""
        self.plot_widget = pg.PlotWidget(background="#ffffff")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.hideButtons()
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.showGrid(x=True, y=True, alpha=0.14)
        self.plot_item.setLabel("bottom", "加工順")

        axis_font = QtGui.QFont()
        axis_font.setPointSize(7)
        for axis_name in ("bottom", "left"):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#7a8492"))
            axis.setTextPen(pg.mkPen("#303846"))
            axis.setStyle(tickFont=axis_font, tickTextOffset=3)

        self.legend = self.plot_item.addLegend(
            offset=(8, 8),
            brush=pg.mkBrush(255, 255, 255, 220),
            pen=pg.mkPen("#c5ccd5"),
        )
        self.legend.setColumnCount(2)
        self.lower_line = self._build_spec_line("#3157a4")
        self.upper_line = self._build_spec_line("#b72e2e")
        self.plot_item.addItem(self.lower_line)
        self.plot_item.addItem(self.upper_line)
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
        line.setZValue(10)
        line.hide()
        return line

    def _populate_equipment_combo(self) -> None:
        """設備選択肢の設定。"""
        blocker = QtCore.QSignalBlocker(self.equipment_combo)
        for equipment in self.catalog.equipments:
            self.equipment_combo.addItem(
                equipment.label,
                equipment.id,
            )
        default_index = self.equipment_combo.findData(self.catalog.default_equipment_id)
        self.equipment_combo.setCurrentIndex(default_index)
        del blocker

    def set_data(self, data: SingleFrameMapData) -> None:
        """キャッシュ済み単一Frameデータの描画。"""
        self.current_data = data
        self.selection_label.setText(data.colname)
        self.frame_label.setText(f"FrameNo {data.frame_no:02d}")
        unit_label = "生値" if not data.unit else f"生値 ({data.unit})"
        self.plot_item.setLabel("left", unit_label)
        self._render_current_equipment()

    @QtCore.Slot(int)
    def _change_equipment(self, _: int) -> None:
        """選択設備の加工パスへの切替。"""
        self._render_current_equipment()

    def _render_current_equipment(self) -> None:
        """選択設備に対応する複数加工パスの描画。"""
        data = self.current_data
        if data is None:
            return

        equipment_id = str(self.equipment_combo.currentData())
        equipment = self.catalog.equipment(equipment_id)
        ng_directions = data.upper_ng_flags.astype(
            np.int8
        ) - data.lower_ng_flags.astype(np.int8)
        self.current_series = build_processing_path_series(
            data.raw_values,
            ng_directions,
            equipment,
        )
        self._clear_path_curves()

        for index, series in enumerate(self.current_series):
            curve = pg.PlotDataItem(
                np.asarray(series.steps, dtype=float),
                np.asarray(series.values, dtype=float),
                pen=pg.mkPen(
                    PATH_COLORS[index % len(PATH_COLORS)],
                    width=1.7,
                ),
                connect="finite",
            )
            curve.setZValue(4)
            self.plot_item.addItem(curve)
            self.legend.addItem(curve, series.label)
            self.path_curves.append(curve)

        self._set_spec_line(self.lower_line, data.spec_lower)
        self._set_spec_line(self.upper_line, data.spec_upper)
        self._set_plot_range(data)

    def _clear_path_curves(self) -> None:
        """設備切替前の加工パス曲線消去。"""
        for curve in self.path_curves:
            self.plot_item.removeItem(curve)
        self.path_curves.clear()
        self.legend.clear()

    @staticmethod
    def _set_spec_line(line: pg.InfiniteLine, value: float) -> None:
        """有限な規格値の表示。"""
        if math.isfinite(value):
            line.setValue(value)
            line.show()
        else:
            line.hide()

    def _set_plot_range(self, data: SingleFrameMapData) -> None:
        """加工順と生値の表示範囲設定。"""
        step_values = np.concatenate(
            [np.asarray(series.steps, dtype=float) for series in self.current_series]
        )
        finite_values = np.concatenate(
            [np.asarray(series.values, dtype=float) for series in self.current_series]
        )
        finite_values = finite_values[np.isfinite(finite_values)]
        specification_values = np.asarray(
            [data.spec_lower, data.spec_upper],
            dtype=float,
        )
        specification_values = specification_values[np.isfinite(specification_values)]
        y_values = np.concatenate([finite_values, specification_values])

        x_min = float(np.nanmin(step_values))
        x_max = float(np.nanmax(step_values))
        x_span = max(1.0, x_max - x_min)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        y_span = max(1e-9, y_max - y_min)
        self.plot_item.setXRange(
            x_min - x_span * 0.03,
            x_max + x_span * 0.03,
            padding=0.0,
        )
        self.plot_item.setYRange(
            y_min - y_span * 0.08,
            y_max + y_span * 0.08,
            padding=0.0,
        )

    @QtCore.Slot()
    def _request_lot_mode(self) -> None:
        """lot集約表示要求の通知。"""
        self.lot_mode_requested.emit()
