"""可視化間で共有するプロット装飾。"""

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

LOT_SEPARATOR_COLOR = "#8f99a6"
LOT_SEPARATOR_WIDTH = 1.0
LOT_SEPARATOR_Z = 25


def make_lot_separator(position: float) -> pg.InfiniteLine:
    """lot境界線の生成。"""
    line = pg.InfiniteLine(
        pos=position,
        angle=90,
        movable=False,
        pen=pg.mkPen(
            LOT_SEPARATOR_COLOR,
            width=LOT_SEPARATOR_WIDTH,
        ),
    )
    line.setZValue(LOT_SEPARATOR_Z)
    return line


class LotSeparatorWidget(QtWidgets.QFrame):
    """分割プロット間のlot境界線。"""

    def __init__(self, bottom_margin: int) -> None:
        super().__init__()
        self.bottom_margin = bottom_margin
        self.line_color = LOT_SEPARATOR_COLOR
        self.line_width = LOT_SEPARATOR_WIDTH
        self.setObjectName("lotSeparator")
        self.setFixedWidth(round(self.line_width))
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """データ領域内の境界線描画。"""
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(
            0,
            0,
            self.width(),
            self.height() - self.bottom_margin,
            QtGui.QColor(self.line_color),
        )


def make_lot_separator_widget(
    bottom_margin: int = 0,
) -> LotSeparatorWidget:
    """分割プロット間のlot境界線生成。"""
    return LotSeparatorWidget(bottom_margin)
