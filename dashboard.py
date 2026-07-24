#!/usr/bin/env python3
"""最終検査データの品質ダッシュボード。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtWidgets

from src.analysis.ng_rate_heatmap import NgRateHeatmapWidget
from src.standardized.quality_data import QualityRepository


class DashboardWindow(QtWidgets.QMainWindow):
    """品質グラフを構成するメイン画面。"""

    def __init__(self, repository: QualityRepository) -> None:
        super().__init__()
        self.setWindowTitle("Quality Dashboard")
        self.resize(1900, 1500)
        self.setMinimumSize(1100, 760)

        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(10)

        self.heatmap = NgRateHeatmapWidget(repository)
        layout.addWidget(self.heatmap, stretch=1)
        self.setStyleSheet(STYLESHEET)


STYLESHEET = """
QWidget#central {
    background: #f3f5f7;
    color: #20262f;
}
QFrame#toolbarCard {
    background: #ffffff;
    border: 1px solid #d5dbe3;
}
QFrame#chartCard {
    background: #ffffff;
    border: none;
}
QLabel#fieldLabel {
    color: #667080;
    font-size: 12px;
    font-weight: 600;
}
QLabel#allLotsLabel {
    color: #08766b;
    background: #e8f5f2;
    border: 1px solid #8fc9c1;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#scopeLabel {
    color: #667080;
    font-size: 12px;
}
QComboBox#categoryCombo {
    color: #20262f;
    background: #ffffff;
    border: 1px solid #aeb7c4;
    padding: 7px 12px;
    min-height: 22px;
    font-size: 12px;
}
QComboBox#categoryCombo:hover {
    border-color: #607086;
}
QComboBox#categoryCombo::drop-down {
    border: none;
    width: 26px;
}
QComboBox#categoryCombo QAbstractItemView {
    color: #20262f;
    background: #ffffff;
    border: 1px solid #aeb7c4;
    selection-color: #ffffff;
    selection-background-color: #287d78;
}
QLabel#ngRateLabel {
    color: #08766b;
    background: #e8f5f2;
    border: 1px solid #8fc9c1;
    padding: 8px 13px;
    font-size: 13px;
    font-weight: 700;
}
QToolButton#navigationButton {
    color: #303846;
    background: #ffffff;
    border: 1px solid #aeb7c4;
    min-width: 48px;
    min-height: 24px;
    padding: 2px 8px;
    font-size: 11px;
}
QToolButton#navigationButton:hover {
    border-color: #607086;
    background: #eef1f4;
}
QToolButton#navigationButton:disabled {
    color: #a6adb7;
    border-color: #d7dce2;
    background: #f3f5f7;
}
QScrollBar#heatmapScrollBar:horizontal {
    background: #e4e8ed;
    border: 1px solid #c2c9d2;
    height: 13px;
}
QScrollBar#heatmapScrollBar::handle:horizontal {
    background: #3a9a93;
    min-width: 42px;
    margin: 1px;
}
QScrollBar#heatmapScrollBar::handle:horizontal:hover {
    background: #267f79;
}
QScrollBar#heatmapScrollBar::add-line:horizontal,
QScrollBar#heatmapScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar#heatmapScrollBar::add-page:horizontal,
QScrollBar#heatmapScrollBar::sub-page:horizontal {
    background: transparent;
}
QScrollArea#heatmapScrollArea,
QWidget#plotsContainer {
    background: #ffffff;
    border: none;
}
QScrollArea#heatmapScrollArea QScrollBar:vertical {
    background: #e4e8ed;
    border: 1px solid #c2c9d2;
    width: 13px;
}
QScrollArea#heatmapScrollArea QScrollBar::handle:vertical {
    background: #3a9a93;
    min-height: 42px;
    margin: 1px;
}
QScrollArea#heatmapScrollArea QScrollBar::handle:vertical:hover {
    background: #267f79;
}
QScrollArea#heatmapScrollArea QScrollBar::add-line:vertical,
QScrollArea#heatmapScrollArea QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollArea#heatmapScrollArea QScrollBar::add-page:vertical,
QScrollArea#heatmapScrollArea QScrollBar::sub-page:vertical {
    background: transparent;
}
QLabel#footerNote {
    color: #737d8b;
    font-size: 11px;
}
"""


def parse_args() -> argparse.Namespace:
    """コマンドライン引数の定義。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "data"
            / "standardized"
            / "quality_data_100lots.parquet"
        ),
        help="標準化済み品質データParquetのパス",
    )
    return parser.parse_args()


def main() -> None:
    """デスクトップアプリケーションの起動。"""
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Quality Dashboard")
    app.setStyle("Fusion")
    window = DashboardWindow(QualityRepository(args.data))
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
