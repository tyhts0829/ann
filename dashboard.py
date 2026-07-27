#!/usr/bin/env python3
"""最終検査データの品質ダッシュボード。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis.fq_map import (
    FqMapData,
    FqMapWidget,
    build_fq_map_data,
)
from src.analysis.frame_map import FrameMapData, build_frame_map_data
from src.analysis.kde import (
    KdeData,
    build_kde_data,
)
from src.analysis.quality_trend import (
    QualityTrendData,
    build_quality_trend_data,
)
from src.dashboard_config import DASHBOARD_CONFIG
from src.standardized.quality_data import QualityRepository


class DashboardScrollArea(QtWidgets.QScrollArea):
    """内容幅を保った縦方向専用スクロール領域。"""

    def setWidget(self, widget: QtWidgets.QWidget) -> None:
        """スクロール対象Widgetの設定。"""
        super().setWidget(widget)
        self._resize_content()
        QtCore.QTimer.singleShot(0, self._resize_content)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """表示領域変更時の内容幅同期。"""
        super().resizeEvent(event)
        self._resize_content()

    def _resize_content(self) -> None:
        """内容幅のviewportへの同期。"""
        content = self.widget()
        if content is None:
            return
        content.resize(
            self.viewport().width(),
            max(
                self.viewport().height(),
                content.sizeHint().height(),
            ),
        )


class DashboardDataWorker(QtCore.QObject):
    """表示用集計データのバックグラウンド読込。"""

    loaded = QtCore.Signal(object, object, object, object)
    failed = QtCore.Signal(str)

    def __init__(self, parquet_path: Path) -> None:
        super().__init__()
        self.parquet_path = parquet_path

    @QtCore.Slot()
    def load(self) -> None:
        """FQmap・Fmap・KDE・F推移集計データの生成。"""
        repository = QualityRepository(self.parquet_path)
        try:
            fq_map_data = build_fq_map_data(repository)
            lot_numbers = fq_map_data.lot_numbers
            frame_map_data = build_frame_map_data(
                repository,
                lot_numbers,
            )
            kde_data = build_kde_data(
                repository,
                lot_numbers,
            )
            quality_trend_data = build_quality_trend_data(
                repository,
                lot_numbers,
            )
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.loaded.emit(
                fq_map_data,
                frame_map_data,
                kde_data,
                quality_trend_data,
            )
        finally:
            repository.close()


class DashboardWindow(QtWidgets.QMainWindow):
    """品質グラフを構成するメイン画面。"""

    data_loaded = QtCore.Signal()
    data_load_failed = QtCore.Signal(str)

    def __init__(self, repository: QualityRepository) -> None:
        super().__init__()
        self.repository = repository
        self.fq_map: FqMapWidget | None = None
        self.dashboard_scroll_area: DashboardScrollArea | None = None
        self._loader_thread: QtCore.QThread | None = None
        self._data_worker: DashboardDataWorker | None = None
        self._close_requested = False
        self.setWindowTitle("Quality Dashboard")
        self.resize(
            DASHBOARD_CONFIG.window_width,
            DASHBOARD_CONFIG.window_height,
        )
        self.setMinimumSize(1100, 760)
        self.close_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Esc"),
            self,
        )
        self.close_shortcut.activated.connect(self.close)

        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        self.content_layout = QtWidgets.QVBoxLayout(central)
        self.content_layout.setContentsMargins(12, 12, 12, 10)
        self.content_layout.setSpacing(10)

        self.loading_widget = self._build_loading_widget()
        self.content_layout.addWidget(self.loading_widget, stretch=1)
        self.setStyleSheet(STYLESHEET)
        QtCore.QTimer.singleShot(0, self._start_data_load)

    def _build_loading_widget(self) -> QtWidgets.QWidget:
        """初期集計中の表示。"""
        container = QtWidgets.QWidget()
        container.setObjectName("loadingContainer")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        card = QtWidgets.QFrame()
        card.setObjectName("loadingCard")
        card.setFixedWidth(420)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(12)

        title = QtWidgets.QLabel("品質データを読み込んでいます")
        title.setObjectName("loadingTitle")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.loading_message = QtWidgets.QLabel(
            "FQmap・Fmap・KDE・F推移の集計中です。"
            "しばらくお待ちください。"
        )
        self.loading_message.setObjectName("loadingMessage")
        self.loading_message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.loading_progress = QtWidgets.QProgressBar()
        self.loading_progress.setObjectName("loadingProgress")
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setTextVisible(False)

        card_layout.addWidget(title)
        card_layout.addWidget(self.loading_message)
        card_layout.addWidget(self.loading_progress)
        layout.addWidget(card)
        return container

    @QtCore.Slot()
    def _start_data_load(self) -> None:
        """バックグラウンド集計の開始。"""
        thread = QtCore.QThread(self)
        worker = DashboardDataWorker(self.repository.parquet_path)
        worker.moveToThread(thread)

        thread.started.connect(worker.load)
        worker.loaded.connect(self._show_dashboard)
        worker.failed.connect(self._show_load_error)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.loaded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._clear_loader)
        thread.finished.connect(thread.deleteLater)

        self._loader_thread = thread
        self._data_worker = worker
        thread.start()

    @QtCore.Slot(object, object, object, object)
    def _show_dashboard(
        self,
        fq_map_data: FqMapData,
        frame_map_data: FrameMapData,
        kde_data: KdeData,
        quality_trend_data: QualityTrendData,
    ) -> None:
        """集計済みデータによるダッシュボード表示。"""
        if self._close_requested:
            return
        self.fq_map = FqMapWidget(
            self.repository,
            fq_map_data,
            frame_map_data,
            kde_data,
            quality_trend_data,
        )
        self.dashboard_scroll_area = DashboardScrollArea()
        self.dashboard_scroll_area.setObjectName("dashboardScrollArea")
        self.dashboard_scroll_area.setWidgetResizable(False)
        self.dashboard_scroll_area.setFrameShape(
            QtWidgets.QFrame.Shape.NoFrame
        )
        self.dashboard_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.dashboard_scroll_area.setWidget(self.fq_map)
        self.content_layout.replaceWidget(
            self.loading_widget,
            self.dashboard_scroll_area,
        )
        self.loading_widget.deleteLater()
        self.data_loaded.emit()

    @QtCore.Slot(str)
    def _show_load_error(self, message: str) -> None:
        """初期集計エラーの表示。"""
        if self._close_requested:
            return
        self.loading_progress.hide()
        self.loading_message.setText(f"品質データを読み込めませんでした。\n{message}")
        self.data_load_failed.emit(message)

    @QtCore.Slot()
    def _clear_loader(self) -> None:
        """読込スレッド参照の解放。"""
        self._loader_thread = None
        self._data_worker = None
        if self._close_requested:
            self.close()
            QtWidgets.QApplication.quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """読込中の安全なウィンドウ終了。"""
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._close_requested = True
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)


STYLESHEET = """
QWidget#central {
    background: #f3f5f7;
    color: #20262f;
}
QWidget#loadingContainer {
    background: #f3f5f7;
}
QFrame#loadingCard {
    background: #ffffff;
    border: 1px solid #d5dbe3;
}
QLabel#loadingTitle {
    color: #20262f;
    font-size: 17px;
    font-weight: 700;
}
QLabel#loadingMessage {
    color: #667080;
    font-size: 12px;
}
QProgressBar#loadingProgress {
    background: #e4e8ed;
    border: 1px solid #c2c9d2;
    height: 8px;
}
QProgressBar#loadingProgress::chunk {
    background: #3a9a93;
}
QScrollArea#dashboardScrollArea {
    background: #f3f5f7;
    border: none;
}
QScrollArea#dashboardScrollArea QScrollBar:vertical {
    background: #e4e8ed;
    border: 1px solid #c2c9d2;
    width: 10px;
}
QScrollArea#dashboardScrollArea QScrollBar::handle:vertical {
    background: #3a9a93;
    min-height: 42px;
    margin: 1px;
}
QScrollArea#dashboardScrollArea QScrollBar::add-line:vertical,
QScrollArea#dashboardScrollArea QScrollBar::sub-line:vertical {
    height: 0;
}
QFrame#mapCard {
    background: #ffffff;
    border: 1px solid #d5dbe3;
}
QFrame#detailGroup {
    background: #ffffff;
    border: 1px solid #aebdce;
}
QFrame#detailSubsection {
    background: #ffffff;
    border: none;
}
QFrame#detailDivider {
    background: #c8d2de;
    border: none;
}
QFrame#frameDetailDivider {
    background: #aebdce;
    border: none;
}
QWidget#frameDetailPage,
QWidget#processingPathWidget,
QWidget#frameDetailPanel {
    background: #ffffff;
}
QWidget#frameDetailToolbar,
QWidget#processingPathHeader,
QWidget#singleFrameKdeHeader {
    background: #f2f6fb;
    border-bottom: 1px solid #d3deeb;
}
QFrame#fqMapSeparator {
    background: #eef1f4;
    border: none;
}
QWidget#toolbarCard {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #d5dbe3;
}
QFrame#chartCard {
    background: #ffffff;
    border: none;
}
QLabel#mapSectionTitle {
    color: #20262f;
    font-size: 16px;
    font-weight: 700;
}
QLabel#mapSectionTitle[sectionRole="detail"] {
    color: #294d91;
}
QLabel#overviewBadge {
    color: #596579;
    background: #eef1f4;
    border: 1px solid #d5dbe3;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 600;
}
QWidget#frameMapLabelPanel,
QWidget#kdeLabelPanel,
QWidget#qualityTrendLabelPanel {
    background: #f2f6fb;
    border-left: 4px solid #3157a4;
    border-right: 1px solid #d3deeb;
}
QLabel#detailBadge {
    color: #294d91;
    background: #eaf0f8;
    border: 1px solid #b9c7d9;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 700;
}
QLabel#frameModeBadge {
    color: #08766b;
    background: #e8f5f2;
    border: 1px solid #8fc9c1;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 700;
}
QLabel#frameMapSelectionLabel,
QLabel#singleFrameSelectionLabel {
    color: #294d91;
    font-size: 11px;
    font-weight: 700;
}
QLabel#frameMapFrameLabel {
    color: #303846;
    font-size: 11px;
    font-weight: 700;
}
QLabel#frameMapOrientationLabel {
    color: #667080;
    font-size: 10px;
}
QLabel#lowerNgKey {
    color: #173f8a;
    font-size: 10px;
    font-weight: 700;
}
QLabel#upperNgKey {
    color: #8f1d1d;
    font-size: 10px;
    font-weight: 700;
}
QLabel#processingPathSelectionLabel,
QLabel#singleFrameKdeSelectionLabel {
    color: #294d91;
    font-size: 11px;
    font-weight: 700;
}
QLabel#processingPathFrameLabel {
    color: #303846;
    font-size: 11px;
    font-weight: 700;
}
QLabel#frameMapMetricLabel {
    color: #667080;
    font-size: 11px;
}
QToolButton#frameMapModeButton {
    color: #294d91;
    background: #ffffff;
    border: 1px solid #9eb0c8;
    min-height: 22px;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 600;
}
QToolButton#frameMapModeButton:hover {
    background: #eaf0f8;
    border-color: #607da5;
}
QToolButton#frameMapModeButton:disabled {
    color: #a6adb7;
    background: #f3f5f7;
    border-color: #d7dce2;
}
QLabel#kdeContextCaption {
    color: #667080;
    font-size: 10px;
}
QLabel#kdeSummaryLabel {
    color: #667080;
    font-size: 11px;
}
QLabel#qualityTrendCaption {
    color: #667080;
    font-size: 10px;
}
QLabel#qualityTrendSelectionLabel {
    color: #294d91;
    font-size: 10px;
    font-weight: 700;
}
QLabel#qualityTrendSummaryLabel {
    color: #667080;
    font-size: 10px;
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
QComboBox#categoryCombo,
QComboBox#visionCombo,
QComboBox#equipmentCombo {
    color: #20262f;
    background: #ffffff;
    border: 1px solid #aeb7c4;
    padding: 7px 12px;
    min-height: 22px;
    font-size: 12px;
}
QComboBox#categoryCombo:hover,
QComboBox#visionCombo:hover,
QComboBox#equipmentCombo:hover {
    border-color: #607086;
}
QComboBox#categoryCombo::drop-down,
QComboBox#visionCombo::drop-down,
QComboBox#equipmentCombo::drop-down {
    border: none;
    width: 26px;
}
QComboBox#categoryCombo QAbstractItemView,
QComboBox#visionCombo QAbstractItemView,
QComboBox#equipmentCombo QAbstractItemView {
    color: #20262f;
    background: #ffffff;
    border: 1px solid #aeb7c4;
    selection-color: #ffffff;
    selection-background-color: #287d78;
}
QLabel#pieceYieldLabel {
    color: #08766b;
    background: #e8f5f2;
    border: 1px solid #8fc9c1;
    padding: 8px 9px;
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
QScrollBar#fqMapVerticalScrollBar:vertical {
    background: #edf1f5;
    border: 1px solid #c2c9d2;
    width: 10px;
}
QScrollBar#fqMapVerticalScrollBar::handle:vertical {
    background: #3a9a93;
    min-height: 28px;
    margin: 1px;
}
QScrollBar#fqMapVerticalScrollBar::handle:vertical:hover {
    background: #267f79;
}
QScrollBar#fqMapVerticalScrollBar::add-line:vertical,
QScrollBar#fqMapVerticalScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar#fqMapVerticalScrollBar::add-page:vertical,
QScrollBar#fqMapVerticalScrollBar::sub-page:vertical {
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
