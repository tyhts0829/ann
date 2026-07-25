"""品質ダッシュボードの外部設定。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


@dataclass(frozen=True)
class DashboardConfig:
    """ダッシュボード表示設定。"""

    visible_lots: int
    window_width: int
    window_height: int
    height_unit: int
    fqmap_height_ratio: float
    fmap_height_ratio: float
    histogram_height_ratio: float
    fqmap_plot_height: int
    left_label_width: int
    color_bar_width: int

    @property
    def fqmap_height(self) -> int:
        """FQmapの固定高。"""
        return round(self.height_unit * self.fqmap_height_ratio)

    @property
    def fmap_height(self) -> int:
        """Fmapの固定高。"""
        return round(self.height_unit * self.fmap_height_ratio)

    @property
    def histogram_height(self) -> int:
        """Histogram予約領域の固定高。"""
        return round(
            self.height_unit * self.histogram_height_ratio
        )


def load_dashboard_config(
    path: Path = CONFIG_PATH,
) -> DashboardConfig:
    """TOML形式の表示設定読込。"""
    with path.open("rb") as file:
        data = tomllib.load(file)

    dashboard = data["dashboard"]
    layout = data["layout"]
    return DashboardConfig(
        visible_lots=int(dashboard["visible_lots"]),
        window_width=int(dashboard["window_width"]),
        window_height=int(dashboard["window_height"]),
        height_unit=int(layout["height_unit"]),
        fqmap_height_ratio=float(layout["fqmap_height_ratio"]),
        fmap_height_ratio=float(layout["fmap_height_ratio"]),
        histogram_height_ratio=float(
            layout["histogram_height_ratio"]
        ),
        fqmap_plot_height=int(layout["fqmap_plot_height"]),
        left_label_width=int(layout["left_label_width"]),
        color_bar_width=int(layout["color_bar_width"]),
    )


DASHBOARD_CONFIG = load_dashboard_config()
