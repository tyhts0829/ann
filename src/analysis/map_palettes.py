"""品質マップの共通カラーパレット。"""

import numpy as np
import pyqtgraph as pg

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

BLUES = (
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
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

MAP_DEFINITIONS = (
    ("ng_rates", "NG率 (%)", REDS),
    ("normalized_mean", "規格逸脱度 平均", BLUES),
    ("normalized_std", "規格位置・使用率 std", PURPLES),
)


def make_color_map(colors: tuple[str, ...]) -> pg.ColorMap:
    """16進色列からの連続カラーマップ。"""
    rgb = np.asarray(
        [pg.mkColor(color).getRgb()[:3] for color in colors],
        dtype=np.ubyte,
    )
    return pg.ColorMap(
        np.linspace(0.0, 1.0, len(colors)),
        rgb,
    )
