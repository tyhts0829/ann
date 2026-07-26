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

DENSITY_STOPS = (
    (0.00, "#ffffff"),
    (0.03, "#eaf3fa"),
    (0.12, "#b9dce9"),
    (0.30, "#48a9bb"),
    (0.52, "#62b88f"),
    (0.68, "#f0d95b"),
    (0.84, "#f18b3b"),
    (1.00, "#c9342f"),
)

MAP_DEFINITIONS = (
    ("ng_rates", "NG率 (%)", REDS),
    ("normalized_mean", "規格位置・使用率 平均", RDBU_R),
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


def make_density_color_map() -> pg.ColorMap:
    """密度表示用の連続カラーマップ。"""
    positions = np.asarray(
        [position for position, _ in DENSITY_STOPS]
    )
    rgb = np.asarray(
        [
            pg.mkColor(color).getRgb()[:3]
            for _, color in DENSITY_STOPS
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(positions, rgb)
