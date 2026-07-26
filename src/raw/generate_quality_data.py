#!/usr/bin/env python3
"""Perlin変動を基調とした再現可能な最終検査rawデータ生成。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SEED = 20260724
DEFAULT_LOTS = 100
PERLIN_PERIOD = 4096
PERLIN_MASK = PERLIN_PERIOD - 1


class Measurement(TypedDict):
    """検査項目の生成条件。"""

    colname: str
    limmin: float | None
    limmax: float | None
    meta_type: str
    meta_ignore: bool
    meta_best: float | None
    meta_category: str
    meta_unit: str


MEASUREMENTS: tuple[Measurement, ...] = (
    {
        "colname": "Foreign_Length_Long",
        "limmin": None,
        "limmax": 0.300,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "異物",
        "meta_unit": "mm",
    },
    {
        "colname": "Foreign_Length_Short",
        "limmin": None,
        "limmax": 0.150,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "異物",
        "meta_unit": "mm",
    },
    {
        "colname": "Foreign_Size",
        "limmin": None,
        "limmax": 0.030,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "異物",
        "meta_unit": "mm²",
    },
    {
        "colname": "Lead_Length_L",
        "limmin": 1.120,
        "limmax": 1.280,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "リード",
        "meta_unit": "mm",
    },
    {
        "colname": "Lead_Length_R",
        "limmin": 1.120,
        "limmax": 1.280,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "リード",
        "meta_unit": "mm",
    },
    {
        "colname": "Lead_Pitch",
        "limmin": 2.490,
        "limmax": 2.590,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "リード",
        "meta_unit": "mm",
    },
    {
        "colname": "Work_Xw",
        "limmin": 3.900,
        "limmax": 4.100,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "PKGサイズ",
        "meta_unit": "mm",
    },
    {
        "colname": "Work_Yw",
        "limmin": 1.950,
        "limmax": 2.050,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "PKGサイズ",
        "meta_unit": "mm",
    },
    {
        "colname": "Work_Center_X",
        "limmin": -0.060,
        "limmax": 0.060,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "PKGサイズ",
        "meta_unit": "mm",
    },
    {
        "colname": "Work_Center_Y",
        "limmin": -0.050,
        "limmax": 0.050,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "PKGサイズ",
        "meta_unit": "mm",
    },
    {
        "colname": "Mark_Center_X",
        "limmin": -0.100,
        "limmax": 0.100,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "標印",
        "meta_unit": "mm",
    },
    {
        "colname": "Mark_Center_Y",
        "limmin": -0.100,
        "limmax": 0.100,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": None,
        "meta_category": "標印",
        "meta_unit": "mm",
    },
    {
        "colname": "Defect_Length_Long",
        "limmin": None,
        "limmax": 0.250,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "欠陥",
        "meta_unit": "mm",
    },
    {
        "colname": "Defect_Length_Short",
        "limmin": None,
        "limmax": 0.120,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "欠陥",
        "meta_unit": "mm",
    },
    {
        "colname": "Defect_Size",
        "limmin": None,
        "limmax": 0.025,
        "meta_type": "spec",
        "meta_ignore": False,
        "meta_best": 0.000,
        "meta_category": "欠陥",
        "meta_unit": "mm²",
    },
)

VISION_COUNT = 3
ROWS_PER_LOT = 24 * 24 * 12 * VISION_COUNT * len(MEASUREMENTS)

ANOMALY_DEFINITIONS = {
    24: {
        "scenario": "vision_2_work_xw_shift",
        "description": "vision_2のWork_Xwが高い",
        "colnames": ["Work_Xw_v2"],
    },
    41: {
        "scenario": "right_edge_mark_shift",
        "description": "右側ほどMark_Center_Xが正方向へずれる",
        "colnames": [
            "Mark_Center_X_v1",
            "Mark_Center_X_v2",
            "Mark_Center_X_v3",
        ],
    },
    58: {
        "scenario": "localized_foreign_cluster",
        "description": "vision_3の中央付近で異物寸法が滑らかに増える",
        "colnames": [
            "Foreign_Length_Long_v3",
            "Foreign_Length_Short_v3",
            "Foreign_Size_v3",
        ],
    },
    72: {
        "scenario": "lead_pitch_x_tilt",
        "description": "Lead_PitchにPositionX方向の傾きがある",
        "colnames": [
            "Lead_Pitch_v1",
            "Lead_Pitch_v2",
            "Lead_Pitch_v3",
        ],
    },
    86: {
        "scenario": "vision_3_work_center_y_shift",
        "description": "vision_3のWork_Center_Yが正方向へずれる",
        "colnames": ["Work_Center_Y_v3"],
    },
    93: {
        "scenario": "lead_length_imbalance",
        "description": "Lead_Length_Lが長く、Lead_Length_Rが短い",
        "colnames": [
            "Lead_Length_L_v1",
            "Lead_Length_L_v2",
            "Lead_Length_L_v3",
            "Lead_Length_R_v1",
            "Lead_Length_R_v2",
            "Lead_Length_R_v3",
        ],
    },
    99: {
        "scenario": "upper_right_corner_damage",
        "description": "右上へ向かって強まる寸法・標印・欠陥の複合異常",
        "colnames": [
            "Work_Xw_v1",
            "Work_Xw_v2",
            "Work_Xw_v3",
            "Mark_Center_X_v1",
            "Mark_Center_X_v2",
            "Mark_Center_X_v3",
            "Defect_Length_Long_v1",
            "Defect_Length_Long_v2",
            "Defect_Length_Long_v3",
            "Defect_Length_Short_v1",
            "Defect_Length_Short_v2",
            "Defect_Length_Short_v3",
            "Defect_Size_v1",
            "Defect_Size_v2",
            "Defect_Size_v3",
        ],
    },
}


def _lot_identity(index: int) -> tuple[str, datetime]:
    """ロット番号と開始時刻の生成。"""
    day = datetime(2026, 6, 1) + timedelta(days=index // 2)
    shift = "A" if index % 2 == 0 else "B"
    hour = 8 if shift == "A" else 20
    minute_jitter = ((index * 17 + 11) % 31) - 15
    start = day.replace(hour=hour, minute=15, second=0) + timedelta(
        minutes=minute_jitter,
        seconds=(index * 37) % 60,
    )
    return f"LOT_{day:%Y%m%d}_{shift}", start


def _fade(value: np.ndarray) -> np.ndarray:
    """Perlin補間曲線。"""
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _lerp(
    weight: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """線形補間。"""
    return start + weight * (end - start)


@lru_cache(maxsize=None)
def _permutation(seed: int, channel: int) -> np.ndarray:
    """Perlin勾配の順列表。"""
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, channel, 7331])
    )
    values = np.arange(PERLIN_PERIOD, dtype=np.int32)
    rng.shuffle(values)
    return np.concatenate([values, values])


def _gradient(
    hash_values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Perlin格子勾配との内積。"""
    gradient_index = hash_values & 15
    first = np.where(gradient_index < 8, x, y)
    second = np.where(
        gradient_index < 4,
        y,
        np.where(
            (gradient_index == 12) | (gradient_index == 14),
            x,
            z,
        ),
    )
    first = np.where((gradient_index & 1) == 0, first, -first)
    second = np.where((gradient_index & 2) == 0, second, -second)
    return first + second


def _perlin_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    channel: int,
) -> np.ndarray:
    """3次元Perlin noise。"""
    permutation = _permutation(seed, channel)
    x_floor = np.floor(x)
    y_floor = np.floor(y)
    z_floor = np.floor(z)
    x_index = x_floor.astype(np.int64) & PERLIN_MASK
    y_index = y_floor.astype(np.int64) & PERLIN_MASK
    z_index = z_floor.astype(np.int64) & PERLIN_MASK

    x_fraction = x - x_floor
    y_fraction = y - y_floor
    z_fraction = z - z_floor
    x_fade = _fade(x_fraction)
    y_fade = _fade(y_fraction)
    z_fade = _fade(z_fraction)

    a = permutation[x_index] + y_index
    aa = permutation[a] + z_index
    ab = permutation[a + 1] + z_index
    b = permutation[x_index + 1] + y_index
    ba = permutation[b] + z_index
    bb = permutation[b + 1] + z_index

    lower = _lerp(
        y_fade,
        _lerp(
            x_fade,
            _gradient(
                permutation[aa],
                x_fraction,
                y_fraction,
                z_fraction,
            ),
            _gradient(
                permutation[ba],
                x_fraction - 1.0,
                y_fraction,
                z_fraction,
            ),
        ),
        _lerp(
            x_fade,
            _gradient(
                permutation[ab],
                x_fraction,
                y_fraction - 1.0,
                z_fraction,
            ),
            _gradient(
                permutation[bb],
                x_fraction - 1.0,
                y_fraction - 1.0,
                z_fraction,
            ),
        ),
    )
    upper = _lerp(
        y_fade,
        _lerp(
            x_fade,
            _gradient(
                permutation[aa + 1],
                x_fraction,
                y_fraction,
                z_fraction - 1.0,
            ),
            _gradient(
                permutation[ba + 1],
                x_fraction - 1.0,
                y_fraction,
                z_fraction - 1.0,
            ),
        ),
        _lerp(
            x_fade,
            _gradient(
                permutation[ab + 1],
                x_fraction,
                y_fraction - 1.0,
                z_fraction - 1.0,
            ),
            _gradient(
                permutation[bb + 1],
                x_fraction - 1.0,
                y_fraction - 1.0,
                z_fraction - 1.0,
            ),
        ),
    )
    return _lerp(z_fade, lower, upper)


def _fractal_perlin_3d(
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    channel: int,
    temporal_scale: float,
    x_scale: float,
    y_scale: float,
    octaves: int = 4,
) -> np.ndarray:
    """複数octaveを合成したPerlin noise。"""
    total = np.zeros(np.broadcast_shapes(time.shape, x.shape, y.shape))
    time_coordinate = time * temporal_scale
    x_coordinate = x * x_scale
    y_coordinate = y * y_scale
    rotated_x = (
        0.812 * time_coordinate
        + 0.451 * x_coordinate
        - 0.371 * y_coordinate
    )
    rotated_y = (
        -0.326 * time_coordinate
        + 0.889 * x_coordinate
        + 0.321 * y_coordinate
    )
    rotated_z = (
        0.484 * time_coordinate
        - 0.087 * x_coordinate
        + 0.871 * y_coordinate
    )
    frequency = 1.0
    amplitude = 1.0
    amplitude_sum = 0.0
    for _ in range(octaves):
        total += amplitude * _perlin_3d(
            rotated_x * frequency + 0.173,
            rotated_y * frequency + 0.417,
            rotated_z * frequency + 0.731,
            seed,
            channel,
        )
        amplitude_sum += amplitude
        frequency *= 2.071
        amplitude *= 0.52
    return 3.2 * total / amplitude_sum


def _perlin_lot_factors(n_lots: int, seed: int) -> np.ndarray:
    """検査項目別のPerlinロット変動。"""
    lot_axis = np.arange(n_lots, dtype=np.float64)
    factors = np.empty((n_lots, len(MEASUREMENTS)))
    for channel in range(len(MEASUREMENTS)):
        factors[:, channel] = _fractal_perlin_3d(
            lot_axis,
            np.full(n_lots, channel + 0.25),
            np.full(n_lots, channel * 0.37 + 0.5),
            seed,
            100 + channel,
            temporal_scale=0.085,
            x_scale=0.11,
            y_scale=0.09,
        )
        if n_lots > 1:
            factors[:, channel] -= factors[:, channel].mean()
            factors[:, channel] /= factors[:, channel].std()
    return factors


def _base_grid() -> dict[str, np.ndarray]:
    """1ロット内の製品座標。"""
    physical_frame = np.repeat(
        np.arange(1, 25, dtype=np.int16),
        24 * 12,
    )
    physical_position_x = np.tile(
        np.repeat(np.arange(1, 25, dtype=np.int8), 12),
        24,
    )
    physical_position_y = np.tile(
        np.arange(1, 13, dtype=np.int8),
        24 * 24,
    )

    frame = np.repeat(physical_frame, VISION_COUNT)
    position_x = np.repeat(physical_position_x, VISION_COUNT)
    position_y = np.repeat(physical_position_y, VISION_COUNT)
    vision_index = np.tile(
        np.arange(VISION_COUNT, dtype=np.int8),
        physical_frame.size,
    )
    vision = np.asarray(
        ["vision_1", "vision_2", "vision_3"],
        dtype=object,
    )[vision_index]

    x_scaled = (position_x.astype(np.float64) - 12.5) / 11.5
    y_scaled = (position_y.astype(np.float64) - 6.5) / 5.5
    edge = np.maximum(np.abs(x_scaled), np.abs(y_scaled))

    return {
        "frame": frame,
        "position_x": position_x,
        "position_y": position_y,
        "vision_index": vision_index,
        "vision": vision,
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "edge": edge,
    }


def _perlin_product_field(
    lot_index: int,
    grid: dict[str, np.ndarray],
    seed: int,
    channel: int,
    temporal_scale: float,
    x_scale: float,
    y_scale: float,
    octaves: int = 4,
) -> np.ndarray:
    """ロット・フレーム・製品位置に連続するPerlin場。"""
    global_frame = (
        lot_index * 24 + grid["frame"].astype(np.float64) - 1.0
    )
    return _fractal_perlin_3d(
        global_frame,
        grid["position_x"].astype(np.float64),
        grid["position_y"].astype(np.float64),
        seed,
        channel,
        temporal_scale,
        x_scale,
        y_scale,
        octaves,
    )


def _generate_values(
    lot_index: int,
    grid: dict[str, np.ndarray],
    lot_factor: np.ndarray,
    seed: int,
) -> np.ndarray:
    """1ロット分の15検査項目の測定値。"""
    rng = np.random.default_rng(np.random.SeedSequence([seed, lot_index, 17]))
    n_products = grid["frame"].size
    vision_index = grid["vision_index"]
    x_scaled = grid["x_scaled"]
    y_scaled = grid["y_scaled"]
    edge = grid["edge"]
    process_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=1,
        temporal_scale=0.035,
        x_scale=0.085,
        y_scale=0.13,
    )
    foreign_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=2,
        temporal_scale=0.060,
        x_scale=0.15,
        y_scale=0.24,
    )
    lead_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=3,
        temporal_scale=0.050,
        x_scale=0.12,
        y_scale=0.20,
    )
    pitch_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=4,
        temporal_scale=0.045,
        x_scale=0.11,
        y_scale=0.19,
    )
    work_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=5,
        temporal_scale=0.042,
        x_scale=0.10,
        y_scale=0.17,
    )
    center_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=6,
        temporal_scale=0.055,
        x_scale=0.14,
        y_scale=0.22,
    )
    mark_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=7,
        temporal_scale=0.060,
        x_scale=0.16,
        y_scale=0.25,
    )
    defect_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=8,
        temporal_scale=0.065,
        x_scale=0.18,
        y_scale=0.27,
    )
    micro_field = _perlin_product_field(
        lot_index,
        grid,
        seed,
        channel=9,
        temporal_scale=0.11,
        x_scale=0.38,
        y_scale=0.55,
        octaves=3,
    )
    unit_latent = (
        0.65 * micro_field
        + 0.35 * rng.normal(size=n_products)
    )

    foreign_machine = np.asarray([0.94, 1.00, 1.08])[vision_index]
    foreign_probability = np.clip(
        0.085
        * np.exp(0.20 * lot_factor[0])
        * foreign_machine
        * np.exp(
            0.32 * foreign_field
            + 0.12 * process_field
            + 0.16 * edge
        ),
        0.015,
        0.32,
    )
    foreign_exists = rng.random(n_products) < foreign_probability
    foreign_long = np.where(
        foreign_exists,
        rng.gamma(2.0, 0.058, n_products)
        * np.exp(
            0.12 * lot_factor[0]
            + 0.24 * foreign_field
            + 0.08 * process_field
        ),
        0.0,
    )
    foreign_short = np.where(
        foreign_exists,
        foreign_long * rng.uniform(0.35, 0.70, n_products)
        + np.abs(rng.normal(scale=0.0035, size=n_products)),
        0.0,
    )
    foreign_size = np.where(
        foreign_exists,
        0.65 * foreign_long * foreign_short
        + np.abs(rng.normal(scale=0.0010, size=n_products)),
        0.0,
    )

    lead_machine = np.asarray([0.000, 0.004, -0.003])[vision_index]
    lead_center = (
        1.200
        + 0.0055 * lot_factor[3]
        + lead_machine
        + 0.007 * lead_field
        + 0.004 * process_field
    )
    lead_length_l = (
        lead_center
        + 0.005 * x_scaled
        - 0.003 * y_scaled
        + 0.004 * unit_latent
        + rng.normal(scale=0.0105, size=n_products)
    )
    lead_length_r = (
        lead_center
        - 0.005 * x_scaled
        + 0.003 * y_scaled
        + 0.004 * unit_latent
        + rng.normal(scale=0.0105, size=n_products)
    )
    lead_pitch = (
        2.540
        + 0.0045 * lot_factor[5]
        + np.asarray([0.000, 0.003, -0.0025])[vision_index]
        + 0.008 * pitch_field
        + 0.003 * process_field
        + 0.0035 * y_scaled
        + 0.003 * unit_latent
        + rng.normal(scale=0.0075, size=n_products)
    )

    work_x_center = (
        4.000
        + 0.006 * lot_factor[6]
        + 0.006 * process_field
        + 0.010 * work_field
    )
    work_y_center = (
        2.000
        + 0.003 * lot_factor[7]
        - 0.003 * process_field
        + 0.006 * work_field
    )

    work_xw = (
        work_x_center
        + np.asarray([0.000, 0.006, -0.004])[vision_index]
        + 0.005 * x_scaled
        + 0.003 * y_scaled
        + 0.004 * unit_latent
        + rng.normal(scale=0.011, size=n_products)
    )
    work_yw = (
        work_y_center
        + np.asarray([0.002, -0.003, 0.003])[vision_index]
        - 0.002 * x_scaled
        + 0.0035 * y_scaled
        + 0.0025 * unit_latent
        + rng.normal(scale=0.0065, size=n_products)
    )
    work_center_x = (
        0.004 * lot_factor[8]
        + np.asarray([0.000, 0.004, -0.003])[vision_index]
        + 0.011 * center_field
        + 0.005 * process_field
        + 0.008 * x_scaled
        + 0.003 * unit_latent
        + rng.normal(scale=0.0085, size=n_products)
    )
    work_center_y = (
        0.004 * lot_factor[9]
        + np.asarray([0.002, -0.003, 0.003])[vision_index]
        - 0.010 * center_field
        + 0.004 * process_field
        + 0.008 * y_scaled
        + 0.003 * unit_latent
        + rng.normal(scale=0.0085, size=n_products)
    )

    mark_center_x = (
        0.0065 * lot_factor[10]
        + np.asarray([0.005, -0.003, -0.006])[vision_index]
        + 0.020 * mark_field
        + 0.006 * process_field
        + 0.012 * x_scaled
        + 0.004 * unit_latent
        + rng.normal(scale=0.013, size=n_products)
    )
    mark_center_y = (
        0.0065 * lot_factor[11]
        + np.asarray([-0.004, 0.003, 0.006])[vision_index]
        - 0.018 * mark_field
        + 0.005 * process_field
        + 0.012 * y_scaled
        + 0.004 * unit_latent
        + rng.normal(scale=0.013, size=n_products)
    )

    defect_machine = np.asarray([0.94, 1.02, 1.08])[vision_index]
    defect_probability = np.clip(
        0.065
        * np.exp(0.20 * lot_factor[12])
        * defect_machine
        * np.exp(
            0.34 * defect_field
            + 0.10 * process_field
            + 0.18 * edge
        ),
        0.015,
        0.28,
    )
    defect_exists = rng.random(n_products) < defect_probability
    defect_long = np.where(
        defect_exists,
        rng.gamma(2.0, 0.048, n_products)
        * np.exp(
            0.12 * lot_factor[12]
            + 0.24 * defect_field
            + 0.08 * process_field
        ),
        0.0,
    )
    defect_short = np.where(
        defect_exists,
        defect_long * rng.uniform(0.35, 0.68, n_products)
        + np.abs(rng.normal(scale=0.0035, size=n_products)),
        0.0,
    )
    defect_size = np.where(
        defect_exists,
        0.65 * defect_long * defect_short
        + np.abs(rng.normal(scale=0.0009, size=n_products)),
        0.0,
    )

    anomaly_texture = np.clip(
        1.0 + 0.22 * process_field + 0.12 * micro_field,
        0.65,
        1.35,
    )
    if lot_index == 24:
        target = vision_index == 1
        work_xw[target] += 0.065 * anomaly_texture[target]
    elif lot_index == 41:
        right_edge = np.clip(
            (grid["position_x"].astype(float) - 14.0) / 10.0,
            0.0,
            1.0,
        )
        mark_center_x += 0.14 * right_edge * anomaly_texture
    elif lot_index == 58:
        cluster_distance = (
            ((grid["frame"].astype(float) - 18.5) / 2.3) ** 2
            + (
                (grid["position_x"].astype(float) - 12.5)
                / 5.5
            )
            ** 2
            + (
                (grid["position_y"].astype(float) - 6.5)
                / 3.2
            )
            ** 2
        )
        cluster = (
            np.exp(-0.5 * cluster_distance)
            * np.clip(1.0 + 0.25 * foreign_field, 0.6, 1.4)
            * (vision_index == 2)
        )
        foreign_long += 0.36 * cluster
        foreign_short += 0.18 * cluster
        foreign_size += 0.042 * cluster
    elif lot_index == 72:
        lead_pitch += (
            0.043
            * x_scaled
            * np.clip(1.0 + 0.12 * lead_field, 0.8, 1.2)
        )
    elif lot_index == 86:
        target = vision_index == 2
        work_center_y[target] += (
            0.065 * anomaly_texture[target]
        )
    elif lot_index == 93:
        imbalance = 0.058 * np.clip(
            1.0 + 0.10 * process_field,
            0.85,
            1.15,
        )
        lead_length_l += imbalance
        lead_length_r -= imbalance
    elif lot_index == 99:
        corner = (
            np.clip(
                (grid["position_x"].astype(float) - 15.0) / 9.0,
                0.0,
                1.0,
            )
            * np.clip(
                (grid["position_y"].astype(float) - 6.0) / 6.0,
                0.0,
                1.0,
            )
            * anomaly_texture
        )
        work_xw += 0.12 * corner
        mark_center_x += 0.16 * corner
        defect_long += 0.30 * corner
        defect_short += 0.15 * corner
        defect_size += 0.038 * corner

    return np.column_stack(
        [
            np.round(np.clip(foreign_long, 0.0, None), 4),
            np.round(np.clip(foreign_short, 0.0, None), 4),
            np.round(np.clip(foreign_size, 0.0, None), 4),
            np.round(lead_length_l, 4),
            np.round(lead_length_r, 4),
            np.round(lead_pitch, 4),
            np.round(work_xw, 4),
            np.round(work_yw, 4),
            np.round(work_center_x, 4),
            np.round(work_center_y, 4),
            np.round(mark_center_x, 4),
            np.round(mark_center_y, 4),
            np.round(np.clip(defect_long, 0.0, None), 4),
            np.round(np.clip(defect_short, 0.0, None), 4),
            np.round(np.clip(defect_size, 0.0, None), 4),
        ]
    )


def _schema(seed: int, n_lots: int) -> pa.Schema:
    """Parquetスキーマ。"""
    metadata = {
        b"dataset": b"discrete_semiconductor_final_inspection",
        b"dataset_stage": b"raw",
        b"dataset_version": b"4.0",
        b"baseline_noise": b"fractal_perlin_3d",
        b"perlin_octaves": b"4",
        b"perlin_period": str(PERLIN_PERIOD).encode(),
        b"generator_seed": str(seed).encode(),
        b"lot_count": str(n_lots).encode(),
        b"rows_per_lot": str(ROWS_PER_LOT).encode(),
        b"visions_per_product": str(VISION_COUNT).encode(),
        b"measurement_base_count": str(len(MEASUREMENTS)).encode(),
        b"unique_colname_count": str(len(MEASUREMENTS) * 3).encode(),
        b"vision_coverage": b"vision_1,vision_2,vision_3:FrameNo 1-24",
    }
    return pa.schema(
        [
            pa.field("vision", pa.string(), nullable=False),
            pa.field("lot_number", pa.string(), nullable=False),
            pa.field("lot_start_time", pa.timestamp("ns"), nullable=False),
            pa.field("FrameNo", pa.int16(), nullable=False),
            pa.field("PositionX", pa.int8(), nullable=False),
            pa.field("PositionY", pa.int8(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("colname", pa.string(), nullable=False),
            pa.field("limmin", pa.float64(), nullable=True),
            pa.field("limmax", pa.float64(), nullable=True),
            pa.field("meta_type", pa.string(), nullable=False),
            pa.field("meta_ignore", pa.bool_(), nullable=False),
            pa.field("meta_best", pa.float64(), nullable=True),
            pa.field("meta_category", pa.string(), nullable=False),
            pa.field("meta_unit", pa.string(), nullable=False),
        ],
        metadata=metadata,
    )


def _long_metadata(
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """製品順に展開した検査項目メタデータ。"""
    n_products = grid["frame"].size
    base_colnames = np.tile(
        np.asarray(
            [item["colname"] for item in MEASUREMENTS],
            dtype=str,
        ),
        n_products,
    )
    vision_suffix = np.asarray(["_v1", "_v2", "_v3"], dtype=str)[
        grid["vision_index"]
    ]

    def nullable_float(
        key: Literal["limmin", "limmax", "meta_best"],
    ) -> np.ndarray:
        return np.tile(
            np.asarray(
                [
                    np.nan if item[key] is None else item[key]
                    for item in MEASUREMENTS
                ],
                dtype=np.float64,
            ),
            n_products,
        )

    return {
        "colname": np.char.add(
            base_colnames,
            np.repeat(vision_suffix, len(MEASUREMENTS)),
        ),
        "limmin": nullable_float("limmin"),
        "limmax": nullable_float("limmax"),
        "meta_type": np.tile(
            np.asarray(
                [item["meta_type"] for item in MEASUREMENTS],
                dtype=object,
            ),
            n_products,
        ),
        "meta_ignore": np.tile(
            np.asarray(
                [item["meta_ignore"] for item in MEASUREMENTS],
                dtype=bool,
            ),
            n_products,
        ),
        "meta_best": nullable_float("meta_best"),
        "meta_category": np.tile(
            np.asarray(
                [item["meta_category"] for item in MEASUREMENTS],
                dtype=object,
            ),
            n_products,
        ),
        "meta_unit": np.tile(
            np.asarray(
                [item["meta_unit"] for item in MEASUREMENTS],
                dtype=object,
            ),
            n_products,
        ),
    }


def _to_table(
    lot_number: str,
    lot_start_time: datetime,
    grid: dict[str, np.ndarray],
    values: np.ndarray,
    long_metadata: dict[str, np.ndarray],
    schema: pa.Schema,
) -> pa.Table:
    """1ロット分のArrowテーブル。"""
    n_measurements = len(MEASUREMENTS)
    n_rows = values.size
    columns = {
        "vision": np.repeat(grid["vision"], n_measurements),
        "lot_number": np.full(n_rows, lot_number, dtype=object),
        "lot_start_time": np.full(
            n_rows,
            np.datetime64(lot_start_time, "ns"),
            dtype="datetime64[ns]",
        ),
        "FrameNo": np.repeat(grid["frame"], n_measurements),
        "PositionX": np.repeat(grid["position_x"], n_measurements),
        "PositionY": np.repeat(grid["position_y"], n_measurements),
        "value": values.reshape(-1),
        **long_metadata,
    }
    arrays = [
        pa.array(
            columns[field.name],
            type=field.type,
            from_pandas=True,
        )
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_manifest(
    output_path: Path,
    seed: int,
    n_lots: int,
    row_count: int,
) -> Path:
    """生成条件と既知異常のマニフェスト保存。"""
    anomalies = []
    for index, definition in ANOMALY_DEFINITIONS.items():
        if index >= n_lots:
            continue
        lot_number, lot_start_time = _lot_identity(index)
        anomalies.append(
            {
                "lot_index": index,
                "lot_number": lot_number,
                "lot_start_time": lot_start_time.isoformat(),
                **definition,
            }
        )

    manifest = {
        "file": output_path.name,
        "seed": seed,
        "lot_count": n_lots,
        "row_count": row_count,
        "rows_per_lot": ROWS_PER_LOT,
        "frames_per_lot": 24,
        "products_per_frame": 24 * 12,
        "visions_per_product": VISION_COUNT,
        "measurements_per_vision": len(MEASUREMENTS),
        "measurements_per_product": len(MEASUREMENTS) * VISION_COUNT,
        "unique_colname_count": len(MEASUREMENTS) * 3,
        "meta_units": {
            item["colname"]: item["meta_unit"]
            for item in MEASUREMENTS
        },
        "baseline_noise": {
            "type": "fractal_perlin_3d",
            "dimensions": [
                "lot_index + FrameNo",
                "PositionX",
                "PositionY",
            ],
            "octaves": 4,
            "persistence": 0.52,
            "lacunarity": 2.071,
            "period": PERLIN_PERIOD,
            "coordinate_rotation": True,
        },
        "colname_suffix_by_vision": {
            "vision_1": "_v1",
            "vision_2": "_v2",
            "vision_3": "_v3",
        },
        "vision_frame_coverage": {
            "vision_1": [1, 24],
            "vision_2": [1, 24],
            "vision_3": [1, 24],
        },
        "known_synthetic_anomalies": anomalies,
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def generate(
    output_path: Path,
    n_lots: int,
    seed: int,
) -> tuple[int, Path]:
    """指定ロット数のParquet生成。"""
    if n_lots < 1:
        raise ValueError("--lots must be at least 1")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    schema = _schema(seed, n_lots)
    grid = _base_grid()
    long_metadata = _long_metadata(grid)
    lot_factors = _perlin_lot_factors(n_lots, seed)
    expected_rows = n_lots * ROWS_PER_LOT

    writer = pq.ParquetWriter(
        temporary_path,
        schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=[
            "vision",
            "lot_number",
            "colname",
            "meta_type",
            "meta_category",
            "meta_unit",
        ],
        write_statistics=True,
        version="2.6",
    )
    try:
        for lot_index in range(n_lots):
            lot_number, lot_start_time = _lot_identity(lot_index)
            values = _generate_values(
                lot_index,
                grid,
                lot_factors[lot_index],
                seed,
            )
            table = _to_table(
                lot_number,
                lot_start_time,
                grid,
                values,
                long_metadata,
                schema,
            )
            writer.write_table(table, row_group_size=ROWS_PER_LOT)
            if (lot_index + 1) % 10 == 0 or lot_index + 1 == n_lots:
                print(f"generated {lot_index + 1:>3}/{n_lots} lots")
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    parquet_file = pq.ParquetFile(temporary_path)
    if parquet_file.metadata.num_rows != expected_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "row count mismatch: "
            f"{parquet_file.metadata.num_rows:,} != {expected_rows:,}"
        )
    if parquet_file.metadata.num_row_groups != n_lots:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "row-group count mismatch: "
            f"{parquet_file.metadata.num_row_groups} != {n_lots}"
        )

    temporary_path.replace(output_path)
    manifest_path = _write_manifest(
        output_path,
        seed,
        n_lots,
        expected_rows,
    )
    return expected_rows, manifest_path


def parse_args() -> argparse.Namespace:
    """コマンドライン引数の定義。"""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "data"
            / "raw"
            / "quality_data_100lots.parquet"
        ),
        help="raw Parquetの出力パス",
    )
    parser.add_argument(
        "--lots",
        type=int,
        default=DEFAULT_LOTS,
        help=f"number of lots (default: {DEFAULT_LOTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    """データ生成の実行。"""
    args = parse_args()
    row_count, manifest_path = generate(args.output, args.lots, args.seed)
    output_path = args.output.resolve()
    print(f"wrote {row_count:,} rows to {output_path}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
