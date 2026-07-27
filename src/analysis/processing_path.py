from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

EQUIPMENT_PATHS_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "equipment_paths.toml"
)
FRAME_SHAPE = (12, 24)

Position = tuple[int, int]


@dataclass(frozen=True)
class ProcessingPath:
    """1本の加工パス定義。"""

    id: str
    label: str
    positions: tuple[Position, ...]


@dataclass(frozen=True)
class EquipmentPaths:
    """設備別の加工パス定義。"""

    id: str
    label: str
    paths: tuple[ProcessingPath, ...]


@dataclass(frozen=True)
class EquipmentPathCatalog:
    """設備別加工パスの一覧。"""

    default_equipment_id: str
    equipments: tuple[EquipmentPaths, ...]

    def equipment(self, equipment_id: str) -> EquipmentPaths:
        """設備IDに対応する加工パス定義。"""
        for equipment in self.equipments:
            if equipment.id == equipment_id:
                return equipment
        raise ValueError(f"未定義の設備IDです: {equipment_id}")

    @property
    def default_equipment(self) -> EquipmentPaths:
        """既定設備の加工パス定義。"""
        return self.equipment(self.default_equipment_id)


@dataclass(frozen=True)
class ProcessingPathSeries:
    """単一Frameの加工パス別測定値系列。"""

    path_id: str
    label: str
    steps: np.ndarray
    position_x: np.ndarray
    position_y: np.ndarray
    values: np.ndarray
    ng_directions: np.ndarray


def load_equipment_path_catalog(
    path: Path = EQUIPMENT_PATHS_PATH,
) -> EquipmentPathCatalog:
    """TOML形式の設備別加工パス定義読込。"""
    with path.open("rb") as file:
        source = tomllib.load(file)

    if source["schema_version"] != 1:
        raise ValueError("未対応の加工パス定義バージョンです。")

    equipments = tuple(
        _parse_equipment(equipment)
        for equipment in source["equipments"]
    )
    catalog = EquipmentPathCatalog(
        default_equipment_id=source["default_equipment_id"],
        equipments=equipments,
    )
    _validate_catalog(catalog)
    return catalog


def build_processing_path_series(
    raw_values: np.ndarray,
    ng_directions: np.ndarray,
    equipment: EquipmentPaths,
) -> tuple[ProcessingPathSeries, ...]:
    """単一Frame値の設備別加工順への展開。"""
    raw = np.asarray(raw_values, dtype=float)
    directions = np.asarray(ng_directions, dtype=np.int8)
    if raw.shape != FRAME_SHAPE or directions.shape != FRAME_SHAPE:
        raise ValueError("単一Frame値は12行×24列で指定してください。")

    series = []
    for path in equipment.paths:
        position_x = np.fromiter(
            (position[0] for position in path.positions),
            dtype=np.int16,
        )
        position_y = np.fromiter(
            (position[1] for position in path.positions),
            dtype=np.int16,
        )
        series.append(
            ProcessingPathSeries(
                path_id=path.id,
                label=path.label,
                steps=np.arange(1, len(path.positions) + 1),
                position_x=position_x,
                position_y=position_y,
                values=raw[position_y - 1, position_x - 1].copy(),
                ng_directions=directions[
                    position_y - 1,
                    position_x - 1,
                ].copy(),
            )
        )
    return tuple(series)


def _parse_equipment(source: dict[str, Any]) -> EquipmentPaths:
    """設備1台分の加工パス定義変換。"""
    paths = tuple(
        ProcessingPath(
            id=path["id"],
            label=path["label"],
            positions=tuple(
                (position[0], position[1])
                for position in path["positions"]
            ),
        )
        for path in source["paths"]
    )
    return EquipmentPaths(
        id=source["id"],
        label=source["label"],
        paths=paths,
    )


def _validate_catalog(catalog: EquipmentPathCatalog) -> None:
    """設備別加工パス定義の整合性検証。"""
    equipment_ids = [equipment.id for equipment in catalog.equipments]
    if not equipment_ids or len(equipment_ids) != len(set(equipment_ids)):
        raise ValueError("設備IDは一意に1件以上定義してください。")
    if catalog.default_equipment_id not in equipment_ids:
        raise ValueError("既定設備IDが設備定義にありません。")

    for equipment in catalog.equipments:
        path_ids = [path.id for path in equipment.paths]
        if not path_ids or len(path_ids) != len(set(path_ids)):
            raise ValueError(
                f"加工パスIDは設備内で一意に定義してください: {equipment.id}"
            )
        for path in equipment.paths:
            if not path.positions:
                raise ValueError(
                    f"加工パスにPositionがありません: {equipment.id}/{path.id}"
                )
            if len(path.positions) != len(set(path.positions)):
                raise ValueError(
                    "加工パス内に重複したPositionがあります: "
                    f"{equipment.id}/{path.id}"
                )
            for step, (position_x, position_y) in enumerate(
                path.positions,
                start=1,
            ):
                if not 1 <= position_x <= FRAME_SHAPE[1] or not (
                    1 <= position_y <= FRAME_SHAPE[0]
                ):
                    raise ValueError(
                        "Positionが範囲外です: "
                        f"{equipment.id}/{path.id}/step {step}"
                    )
