"""事前集計済み品質データの参照窓口。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


class QualityRepository:
    """分析データbundleの参照窓口。"""

    def __init__(self, analysis_dir: Path) -> None:
        self.analysis_dir = analysis_dir.resolve()
        manifest_path = self.analysis_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.detail_path = (
            self.analysis_dir / self.manifest["source_detail"]
        ).resolve()
        self.paths = {
            name: self.analysis_dir / file_info["file"]
            for name, file_info in self.manifest["files"].items()
        }
        self.kde_bins = int(self.manifest["kde_bins"])
        bit_mapping = self.manifest["ng_bit_mapping"]
        self.bit_colnames = tuple(item["colname"] for item in bit_mapping)
        self.bit_by_colname = {
            item["colname"]: np.uint64(item["bit_value"]) for item in bit_mapping
        }
        self.connection = duckdb.connect(database=":memory:")

    def lots(self) -> list[tuple[str, datetime]]:
        """開始時刻順のlot一覧。"""
        query = """
            SELECT lot_number, lot_start_time
            FROM read_parquet(?)
            ORDER BY lot_start_time
        """
        return self.connection.execute(
            query,
            [str(self.paths["lots"])],
        ).fetchall()

    def close(self) -> None:
        """データベース接続の終了。"""
        self.connection.close()

    def ng_rate_by_frame(
        self,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・Frame・検査項目別の事前集計取得。"""
        lot_filter, lot_parameters = self._lot_filter(lot_numbers)
        query = f"""
            SELECT
                lot_number,
                FrameNo,
                vision,
                colname,
                meta_category,
                total_count,
                ng_count,
                ng_rate,
                normalized_mean,
                normalized_std
            FROM read_parquet(?)
            WHERE TRUE {lot_filter}
        """
        return self.connection.execute(
            query,
            [str(self.paths["frame_item_stats"]), *lot_parameters],
        ).df()

    def piece_ng_masks(
        self,
        colnames: tuple[str, ...],
        lot_numbers: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """個片別の検査項目NGビットマスク取得。"""
        lot_filter, lot_parameters = self._lot_filter(lot_numbers)
        query = f"""
            SELECT ng_mask
            FROM read_parquet(?)
            WHERE TRUE {lot_filter}
            ORDER BY lot_start_time, FrameNo, PositionY, PositionX
        """
        masks = self.connection.execute(
            query,
            [str(self.paths["piece_ng"]), *lot_parameters],
        ).fetchnumpy()["ng_mask"]
        if colnames == self.bit_colnames:
            return masks

        remapped = np.zeros(masks.shape, dtype=np.uint64)
        for output_bit, colname in enumerate(colnames):
            selected = masks & self.bit_by_colname[colname]
            remapped[selected != 0] |= np.uint64(1) << np.uint64(output_bit)
        return remapped

    def metrics_by_colname_position(
        self,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・検査項目・製品座標別の事前集計取得。"""
        lot_filter, lot_parameters = self._lot_filter(lot_numbers)
        query = f"""
            SELECT
                lot_number,
                colname,
                PositionX,
                PositionY,
                total_count,
                ng_count,
                ng_rate,
                normalized_mean,
                normalized_std
            FROM read_parquet(?)
            WHERE TRUE {lot_filter}
        """
        return self.connection.execute(
            query,
            [str(self.paths["position_item_stats"]), *lot_parameters],
        ).df()

    def values_by_colname_frame(
        self,
        lot_number: str,
        frame_no: int,
        colname: str,
    ) -> pd.DataFrame:
        """単一Frameの判定済み個片測定値取得。"""
        query = """
            SELECT
                PositionX,
                PositionY,
                value,
                normalized_value,
                is_ng,
                ng_direction,
                limmin,
                limmax,
                meta_best,
                meta_unit
            FROM read_parquet(?)
            WHERE is_judgement_target
              AND lot_number = ?
              AND FrameNo = ?
              AND colname = ?
            ORDER BY PositionY, PositionX
        """
        return self.connection.execute(
            query,
            [
                str(self.detail_path),
                lot_number,
                frame_no,
                colname,
            ],
        ).df()

    def quantiles_by_colname_frame(
        self,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・Frame・検査項目別の事前計算済み分位点取得。"""
        lot_filter, lot_parameters = self._lot_filter(lot_numbers)
        query = f"""
            SELECT
                lot_number,
                FrameNo,
                colname,
                sample_count,
                ng_count,
                minimum,
                p05,
                p25,
                p50,
                p75,
                p95,
                maximum,
                spec_lower,
                spec_upper,
                meta_unit
            FROM read_parquet(?)
            WHERE TRUE {lot_filter}
        """
        return self.connection.execute(
            query,
            [str(self.paths["frame_item_stats"]), *lot_parameters],
        ).df()

    def kde_bins_by_colname_lot(
        self,
        bins: int,
        lot_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """lot・検査項目別の事前集計済みbin取得。"""
        if bins != self.kde_bins:
            raise ValueError(f"KDE bin mismatch: data={self.kde_bins}, config={bins}")
        lot_filter, lot_parameters = self._lot_filter(lot_numbers)
        query = f"""
            SELECT
                lot_number,
                colname,
                bin_index,
                plot_min,
                plot_max,
                spec_lower,
                spec_upper,
                spec_best,
                count
            FROM read_parquet(?)
            WHERE TRUE {lot_filter}
            ORDER BY colname, lot_number, bin_index
        """
        return self.connection.execute(
            query,
            [str(self.paths["lot_item_histogram"]), *lot_parameters],
        ).df()

    @staticmethod
    def _lot_filter(
        lot_numbers: tuple[str, ...] | None,
    ) -> tuple[str, list[object]]:
        """任意のlot絞込SQLとパラメータ。"""
        if not lot_numbers:
            return "", []
        placeholders = ", ".join("?" for _ in lot_numbers)
        return (
            f"AND lot_number IN ({placeholders})",
            list(lot_numbers),
        )
