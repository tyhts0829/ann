# 品質データ生成・加工フロー

この文書は、現行実装のデータ生成、標準化、判定、事前集計の仕様をまとめたものである。実装変更で列、式、粒度、出力ファイルが変わる場合は、この文書も同時に更新する。

## 1. 全体フロー

```text
合成データ生成
  ↓
raw: 1測定値1行
  ↓ 規格基準の標準化
standardized: spec_position・spec_usage追加
  ↓ 判定列付与
judged: 統一正規化値・規格逸脱度・NG判定・NG方向追加
  ↓ 粒度別の事前集計
analysis bundle: 5種類の集計Parquet + manifest
  ↓
ダッシュボードは集計済みParquetを抽出・整列して使用
```

ダッシュボード起動時に、判定済み詳細データから`GROUP BY`、分位点計算、bin集計、個片NGマスク集計は行わない。選択された単一Frameの288個片の詳細抽出だけは、判定済み詳細Parquetからオンデマンドで行う。

## 2. 粒度の定義

粒度は「1行が何を表すか」を定義する列の組み合わせである。

| 表記 | 1行の意味 |
|---|---|
| `lot × FrameNo × colname × PositionX × PositionY` | 1個片の1検査項目 |
| `lot × FrameNo × colname` | 1 Frame内の1検査項目の統計 |
| `lot × colname × PositionX × PositionY` | 1 lotの同じXY位置における1検査項目のFrame横断統計 |
| `lot × colname × bin` | 1 lotの1検査項目の値域別件数 |
| `lot × FrameNo × PositionX × PositionY` | 1個片の全検査項目NG結果 |

現行の`colname`には`_v1`、`_v2`、`_v3`が含まれ、1つのVisionを一意に識別する。そのため、一意キーに`vision`を重複して加えていない。

## 3. raw生成

### 入出力

- 実装: `src/raw/generate_quality_data.py`
- 出力: `data/raw/quality_data_100lots.parquet`
- manifest: `data/raw/quality_data_100lots_manifest.json`
- dataset stage/version: `raw` / `4.0`

### 1行の粒度

```text
lot_number × FrameNo × PositionX × PositionY × colname
```

`colname`がVision別に一意であるため、この組み合わせで1測定値が一意となる。

### データ量

```text
24 Frame
× 24 PositionX
× 12 PositionY
× 3 Vision
× 15検査項目
= 311,040行 / lot
```

現行は100 lotで、31,104,000行、100 row groupsである。1 row groupが1 lotに対応する。

### 主な列

| 種別 | 列 |
|---|---|
| lot | `lot_number`, `lot_start_time` |
| 個片位置 | `FrameNo`, `PositionX`, `PositionY` |
| 検査 | `vision`, `colname`, `value` |
| 規格 | `limmin`, `limmax`, `meta_best` |
| メタデータ | `meta_type`, `meta_ignore`, `meta_category`, `meta_unit` |

### 生成内容

- 再現用seedの既定値: `20260724`
- lot、FrameNo、PositionX、PositionYに連続するfractal 3D Perlin noise
- Vision別の固定偏り、工程変動、個片変動、乱数の合成
- 15検査項目 × 3 Vision = 45種類の`colname`
- 特定lotへの既知の合成異常付与
- 異物・欠陥値の0以上へのclipと小数4桁丸め

## 4. 標準化

### 入出力

- 実装: `src/standardized/standardize_quality_data.py`
- 入力: raw Parquet
- 出力: `data/standardized/quality_data_100lots.parquet`
- manifest: `data/standardized/quality_data_100lots_manifest.json`
- dataset stage/version: `standardized` / `4.1`

### 変換

rawの行数、行の粒度、row group数、`value`、メタデータを維持し、`value`の直後に次の2列を追加する。

#### `spec_position`

両側規格の行だけに設定する。

```text
(value - (limmin + limmax) / 2) / ((limmax - limmin) / 2)
```

- 規格中心: `0`
- 下限: `-1`
- 上限: `1`
- 対象外: null

#### `spec_usage`

`limmin`がなく、`meta_best`と`limmax`がある片側上限規格の行だけに設定する。

```text
(value - meta_best) / (limmax - meta_best)
```

- 最良値: `0`
- 上限: `1`
- 対象外: null

## 5. 判定列付与

### 入出力

- 実装: `src/judged/add_quality_judgement.py`
- 入力: standardized Parquet
- 出力: `data/judged/quality_data_100lots.parquet`
- manifest: `data/judged/quality_data_100lots_manifest.json`
- dataset stage/version: `judged` / `4.2`

### 変換

standardizedの行数、行の粒度、row group数、全列を維持し、次の5列を追加する。

| 列 | 型 | 定義 |
|---|---|---|
| `normalized_value` | nullable float64 | `spec_position`があればその値、なければ`spec_usage` |
| `normalized_deviation` | nullable float64 | 両側は`abs(spec_position)`、片側上限は`max(spec_usage, 0)` |
| `is_judgement_target` | bool | `meta_type == "spec" and not meta_ignore` |
| `is_ng` | bool | 判定対象で、`value < limmin`または`value > limmax` |
| `ng_direction` | int8 | 下限NG=`-1`、OKまたは判定対象外=`0`、上限NG=`1` |

規格値と完全に等しい値はOKとする。`normalized_value`は符号付きの位置・使用率、`normalized_deviation`は「0=最良または規格中心、1=規格限界」の非負指標である。

## 6. 粒度別の事前集計

### 入出力

- 実装: `src/aggregated/build_quality_aggregates.py`
- 入力: judged Parquet
- 出力ディレクトリ: `data/analysis/quality_data_100lots/`
- 完成マニフェスト: `data/analysis/quality_data_100lots/manifest.json`
- 集計対象: `is_judgement_target == True`

manifestには、judged詳細Parquetへの相対パス、元行数、KDE bin数、各Parquetの粒度と行数、`colname`とNG bitの対応を保存する。

### 6.1 `lots.parquet`

| 項目 | 内容 |
|---|---|
| 粒度 | `lot_number` |
| 現行行数 | 100 |
| 元データ | 1 lotの311,040測定 |
| 列 | `lot_number`, `lot_start_time` |

`lot_start_time`昇順のlot一覧。起動時に詳細Parquetからlot一覧を集計し直さないためのデータである。

### 6.2 `frame_item_stats.parquet`

| 項目 | 内容 |
|---|---|
| 粒度 | `lot_number × FrameNo × colname` |
| 1行にまとめる元行 | 通常288個片 |
| 現行行数 | `100 × 24 × 45 = 108,000` |

列:

- キー・メタ: `lot_number`, `lot_start_time`, `FrameNo`, `vision`, `colname`, `meta_category`, `meta_unit`
- 規格: `spec_lower`, `spec_upper`, `spec_best`
- 件数: `total_count=count(*)`, `sample_count=count(value)`, `ng_count=count_if(is_ng)`
- NG率: `ng_rate = 100 * ng_count / total_count`
- 正規化統計: `normalized_mean=avg(normalized_deviation)`, `normalized_std=stddev_pop(normalized_value)`
- 生値統計: `minimum`, `p05`, `p25`, `p50`, `p75`, `p95`, `maximum`

Frame別NG指標とFrame別分位点は粒度が同じため、1ファイルに統合する。`total_count`は全行数、`sample_count`は生値がnullでない件数である。

### 6.3 `position_item_stats.parquet`

| 項目 | 内容 |
|---|---|
| 粒度 | `lot_number × colname × PositionX × PositionY` |
| 1行にまとめる元行 | 通常24 Frame |
| 現行行数 | `100 × 45 × 24 × 12 = 1,296,000` |

列:

- キー・メタ: `lot_number`, `lot_start_time`, `vision`, `colname`, `meta_category`, `meta_unit`, `PositionX`, `PositionY`
- 件数: `total_count`, `sample_count`, `ng_count`
- 指標: `ng_rate`, `normalized_mean`, `normalized_std`

`FrameNo`をまとめ、同じXY位置のFrame横断統計とする。

### 6.4 `lot_item_histogram.parquet`

| 項目 | 内容 |
|---|---|
| 粒度 | `lot_number × colname × bin_index` |
| 1行にまとめる元行 | 同じbinに入る測定値 |
| 現行行数 | 170,537 |
| 最大行数 | `100 × 45 × (60 bin + 範囲外) = 274,500` |

列:

- キー: `lot_number`, `colname`, `bin_index`
- 値域: `plot_min`, `plot_max`, `bin_left`, `bin_right`, `bin_center`
- 規格: `spec_lower`, `spec_upper`, `spec_best`
- 件数: `count`

bin数は`config.toml` の`kde_bins`と同じ60。測定値が`plot_min`未満または`plot_max`超過の場合は`bin_index=null`の1グループとして件数を残す。ゼロ件のbinはParquetに保存せず、読込時に0で補完する。

### 6.5 `piece_ng.parquet`

| 項目 | 内容 |
|---|---|
| 粒度 | `lot_number × FrameNo × PositionX × PositionY` |
| 1行にまとめる元行 | 通常45検査項目 |
| 現行行数 | `100 × 24 × 24 × 12 = 691,200` |

列:

- キー: `lot_number`, `lot_start_time`, `FrameNo`, `PositionX`, `PositionY`
- 件数: `total_item_count`, `ng_item_count`
- 判定: `ng_mask`, `is_ng`

`ng_mask`は64 bit符号なし整数である。`src/analysis/quality_columns.py`の`SPEC_ORDER`順に検査項目をbit 0から割り当て、NG項目のbit ORを保存する。割当てはmanifestの`ng_bit_mapping`にも保存する。現行は45項目であるため64 bit内に収まる。

## 7. 参照層

- 実装: `src/quality_repository.py`
- 入力: analysis bundleディレクトリ

`QualityRepository`はmanifestから各Parquetとjudged詳細Parquetのパスを取得する。起動時のメソッドは集計済みParquetへの`SELECT`、lot絞込、並べ替えだけを行う。

| Repositoryメソッド | 参照先 |
|---|---|
| `lots()` | `lots.parquet` |
| `ng_rate_by_frame()` | `frame_item_stats.parquet` |
| `quantiles_by_colname_frame()` | `frame_item_stats.parquet` |
| `metrics_by_colname_position()` | `position_item_stats.parquet` |
| `kde_bins_by_colname_lot()` | `lot_item_histogram.parquet` |
| `piece_ng_masks()` | `piece_ng.parquet` |
| `values_by_colname_frame()` | judged詳細Parquetから選択Frameのみ |

DuckDBは重い集計を起動時に実行するためではなく、バッチ時の事前集計と、起動時の小さなParquetからの抽出に使用する。

## 8. 再生成手順

プロジェクトルートで次を順番に実行する。

```bash
.venv/bin/python src/raw/generate_quality_data.py
.venv/bin/python src/standardized/standardize_quality_data.py
.venv/bin/python src/judged/add_quality_judgement.py
.venv/bin/python src/aggregated/build_quality_aggregates.py
```

事前集計が完成した後のダッシュボード起動:

```bash
.venv/bin/python dashboard.py
```

## 9. 変更時の更新対象

次の変更を行う場合は、実装、テスト、この文書、analysis manifestを一致させる。

- rawの列または検査項目の追加・変更・削除
- 標準化式の変更
- NG判定または判定対象条件の変更
- 可視化・分析に必要な列の追加
- 新しい集計粒度やParquetの追加
- 既存集計のキー、式、列、bin数の変更
- `SPEC_ORDER`の変更に伴うNG bit割当ての変更

集計結果は元データや計算式の変更を自動反映しない。上流のデータまたは仕様を変更した場合は、後続段階を再実行する。
