ディスクリート半導体後工程の出荷前最終検査で得られる製品品質データのダッシュボードをつくります。

# データモデル

製品は1枚のフレームに、24 x 12で配置されています。1lotは24フレームです。
データは外観検査機で取得されたもので、3つのvision=機台があります。
rawデータはpandas dataframe形式で、

vision
lot_number
lot_start_time
FrameNo
PositionX
PositionY
value
colname: データの名前。検査項目spec、または補助測定値aux
limmin
limmax
meta_type: auxまたはspec
meta_ignore: TrueまたはFalseで可視化で無視するかどうかを指定
meta_best: 片側規格しかない場合に、最良値を指定する。例えば異物数であれば、0が最良値となる。
meta_category: データのカテゴリ。例えば、異物、標印、PKGサイズ、欠陥 など

標準化処理で、`value`の直後に次の2列を追加します。

spec_position: 両側規格内での位置。規格中心を0、規格上限を1、規格下限を-1とする。
spec_usage: meta_bestが定義された片側上限規格の使用率。meta_bestを0、規格上限を1とする。

```text
     vision    lot_number      lot_start_time  FrameNo  PositionX  PositionY   value  spec_position  spec_usage                    colname  limmin  limmax meta_type  meta_ignore  meta_best meta_category
0  vision_1  LOT_20260701_A  2026-07-01 08:15:23        1          1          1  0.0820            NaN      0.2733     Foreign_Length_Long_v1     NaN   0.300      spec        False      0.000            異物
1  vision_1  LOT_20260701_A  2026-07-01 08:15:23        1          1          1  1.2040           0.05         NaN            Lead_Length_L_v1   1.120   1.280      spec        False        NaN          リード
2  vision_1  LOT_20260701_A  2026-07-01 08:15:23        1          1          1  4.0120           0.12         NaN                  Work_Xw_v1   3.900   4.100      spec        False        NaN        PKGサイズ
3  vision_2  LOT_20260701_A  2026-07-01 08:15:23        9          7          4  0.0180           0.18         NaN            Mark_Center_X_v2  -0.100   0.100      spec        False        NaN            標印
4  vision_3  LOT_20260701_A  2026-07-01 08:15:23       17         14          8  0.0040            NaN      0.1600                Defect_Size_v3     NaN   0.025      spec        False      0.000            欠陥
```

データフレームの意味は以下のとおりです。

- 1ロットには`FrameNo = 1～24`の24枚のフレームが存在する。
- 各フレームには`PositionX = 1～24`、`PositionY = 1～12`の288個の製品が存在する。
- 同じ製品に対して複数の検査項目があるため、`vision, lot_number, FrameNo, PositionX, PositionY`の組み合わせは一意ではない。
- 1件の測定値を一意に識別するキーは、原則として次の組み合わせである。

vision
lot_number
FrameNo
PositionX
PositionY
colname

- `meta_type == "spec"`の項目は規格判定の対象とする。
- `meta_type == "aux"`の項目は規格判定を行わないが、分布、傾向、異常度などの解析対象にはできる。
- `limmin`と`limmax`の両方が設定されている場合は両側規格である。
- `limmin`のみ設定されている場合は下限規格である。
- `limmax`のみ設定されている場合は上限規格である。
- `limmin`と`limmax`の両方が欠損している場合は規格なしである。
- 規格値ちょうどの値は規格内として扱う。
- `meta_best`は、片側規格項目について望ましい方向を定義する。例えば異物数は小さいほどよいため`meta_best = 0`とする。
- 両側規格の`spec_position`は、`(value - (limmin + limmax) / 2) / ((limmax - limmin) / 2)`で計算する。規格中心は0、上限は1、下限は-1となる。
- `meta_best`が定義された片側上限規格の`spec_usage`は、`(value - meta_best) / (limmax - meta_best)`で計算する。最良値は0、規格上限は1となる。
- `spec_position`または`spec_usage`の対象外となる行には欠損値を格納する。
- `meta_ignore == True`の項目は、データ自体は保持するが通常のダッシュボードでは表示候補から除外する。
- `meta_category`は検査項目をダッシュボード上で分類・絞り込みするために使用する。
- `lot_start_time`は同一ロット内では同じ値とする。
- `value`は検査項目によって単位や意味が異なるため、異なる`colname`間で生値を直接比較しない。
- 3台の検査機は`vision_1`、`vision_2`、`vision_3`として表現する。
- colnameは機台間で重複させず、`vision_1`は`_v1`、`vision_2`は`_v2`、`vision_3`は`_v3`を末尾に付与する。
- 3台のvisionはそれぞれ`FrameNo = 1～24`の全フレームを検査する。

検査項目のベース名は以下の15項目である。vision別サフィックスを含めた一意なcolnameは45項目となる。

- 異物: `Foreign_Length_Long`、`Foreign_Length_Short`、`Foreign_Size`
- リード: `Lead_Length_L`、`Lead_Length_R`、`Lead_Pitch`
- PKGサイズ: `Work_Xw`、`Work_Yw`、`Work_Center_X`、`Work_Center_Y`
- 標印: `Mark_Center_X`、`Mark_Center_Y`
- 欠陥: `Defect_Length_Long`、`Defect_Length_Short`、`Defect_Size`

完全な1ロットの想定行数は次のとおりである。

24 frames × 24 positions X × 12 positions Y × 3 visions × 15 measurements
= 311,040 rows / lot

# データ処理

データ処理はraw生成と標準化の2段階に分ける。

1. `src/raw/generate_quality_data.py`で生成し、`data/raw/quality_data_100lots.parquet`へ保存する。この段階では`spec_position`と`spec_usage`を持たない。通常変動はlotとFrameNoを連続時間軸、PositionXとPositionYを空間軸としたfractal 3D Perlin noiseを基調とし、複数octaveの緩やかな工程変動と局所変動を重ねる。短周期や座標軸に沿う人工的な模様を避けるため、Perlinの周期は4096、lacunarityは2.071とし、入力座標を回転してから評価する。
2. `src/standardized/standardize_quality_data.py`で規格正規化列を追加し、`data/standardized/quality_data_100lots.parquet`へ保存する。
3. FQマップは`src/analysis/fq_map.py`、フレームマップは`src/analysis/frame_map.py`に分けて実装し、`dashboard.py`が両者を構成する。

```bash
.venv/bin/python src/raw/generate_quality_data.py
.venv/bin/python src/standardized/standardize_quality_data.py
```

---

品質データモデルはQ(f, x, y)
Qは品質ベクトル
xはPositionX
yはPositionY
fはFrameNoである。

# ダッシュボード

ダッシュボードの機能は以下の4つである。

把握：どの品質項目が、どの単位、位置、順序で、どのような分布・推移を示しているか？

- 製品階層：ロット、フレーム、個片
- 品質指標：良否判定、規格マージン、生の計測値
- 検査構成：ビジョン、検査カテゴリ、検査項目
- 分析方向：
  - F方向：フレーム順序分析
  - XY方向：フレームマップ
  - P方向：個片順序分析

判定：どのロットが、過去の通常状態に対し調査する価値があるほど異なるか？

- 規格適合性、通常性の2軸で評価
- 横軸を規格適合率、縦軸を通常率または異常度としてロットごとに配置してもよさそう
- 一覧のなかで以下が分かるように異常を目立たせるイメージ
- 規格不適合だから目立っているのか
- 通常状態から外れているから目立っているのか
- 両方に該当するのか
- どの程度調査優先度が高いのか

診断：異常判定したロットが、何が、どこで、いつから、どのように通常と異なるか？

- what：何の指標が異常か一覧し、異常が現れている品質指標や検査項目を特定する
  - ビジョン、検査カテゴリ、検査項目
  - 良否判定、規格マージン、q間の関係、q内分布hist
- how：特定した指標が、通常状態との差がどのような形で現れているかを確認する
  - NG率の増加、規格マージンの減少、相関の出現・消失、周期性の出現・消失、分布の変化、ばらつきの増加・減少、水準の上昇・低下
  - 対象ロットと通常状態を並べるか重ね、差の種類を説明できる可視化が必要
- where：どこで異なるか、異常が現れている空間または加工上の位置を特定
  - フレームマップ、mフレーム座標、加工パス上の位置、フレーム順序、ロット順序
- when：いつから異なるか、異常が現れ始めた時期を特定
  - ロット順序、フレーム順序、個片順序

提案：その異常は、過去のどのロットと類似しているか？（そのロットはどのような原因で、どのような対応をしたか？）

- 正常ロット、類似異常ロット、対象ロットの3者を上記と同様のビューで比較
- 総合類似度と、その内訳を表示

現在の概要画面では、各`lot_number × FrameNo × colname`について次の3段をFQマップと呼ぶ。

1. NG率（Reds相当）
2. `spec_position`または`spec_usage`の平均（RdBu_r相当）
3. `spec_position`または`spec_usage`の母標準偏差（Purples相当）

各列は`FrameNo`単位で集計するが、x tick labelにはロット番号だけを表示する。ロット番号は最上段のFQマップ上側に1回だけ表示し、2段目と3段目には重複表示しない。一度に5ロットを表示し、全100ロットは横スクロールで移動する。FQマップは低い高さの概要表示とし、`meta_category`が`None`の場合は全項目、それ以外は選択カテゴリの`colname`だけを表示する。

raw、standardized、FQマップのすべてで、末尾に`_v1`、`_v2`、`_v3`を持つ45個の一意な`colname`を維持する。全カテゴリ表示では3 visionの行をベース検査項目単位でまとめて目盛表示し、カテゴリを選択した場合は各colnameを個別表示する。セル選択時と詳細表示では常に完全なvision別colnameを示す。

FQマップの下には、選択したセルの行に対応するvision別検査項目について、`PositionX = 1～24`、`PositionY = 1～12`のフレームマップを次の3段で表示する。

1. NG率（Reds相当）
2. `spec_position`または`spec_usage`の平均（RdBu_r相当）
3. `spec_position`または`spec_usage`の母標準偏差（Purples相当）

フレームマップは1ロット内の24フレームを製品座標ごとに集計し、FQマップと同じ横スクロール位置の5ロットを並べる。各製品セルの境界には白いグリッド線を表示する。3種類のどのFQマップをクリックしても、ダイヤ型の選択マーカーとフレームマップ対象項目を同期する。選択ロットはフレームマップ上端の細いインディゴ線で控えめに示す。

画面はライトテーマとし、3段のFQマップ間にカード枠や区切り余白を設けない。

不良は規格不適合、異常は過去の通常状態からの逸脱として定義し、それらは必ずしも一致しない。
過去の通常状態は、対象ロットを含まない過去のNロットから計算する。
通常中心は中央値、通常ばらつきは1.4826 × MAD（中央値絶対偏差）で定義する。
異常度は x-通常中心 / 通常ばらつきで定義する。
