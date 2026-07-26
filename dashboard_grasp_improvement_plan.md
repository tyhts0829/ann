# 「把握」ダッシュボード改善 実装計画

作成日: 2026-07-26

## 1. 目的

`concept.md`の「把握」を、現在のFQmap・Fmap・KDEを基礎に段階的に改善するための実装計画。

各提案は原則として1件ずつ実装・評価し、採用判断後に次の提案へ進む。異常度、通常状態との比較、調査優先度、類似ロットなどは「判定」「診断」「提案」の責務とし、本計画には含めない。

## 2. 現状の基準

- 対象データ: 100 lot、24 frame/lot、24 × 12個片/frame、45 spec項目
- FQmap:
  - 行: vision別検査項目
  - 列: lot × FrameNo
  - 指標: 測定NG率、規格位置・使用率の平均、母標準偏差
- Fmap:
  - 選択項目をlot内24 frameで集約したXYマップ
- KDE:
  - 選択項目のlot別生値分布
- 現行表示数: 8 lot
- 現行の「総合NG率」: NG測定件数 ÷ 全測定件数
- 現行集計対象: `meta_type == "spec"`かつ`meta_ignore == False`
- 現在保存されているraw・standardized Parquetには`meta_unit`があるが、`concept.md`とデータ生成ソースには定義がない。現状のままデータを再生成すると単位列が失われるため、P03で再現性を修正する。
- 2026-07-26時点の基準テスト: 11 passed

主な変更対象:

- `dashboard.py`
- `config.toml`
- `concept.md`
- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/kde.py`
- `src/analysis/map_palettes.py`
- `src/analysis/quality_columns.py`
- `src/standardized/quality_data.py`
- `src/standardized/standardize_quality_data.py`
- `src/raw/generate_quality_data.py`
- `tests/test_dashboard.py`

## 3. 1提案ごとの進め方

各提案を次の単位で試す。

1. 実装前に現行テストを実行する。
2. 当該提案に必要な変更だけを実装する。
3. 集計ロジックのテストとUIテストを追加する。
4. 1900 × 1550、1600 × 900、最小サイズ1100 × 760で画面を確認する。
5. 現行データ中の既知の変化ロットで読み取りやすさを確認する。
6. 完了条件と評価観点に基づき、採用・修正・見送りを決める。
7. 採用時のみ`concept.md`と設定説明を更新する。

共通確認コマンド:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python dashboard.py
```

## 4. 推奨実装順

| 順序 | ID  | 提案                             | 規模 | 主な依存              |
| ---: | --- | -------------------------------- | ---- | --------------------- |
|    1 | P00 | 表示lot数と仕様記述の統一        | 小   | なし                  |
|    2 | P01 | 測定NG率と個片NG率の分離         | 中   | なし                  |
|    3 | P02 | 表示範囲・時刻・Frame順序の明示  | 小   | なし                  |
|    4 | P03 | 測定単位と項目メタデータの追加   | 中   | データ再生成          |
|    5 | P04 | 選択項目の分位点トレンド         | 大   | P03推奨               |
|    6 | P05 | FQmapセルと単一Frame Fmapの連動  | 大   | なし                  |
|    7 | P06 | 個片ドリルダウン                 | 大   | P03、P05              |
|    8 | P07 | P方向の定義と順序プロット        | 大   | 業務上の順序定義、P05 |
|    9 | P08 | 規格マージン・裾指標の追加       | 中   | データ再生成          |
|   10 | P09 | 色尺度の固定・自動切替           | 中   | P08とは独立実装可     |
|   11 | P10 | 欠測・測定数・カバレッジ表示     | 中   | なし                  |
|   12 | P11 | カテゴリ・項目・Vision階層の明示 | 中   | なし                  |
|   13 | P12 | Fmap・KDEのホバー詳細            | 中   | P05後が望ましい       |
|   14 | P13 | 余白削減とフォーカス表示         | 中   | P04後が望ましい       |
|   15 | P14 | spec・aux表示モードの分離        | 大   | aux定義、P03、P04推奨 |

---

## P00. 表示lot数と仕様記述の統一

対処済みのため削除

---

## P01. 測定NG率と個片NG率の分離

### 目的

測定単位のNG率を製品歩留まりと誤認しないようにし、製品階層の最小単位である個片の品質状態も把握できるようにする。

### 初回仕様

- **測定NG率**: 表示対象のNG測定件数 ÷ 表示対象の測定件数
- **個片NG率**: 全spec項目のうち1項目以上がNGの個片数 ÷ 全個片数
- 個片キー: `lot_number, FrameNo, PositionX, PositionY`
- 初回は個片NG率を全spec項目固定とし、カテゴリ・Visionフィルターには追従させない。
- KPI名に集計範囲を明記する。

### 変更対象

- `src/standardized/quality_data.py`
- `dashboard.py`
- `src/analysis/fq_map.py`
- `tests/test_dashboard.py`

### 実装手順

1. 個片ごとにspec NGの有無を集約するRepositoryメソッドを追加する。
2. 個片総数、NG個片数、個片NG率を保持する小さなデータクラスを追加する。
3. 起動時のバックグラウンド集計へ個片集計を追加する。
4. 現行ラベルを「測定NG率」へ変更する。
5. 「個片NG率」または「個片歩留まり」を別ラベルで追加する。
6. ツールチップまたは補足文に分子・分母を表示する。

### テスト

- 現行データで次を確認する。
  - 測定NG: 78,126 / 31,104,000、0.251%前後
  - 個片NG: 37,989 / 691,200、5.496%前後
- 同じ個片に複数NGがあってもNG個片数を1件と数えること。
- 規格値ちょうどをNGにしないこと。
- `meta_ignore == True`とauxを個片NG判定から除外すること。
- UIに2種類のKPI名と母数が表示されること。

### 完了条件

- 操作者が2つのNG率の分母を画面だけで説明できる。
- 「総合NG率」という曖昧な名称が残っていない。
- 起動時の応答性が現状から大きく悪化しない。

### 今回含めないもの

- フィルター条件別の個片NG率
- ロット判定、異常度、調査優先度

---

## P02. 表示範囲・時刻・Frame順序の明示

完了

---

## P03. 測定単位と項目メタデータの追加

一旦不要のため削除

---

## P04. 選択項目の分位点トレンド

### 目的

FQmapで見つけた変化について、lot・Frame順に生値の水準、幅、裾を正確に読めるようにする。

### 初回仕様

- 対象: 選択中の1 vision別colname
- 横軸: 表示中lot × FrameNo
- 縦軸: 生測定値
- 表示:
  - 中央値
  - P25–P75帯
  - P05–P95帯
  - 規格上限・下限
- FQmap・Fmap・KDEとlotスクロールを同期する。
- lot間でx軸、同一項目の全lotでy軸を共通にする。

### 変更対象

- 新規 `src/analysis/quality_trend.py`
- `src/standardized/quality_data.py`
- `src/analysis/fq_map.py`
- `dashboard.py`
- `config.toml`
- `tests/test_dashboard.py`

### データ集計

`lot_number, FrameNo, colname`ごとに次を集計する。

- count
- NG count
- P05
- P25
- P50
- P75
- P95
- limmin
- limmax
- meta_unit

45 × 100 × 24の集計結果を起動時にバックグラウンド生成する。

### 実装手順

1. 分位点集計用Repositoryメソッドを追加する。
2. 配列化した`QualityTrendData`を作成する。
3. 中央値線と2段の帯を持つWidgetを追加する。
4. 選択colnameとlotスクロール位置を同期する。
5. 現在のFQmap下部余白を減らし、全体の高さを増やさず配置する。
6. ホバーでlot、FrameNo、各分位点、N、NG件数を表示する。

### テスト

- 分位点の大小関係が`P05 <= P25 <= P50 <= P75 <= P95`であること。
- 1セルの測定数が通常288件であること。
- 選択項目変更でデータと規格線が切り替わること。
- 横スクロール後もFQmapとlot境界が一致すること。
- 片側規格で架空の規格中心を表示しないこと。

### 完了条件

- FQmapの色が濃い箇所について、水準変化か幅の増加か裾の変化かを読み分けられる。
- 8 lot表示時も帯と規格線を判読できる。

### 採用評価

- KDEと情報が重複しすぎないか。
- FQmapから詳細理解までの操作数が減るか。
- 画面密度が過剰にならないか。

---

## P05. FQmapセルと単一Frame Fmapの連動

### 目的

`Q(f, x, y)`のうち、現在分離しているFrame方向とXY方向をつなぐ。

### 初回仕様

- FQmapクリックで次を同時選択する。
  - colname
  - lot
  - FrameNo
- 行選択の強調は維持し、選択セルには追加の枠を表示する。
- Fmapに次の表示切替を追加する。
  - lot集約
  - 選択Frame
- 単一Frame表示では1セル1個片とする。
- 単一Frameの各XYセルには1測定しかないため、標準偏差段は表示しない。
- 単一Frameでは選択した1 Frameを大きく使い、NG判定、規格正規化値、生値の3マップを横並びにする。
- 起動直後は従来のlot集約表示とし、FQmapセル選択時に選択Frame表示へ切り替える。

### 変更対象

- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/standardized/quality_data.py`
- `tests/test_dashboard.py`

### データ取得方針

- 全colname・全lot・全FrameのXY配列を事前保持しない。
- 選択したcolname・lot・FrameNoの288行だけを取得する。
- 現行Parquetはlotごとのrow groupであるため、まずオンデマンド取得で応答時間を測る。
- 実測してクリック応答に問題がある場合のみ、専用workerへ移す。

### 実装手順

1. `selected_lot_index`と`selected_frame_no`をFQmap状態へ追加する。
2. FQmapセル選択枠を3段で同期する。
3. 指定colname・lot・FrameNoの生値、規格正規化値、NG判定を取得するRepositoryメソッドを追加する。
4. `FrameMapWidget`へlot集約3段／単一Frame横並び3マップのモードを追加する。
5. 選択lot・FrameNoをFmap左欄へ表示する。
6. FmapへX/Y目盛と原点・向きの表示を追加する。

### テスト

- クリックした列からlotとFrameNoを正しく復元すること。
- 3段FQmapで同じセル枠が表示されること。
- 単一Frameの3マップがそれぞれ12 × 24であること。
- 単一Frameの集計元が288個片であること。
- 単一Frameモードで意味のない標準偏差0マップを表示しないこと。
- 生値マップと規格正規化値マップが元データと一致すること。
- lot集約へ戻したときに現行結果と一致すること。

### 完了条件

- 色の強いFQmapセルから、同じFrameのXY分布へ1操作で移動できる。
- lot集約とFrame固有パターンを明確に区別できる。

---

## P06. 個片ドリルダウン

### 目的

集約値の根拠となる個片まで追跡し、製品階層のlot → Frame → 個片を完成させる。

### 初回仕様

- 単一Frame Fmapのセルクリックで個片を選択する。
- 個片キー:
  - lot_number
  - FrameNo
  - PositionX
  - PositionY
- 詳細表示:
- 45 spec項目の規格内位置
- NG項目
- 生値、規格値、単位
- category、vision、colname
- 異なるcolnameの生値は同じ軸で比較しない。
- 詳細は初期状態で非表示の右側`QDockWidget`とし、メイン画面の選択状態から独立して閉じられるようにする。
- 別のlotまたはFrameを選択したときは、古い個片選択を解除する。

### 変更対象

- 新規 `src/analysis/product_profile.py`
- `src/analysis/frame_map.py`
- `src/analysis/fq_map.py`
- `src/standardized/quality_data.py`
- `tests/test_dashboard.py`

### 実装手順

1. 個片キーで全項目を取得するRepositoryメソッドを追加する。
2. Fmapにセル選択枠とクリック処理を追加する。
3. 右側`QDockWidget`を追加する。
4. 正規化値のドットプロットと生値表を表示する。
5. category・visionによる絞り込みを詳細内にも反映する。
6. 選択状態をヘッダーのパンくずへ表示する。

### テスト

- 完全データの1個片から45行取得できること。
- NG判定がFQmapの条件と一致すること。
- PositionX/Yのクリック位置変換が端部でも正しいこと。
- 項目選択や横スクロール後も不正な個片情報を残さないこと。

### 完了条件

- FQmap → Frame → XYセル → 生測定値まで追跡できる。
- 個片詳細だけでNG項目と規格との差を確認できる。

---

## P07. P方向の定義と順序プロット

### 目的

個片の検査・加工順序に沿った連続変化を把握できるようにする。

### 実装開始前の必須判断

P方向が次のどれを意味するかを業務定義として確定する。

- 検査取得順
- 加工順
- 搬送順
- フレーム内の論理個片番号

XYから暗黙に順序を推測しない。

### 推奨データモデル

- rawデータへnullableでない`int16`の`product_order`を追加する。
- 各`lot_number, FrameNo, vision, colname`内で1～288を持つ。
- 同じ物理個片ではvision・colnameが変わっても同じ順序値を持つ。
- 蛇行走査などの場合は変換規則を`concept.md`へ明記する。
- lot内表示用の順序は保存せず、`(FrameNo - 1) × 288 + product_order`で1～6,912を導出する。

### 変更対象

- `concept.md`
- `src/raw/generate_quality_data.py`
- `src/standardized/standardize_quality_data.py`
- `src/standardized/quality_data.py`
- 新規 `src/analysis/product_order.py`
- `src/analysis/fq_map.py`
- `tests/test_dashboard.py`
- raw・standardized Parquetとmanifest

### 初回可視化

- 対象: 選択colname・lot
- 横軸: FrameNoごとに区切った`product_order`
- 縦軸: 生値
- 規格線とFrame境界を表示
- FQmapで選択中のFrameに対応する288点を背景強調する。
- 点が過密な場合は線ではなく低不透明度の点またはbin集約を使用

### テスト

- 各Frameに1～288が重複なく存在すること。
- PositionX/Yとの変換規則が期待値と一致すること。
- vision・colname間で同じ個片の順序が一致すること。
- lot内表示順が1～6,912となり、FrameNo 2の先頭が289であること。
- Frame境界とFQmapのFrameNoが同期すること。

### 完了条件

- P方向の意味を画面と文書から一意に説明できる。
- XYパターンと順序パターンを混同せず比較できる。

### 中止条件

- 実データで信頼できる順序列を取得できない場合は実装しない。

---

## P08. 規格マージン・裾指標の追加

### 目的

規格位置平均の正負相殺を避け、規格境界に近い裾側の余裕を把握する。

### 指標案

`spec_margin`は規格境界で0、中心または最良値で1、規格外で負とする。

- 両側規格: `1 - abs(spec_position)`
- 片側上限規格: `1 - spec_usage`
- 片側下限規格: `meta_best`確定後、下限から最良値までを同様に正規化

FQmapでは平均ではなく、Frame内の**P05規格マージン**を初回候補とする。

### 初回仕様

- 現行3段は削除しない。
- 「規格位置平均」と「P05規格マージン」を切り替えられるようにする。
- マージン0を色尺度上の明示的な境界にする。

### 変更対象

- `concept.md`
- `src/standardized/standardize_quality_data.py`
- `src/standardized/quality_data.py`
- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/map_palettes.py`
- `tests/test_dashboard.py`
- standardized Parquetとmanifest

### 実装手順

1. `spec_margin`の式と対象条件を確定する。
2. standardizedデータへ列を追加する。
3. Frame別・Position別にP05を集計する。
4. FQmapとFmapへ指標切替を追加する。
5. ホバーにP05値と0の意味を表示する。

### テスト

- 規格中心または最良値で1となること。
- 規格境界ちょうどで0となること。
- 規格外で負となること。
- 上限側・下限側が両側規格で対称になること。
- P05集計が手計算結果と一致すること。

### 完了条件

- 平均0だが両裾が規格近傍にあるケースを見落とさない。
- 規格位置平均と規格余裕の意味がラベル上で混同されない。

---

## P09. 色尺度の固定・自動切替

### 目的

フィルター変更で同じ数値の色が変わる問題を抑え、比較時の色の意味を安定させる。

### 初回仕様

- **比較尺度**: 全100 lot・全対象項目から算出した尺度を維持
- **自動尺度**: 現在のフィルター範囲から算出
- 既定値は比較尺度
- 現在のモードと数値範囲を常時表示
- 尺度外値は上限・下限へclipし、セルまたは凡例にclip発生を表示する。

### 変更対象

- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/map_palettes.py`
- `config.toml`
- `src/dashboard_config.py`
- `tests/test_dashboard.py`

### 実装手順

1. 全体尺度とフィルター尺度を別に計算する。
2. ツールバーへ比較／自動切替を追加する。
3. FQmap・Fmapの尺度モードを同期する。
4. clip件数を集計し、凡例またはフッターへ表示する。
5. 指標別の尺度設定を`config.toml`へ持たせるかは試用後に決める。

### テスト

- 比較尺度ではカテゴリ・Vision変更後もlevelsが変わらないこと。
- 自動尺度では対象データから再計算されること。
- 0件、定数値、尺度外値の表示が破綻しないこと。

### 完了条件

- 同じ値がフィルター変更だけで別の色にならないモードを選べる。
- 凡例から現在の尺度モードを判断できる。

---

## P10. 欠測・測定数・カバレッジ表示

### 目的

0、低NG率、未測定、母数不足を区別し、集約結果の信頼性を確認できるようにする。

### 初回仕様

- FQmapホバーへ測定数と期待測定数を表示する。
- Fmapホバーへ集約Frame数と測定数を表示する。
- KDEへlot別の測定数と表示範囲外率を持たせる。
- 欠測セルは白ではなく灰色オーバーレイで表示する。
- `meta_ignore`件数はヘッダーのデータ範囲情報に表示する。
- 完全データでの期待件数は、FQmap 1セル288、lot集約Fmap 1セル24、KDE 1列6,912とする。

### 変更対象

- `src/standardized/quality_data.py`
- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/kde.py`
- `tests/test_dashboard.py`

### 実装手順

1. Fmapデータにも`total_counts`を保持する。
2. 各集計粒度の期待件数を明文化する。
3. 欠測・部分測定マスクを生成する。
4. ImageItem上へ非データ用のRGBAマスクを重ねる。
5. hoverとサマリーへN、coverage、ignore件数を追加する。

### テスト

- 一部行を除いた小規模Parquet fixtureを作る。
- 0% NGと欠測が異なる表示になること。
- 測定数が期待値未満のセルを検出できること。
- 完全データでは不要な欠測マスクが表示されないこと。

### 完了条件

- 色が薄い理由を「良好」「未測定」「母数不足」に区別できる。
- FQmap、Fmap、KDEで母数表記が矛盾しない。

---

## P11. カテゴリ・項目・Vision階層の明示

### 目的

45行の関係を読みやすくし、カテゴリ、ベース項目、Visionを視覚的に区別する。

### 初回仕様

- 行順は現行の`SPEC_ORDER`を維持する。
- category境界に太めの区切り線を表示する。
- ベース項目ごとに3 visionを1グループとして表示する。
- 全項目表示時も各vision行を識別できる短縮ラベルを付ける。
- 折りたたみ機能は初回には含めず、静的階層表示を先に評価する。

### 変更対象

- `src/analysis/quality_columns.py`
- `src/analysis/fq_map.py`
- `tests/test_dashboard.py`

### 実装手順

1. category・base colname・visionの行メタデータを明示的に持たせる。
2. categoryとbase colname境界を描画する。
3. y軸ラベルを階層が分かる形式へ変更する。
4. フィルター時は不要な上位ラベルを省略する。
5. 視認性が不足する場合のみ次段階で展開・折りたたみを検討する。

### テスト

- 45行、15グループ、5カテゴリの境界位置が正しいこと。
- category・Visionフィルター後に境界が再計算されること。
- 完全なvision別colnameをホバーと選択欄で維持すること。

### 完了条件

- 全項目表示でも任意の1行がどのcategory・base項目・visionか判断できる。
- 文字量が増えてヒートマップ幅を過度に圧迫しない。

---

## P12. Fmap・KDEのホバー詳細

### 目的

詳細領域でも、集約値とその位置・lotを画像から直接確認できるようにする。

### Fmapホバー仕様

- lot_number
- 表示モード
- FrameNoまたはlot集約
- PositionX、PositionY
- 指標値
- 測定数

### KDEホバー仕様

- lot_number
- x値と単位
- 密度
- lotのP05、中央値、P95
- 規格範囲外件数

### 変更対象

- `src/analysis/frame_map.py`
- `src/analysis/kde.py`
- `src/analysis/fq_map.py`
- `tests/test_dashboard.py`

### 実装手順

1. scene座標からlot列とXYセルまたはKDE x値を取得する。
2. 共通フッターまたは各詳細欄へhover情報を表示する。
3. FQmapから別領域へ移動したときに古いhover情報を消す。
4. ホバーとクリック選択の見た目を区別する。

### テスト

- X/Yの四隅とlot列境界で正しい値を返すこと。
- KDEの共通x軸と単位が正しいこと。
- 表示範囲外で補足文へ戻ること。

### 完了条件

- 詳細値を確認するために別画面やログを開く必要がない。
- ホバーが選択状態を変更しない。

---

## P13. 余白削減とフォーカス表示

### 目的

FQmap固定領域下部の余白を減らし、KDEや追加トレンドを必要に応じて拡大できるようにする。

### 初回仕様

- FQmapの高さをツールバー、3プロット、区切り帯から算出し、空白を残さない。
- FQmap、Fmap、KDE、分位点トレンドにフォーカスボタンを付ける。
- フォーカス中は対象を拡大し、解除すると設定比率へ戻る。
- 初回は自由なドラッグリサイズより、状態が明確なフォーカス切替を優先する。

### 変更対象

- `dashboard.py`
- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/kde.py`
- `src/dashboard_config.py`
- `config.toml`
- `tests/test_dashboard.py`

### 実装手順

1. 現行固定高と実コンテンツ高の算出箇所を一本化する。
2. 通常表示で余白を除去する。
3. 各セクションのフォーカス状態を追加する。
4. フォーカス切替後にplotのViewBoxと列幅を再計算する。
5. 必要性が確認できた場合のみ、次段階でQSplitterを検討する。

### テスト

- 1900 × 1550、1600 × 900、最小ウィンドウで重なりがないこと。
- 通常表示で3段FQmapを同時表示できること。
- フォーカス解除後にlot列位置が再び一致すること。
- 既存の高さ固定テストを新しいレイアウト仕様へ更新すること。

### 完了条件

- 通常表示に用途のない大きな空白がない。
- KDEやトレンドを一時的に詳細表示できる。

---

## P14. spec・aux表示モードの分離

### 目的

規格判定を持たない補助測定値も把握対象にしつつ、NGや規格マージンを誤って適用しない。

### 実装開始前の必須判断

- 実際に扱うaux項目名、単位、categoryを定義する。
- auxを全項目俯瞰する必要があるか、選択項目詳細だけでよいかを決める。
- 生値の異なるaux項目を共通色尺度で比較しない。

### 推奨初回仕様

- ツールバーに`spec / aux`切替を追加する。
- specモードは現行のNG・規格位置・ばらつきを維持する。
- auxモードは1項目を選択し、次を表示する。
  - 生値の分位点トレンド
  - lot別KDE
  - XY平均
  - XY標準偏差
- auxモードではNG率、規格線、規格マージンを表示しない。

### 変更対象

- `concept.md`
- `src/raw/generate_quality_data.py`
- `src/standardized/standardize_quality_data.py`
- `src/standardized/quality_data.py`
- `src/analysis/quality_columns.py`
- `src/analysis/fq_map.py`
- `src/analysis/frame_map.py`
- `src/analysis/kde.py`
- `src/analysis/quality_trend.py`
- `tests/test_dashboard.py`
- raw・standardized Parquetとmanifest

### 実装手順

1. aux測定マスターを定義する。
2. 生成データへaux行を追加し、標準化列が欠損になることを確認する。
3. Repositoryの各クエリをspec用とaux用に分ける。
4. type切替とaux項目選択を追加する。
5. aux用Fmap・KDE・トレンドを生値尺度で表示する。
6. モード切替時に不適切な凡例、規格線、KPIを隠す。

### テスト

- aux行の`spec_position`と`spec_usage`が欠損であること。
- auxが個片NG率と測定NG率に含まれないこと。
- auxモードで規格線とNG色尺度が表示されないこと。
- specへ戻したときに現行表示が復元されること。

### 完了条件

- specとauxの意味がUI上で混同されない。
- auxの生値変化、分布、XY位置を同じ項目内で比較できる。

---

## 5. 提案間の依存関係

```text
P03 測定単位 ──> P04 分位点トレンド
       └───────> P06 個片ドリルダウン
       └───────> P14 auxモード

P05 単一Frame Fmap ──> P06 個片ドリルダウン
              └──────> P07 P方向
              └──────> P12 Fmapホバー詳細

P04 分位点トレンド ──> P13 レイアウト調整

P08 規格マージン ──> P09 色尺度設定へのマージン対応
```

P00、P01、P02、P10、P11は他提案から独立して試せる。

## 6. 各試行で記録する内容

各提案の実装後、次を短く記録する。

- 実装したID
- 使用したデータ
- 変更前後のスクリーンショット
- テスト結果
- 起動・初期集計時間
- 読み取りやすくなった問い
- 新たに分かりにくくなった点
- 採用、修正、見送りの判断

## 7. 本計画の対象外

以下は「把握」ではなく後続画面で扱う。

- 過去N lotを基準とした異常度
- 通常状態との差分マップ
- 調査優先度ランキング
- 原因候補の推定
- 相関の出現・消失
- 類似異常lot検索
- 対応履歴と推奨アクション
