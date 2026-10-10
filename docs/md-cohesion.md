# MDの凝集エネルギー観測値を準備する

`md-cohesion-prepare` は、許可済みのローカルCSVからMDの観測値と材料メタデータを
読み込むコマンドです。物理計算の実行・学習・HSP3成分への変換は行いません。
外部データやRadonPy・力場ファイルは配布物に含めません。

今回調べた著者データは [PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics)、
固定した版は `041e5834ea1a48682fae12dc39ccd723bcd4f771`、宣言ライセンスはCC BY 4.0です。
[一次論文](https://arxiv.org/pdf/2511.11626v1)のPDF15ページ目と58ページ目の式23は、
GAFF2の電気項を √(δP²＋δH²) に対応させています。
分散・電気・全体という3列は、HansenのδD・δP・δHという3つの独立した正解ではありません。
同じ電気項に対して極性項と水素結合項の分け方は複数あり、別の校正情報が必要です。
短距離・長距離のCoulomb項も、δP・δHには置き換えません。

## データと計算方式の確認

次の著者CSVを、固定した版から取得し、発行者のLFS SHA-256とバイト数を確認しました。

| CSV | 実際の行数 | 確認した用途 |
| --- | ---: | --- |
| `general_polymers_with_sp_abbe_dynamic-dielectric.csv` | 95,335 | MDの凝集エネルギー等の観測値 |
| `small_molecules.csv` | 28,159 | 物性・条件の一覧のみ。`sp_total/sp_vdw/sp_ele` 列がなく、この準備コマンドは拒否 |

著者カードの73,045高分子・23,361小分子という説明は、この版の実際の行数とは異なります。
高分子表には78,379種類の未正規化繰返し単位文字列があり、95,335個の独立した材料という
数え方はしていません。複数の計算・重複する構造・系列の確認が必要です。
元データの全体を1,000件のHSP実測正解として数えることもできません。

高分子表の95,335行を読み、以下のように準備しました。

- 23,481行は必要な計算方式のメタデータが欠け、準備対象から除外。
- 6行は必要な観測値が欠測または非有限で、準備対象から除外。
- 71,848行は4項すべてが有限・非負で、300 K・1 atm、GAFF2_mod、`preset_sp_ver=0.1.1`。
- 48行には、成分平均の二乗和が平均凝集エネルギー密度を上回る診断フラグ。
  最大差は0.235162 J/cm³。値を補正せず保持し、原因は生の時系列等で確認する必要があります。
- 準備した全71,848行で、第2末端の明示欄が空。推測で第1末端を複製していません。

現行の[RadonPy静的ソース](https://github.com/RadonPy/RadonPy/blob/5d14893515376a4518e9f1373a1ebc4bb756db14/radonpy/sim/preset/sp.py)を
読み、エネルギーをJ/mol、体積をcm³/molとして求める約束を確認しました。
ソースは別々に計算した平方根を時間平均し、エネルギー密度も別に平均します。
したがって平均平方根の二乗を平均エネルギーに厳密一致させる操作は行いません。
診断の1e-6 J/cm³は差を記録する計算上の閾値で、物理精度の合格基準ではありません。
現行ソースには集計範囲を固定する処理もあり、同じ版文字列だけで各行の過去の実行コード・
サンプル数・収束を証明したことにはしていません。上流コードは実行していません。

現行の[末端処理ソース](https://github.com/RadonPy/RadonPy/blob/5d14893515376a4518e9f1373a1ebc4bb756db14/radonpy/core/poly.py)には、
第2末端が未指定なら第1末端を使う規則があります。ただし材料を確定するには、各計算の
実行版・呼出し経路・Mn・DP・保存構造との照合が必要です。
この準備機能は空欄をその規則で埋めず、元の未知の状態を残します。
実際の系列・繰返し単位の切り方・末端・立体情報・相状態の確認前には学習を有効にしません。

別プロセスで95,335行の採否を再計算し、71,848行の全観測値・材料メタデータ・
診断フラグ・チェックサムが一致することを確認しました。
温度300 Kを、PoCの既定298.15 Kへ変換したという扱いもしません。

## ローカル実行

CSVの必須列は `UUID,temp,press,check_eq,smiles_list,forcefield,RadonPy_ver,preset_sp_ver`
と `sp_ced,sp_total,sp_vdw,sp_ele` です。温度はK、圧力はatm、
`sp_ced` は `J/cm^3`、他の3項は `MPa^0.5` と明示します。
`check_eq` は文字列 `True` の行だけを準備し、欠測・非有限・負の観測値や非正の条件を拒否します。
この上流の平衡フラグは独立した収束監査の代わりにはなりません。

レビューJSONには `sha256,source,declared_license,permission_evidence,approved_by,audit_basis,`
`rights_status,local_preparation_allowed,source_units,definition_evidence` が必要です。
`rights_status="approved"`、`local_preparation_allowed=true` と宣言し、使用根拠を確認します。
`audit_basis` は `documented_permission` または `operator_assumption`。
`source_units` は上記の4項と単位を正確に指定します。
`training_allowed` と `public_redistribution` は `false` に限定します。
構造と用途を確認した後の学習には別の実装・計画・許可が必要です。

```sh
uv run hansenkit md-cohesion-prepare --data local/author-md.csv --review local/md-review.json --out runs/md-observations
```

入力ファイルのSHA-256を確認し、空欄のメタデータは `null` として保持します。
出力は `observations.jsonl` と、条件・採否数・出典・ハッシュ・用途制限を記録する
`preparation.json`。出力先が存在すれば上書きせず拒否します。
CSVを順に読み込むので、Hugging Faceの自動型推論やpandasのインストールは不要です。
既存のHSP学習ローダーはこのMDのラベル種類・列を受け付けず、自動的な列名置換もありません。

観測値の種類は `computed_md_cohesion`、`hsp_predictions=null`、
`individual_polar_hydrogen_components_identified=false`。
材料同定・予約群との系列分離・物理HSPバックエンドの資格は未確認のまま保持します。
追加のHSP学習正解は0件で、1,000化合物のHSP精度目標はまだ達成していません。

## 独立実験資料の追加調査

[高分子溶解性の測定品質を比較した一次論文](https://doi.org/10.1002/marc.202500454)には、
視認とCrystal16による濁度測定、温度・濃度・分子量の扱いが記載されています。
これは条件付きの溶解性分類であり、HSP3成分そのものの正解ではありません。
公開の補足ZIPの取得は403で拒否され、実測ベンチマークにはまだ採用していません。
著者へ連絡したり、認証・アクセス制限を回避したりはしていません。

対応Issue: [#35](https://github.com/myu65/hansenkit/issues/35)。
