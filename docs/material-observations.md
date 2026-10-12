# 材料gradeごとの溶解観測を照会する

`material-lookup`は、レビュー済みのローカル台帳から、指定した材料gradeと溶媒の
報告済み観測を返します。高分子の溶媒候補を検討するときに、定性観測、試験条件、
原資料の行番号を一緒に確認するための機能です。モデルの実行やHSPの推算は行いません。

入力は原資料の`material_id`と、任意の`solvent_name_as_reported`に限定します。
SMILES、DP、末端、Mn、温度から別のgradeへ対応づける処理はありません。
未知の条件は`null`のまま返し、MwをMnへ変換しません。記載された濃度、試料質量、
添加溶媒量はそれぞれ保持し、最終溶液体積が不明な試験の濃度を計算し直しません。

## ローカルで使う

元資産、準備時のレビュー、観測台帳をローカルに保存し、
[別の照会レビュー](../examples/material-lookup-review.template.json)を作ります。
準備や評価の許可だけでは照会できません。元資産と準備レビューのハッシュ、
台帳のハッシュ、源ファイルごとのgradeと行番号の範囲を確認し、
`read_only_lookup_allowed=true`を含む資産固有の許可を明示してください。
相対的な元資産・準備レビューのパスは、この照会レビューの位置から解決します。

```sh
uv run hansenkit material-lookup --data local/material-observations.json --review local/material-lookup-review.json --material-id SOURCE_GRADE_ID --solvent "SOURCE SOLVENT NAME" --output runs/material-evidence.json
```

`--solvent`を省略すると、そのgradeの全観測を返します。照合は原文字列との完全一致です。
一致がなければ`no_reported_observation`と空の`records`を返します。
出力ファイルは上書きせず、標準出力には件数と状態だけを表示します。

Pythonからは`SourceObservationLookup(data_path, review_path).lookup(query)`を使えます。
`query`は`{"material_id": "SOURCE_GRADE_ID", "solvent_name_as_reported": "SOURCE SOLVENT NAME"}`
の形です。未指定の試験条件を、照合済みの条件とみなしません。

## 観測の読み方

同じgrade・溶媒に複数の行があれば、全て返します。二値ラベルの食い違いは
`same_query_key_outcome_disagreement`に記録し、多数決や平均によって一つの回答にしません。
未知の条件がある場合、この食い違いだけから物理的な矛盾とは判断できません。
部分溶解やコロイドを含む定義と元の定性注記も保持します。

各結果に、問い合わせ、台帳、照会レビュー、元資産、準備レビューのハッシュと
元の`label_kind`・`allowed_role`が残ります。元の観測が候補台帳に分類されていれば、
照会してもその資格は変わりません。常に`model_execution=false`、
`hsp_predictions=null`、`physical_accuracy_validated=false`です。
台帳にHSP数値が入った行や、レビュー外のgrade・源行は拒否します。

[PolyCharの4材料・64観測](polymer-scope-extension.md)は、この用途のローカル候補です。
原資産は[著者のZenodo記録](https://zenodo.org/records/16536864)、
測定法は[一次論文](https://doi.org/10.1039/D5DD00219B)を確認してください。
源の二値定義は、正が溶解またはコロイド状の溶解、負が部分溶解または不溶です。
記載されたMwの4gradeに対する報告を、任意の末端・鎖長やEO/PO材料へ広げません。

外部の台帳・元資産・結果は同梱しません。現在の照会は科学研究・私用・非商用の
ローカル用途に限定し、学習、派生ラベル、公開重み、資産の再配布を許可しません。
公開テストはプロジェクトで作った架空の観測だけを使用します。
通常のHSP推算の適用範囲は、この照会とは別に検証します。

対応Issue: [#86](https://github.com/myu65/hansenkit/issues/86)。
