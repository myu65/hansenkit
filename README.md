# hansenkit

MIT志向の、HSPiPに依存しないHansen溶解度パラメータ推算のPython研究プロジェクト。
SMILESから独自の化学特徴量と構造表現を組み合わせ、将来は高分子・EO/PO分布・GPU物理計算へ拡張します。

**開発中。初期PoCは人工的に作る合成ラベルだけを使用し、実測HSPの精度は検証していません。**
δD・δP・δHの出力を実験値・製品適合性の根拠に使わないでください。
権利未確認の教師データ・係数表・外部重みを同梱しません。

動くものは、RDKitによる正規化、独自の官能基・原子所属特徴量、厳密な入力スキーマ、
権利宣言とチェックサムを検証するローカルCSV読込み、骨格・系列分割、3成分評価、
OOD・外挿判定、校正用データを分離した予測区間、CLIとCSV入出力です。

ローカルでの実データ研究を開始しました。ExcelとHSP評価対象の分子・骨格を先に予約し、
QM9の量子物性を別タスクとして学習する `auxiliary-fit` を追加しています。
[実験条件と未達の目標](docs/local-research-protocol.md) を参照してください。
補助学習の件数や既存手法の比較結果は、hansenkitの実測HSP精度を証明するものではありません。
`chain-features` で繰返し単位・末端基・Mn・EO/POの構造と質量を検証できるようになりました。
[長鎖の計算手順](docs/chain-features.md) を参照してください。構造計算はHSP精度の検証とは区別します。
[参照データの構造照合](docs/identity-audit.md) では、1,210識別子の一括照合、
Excel評価予約の拡張、量子補助評価の再集計、関連する2026年の公開リポジトリを記録しています。
`solquest-prepare` で、ローカルの計算溶媒和データを予約骨格・重複・単位を検査して準備できます。
[溶媒相互作用の補助学習](docs/solvation-auxiliary.md) を参照してください。HSPの実測正解とは区別します。
[公開実測データの候補](docs/public-measurement-candidates.md) に、実験で求めた高分子・医薬分子の
資料と、既存表を再編集した資料の違い、採用前の確認事項を記録しています。
[互変異性体と骨格の分割](docs/tautomer-isolation.md) を強化しました。
旧EGP・GDB17モデルの予約群・分割間の重なりを監査し、新しい条件で再学習しました。
GDB17の計算溶媒和ではハイブリッドのMAEが官能基回帰より11.5%低く、EGPでは改善を確認できませんでした。
[QM9も旧学習集団を監査し、61,622分子で再学習しました](docs/qm9-current-policy.md)。
新しいGDB17モデルをEGPへ転移すると全方式・全39溶媒のR²が負で、区間被覆率も不足しています。
[溶媒を固定した参照溶解度の補助学習](docs/fixed-coordinate-solubility.md)では、
263化合物の17,129測定行でA/B/Cを比較しました。B/Cの改善は示せず、HSPの学習・精度証明には使っていません。
[許可済み係数による参照計算](docs/reference-gc.md)を通常の予測と分けて追加しました。
新しい論文の6基団に完全対応する112件を評価し、論文例の再現と長鎖の計算一致を確認しました。
1,030件のうち918件は拒否し、独立実測の精度保証・1,000化合物の目標は未達です。

| 方式 | 初期PoC |
| --- | --- |
| A | 官能基＋記述子＋Ridge回帰。LightGBMは追加オプション |
| B | 固定Morgan表現、または確認済みMoLFormerの固定Embedding＋回帰 |
| C | 官能基の線形寄与＋骨格群を分離して作った残差の構造表現による補正 |

MoLFormerの公開10%版を版番号・ファイルハッシュ・依存バージョンを固定して確認しました。
追加オプションで有効化でき、重みを固定して回帰部分だけを学習します。
既定ではMorgan表現を使い、外部重みを自動ダウンロードしません。
[確認内容とMoLFormerの実行手順](docs/molformer-review.md) を参照してください。

Python 3.11または3.12と [uv](https://docs.astral.sh/uv/) を使います。

```sh
git clone https://github.com/myu65/hansenkit.git
cd hansenkit
uv sync --locked
uv run hansenkit synthetic --out runs/demo-data
uv run hansenkit compare --data runs/demo-data/synthetic.csv --manifest runs/demo-data/manifest.json --out runs/demo-models
uv run hansenkit predict --model runs/demo-models/A-ridge.json --input examples/predict.csv --output runs/demo-predictions.csv
uv run pytest
```

`runs/demo-models/comparison.json` に同じ骨格分割でのMAE/RMSE/R²、OOD・外挿別の指標、
区間幅と被覆率が出ます。`runs/demo-predictions.csv` は適用外の行を理由付きで拒否し、
数値欄を空にします。例のエタノール・ベンゼンは学習データに含まれる場合があり、
CSVの例は動作確認用です。精度評価は独立したテスト分割のレポートで行います。
出力先が既にある場合は上書きせず、新しい名前を指定します。

LightGBMも比較する場合は `uv sync --locked --extra lightgbm` を実行し、`compare` に
`--include-lightgbm` を追加します。

```sh
uv run hansenkit features --smiles "CC(=O)OC"
uv run hansenkit schema --output runs/input-schema.json
uv run hansenkit validate --input examples/polymer.json
uv run hansenkit train --data runs/demo-data/synthetic.csv --manifest runs/demo-data/manifest.json --mode C --model models/hybrid.json --report runs/hybrid-report.json
```

正式な教師データを使う際は [権利宣言の雛形](examples/cleared-manifest.template.json) と
[データ方針](docs/data-policy.md) を読み、許諾確認済みのデータを `local/` に置いてください。
必須CSV列は `sample_id,smiles,delta_d,delta_p,delta_h,label_kind`、単位はMPa^0.5です。
任意列は `temperature_k` と `polymer_series`。初期温度は298.15 Kに限定します。
真のテスト用データは `allowed_role="evaluation_only"` と宣言すると、学習への使用を拒否します。
HSPiTのExcelの分類と、評価用の分子・骨格を学習から除外した記録は
[Excelの確認記録](docs/hspit-classification.md) にあります。係数表を実測正解とは扱いません。
外部の独立評価は `hansenkit evaluate --model ... --data ... --manifest ... --report ...` で実行し、
教師再現性と実測評価を別レポートに記録します。

高分子・EO/PO分布・末端基・Mnのスキーマを用意しましたが、高分子・長鎖EO/PO活性剤・
イオン性分子・混合物への推算はまだ拒否します。Uni-Mol2、MiniMol、CheMeleon、QM9、
OpenMM/RadonPy、GPU-MD、能動学習は今後の接続候補です。

開発方針は [AGENTS.md](AGENTS.md)、[Codexで続ける手順](docs/codex.md)、
[ロードマップ](docs/roadmap.md)、[実験条件と合成データの結果](docs/evaluation.md)、
[権利・出典台帳](docs/rights-ledger.md)、[構成](docs/architecture.md) を参照してください。
次の本格実験は、権利を確認した中性小分子の独立実測データでAを評価し、
その後に確認済みMoLFormerを同じ分割でB/Cと比較することです。

コードと独自の文書はMIT。依存ソフト・重み・数表・教師データの条件は別々に管理します。
