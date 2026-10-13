# hansenkit

MIT志向の、HSPiPに依存しないHansen溶解度パラメータ推算のPython研究プロジェクト。
SMILESから独自の化学特徴量と構造表現を組み合わせ、将来は高分子・EO/PO分布・GPU物理計算へ拡張します。

**開発中。初期PoCは人工的に作る合成ラベルだけを使用し、実測HSPの精度は検証していません。**
δD・δP・δHの出力を実験値・製品適合性の根拠に使わないでください。
権利未確認の教師データ・係数表・外部重みを同梱しません。

**2026-10-12のローカル参照評価では、1,013化合物で各成分の点MSEがHSPiT比較値以下になりました。**
[757行への学習拡張](docs/mse-reference-experiment.md)に続き、
[現行HSP系列を避けた53,670分子のQM9転移](docs/expanded-hsp-auxiliary-refit.md)で、
δD／δP／δH MSEは1.472／11.134／7.219、比較値は1.786／16.837／13.721でした。
Excelは評価専用のまま、設定は学習系列内で選びました。これは系譜不明の公開参照値との比較です。
独立実測の精度、OODのδD、高分子への適用や校正済み不確実性は未確認です。
主方式にはδPの負値6件が残ります。[森林・応答変換の比較](docs/positive-response-comparison.md)に続き、
[学習する非負応答](docs/learned-nonnegative-response.md)でMSE 1.484／11.183／7.158、
評価1,013行の負値・数値拒否0件を確認しました。旧主方式は変更していません。
新方式もOODのδDは比較値を超え、独立実測・高分子・活性剤の実用資格は未確認です。
直接xTBを5,253回計算して比較しました。現在の選択は化学特徴・固定MoLFormer・3D proxyのkernelに、
δD／δHだけ新しいQM9座標を加えた方式です。量子学習件数をHSP実測教師へ加えていません。
旧補助モデルはHSP系列拡張後に再監査し、QM9・GDB17・EGP・Abraham・高分子MDの5資料を再学習しました。
源データ・外部資産・学習済み重みは科学研究・私用・非商用のローカル限定で、Gitへ含めません。

動くものは、RDKitによる正規化、独自の官能基・原子所属特徴量、厳密な入力スキーマ、
権利宣言とチェックサムを検証するローカルCSV読込み、骨格・系列分割、3成分評価、
OOD・外挿判定、校正用データを分離した予測区間、CLIとCSV入出力です。

ローカルでの実データ研究を開始しました。ExcelとHSP評価対象の分子・骨格を先に予約し、
QM9の量子物性を別タスクとして学習する `auxiliary-fit` を追加しています。
以下の各実験は実施時点の結果です。旧少数データでの目標未達と、上記の757行による参照MSE通過を区別します。
[実験条件と未達の目標](docs/local-research-protocol.md) を参照してください。
補助学習の件数や既存手法の比較結果は、hansenkitの実測HSP精度を証明するものではありません。
`chain-features` で繰返し単位・末端基・Mn・EO/POの構造と質量を検証できるようになりました。
[長鎖の計算手順](docs/chain-features.md) を参照してください。構造計算はHSP精度の検証とは区別します。
[追加高分子資料](docs/polymer-scope-extension.md)で2,088溶解性観測と4材料×16溶媒の64観測を確認しましたが、
HSP3成分や未知の末端・Mnへ置き換えません。[二量体源データ](docs/dimer-auxiliary-intake.md)も
両モノマーを予約と照合し、採用可能な系列数が不足しているため学習していません。
[高分子の溶媒適合性を別に学習しました](docs/polymer-compatibility.md)。予約系列を避けた190観測を保持し、
適用可能な訓練73行で官能基と固定MoLFormerを比較しました。別系列の18観測では、
どちらも溶媒ごとの訓練割合による基準に届きませんでした。通常の高分子HSP推算は未認定です。
[構造補正の追加比較と原資料の確認](docs/polymer-development-followup.md)を実行しました。
訓練系列内で7候補を比較しても構造補正は対照に届かず、最終選択は基準値だけでした。
χ、CMT、CMC、画像からの溶解分類とHSP3成分の教師を区別し、原表の未解決箇所を保持しています。
[三方式の追加学習・新しい高分子参照資料・MDの全反復診断](docs/polymer-research-followup.md)も実行しました。
構造補正は系列分割で基準を改善せず、単一δ/CEDの53候補系列と911行の注釈CSVを別に確認しています。
MDは3試料各3反復・全73,728点でエタノールだけ統計基準を通過し、PEG・EO/POは未通過です。
参照データの候補数やMDの統計通過を、高分子HSPの実用資格へ置き換えていません。
[単一δ/CEDの系列を分けた回帰](docs/scalar-polymer-reference.md)では、単一δは別10系列15行で
系列等重みRMSE3.772、訓練平均基準4.751でした。CEDは基準より悪化しました。
二つの原宣言単位・教師種を保ち、HSP3成分へ分解したり実用資格へ置き換えたりしません。
[固定MoLFormerと残差補正の追加比較](docs/polymer-fixed-embedding-comparison.md)では、
同じ単一δの系列等重みRMSEがBで2.176、Cで2.647となり、化学Aの3.772を改善しました。
CEDはB/Cとも平均値基準より悪く、保留構造の全768列OODも未校正です。
既に見た参照評価での開発比較であり、高分子・活性剤のHSP3成分の実用資格は未確認です。
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
[MDの凝集エネルギー観測値の準備](docs/md-cohesion.md)も追加しました。
PolyOmicsの71,848計算行を単位・条件付きで準備・照合しましたが、電気項からδP/δHを分離できず、
材料・系列の資格確認前なので学習は無効です。HSP3成分の教師データとして数えていません。
[周期系列を分けた計算凝集物性の学習](docs/periodic-cohesion-learning.md)を追加しました。
末端規則とDP/Mnを照合した17,833計算行でA/B/Cを学習し、分散項はB、電気項はCがAより低いMAEでした。
この学習は別のローカル許可に限定し、δP/δHの分離・実測HSP精度の証明には使っていません。
[レビュー済み材料のMD参照推論](docs/md-cohesion-inference.md)では、再学習したA/B/Cを
`md-cohesion-predict`から利用できます。出典のUUID・繰返し構造・DP・Mn・立体規則、
明示した両端メチルと計算条件を照合し、保存済みの同じ特徴・固定ベクトルだけを使います。
同じ構造の複数DP観測はないため、全体のDP範囲内でも未知の鎖長や末端への数値推論を拒否します。
出力は分散・電気の計算参照2項です。高分子・EO/POのHSP3成分の適用資格はまだありません。
[MDの固定構造応答を小分子HSPへ転移した比較](docs/md-response-transfer.md)も実施しました。
同じ1,013構造でδDは全方式で悪化し、3成分の非劣性は未達です。周期材料からの適用範囲は広げません。
[現在のExcel予約でのHSP3成分学習](docs/current-hsp-fewshot.md)も検証しました。
11件の公開参照値で学習し、別の1,013構造を評価しましたが、全3成分の非劣性は未達です。
QM9特徴追加とCはδPの基準だけを通過し、δD・δHと独立実測の裏付けが不足しています。
[実験由来の21ポリエステルと原料処方の回帰](docs/polyester-recipe-exploration.md)も比較しました。
Mnなどの追加でMAEは小さくなりましたが、強い3系列の分割では全成分のR²が負でした。
未知の配列・末端・試験温度を保持し、SMILES推算や通常の高分子HSPバックエンドには転用しません。
[学習系列だけで正則化を選ぶ処理](docs/train-only-selection.md)も追加し、同じ1,013構造で検証しました。
δHの一部は改善しましたがδPは悪化し、全3成分の非劣性は未達です。選択処理は配列用の明示的な関数です。
[Abraham参照記述子の補助学習](docs/abraham-auxiliary.md)では、HSP予約を避けた435構造で
5記述子を学習しました。推定特徴の追加でδD MAEは1.435→1.300へ下がりましたが、全3成分の非劣性は未達です。
研究・私用限定の条件を保持し、参照記述子をHSPや量子物性の学習件数へ足しません。

[OpenMMのCPU凝集エネルギー観測器](docs/openmm-cohesion-observer.md)を追加しました。
エタノール・PEG8・指定したEO/PO鎖で各2 psのMDと保存座標の独立照合を実行しました。
平衡化前の動作確認で、HSP3成分の推算・実測精度・市販高分子の適用資格は未確認です。
[MD時系列の診断](docs/md-diagnostics.md)で、自己相関・収集時間・反復間の差を調べられます。
独立配置からの追加MDを評価専用で実行し、未収束の結果を保持してチェックポイントから延長しています。
統計量が安定しても、力場精度やHSP3成分の分離を確認したことにはなりません。
[双極子の配列診断](docs/dipole-diagnostics.md)も追加しました。
共分散・平均体積・配向の相関を調べるAPIです。実測誘電率やHSPへの適用は未検証です。
[追加HSP資料の出典調査](docs/hsp-source-followup.md)では、材料の適合性観測、
公開参照値、構造推算値を分類し、既存の評価予約を保持しています。
[参照HSP学習を50構造・40系列へ増やしました](docs/reference-data-expansion.md)。
1,013構造の比較ではδHの一部が改善しましたが、δD・δPは悪化し、全3成分の非劣性は未達です。
測定系譜が不明な参照値のローカル学習で、独立実測の精度証明とは扱いません。
[同じ50構造でアーキテクチャを比較しました](docs/architecture-comparison.md)。
続けて[水素結合・電荷・3D特徴と二乗回帰の6条件](docs/physics-feature-comparison.md)を比較しました。
サイト・電荷の追加でδH MAEは2.458→2.171、正の木モデルでδDは1.680→1.097でした。
3成分すべての固定基準は未達で、3D追加・二乗回帰の全体の改善も確認できませんでした。
補助学習はHSPの全系列を避けて再監査し、Excelは評価専用、独立実測の資格確認は未完了です。
結合Ridge・RBF・ExtraTrees・MLPを試し、参照評価でRBFのδPだけが非劣性基準を通過しました。
全3成分の精度目標、MLPの収束、独立実測の裏付けは不足しており、通常の適用範囲は広げません。
新しく資格を確認した独立実測の分子HSP学習ラベルは、今回の調査では0件です。
[18件・15系列への参照学習の拡張](docs/expanded-reference-head.md)も実行しました。
固定MoLFormerを含む4方式の1,013構造評価は独立再計算と一致しましたが、全成分の非劣性は未達です。
同じ18件で学習系列内の設定選択も比較しました。CのδHは改善しましたが、今回も全3成分の基準を満たしていません。
[NIST ThermoMLの補助物性調査](docs/thermoml-auxiliary-review.md)では固定アーカイブを照合し、
評価予約を避けた密度248点・21分子・16系列を候補として確認しました。
系列分割でA/B/Cを比較し、測定点と分子・系列の重み付けで結果が変わることを記録しています。
密度・蒸発熱はHSP3成分の正解ではなく、一次数表との独立照合や物理的な適用資格は未確認です。

| 方式 | 初期PoC |
| --- | --- |
| A | 官能基＋記述子＋Ridge回帰。LightGBMは追加オプション |
| B | 固定Morgan表現、または確認済みMoLFormerの固定Embedding＋回帰 |
| C | 官能基の線形寄与＋骨格群を分離して作った残差の構造表現による補正 |

追加の`SoftplusKernelHead`は、許可済み・系列分離済みの配列を受け取る研究用Python APIです。
化学105特徴・固定768次元構造表現・物理特徴から非負応答を学習し、保存復元時も数値条件と
出典宣言を検査します。通常CLIの合成PoCや高分子等の適用判定は変更しません。
外部重み・データの読込みや自動取得を行いません。[方式と比較結果](docs/learned-nonnegative-response.md)を参照してください。

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
イオン性分子・混合物への推算はまだ拒否します。QM9はローカル補助学習とHSP転移を実装しています。
Uni-Mol2、MiniMol、CheMeleon、RadonPy、GPU-MD、能動学習は今後の接続候補です。
OpenMMはCPU観測器のみ追加しています。

開発方針は [AGENTS.md](AGENTS.md)、[Codexで続ける手順](docs/codex.md)、
[ロードマップ](docs/roadmap.md)、[実験条件と合成データの結果](docs/evaluation.md)、
[権利・出典台帳](docs/rights-ledger.md)、[構成](docs/architecture.md) を参照してください。
次の本格実験は、権利を確認した中性小分子の独立実測データでAを評価し、
その後に確認済みMoLFormerを同じ分割でB/Cと比較することです。

コードと独自の文書はMIT。依存ソフト・重み・数表・教師データの条件は別々に管理します。
