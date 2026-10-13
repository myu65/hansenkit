# Roadmap

## M0 — Rights-clean runnable PoC

- RDKit normalization; original SMARTS features; explicit atom ownership and coverage.
- Strict molecule/polymer/EO-PO schema; refuse unsupported prediction types.
- A: chemical features + ridge (optional LightGBM); B-proxy: frozen Morgan fingerprint + ridge;
  C-proxy: chemical linear contribution + group-cross-fitted structural residual correction.
- Licensed pinned frozen MoLFormer is now available as an explicit optional local adapter;
  default Morgan path remains lightweight. Real checkpoint training on synthetic labels is recorded.
- Scaffold/polymer-family splitting, 3-component MAE/RMSE/R², extrapolation/OOD and uncertainty diagnostics.
- Cleared local CSV ingestion; deterministic synthetic CLI/CSV demonstration; tests and CI.

## M1 — Independent experimental small-molecule benchmark

- [拡張した全HSP予約による補助モデルの再監査・再学習](expanded-hsp-auxiliary-refit.md)で、
  QM9 53,670構造、GDB17 95,787構造、EGP 1,075構造、Abraham 222構造、
  PolyOmics 15,940計算行をそれぞれ別の教師種・レビュー・系列分割で学習した。
  全旧学習衝突を再fitで修復し、全5資料の独立再計算が一致。
  新しいQM9転移を学習内でδD/Hへ採用し、1,013参照構造の主MSEは
  1.472484／11.133871／7.219472。HSP教師は757参照行のままで、実測や高分子の資格は未達。
  [#71](https://github.com/myu65/hansenkit/issues/71)。

- [757行・746構造・402系列への拡張と直接電子物性のMSE比較](mse-reference-experiment.md)で、
  変更していない1,013構造の3成分の点MSEが全てHSPiT比較値以下となった。
  ユーザーが指定した参照MSE目標は達成。53候補の選択は内側系列分割だけで行い、
  xTB条件を評価後に成分別採用する操作はしない。δDのbootstrap優越、OODのδD、
  独立実測・市販材料・不確実性校正は未確認で、M1の物理的資格を達成したとは扱わない。
  次は現データ・分割を固定したデータ量／kernel／proxyの寄与分離、
  予約両モノマーを避けた非共有結合補助データの資格確認、別系列での外挿検証。
  [#69](https://github.com/myu65/hansenkit/issues/69)。

- [Physics-feature and squared-target comparison](physics-feature-comparison.md) tests site/charge,
  frozen quantum/reference descriptor coordinates and bounded conformers on the same50 structures.
  Every HSP family is reserved before refitting the auxiliary descriptor model. Six fixed cases
  use nested training-family selection before the exposed1013-reference comparison; no physical
  backend, measured-data clearance or material inference follows from these calculations.
  Site/charge augmentation lowers H-bond reference MAE and the positive direct forest
  lowers dispersion MAE; squared targets worsen all components. All-three fixed
  noninferiority remains unmet. Independent native replay agrees.
  Tracked in [#67](https://github.com/myu65/hansenkit/issues/67).

- Clear a small neutral-molecule measured dataset and publish its provenance/allowed uses.
- [Additional primary-source qualification](hsp-source-followup.md) distinguishes material
  compatibility outcomes and isolated published references from measured molecular HSP.
  New independent measured training rows remain zero; the exact local reference-head
  comparison is tracked in [#53](https://github.com/myu65/hansenkit/issues/53).
- The separately reviewed [18-row / 15-family reference fit](expanded-reference-head.md)
  uses fresh uniform CPU embeddings and the unchanged 1,013-structure evaluation. All
  four models fail every component's fixed numerical screen; independent replay agrees.
  More source-resolved, representative calibration data remain necessary.
  The finite follow-up applies the existing train-only selection to those same 18 rows;
  all 12 component screens fail again. No test-based refit or scope expansion follows.
- Pre-register scaffold holdouts, units, temperature handling and calibration; compare A/B/C over seeds.
- Exact MoLFormer release review and frozen adapter are implemented; evaluate B/C on cleared measurements.
- Report teacher reproduction separately if an authorized teacher is used. Never replace real accuracy with it.
- [Opt-in six-group source-reference calculation](reference-gc.md) reproduces a new primary article's
  examples and covers 112/1,030 reference structures; it is not independent measured validation.
  Current Excel reservation leaves 11 structure candidates in 10 families, still awaiting measurement qualification.
- [Current-policy few-shot HSP reference fit](current-hsp-fewshot.md) learns those 11 records
  locally and compares 1,013 other structures. C and QM9 augmentation pass the numerical
  polar criterion only; all methods fail the dispersion/H-bond criteria. Independent measured
  noninferiority remains unproven; no calibrated interval or qualified material backend follows.
- [Train-only component selection](train-only-selection.md) adds a constant/ridge family-CV
  helper and audits nested hybrid residual selection. It modestly lowers some H-bond errors
  but worsens polar transfer; the same 1,013-structure experiment still fails all-three noninferiority.
- [Abraham reference auxiliary transfer](abraham-auxiliary.md) fits 435 HSP-isolated solutes
  and evaluates 186 others. Descriptor augmentation lowers some HSP dispersion errors,
  but D/H noninferiority still fails; source commercial restrictions and missing lineage
  prevent a public/physical model claim.
- [Frozen MD structural-response transfer](md-response-transfer.md) reuses the 17,833-row
  computed-cohesion heads without guessing chain conditions. Three fixed response augmentations
  fail all-three noninferiority on the same 1,013 reference structures and worsen dispersion;
  source-domain calibration and independently measured molecular HSP remain necessary.
- [Measured-property auxiliary inventory](thermoml-auxiliary-review.md) verifies the fixed
  NIST archive and retains 248 density points in 21 graphs/16 isolated families
  as archived-measurement candidates. A separately reviewed 16-family density pilot
  compares A/B/C and distinct point/graph/family weighting. Primary table/state qualification
  remains; density/Hvap counts are not HSP truth counts. Tracked in [#55](https://github.com/myu65/hansenkit/issues/55).

## M2 — Polymers and long-chain surfactants

- [活性剤の公称組成と原試料の確認](solubilization-source-qualification.md)で、原報6gradeの公称EO数と
  メーカーの説明を照合した。現行製品のMnや掲載SMILESを2021年試料へ移さない。
  Eq.1の混合媒体M2は未解決で、3分類は系列をまとめる候補だけ。公称薬物のPubChem照合は
  既存HSP予約系列に一致し、75点は候補・評価専用のまま換算・学習しない。[#110](https://github.com/myu65/hansenkit/issues/110)。

- [活性剤の可溶化原表](surfactant-solubilization-intake.md)を全75点のローカル候補に整理した。
  原画像と独立転記を照合し、計算による理想値・HSPiP値を除外、相対不確かさをそのまま保持した。
  全件の保存復元と元条件での検索が通過。構造・系列分離と測定定義を確認するまでは学習しない。
  実測HSP3成分の追加や通常の適用範囲拡張には数えない。
  [#108](https://github.com/myu65/hansenkit/issues/108)。

- [PMMAの原観測と未知単位の保存](pmma-source-observations.md)を実装した。
  既存のg/mol値を保ち、単位未確認のMwを原文・未知単位のまま保存復元できる。
  公式SIの1試料28溶媒観察はローカル評価候補だけとして全行読込み・CLI参照した。
  未報告gradeへの一般化、溶媒HSP列・χの取込み、学習は行わない。
  次は活性剤の可溶化原観測の単位・条件・系列を確認する。HSP3成分とは別タスク。
  [#106](https://github.com/myu65/hansenkit/issues/106)。

- [密度参照の全件照合](polymer-density-isolation.md)で、全1,699行と旧623行・69系列・
  HSP予約5,176別名を保持した。暫定候補は535行・371系列で、未解決20構造は隔離したまま。
  旧学習の1系列に新しい未適用構造のつながりがあり、系列全体を隔離してA/B/Cを再学習した。
  単一δの別10系列RMSEはB2.293、C3.539、平均4.722。CEDは全方式が平均より悪い。
  数値を読む前に密度系列を分割し、訓練314行だけで選択・学習した密度Aの
  別74系列104行RMSEは0.0842 g/cm³、平均基準0.1224。保存予測と元CSV指標を別々に検査する。
  [密度推定値を20列目に加える固定比較](polymer-density-transfer.md)も実行した。
  単一δの外側訓練MSEは1.49%改善したが、別10系列RMSEは4.875→4.917と悪化した。
  CEDは訓練条件で停止し、最終モデル・保留評価を作らない。保留値で追加調整しない。
  元資料・モデルは非公開、密度と実測HSPの件数は別。通常の推算範囲は変えない。
  [#102](https://github.com/myu65/hansenkit/issues/102)。
  次に必要なのは、材料・測定条件の対応と独立なHSP3成分の検証で、密度列だけの拡張では解決しなかった。
  [#104](https://github.com/myu65/hansenkit/issues/104)。

- [PolySolの系列分割による二値適合性学習](polymer-compatibility.md)を実行した。
  全5,176予約とホモ／コポリマー系列を維持し、候補190観測のうち7カテゴリー・訓練73行でA/Bを学習。
  別系列18観測では両方式がカテゴリー別の訓練割合に届かなかった。全80拒否も分母へ残した。
  公開するのは元のMIT配列APIと自作テストで、外部資料・ローカルモデルは含めない。
  次は材料の構造・条件を対応づけた資料と、新しい系列で構造表現の利点を検証する。
  HSP3成分や実用的な高分子／活性剤の資格は未達。[#92](https://github.com/myu65/hansenkit/issues/92)。

- [カテゴリー基準値に構造補正を加える比較と原資料の確認](polymer-development-followup.md)を実行した。
  元の訓練系列内で基準値・補正・選択を作り直し、対照だけの候補も残す。
  7候補の最終選択は対照のみで、構造補正の利点を確認できなかった。外側評価の系列数不足も保存した。
  χ、CMT、CMC、原実験の溶解観測を別タスクとし、既存溶媒表や画像モデルの出力をHSP教師へ足さない。
  材料・条件の対応と新しい系列での利点を確認してから適用範囲を広げる。
  [#94](https://github.com/myu65/hansenkit/issues/94)。

- [三方式の追加学習と高分子参照資料・MDの確認](polymer-research-followup.md)で、
  共同ロジスティック、Beta事前分布を保つ相互作用、共有26係数を比較した。
  外側の学習系列ではいずれも基準を改善せず、通常の高分子HSP推論へ採用しない。
  単一δ/CEDの全623行を照合し、予約を避ける候補は53系列・177行。
  χ1,208行の構造ペアは全て予約成分に接続するため、追加学習は0行。
  画像ZIPの911行注釈CSVは原文字列を保持し、値の由来・条件は別に確認する。
  MD全73,728点の保存診断は独立再計算と一致したが、PEG・EO/POは統計基準未通過。
  次は別単位の補助学習と材料・測定系譜の確認。[#96](https://github.com/myu65/hansenkit/issues/96)。

- [単一δ/CEDの別タスク回帰](scalar-polymer-reference.md)で、両タスク共有の53系列を先に分割した。
  訓練系列内で固定6候補を比較し、単一δは別10系列15行で系列等重みRMSE3.772対基準4.751、
  R²0.352。CEDは別8系列12行で基準より悪化した。原宣言単位・未知の測定系譜を保持する。
  公開するのは独自の配列回帰・原単位特徴と自作テストで、資料・派生重みはローカルに残す。
  次は材料・出典確認と当初の固定表現比較。高分子HSP3成分の実用資格は未達。
  [#98](https://github.com/myu65/hansenkit/issues/98)。

- [高分子の固定構造表現B/C](polymer-fixed-embedding-comparison.md)を同じ共有系列分割で学習した。
  Cは各候補採点の訓練側で4系列foldの残差と5系列foldの基底選択を作り直す。
  単一δは系列等重みRMSEがA3.772からB2.176/C2.647へ改善したが、CEDは平均値基準に届かない。
  少数の同じ参照評価を使う探索的比較で、768列のOOD・材料・測定系譜は未校正／未確認。
  次は原単位・出典確認と、全対象系列を予約したうえでの密度等の補助学習の可否確認。
  高分子・活性剤HSP3成分の実用資格は未達。[#100](https://github.com/myu65/hansenkit/issues/100)。

- [追加材料資料の実際の確認](polymer-scope-extension.md)でPolySol2,088観測と
  PolyChar64観測を準備した。Mn/末端/温度/二端構造が未確定な材料へ数値HSPを出さない。
  新しい隔離MDモデルは分散・電気2項のcomputed-referenceであり、δP/δHは分離しない。
  [計算参照の推論CLI](md-cohesion-inference.md)は、出典に対応するUUID・構造・DP・Mn・
  立体規則と両端メチル・計算条件を別のhash-bound reviewで照合する。
  同じ構造の複数DP観測は0件であり、全体のDP範囲を鎖長依存の検証へ置き換えない。
  次に必要なのは材料を固定した鎖長／末端／配列の定量的な資料とHSP3成分の資格確認。
  通常のHSP拒否やMD2項の出力を、未達の高分子HSP目標の代替成果にしない。

- Cleared polymer measurements; repeat units, Mn, dispersity, end groups and EO/PO distribution.
- Two public experimental studies are inventoried as [28 material/API candidates](public-measurement-candidates.md);
  grades, methods and conditions remain to qualify before a material benchmark.
- Polymer-series and surfactant-family held-out evaluation, then an evidence-based scope expansion.
- A [21-material experimental-sphere recipe exploration](polyester-recipe-exploration.md)
  compares feed composition with Mn metadata in 11-family and stronger three-family holdouts.
  Metadata lowers MAE, but the stronger split has negative R² for all components;
  unknown product graphs/temperature/ends prevent qualified material inference.
- Ionic surfactants and mixtures need separate physical definitions and validation before numerical output.

## M3 — Optional physics and active learning

- License-reviewed Uni-Mol2, MiniMol or CheMeleon encoder; QM9 auxiliary tasks after rights review.
- OpenMM/RadonPy backend: units, temperature, force-field coverage, charge treatment, convergence,
  cohesive energy definitions and uncertainty. No GPU jobs in routine CI.
- [Descriptive MD diagnostics](md-diagnostics.md) now report drift, conditional ESS/block errors
  and separate replica spread. Local checkpoint continuations preserve unsuccessful stages;
  convergence, kinetic sampling, cell-size effects and measured density/evaporation validation
  remain prerequisites to physical/HSP qualification.
- Active learning only after calibrated uncertainty and independent validation exist.
- [Array-only dipole diagnostics](dipole-diagnostics.md) report population covariance,
  mean-volume fixed-charge response and whole/block/half correlation diagnostics;
  source, ensemble, polarization, finite-size and physical qualifications remain caller tasks.

Do not implement all future backends in M0. Each milestone should advance one reviewable experiment.
