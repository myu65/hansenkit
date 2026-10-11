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
