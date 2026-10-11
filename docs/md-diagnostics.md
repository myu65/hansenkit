# MD時系列の収集不足を調べる

`hansenkit.md_diagnostics`は、等間隔に保存した観測値について、ドリフト、
自己相関、有効サンプル数、ブロック平均のばらつきと反復間の差を返します。
NumPyだけを使う小さなPython APIで、MDの実行・平衡区間の選択・HSP推算は行いません。

```python
import numpy as np
from hansenkit.md_diagnostics import diagnose_trace, diagnose_replicates

# 独自に作った人工時系列。物理モデルや実測データではない。
time_ps = np.arange(1024) * 0.2
rng = np.random.default_rng(81)
first = {"density": rng.normal(0.8, 0.01, len(time_ps))}
second = {"density": rng.normal(0.81, 0.01, len(time_ps))}
one = diagnose_trace(time_ps, first, block_count=16)
separate = diagnose_replicates(
    {"seed_1": (time_ps, first), "seed_2": (time_ps, second)}, block_count=16
)
```

## 入力と返却値

時刻は有限・狭義単調増加・等間隔、各観測値は同じ長さの有限実数列に限定します。
32点以上、4個以上の等サイズブロック、1ブロック8点以上が必要です。
端数を切り捨てず、不正な入力を拒否します。時間・観測値の単位は呼出側が管理します。

平均、標準偏差、前後半の平均差、全区間の線形傾向、ブロック平均と分散を返します。
自己相関から統計的非効率度とESSを推定し、ESSは1..保存点数に制限します。
一定値や数値的に分散を解決できない列では、ESSと平均の標準誤差を`None`にします。
変動がないことを十分な混合やゼロ不確実性の証拠にはしません。

各反復は個別に解析します。長さや保存間隔が異なる反復も受け付けますが、
末尾と先頭を連結したりESSを足したりしません。
反復平均の等重み平均と反復間の分散・条件付き標準誤差を別に返します。
独立した初期状態と同一の材料・条件は呼出側の責任で、APIは確認していません。

## 統計量の前提

自己相関の正のペア列を単調化する方法について、
[Stanの一次資料](https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size)
に記載された式を独自に実装しています。Stanの多鎖推定器ではありません。
ESSと平均誤差には定常性・有限分散、ペア推定法には可逆性の前提があります。
特定のMD積分器や観測量で前提が成立するかは別に確認が必要です。
ブロック平均の誤差には、ブロックが十分長く、ほぼ独立であることも必要です。

探索用の警告は前後半差が0.5標準偏差超、傾向が1標準偏差超、ESSが20未満、
ブロック長が非効率度の10倍未満などです。物理的な合格判定ではありません。
全出力は`equilibrium_qualification="not_assessed"`、
`physical_accuracy_validated=False`、`hsp_predictions=None`を保持します。
小さいドリフトや大きいESSだけで平衡・力場精度・HSP適用範囲を認めません。

## ローカル実験との関係

各2 psの[CPU動作試験](openmm-cohesion-observer.md)では、平衡化と収集時間が不足していました。
追加の評価専用実験では独立した配置を各3個用意し、298.15 K / 1 atmで、
100 psのburn-in後に各102.4 psを収集しました。全てHSP評価予約の系列で、学習には使いません。
条件・コード・パラメータ・解析ゲートを実行前にハッシュで固定し、
未通過時の延長はチェックポイントから同じ条件で行います。
以前の収集行・失敗・座標・速度・状態を上書きせず、旧行をそのまま残します。

初期のエタノール液相3反復は温度の平均と反復間の平均差の基準を通過しましたが、
密度・エネルギーの相関に対してブロックが短く、収集時間の基準は未通過でした。
保存点数を独立な教師データの件数には数えません。
延長結果と物理照合の状態は、確定した集計だけを追記します。

公開APIはCPU/Referenceのフレーム観測器と、この統計処理だけです。
ローカルの積分には明示したOpenCLデバイスを使い、同じ保存座標の標準力・力ベクトルを
Referenceと照合しました。[OpenMMのプラットフォーム資料](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html)
に従いmixed精度を明示しました。数値照合の通過は物理精度の資格ではありません。
OpenCLはLGPL、ドライバ・SDK・力場の条件は個別に記録し、資産・軌跡・重みを公開しません。

対応Issue: [#51](https://github.com/myu65/hansenkit/issues/51)。
