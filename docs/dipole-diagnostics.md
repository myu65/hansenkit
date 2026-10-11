# 双極子時系列の条件付き診断

`hansenkit.dipole_diagnostics`は、総双極子ベクトルと体積の配列から
母共分散、平均体積、固定電荷モデルの条件付き応答を返します。
NumPyだけの配列APIです。座標・電荷・ファイルを読み込まず、MDやHSP推算も行いません。

```python
import numpy as np
from hansenkit.dipole_diagnostics import fluctuation_moments, diagnose_dipole_trace

# 独自の人工配列。実測値・MD結果ではない。
rng = np.random.default_rng(1187)
M_e_nm = rng.normal(0, 0.03, (256, 3))
volume_nm3 = 4 + rng.uniform(0, 0.1, 256)
moments = fluctuation_moments(M_e_nm, volume_nm3, configured_temperature_k=310)
report = diagnose_dipole_trace(
    np.arange(256) * 0.2,
    M_e_nm,
    volume_nm3,
    configured_temperature_k=310,
    block_count=16,
)
```

入力は有限の実数`M`（N×3、e·nm）、正の体積（N、nm³）、明示した設定温度（K）です。
単純なモーメントは2点以上、時系列診断は32点以上で、4個以上の等サイズブロック、
各8点以上が必要です。既定の16ブロックには128点以上が必要です。
時刻は等間隔・狭義単調増加に限定し、端数・不正値を捨てず拒否します。

共分散は`mean[(M−mean(M)) outer (M−mean(M))]`、分母はNです。
ベクトルを中心化し、双極子の大きさの分散には置き換えません。
SIへ変換した共分散のトレースから、導体境界を仮定した記述量
`1 + trace(Cov(M_Cm)) / (3 ε0 kB Tconfigured mean(V_m3))`を求めます。
NPTの体積はアンサンブル平均を分母に使い、各フレームの`1/V`で重み付けしません。
この規約は[Beauchampら（2015）、式3](https://doi.org/10.1021/acs.jpcb.5b06703)と
[Fingerhutら（2021）、節8・式44–45](https://doi.org/10.1016/j.cpc.2021.107860)を参照しています。
定数は[NISTの2022 CODATA](https://physics.nist.gov/cuu/pdf/all.pdf)に基づきます。
原著のコード・数表・測定データは同梱していません。

時系列ではMの3成分、M²、中心化したM²、体積、共分散/平均体積比の一次影響列、
中心化したテンソルの6成分、計13列を既存の[MD診断](md-diagnostics.md)へ渡します。
全点を保持し、固定ブロックと前後半は各区間の平均で共分散を中心化します。
短い区間が遅い配向変化を見落とす共分散バイアスは補正せず、結果と前提を保持します。
反復を連結せず、独立した反復は呼出側が個別に扱います。

分子の中性、周期境界での原点、電荷と単位の出典、導体境界、温度・体積の
アンサンブル、等方性は呼出側で確認する必要があります。APIは確認していません。
基準値1は固定電荷の規約で、誘起電子分極や光学応答を追加しません。
有限サイズや相関の影響、収集時間の十分さ、区間被覆率、力場精度も未検証です。
一定値のESSは未推定のままで、十分な混合の証拠にはしません。

出力の適用範囲・アンサンブル・平衡・物理精度・被覆率・学習資格は全て未認定です。
`hsp_predictions=None`、`experimental_dielectric_prediction=None`、`training_allowed=False`を保持します。
実測誘電率との比較、δP/δHへの分解、モデル学習の許可は与えません。

対応Issue: [#57](https://github.com/myu65/hansenkit/issues/57)。
