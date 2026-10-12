# 高分子の溶媒適合性を別タスクで学習する

2026-10-13、PolySolの公開参照二値ラベルで官能基記述子と固定MoLFormerを実際に学習・評価しました。
**今回の構造モデルは溶媒ごとの単純な基準を上回らず、実用精度を確認できていません。**
HSP3成分の学習や、未知の鎖長・末端・温度の推論には転用しません。

## 資料と系列分割

[一次論文](https://doi.org/10.1039/D4DD00290C)、
[固定版リポジトリ](https://github.com/cstubb/PolySol/tree/5d2fc43be8f547acdbe2441b805969e653244d62)と
[源資料の確認](polymer-scope-extension.md)を参照してください。
コードのMIT、論文のCC BY-NC 3.0、編集元Handbookの権利は別です。
原CSV、特徴、モデルは`audit_basis="operator_assumption"`による科学研究・私用・非商用のローカル限定です。
同梱・再配布せず、ラベルは`published_reference_polymer_compatibility_auxiliary`とします。

HSP評価Excelと現行HSP系列の全5,176別名を先に予約しました。
ホモ・コポリマー成分、既存別名、正規化同一性、材料名の接続をまとめると、
以前の52表現・194観測から予約系列につながる1表現・4観測がさらに除外されます。
今回の候補は**51源単位表現・190観測・47系列・40固定溶媒カテゴリー**です。
源表現には接続ポートがなく、材料の完全な繰返し単位・Mn・末端・試験温度は未確認です。
既知の不完全な元記録も保全し、資料全体の完全な系列独立性は認定していません。
溶媒の構造はHSP予約と重なるため、固定カテゴリーとしてだけ使い、溶媒の分子特徴を学習しません。

原47系列の28／9／10を訓練／校正用報告／テストへ固定しました。
カテゴリーは訓練系列2以上、各内側訓練分割でも2以上、校正用とテストで各1以上を必要とします。
二値ラベルを使わず、分割を変えずに全190行の適用・拒否を保存しました。
7カテゴリーだけが通過し、訓練73／133、校正用19／32、テスト18／25行が適用可能です。
残る80行も削除せず、カテゴリーの系列数不足による拒否として分母に残します。

## 固定した比較と実際の結果

- A：元素数、源単位の分子量、環、HBD/HBA、TPSAなど独自19記述子とカテゴリーのL2ロジスティック回帰。
- B：監査済みMoLFormer10%版の固定768次元表現と同じカテゴリーの回帰。
- 対照：全訓練の割合と、カテゴリーごとの訓練割合。系列・材料ペアの重みとBeta(1,1)擬似計数を使用。

A全集団とA/B共通集団を分けました。今回は全51表現で両特徴が可能で集団は一致しました。
202トークン上限で切り詰めず、欠落Embeddingを補完しません。
alpha 0.1／1／10を元訓練系列内5分割のloglossだけで選び、Aは0.1、Bは10でした。
前処理、重み、範囲外参照も各訓練部分だけから求めました。Cは初回比較で未実装です。
CSV全行を字句走査しますが、学習で二値ラベルへ変換するのは選別済み訓練73行だけです。
全190行の予測・拒否・モデルをハッシュで固定してから、別の評価操作で正解を読みました。

同じ適用可能なテスト18観測・9系列の比較です。元25観測のうち7観測は対象外です。
指標は系列・元材料ペアの重みを使います。Brierとloglossは小さいほど良い値です。

| 方式 | Brier | logloss | Balanced accuracy |
| --- | ---: | ---: | ---: |
| 全訓練の割合 | 0.239390 | 0.671883 | 0.500000 |
| 溶媒カテゴリーごとの割合 | 0.097649 | 0.342577 | 0.868421 |
| A：19記述子 | 0.233043 | 0.654936 | 0.543233 |
| B：固定MoLFormer | 0.255958 | 0.700228 | 0.319549 |

構造特徴の利点を確認できませんでした。訓練特徴の各列の最小・最大から外れるテスト行は
Aで4／18、Bで18／18です。校正済み不確実性や材料の適用資格とは区別します。
校正用19行でもA/BのBrierは0.237018／0.291946、カテゴリー対照は0.113937でした。
確率校正・被覆率は認定していません。選別後の190行の重複超過・矛盾ペアは各0で、
元資料全体の重複・矛盾を修正した結果ではありません。
保存したAモデルは約22 kB、Bヘッドは約194 kBで、固定エンコーダ約187 MBは別です。
これらの外部資料から学習したローカルモデルは公開しません。

| 保全した証拠 | SHA-256 |
| --- | --- |
| 実行前の固定計画v2 | `5b37b10f8d67d3101a5176db1588a2b2d7b592dbaeb523d331923f936913d347` |
| 正解評価前のモデル・予測固定 | `dcd8f89366657484b76b8f060433c645c6ffa57a847adef17317120b77e8d3ef` |
| 別操作の評価報告 | `ff00dff9620497653e87b724601527d346a04a6a2c7a90cf842a759e06db949b` |

学習前レビューで実行権限の個別フラグ矛盾を見逃す旧実装を検出し、記録を保持したv2で
初めて実データを学習しました。自作19テストは権限の早期拒否、全190行の保全、
B全件拒否でもAが継続すること、訓練系列内選択、勾配、モデル復元を含みます。
非実行の凍結用メタデータ生成スクリプトには行長2件の指摘が残り、数値コードと区別します。

保存復元の独立NumPy計算では、全190行・3モデルのlogit最大差が5.55×10⁻¹⁶でした。
元CSVや正解を読み直さず、全予測・拒否・範囲外マスクと出典が保たれたことを確認しました。
独立記録のSHA-256は`8c7994eec678f32849e862b6019c4bbe485f1a0cd8f891453033301973439981`です。
実測精度の独立再検証とは区別します。

追加の訓練系列内診断でも、カテゴリー対照のOOF loglossは0.442503、
全訓練対照は0.709716で、A/Bの選択時OOF値0.641089／0.687426より良い結果でした。
同じ訓練73行・元5分割だけで集計し、校正・テストの正解や予測を読み直さず、
既存の構造モデルや設定を変更していません。
訓練診断のSHA-256は`4df01b8b0581f6f3f81082fb4a833c391c27e32360c66f133d338a3dd0e7df3d`です。

次の比較も訓練系列内で計画と選択を固定し、今回のテスト結果で設定を選び直しません。
実用化には材料構造・温度・鎖長・末端・観察条件を対応づけた資料と新しい系列検証が必要です。
対応Issue：[#92](https://github.com/myu65/hansenkit/issues/92)。

## 自分で確認した配列を使うAPI

`CompatibilityClassifier`は配列、元行ID、材料ペアID、系列ID、固定溶媒IDと権利宣言を受け取ります。
元CSVの読込み、構造の組立て、外部Embeddingの取得は行いません。
呼出側で正規化・完全な系列予約を行い、訓練用の配列だけを`select`へ渡してください。
各溶媒は各訓練分割で2系列以上を必要とし、満たさない候補を拒否します。
校正・テストの配列やラベルを選択へ渡す引数はありません。

以下はこの文書のために作った架空の配列です。実際の材料や物性を表しません。

```python
import hashlib
import numpy as np
from hansenkit.compatibility import (
    CompatibilityClassifier,
    CompatibilityProvenance,
    CompatibilityRow,
)

rows = [
    CompatibilityRow(f"r{i}-{j}", f"f{i}", f"p{i}", f"s{j}") for i in range(10) for j in range(2)
]
x = np.array([[i / 10, j] for i in range(10) for j in range(2)], dtype=float)
y = np.array([(i + j) % 3 == 0 for i in range(10) for j in range(2)], dtype=float)
own_bytes = x.astype("<f8").tobytes() + y.astype("<f8").tobytes()
provenance = CompatibilityProvenance(
    label_kind="synthetic_polymer_compatibility",
    dataset_id="authored-demo",
    source="Arrays authored in this example",
    source_sha256=hashlib.sha256(own_bytes).hexdigest(),
    license="MIT",
    rights_status="approved",
    permission_evidence="Original authored fixture",
    approved_by="Fixture author",
    audit_basis="documented_permission",
    training_allowed=True,
    derived_weights_allowed=True,
    local_evaluation_allowed=True,
    measurement_lineage="authored",
    input_representation="authored_array_fixture",
)
model = CompatibilityClassifier.select(x, rows, y, provenance=provenance, folds=3)
restored = CompatibilityClassifier.from_json(model.to_json())
result = restored.predict([[0.5, 1], [0.5, 1]], ["s1", "unseen"])
print(result["valid"].tolist())  # [True, False]
```

保存する場合は`to_json()`の文字列を無視対象の`local/`へ置きます。
復元時に権利、教師種、前処理、訓練指紋、収束条件、数値依存の版を検査します。
予測結果にも出典・教師種・利用条件とモデルハッシュが残ります。特徴列の意味と順序は呼出側で学習時と揃えます。
未知の溶媒や非有限入力には数値を返さず、各行に理由を残します。
`valid`は有限計算と既知カテゴリーの状態で、物理精度や材料適用の認定ではありません。
`OOD`も訓練特徴の範囲診断です。通常のHSP予測の適用範囲は変更しません。
