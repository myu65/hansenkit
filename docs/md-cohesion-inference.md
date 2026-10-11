# 観測済み源材料に限定したMD計算参照の推論

`md-cohesion-predict`は、ローカルでレビューしたPolyOmicsの源材料について、
保存済みA/B/Cモデルの`sp_vdw`と`sp_ele`を返す追加コマンドです。
この2項は源MDの計算参照値で、HansenのδD・δP・δHではありません。
通常の`predict`の高分子・EO/PO・イオン・混合物の適用範囲は変えません。
実際の新材料に対するHSP3成分の推算資格は未認定です。

## 数値を返せる材料

[再学習した集団](expanded-hsp-auxiliary-refit.md)の26,867行を、
源UUID・元の繰返し単位・DP・Mn・立体規則・特徴・固定Embeddingだけの台帳へ出力しました。
学習15,940行、校正5,308行、テスト5,619行の元順序を保持しています。
台帳にはMD教師値もHSP教師値も含めません。
すでに観測された源材料のモデル計算を利用するための台帳で、
この全件を新しい独立評価や新材料の予測結果として数えません。

入力は全欄が必須です。省略時の温度・末端・DPの補完はありません。

| 入力欄 | 必須の内容 |
| --- | --- |
| `source_uuid` | 台帳に実在する一意な源UUID |
| `repeat_unit_smiles` | そのUUIDの元の二端繰返し単位と正規化後に同一。実化学・ポート・立体を保持 |
| `dp` | 源資料と同じ正の整数。小数・boolを受け付けない |
| `mn_g_mol` | 源資料のMnと一致し、二端構造の質量計算と整合。許容差0.05 g/mol |
| `dispersity` | 1.0。源集団のMn=Mwの単分散鎖に限定 |
| `left_cap`, `right_cap` | ともに明示した`*C`。未知末端やOH末端へ読み替えない |
| `tacticity` | 源資料と同じ`none`または`atactic` |
| `architecture` | `linear`。単一繰返し単位の源材料 |
| `temperature_k`, `pressure_atm` | 明示300.0 K、1.0 atm |
| `forcefield`, `sp_module_version` | 明示`GAFF2_mod`、`0.1.1` |

元の第2末端の空欄を新しい入力の既定値には使いません。
台帳に採用した源資料だけで、旧版ソースの規則とDP/Mnを照合した両端メチルの根拠を保持します。
ユーザー入力には両端を明示する必要があります。商用gradeのMwをMnへ変更したり、
平均Mnを整数DPへ丸めたりして通すこともありません。

源UUIDと元の繰返しを照合し、組み立てた末端付き1単位の109特徴が台帳と一致することを確認します。
イオン、ラジカル、同位体、混合物、源材料と違う末端・配列・DP・条件は拒否します。
EO/PO共重合体・界面活性剤について、今回の台帳には数値出力を資格確認した源材料がありません。

### DPの全体範囲は材料の支持範囲ではない

26,867行には48種類のDPがあり、全体では7–165です。
しかし、同じ末端付きcanonical構造で複数のDPを観測した例は0件でした。
周期系列内には複数DPがあっても、別の化学・繰返し表現を含むため、
同じ材料の鎖長依存を測った実験とは扱えません。
たとえば観測済み材料のDPを16から32へ変えた入力は、全体範囲内でも拒否します。
新しいDP・末端の定量的な応答を、今回のモデルで検証したとは説明しません。

源UUIDは26,867種類、繰返し・DP・Mn・立体規則の材料組合せは25,366種類です。
末端付きcanonical構造は25,357種類で、9組では異なる正規化繰返しが同じ末端付き構造になります。
そのうち同じ周期位相キーとなるのは5組です。
同じ末端付き構造やグループキーだけで別の繰返しを承認せず、全行の元の表現を残します。

## ローカルの利用方法

モデル、推論レビュー、支持台帳、源材料の入力JSONをローカルに用意します。
これらの外部由来ファイルは公開リポジトリへ含めていません。
実施したローカル台帳を使う場合のコマンドは次の形です。
`local/source-material-input.json`には、上の全欄を含む確認済み源材料の単体または配列を保存します。

```sh
uv run hansenkit md-cohesion-predict --model local/expanded-hsp-auxiliary-refit-v1/polyomics-refit-v1/models.local-only.json --review local/md-cohesion-inference-v1/inference-review.local-only.json --support local/md-cohesion-inference-v1/observed-material-support.local-only.npz --input local/source-material-input.json --output runs/md-source-reference.json --mode A
```

`--mode B`または`--mode C`も、台帳の固定Embeddingを使います。
エンコーダの取得・実行・重み更新やMDの実行は行いません。
巨大な鎖をモデルへ渡さず、源資料と同じ末端付き1単位と明示された鎖条件を使います。
出力先が存在すれば上書きしません。

Pythonからは`hansenkit.md_inference.MDCohesionPredictor(model_path, review_path, support_path)`を作り、
`predictor.predict(record, mode="A")`を呼びます。
構造・条件や支持範囲の問題では`status="refused"`と理由を返し、
`predictions`と`prediction_intervals`を`None`にします。
学習109座標の範囲外、非有限値、負の回帰予測も数値出力を拒否します。
負値や区間下限をゼロへ切り上げる修正はありません。

受理する出力は`status="computed_reference_prediction"`で、
`predictions`の2項と`prediction_intervals`を返します。
拒否結果も含め、モデル・台帳・レビュー・学習計画・源資料・源資格・予約のハッシュを保持します。
`hsp_predictions=None`、P/H分離なし、物理HSPバックエンド未認定という記録を常に維持します。

## 区間と検証が示す範囲

区間は源計算参照系列の校正です。学習だけの重み付きターゲット標準偏差で規格化し、
各校正系列の最大残差を使った同時90%区間です。
[固定テスト系列の被覆率](expanded-hsp-auxiliary-refit.md)はA/B/Cが0.8961／0.8912／0.8932で、
全方式で名目90%を下回りました。新しい材料・実物・HSPへの90%保証ではありません。
区間が有限でも、平衡・力場精度・相・分布・蒸発エネルギーを資格確認したことにはなりません。

別に、独自に指定したPEGと立体を指定したPOのDP4／8／16／32／64／128、
両端メチルと両端OH、およびEO/POブロックの両順序について36構造を実際に組み立てました。
最大256単位・905重原子、全官能基原子カバレッジが1で特徴が有限、
独立した残基・末端の質量計算との最大差は3.00×10⁻¹¹ g/molでした。
この試験では回帰ヘッドもMDも実行せず、HSP／`sp`予測はありません。
鎖長・末端・分布による物性応答の検証に読み替えません。

支持台帳の元順序・特徴・Embedding・源Mnを保持し、源材料への対応を照合しました。
別実装で全26,867源行、25,366繰返し、25,357種類の末端付き109特徴を再構成しました。
プロジェクトの組立て・特徴関数を呼ばず、dummy原子の独立したメチル置換、
残基＋2末端の質量、元SMARTS宣言の原子所属、インストール済みCrippen表の完全な原子型を確認しました。
質量の最大差3.18×10⁻¹¹ g/mol、109特徴の最大差5.33×10⁻¹⁵で、
既存特徴とfloat32のEmbeddingは元配列とbit単位で一致しました。
公開APIの全源材料の数値・拒否検証は、別の固定計画による確定記録を追記します。

## 権利と固定記録

推論レビューはモデル・台帳・学習計画・学習許可・予約・選別・源資料・源資格のSHA-256と、
NumPy／scikit-learn／RDKitの実行版を検査します。
`local_cached_inference_allowed=true`、`use_scope="scientific_private_noncommercial"`、
`public_redistribution=false`、`encoder_execution_allowed=false`、`hsp_inference_allowed=false`が必要です。
記事や配布CSVのCC BY宣言を、上流力場・数表の商用利用や派生重みの再配布許可へ読み替えません。
今回は`audit_basis="operator_assumption"`による正確な源材料のローカル研究・私用・非商用に限定します。
元の準備許可・旧モデル・源データは変更していません。

| ローカル記録 | SHA-256 |
| --- | --- |
| 源材料台帳の出力計画 | `b379cdfe1b03a0e0359ebbe048fb1672270f79f7651bc30e93265f7126cf979d` |
| 観測済み源材料の支持台帳 | `7d86ab473f0f0d544f17f5389012911efc505ec23bee18f05feafb621bac2524` |
| 推論レビュー | `0fa5fc34d8c86eb10b66d1d185902a18102cb6a7ab8a49ab3f4cfdb1927a411e` |
| 現在のMD2項モデル | `3d23a43816b7e8dff00ddd331f02cb46b57e653b856f07cde21e211dd0cef1db` |
| 全源材料・質量・109特徴の独立照合 | `3e2a4e12818a5bfbc1f385f239f035e08950e1fb3250f5a49d1811fe56ffe88a` |
| 構造負荷試験の計画 | `bfb0f459878a1c45a5a85f091693a9c87ec43cedb9510c028b232d61b1155864` |
| 36構造の負荷試験結果 | `6b81e83eeaec07b21e940fb00eed4a3d4de9dd5f6da149fb5cb3c2fb8bc3ccab` |

公開物は独自コード・テスト・方法と集計の文書だけです。
高分子・長鎖界面活性剤のHSP3成分を新しい材料から推算する本来の目標は未完了です。
