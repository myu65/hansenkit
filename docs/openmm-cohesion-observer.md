# CPUの分子セルと凝集エネルギー観測

`physics`オプションと`hansenkit.openmm_cohesion.FrameCohesionObserver`を追加しました。
呼出側で権利・構造・パラメータを確認したOpenMM Systemの座標を受け取り、
Lennard-Jones項とCoulomb項の凝集エネルギー密度を別々に返す、小さなPython APIです。
外部資産を取得せず、MDの積分も行いません。通常のHSP推算の適用範囲は広げません。

```sh
uv sync --locked --extra physics
uv run --extra physics pytest tests/test_openmm_cohesion.py
```

System、分子ID、結合、nm単位の座標と直交セルを用意した呼出側では、次の形で使います。
この例はパラメータ作成や平衡化の手順を省略しています。

```python
from hansenkit.openmm_cohesion import FrameCohesionObserver

with FrameCohesionObserver(system, molecule_ids, bonds, platform="CPU", threads=2) as observer:
    frame = observer.snapshot(positions_nm, box_vectors_nm)
```

## 観測値と範囲

各フレームの分子を結合に沿ってアンラップし、同じ瞬間の分子内配置のまま、
孤立分子のNoCutoff非結合エネルギーを計算します。元の1–4例外を保持します。
バルク側は元のPMEと分散補正を保持してLJ、Coulomb、元の非結合全項を照合します。
静的な照合用SystemをMDへ流用してはいけません。

セル全体のエネルギーについて、各項の観測値は
`(E_isolated − E_bulk) / V × 10²⁴ / N_A`、単位はMPaです。
負の値も保持します。気体分子を別に緩和した蒸発エンタルピーではありません。
LJ項をそのままδDとみなす資格はなく、Coulomb項からδPとδHを分離できません。
`observable_kind="computed_frame_nonbonded_cohesion"`、`hsp_predictions=None`、
`physical_accuracy_validated=False`を常に保持します。

明示したCPU/Referenceのみ、2..8192実粒子、少なくとも2分子、中性の部分電荷和、
連結した分子内結合、標準的な加算型非結合力、PME/NoCutoffと直交セルに限定します。
各分子の部分電荷和の許容差は1e−4 eです。GPUの暗黙選択、仮想粒子、分極力場、
パラメータオフセット、別の力グループに分けたPME逆空間項、分子間例外、
周期的に巻き付いた鎖などは拒否します。
これらは数値契約の検査です。呼出側には完全なパラメータ、分子分割、状態と権利の確認が必要です。

## 実行したローカル試験

2026-10-11、独自に構造を指定した3種類をCPU上でパラメータ化し、実際にMDを実行しました。
全てHSP評価予約の系列に入るため、今回の役割は`evaluation_only`、学習許可はありません。
以下は計算の動作確認であり、実測HSPの検証ではありません。

| 分子 | セル内分子数 / 原子数 | MD時間 | 座標保存フレーム | 運動温度の範囲 / K | 密度の範囲 / g cm⁻³ |
| --- | --- | --- | --- | --- | --- |
| エタノール | 64 / 576 | 2 ps | 10 | 187.50–280.90 | 0.6860–0.7394 |
| HO–(CH₂CH₂O)₈–H | 27 / 1593 | 2 ps | 10 | 178.33–285.29 | 0.7595–0.8438 |
| C8–EO3–PO1–OH、POの立体配置を指定 | 27 / 1566 | 2 ps | 7 | 175.05–288.57 | 0.7623–0.8415 |

条件はGAFF2 2.2.20、AM1-BCC、HBonds拘束、PME 0.75 nm、
Ewald許容差5e−4、分散補正あり、LangevinMiddle 1 fs、摩擦1 ps⁻¹、
目標300 K / 1 atm、バロスタット25ステップ、CPU 2スレッドです。
初期構造はETKDGv3、Packmolで配置しました。GPUのジョブ・設定は操作していません。
温度と密度はまだ変化しており、平衡・物性精度・独立反復の収束は確認できません。
PEG8と指定したEO/POオリゴマーは、分布や末端が不明な市販品の代表値ではありません。

元のAM1-BCC出力は部分電荷和が+0.000001 / +0.001 / −0.000998 eでした。
検証済みの中性構造に対し、原子当たり−1.11e−7 / −1.69e−5 / +1.72e−5 eの
一様補正を明示して正規化しました。元ファイルを保持し、電荷判定の閾値は緩めていません。
原子名・元素・結合次数・指定立体配置・全コピーのパラメータを照合しました。
この操作は構造の中性化やHSPに合わせたパラメータ調整ではありません。

## 保存した失敗と独立照合

EO/PO鎖の800ステップ目で、LJ −2130.70325、Coulomb +2462.96476 kJ/molが
相殺し、CPUの分割和と元の全項に−4.75e−5 kJ/molの差が出ました。
全項の大きさで相対誤差を判定していた旧閾値は、このフレームを拒否しました。
同じ保存座標のReference計算では閉合誤差1.64e−11 kJ/molです。
分割の判定を`1e−5 + 1e−7 × (|E_LJ| + |E_Coulomb|)`へ修正し、
誤った全項を拒否するテストも追加しました。各返却値に誤差と判定閾値を残します。
物理パラメータを変更せず、保存チェックポイントから残り1200ステップを実行しました。

完了済みのエタノールとPEG8はハッシュ照合して再利用しました。
EO/PO鎖は最初の失敗実行でチェックポイントがなく、一度だけ診断用に再実行しています。
再実行でも800ステップ目で拒否され、その保存状態から再開しました。
初期200/400/600ステップは集計値だけが残り、座標の再構成や再検証はしていません。
合計30集計フレーム、27保存座標、独立した構造は3種類です。

27座標について、元のモノマーの標準NonbondedForceと独立したNumPyの全分子内対和で
1–4例外を含むLJ/Coulombを照合しました。最大差は標準力との間で8.19e−12、
Reference観測器との間で8.64e−11 kJ/molです。
Referenceの分割和と独立した元のPME全項の最大差は4.83e−11 kJ/molでした。
保存CPU値との最大差は孤立項4.48e−4、バルク成分0.0351 kJ/molで、
バルク成分の相対差は9.91e−6未満です。CPUとReferenceの値が完全一致するとは主張しません。
30集計行の単位換算の最大差は5.69e−14 MPaで、密度換算も一致しました。
EO/POの27コピー×7フレーム、189分子で指定したS立体配置を座標から再確認しました。

照合プログラムの初回はSI→nm・kJの変換係数の向きを誤り、独立対和が失敗しました。
そのプログラムを保存し、単位式を修正しました。元の観測器・MD・保存値は変更していません。
元の失敗、旧コード、Reference証明、実行計画、チェックポイントと照合結果を全てローカルに保持します。

## 資産と次の実験

OpenMM 8.6.1の[固定版ライセンス](https://github.com/openmm/openmm/blob/b399af4725573963b46d6c1083fdcf7a37615857/docs-source/licenses/Licenses.txt)
はAPI/CPU/ReferenceがMIT、CUDA/OpenCLがLGPLで、同梱部品の追加通知もあります。
Windows CPython 3.12ホイールSHA-256は
`81d2db56d6cfc9d7ade0adc0039020cb995d3606915087fab3563499981e739a`。
通知ファイルSHA-256は`437b7168cc997abea3b5f2a9e0fb6894f96de77b9c69be428ccfcfe9bed58293`。
公開リポジトリは依存参照と独自コードだけを含み、SDKバイナリを同梱しません。

別のローカルWSL環境で[AmberTools](https://ambermd.org/AmberTools.php) 25.3の
conda-forge `cuda_None_nompi_py312hd652fd9_100`を使用しました。
同パッケージの宣言はGPL-3.0-or-later / LGPL-3.0-or-later / BSD-3-Clause / MIT、
同梱Packmol 21.0.1はMITです。micromamba 2.9.0はBSD-3-Clause。
環境にはIntel MKLの独自ライセンスもあり、全環境をMITや全てOSSとは扱いません。
138パッケージのURL・MD5・SHA-256、使用した実行ファイル・力場のハッシュを記録しました。
力場・電荷・生成トポロジーの今回の使用は個別のローカル`operator_assumption`に限定し、
ソース表、数値パラメータ、軌跡、モデルを公開していません。RadonPyは実行していません。

HSP Excelの数値・係数・教師予測は、このMD・観測器・照合・調整に使っていません。
追加HSP教師ラベルと独立実測ラベルは0件です。以前の1,013構造での非劣性未達も保持します。
次は状態・温度・条件を揃えた純液体で、十分な平衡化、複数反復、サイズ依存性、
密度・蒸発エネルギーの独立資料との照合を先に行います。
その後もHSP3成分の分離・校正と、高分子のMn/末端・EO/PO分布の物性確認は別に必要です。

対応Issue: [#49](https://github.com/myu65/hansenkit/issues/49)。
