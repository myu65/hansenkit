# 計算化学で参照HSPのMSEを下げるための調査

2026-10-11の追加調査です。今回の目標は、独立実測HSPの取得を達成条件にせず、
同じ1,000構造以上の公開参照集団で、δD・δP・δHのMSEをそれぞれ保存済みHSPiT比較値以下にすることです。
これは`published_reference`との一致を調べる目標です。実験精度、高分子への適用資格、
液体の水素結合エネルギーを保証する目標とは区別します。
以下の方法・追加データの一覧は調査と提案です。
末尾に記録したxTBの事前確認以外の新規計算、大規模収集、追加データ学習の完了記録ではありません。

既に[50構造での特徴比較](physics-feature-comparison.md)では、サイト・電荷特徴によるδHの改善を確認しました。
3D体積・MMFF双極子を追加しただけでは全成分の改善を示せず、δ²回帰は全成分で悪化しました。
次の計算では、同じ特徴数をさらに増やすことより、既存の予測代理量を直接計算で置き換えることを優先します。
全評価値を以前の実験で確認済みであるため、今回も探索的な再評価です。
方式の採用は学習系列内で決め、評価結果のよい成分を後から組み合わせません。

## 理論から直接得られる範囲

[IUPACの定義](https://goldbook.iupac.org/terms/view/ST07466)では、総溶解度パラメータは
モル蒸発内部エネルギーとモル体積から求めます。
低圧の理想気体近似を明示すると、同じ温度・状態の蒸発エンタルピーと液体モル体積について

\[
\delta_T^2 \simeq (\Delta H_{vap}-RT)/V_m,
\qquad \delta_T^2=\delta_D^2+\delta_P^2+\delta_H^2
\]

という総量の関係を検査できます。
[Hansen本人の解説](https://www.hansen-solubility.com/HSP-science/basics.php)もこの総量と3成分の関係を示しています。
総量1つから3成分を一意に求めることはできません。気相単分子の絶対エネルギー、
幾何体積、ALPB溶媒和自由エネルギーをこの式へ代入してHSP3成分と呼ぶ処理は採用しません。
特に溶媒和自由エネルギーにはエントロピーや標準状態が関わり、蒸発内部エネルギーとは異なります。

分散に関しては、[IUPACのモル屈折](https://goldbook.iupac.org/terms/view/M03979/pdf)と分極率・屈折率・体積の関係が特徴設計の根拠になります。
ただし、分極率だけでは液体の距離分布、充填、密度、反発、多体効果を決められません。
HOMO–LUMO gapをイオン化エネルギーと同一視したLondon式の代入も避けます。
分極率／幾何体積、C6／幾何体積のべきなどを使う場合は、単位と仮説を持つ回帰特徴とし、
MPa単位の凝集エネルギー密度として出力しません。

## 優先する計算

| 優先 | 方法 | 新しく得る情報 | 最初の実験 | 採用条件と限界 |
| --- | --- | --- | --- | --- |
| 1 | GFN2-xTB気相singlepoint | 電子密度の双極子、原子電荷、環境依存C6・分極率、gap | 保存済みの完全な3コンフォマーへ同一条件で実行し、既存のQM9推定特徴・MMFF双極子との置換と追加を比較 | SCC・完全原子対応・全3体の成功が必要。半経験的計算で、液体物性ではない |
| 2 | GFN2-xTB＋固定ALPB溶媒パネル | 分子の溶媒応答、電気・表面・経験的HB項、溶媒間差 | 気相と同一座標で水・ヘキサン・アセトニトリル・メタノールなどを計算 | 溶媒、パラメータ、標準状態を先に固定。Gelec/GhbをδP/δHへ変換しない |
| 3 | 固定プローブとの相互作用 | HBサイトの環境・遮蔽・方向性、相互作用の分布 | 水・固定ドナー・固定アクセプター・非極性プローブについて少数の事前定義姿勢 | 姿勢・距離規則・最適化予算を固定。1つの最低エネルギーを液体のHB強度と呼ばない |
| 4 | DFTの有限場応答／COSMO表面 | 半経験的特徴の誤差、表面電荷の分布、分極率の異方性 | 学習側の構造だけで選んだ小規模パイロットをxTBと比較 | 精度・基底・積分格子・面積規約・収束・依存資産のレビューが必要 |
| 5 | SAPT0 | 固定ダイマーのelst・exch・ind・disp | プローブ計算の少数例を物理項別に点検 | 3成分HSPの直接対応表を作らない。全1,000構造への初手には計算負担が大きい |
| 別計画 | 液体MD／状態方程式 | 充填・温度・密度・総凝集の条件依存 | 凍結済みの過去MD失敗を保持し、必要なら新しい独立計画を立てる | 過去HSP結果で力場やburn-inを修正しない。今回の最短経路にはしない |

優先順位は本プロジェクトの小規模学習、現環境、既存の特徴比較からの判断です。
各方法による参照MSEの改善が証明済みという意味ではありません。

## 実行できる最短経路と版の証拠

読取り時のnative研究環境にはRDKit・OpenMM・PyTorchがあります。
`pyscf`、`psi4`、`xtb`、`tblite`、`dftd4`は未インストールで、PATH上にもxTB・Psi4・CRESTはありませんでした。
UbuntuのWSLと`/usr/bin/python3` 3.14.4はあります。環境や設定を変更していません。

[xTBの公式リリースv6.7.1](https://github.com/grimme-lab/xtb/releases/tag/v6.7.1)にWindows用standaloneがあります。
取得候補は`xtb-6.7.1pre-windows-x86_64.zip`、37,550,920 bytesです。
[公式チェックサム](https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1pre-windows-x86_64.zip.sha256)は
`043e578da4a7e114a4d584972959a875e3ffb9f2767a86723b95aa6719d28d9c`です。
タグのcommitは`26b28010e805f7d1aeeef39813feb473e69cc4be`ですが、配布名は`6.7.1pre`です。
ZIPの一致だけでタグと同一buildと断定せず、実際の`--version`、build commit、実行ファイル・DLL・数表のハッシュを保持します。
[公式セットアップ資料](https://xtb-docs.readthedocs.io/en/latest/setup.html)に沿ってbinaryの版とパラメータ探索先を確認します。

[固定版README](https://github.com/grimme-lab/xtb/blob/26b28010e805f7d1aeeef39813feb473e69cc4be/README.md)はLGPL-3.0-or-laterを宣言します。
[GFN2数表](https://github.com/grimme-lab/xtb/blob/26b28010e805f7d1aeeef39813feb473e69cc4be/param_gfn2-xtb.txt)は方法名とDOIを記録しています。
[ALPB水の埋め込み表](https://github.com/grimme-lab/xtb/blob/26b28010e805f7d1aeeef39813feb473e69cc4be/include/param_alpb_water.fh)なども別資産として記録します。
数表自身に独立した許諾表示が確認できない場合は、上流ソースと対応方法の出典、
ユーザーの`audit_basis="operator_assumption"`による正確なローカル範囲を区別します。
数表・依存binary・モデルをGitへ同梱したり、hansenkit全体がMITだから再配布可能と説明したりしません。

[tblite 0.7.0](https://github.com/tblite/tblite/releases/tag/v0.7.0)はcommit
`fa8a4416e8fe093d0075bc10ac875494c2a449a9`、LGPL-3.0-or-laterです。
[公式PyPIメタデータ](https://pypi.org/pypi/tblite/0.7.0/json)にはWindows wheelが0本でした。
WSL用wheelはありますが、環境追加よりnative xTB standaloneの方がこの端末では経路が短いと判断します。
[Python Result API](https://tblite.readthedocs.io/en/latest/_modules/tblite/interface.html)には双極子・電荷などがありますが、
直接`polarizability`を取り出すgetterは確認できませんでした。C6・αまで必要な初回にはxTB CLIを優先します。

## GFN2計画の固定案

まず学習側の51行と、このプロジェクトで書いた小分子だけを使い、
実行時間、出力キー、原子順、SCC失敗の扱いを確認します。HSP評価数値は開きません。
その後、数値結果に応じた救済を追加せず、全1,064構造について同じ計画を凍結します。

1. 既存のMMFF計画で3体すべてが条件を通過した座標を再利用します。
   MMFF不成功・サイズ範囲外の行は保持し、量子特徴を欠損にします。
   xTBの成功に合わせて過去の幾何選別規則を変更しません。
2. 気相、GFN2、全電荷0、不対電子0、accuracy 1.0、SCC上限250、電子温度300 K、
   1スレッド、再起動なしのsinglepointを固定します。これは液体温度300 Kの計算ではありません。
   [公式singlepoint仕様](https://xtb-docs.readthedocs.io/en/latest/sp.html)に従い、SCCの終了と数値を別々に確認します。
3. 分子・コンフォマーごとに専用作業ディレクトリを作り、`.CHRG`、`.UHF`、`.xtbrc`、
   `xtbrestart`などの暗黙入力を禁止します。`XTBPATH`、環境、実行コマンド、入力座標、
   stdout/stderr、終了コード、全出力、実行時間をローカルのレシートへ保存します。
4. [物性出力の仕様](https://xtb-docs.readthedocs.io/en/latest/properties.html)から、
   full multipoleの双極子を使用し、q-only双極子を区別します。
   GFN2のC6・C8・αはD4の環境依存モデル推定として保存し、DFT有限場応答と呼びません。
   電荷は原子数・順序・全有限値・総和を点検します。規格化による修正を行いません。
5. 全3体の正常終了が必要です。失敗した体を捨てた平均、追加温度でのSCC救済、
   他の力場・量子方法への自動代替はありません。生の失敗理由とNaNを保持します。
   学習分割だけから求めた補完と欠損マスクを使い、評価行を減らしません。
6. 生のµ・C6・α・gap、重原子数または幾何体積で割った特徴、
   サイト電荷の分布を少数の固定ブロックにします。コンフォマー平均とばらつきは記述量です。
   3体はBoltzmann標本ではなく、最小値だけを後から採用しません。
7. 対照、直接気相物性への置換、直接物性の追加、ALPB追加を分けます。
   全正規化・PCA・補完・設定選択は学習系列の内側で実行します。
   方式選択の指標は今回の目標に合わせ成分別MSEとして事前記録します。

実行形の候補は次の通りです。実際のbinaryのhelp・出力とローカルレビューを照合してから実行します。

```text
xtb geometry.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --iterations 250 --etemp 300 --norestart --input controls.inp --parallel 1
```

`controls.inp`の`$write`ブロックで`json=true`を指定します。
JSONだけで必要なC6・αが得られない版では、生のstdoutと対応する固定版parserを併用します。
新しい自動ダウンロードや遠隔コード実行を通常のCLIへ加えません。

## 水素結合を特徴にする方法

現状のドナー／アクセプター数とGasteiger電荷は存在と粗い環境を表します。
次段ではGFN2の原子電荷、原子multipoleまたはESPの極値、サイトのアクセス可能性、
固定プローブ相互作用を別ブロックにします。単純な「ドナー数×アクセプター数」だけでは
酸性・塩基性の差、方向性、立体遮蔽、分子内HBによるサイト消費を表せないためです。
これは特徴仮説で、結合形成率や液体中の実測エネルギーの認定ではありません。

[Hunterの一次研究](https://doi.org/10.1002/anie.200301739)は、気相・固体・溶液の相互作用を
同じ定量的な視点で扱う道筋を示しています。
この方向を調べる際も、論文の官能基別定数を転載して未確認の教師にすることはありません。
[xTBのESP計算](https://xtb-docs.readthedocs.io/en/latest/commandline.html)は`--esp`でvdW格子上の電位を返すため、
原子電荷だけで失われる方向性を点検する次の候補です。格子・原子対応・電位単位を固定し、
ESPから経験的な酸性／塩基性尺度へ変換する係数は別レビュー対象とします。

[ALPB仕様](https://xtb-docs.readthedocs.io/en/latest/gbsa.html)には溶媒ごとのパラメータと標準状態が記載されています。
溶媒パネルはHSP評価結果で選ばず、同一幾何の気相基準との差を保存します。
`gsolv`、`bar1mol`、`reference`を明示し、既定値に頼りません。
Gsolv、Gelec、Gsasa、Ghb、Gshiftは方法の計算量として区別します。
経験的HB項を独立の実測HB強度、GelecをδP²、GhbをδH²と対応させません。

[Psi4のSAPT仕様](https://psi4.github.io/psi4docs/1.8.x/sapt.html)は、固定ダイマーの相互作用を
静電・交換・誘起・分散などへ分解します。
HBはそれらの複数項を含み、誘起項だけをHBと扱う定義ではありません。
基底、密度フィッティング、delta-HF、スケーリングなどで値が変わるため、すべて固定します。
固定プローブの特徴を採用しても、液体の多体環境、温度、競争的HBを直接検証したことにはなりません。

## DFT・COSMO・状態方程式の位置付け

[PySCFはnative WindowsをサポートせずWSLを案内](https://pyscf.org/user/install.html)します。
[PySCF本体](https://github.com/pyscf/pyscf)はApache-2.0ですが、Libxc、積分ライブラリ、
基底・溶媒数表・拡張モジュールの出典と許諾は別に記録します。
[PySCFの溶媒モデル](https://pyscf.org/user/solvent.html)はPCM、ddCOSMO、SMDなどを提供します。
SMDの溶媒記述子・CDS項も外部資産なので、権利レビューなしの置換実行はしません。

COSMOの表面電荷momentからHSPを回帰する
[2011年の一次論文](https://www.sciencedirect.com/science/article/abs/pii/S0378381211003116)があり、
表面積、電荷の正負分布、酸性・塩基性領域を特徴にする根拠になります。
論文係数や元のHSP表を転記する計画ではありません。
格子・半径・surface smoothingなどの規約が異なる出力を同じ特徴として混在させません。

[openCOSMO-RS 24aの一次論文](https://doi.org/10.1016/j.fluid.2024.114250)は
溶媒和・分配・無限希釈活量を対象とします。
[公式conformer pipeline](https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline)はORCA 6.1+とxTBを要求します。
そのためLGPLのCOSMO-RS実装だけを見て完全なMIT系量子計算経路と判断しません。
PySCFのCOSMO出力へ置き換えるなら、面・量子方法・半径・パラメータの対応を別に資格確認します。

[3成分を状態方程式から扱う2006年の一次研究](https://doi.org/10.1016/j.fluid.2005.12.020)もあります。
ただし流体のパラメータと状態情報が必要で、分子構造だけで未知パラメータが解消するわけではありません。
論文のパラメータ数表、HSPiTのExcel係数、参照GCの出力を新しい学習や特徴へ流すことはありません。

## 追加データ候補と数え方

| 一次データ | 得られる補助量 | 版・権利の読取り | 優先と隔離 |
| --- | --- | --- | --- |
| [DES370K著者Figshare](https://api.figshare.com/v2/articles/12692714) | CCSD(T)/MP2の相互作用、SAPT成分、ダイマー構造 | v1、DOI `10.6084/m9.figshare.12692714.v1`、asset license CC0。CSV 293,432,674 bytes、MD5 `de78d5478ece522bb6c71631c7082374` | 相互作用の補助学習候補。各モノマーとその系列を両側とも予約照合。1つでも衝突すればそのダイマーを学習しない |
| [QM7-X](https://zenodo.org/records/4288677) | PBE0+MBDの分極率tensor、分散係数、双極子など | Zenodo v2、DOI `10.5281/zenodo.4288677`、metadata license CC-BY-4.0、圧縮計9.6 GB | QM9からS/Cl・非平衡へ拡張する候補。ただし7重原子まで。全コンフォマーを同一化学系列へ束ねる |
| [SPICE 2.0データ](https://zenodo.org/records/10835749) | SCF双極子、MBIS電荷・multipole、エネルギー・力 | DOI `10.5281/zenodo.10835749`、asset license CC0、HDF5 37.5 GB。[著者README](https://github.com/openmm/spice-dataset/tree/2.0.0)もデータCC0を明記 | 大分子・Si/Bの候補。欠けたMBIS結果を0で埋めず、イオン・複合体は別扱い。software DOIとの混同を避ける |
| [QMugs著者ETHデータ](https://www.research-collection.ethz.ch/entities/researchdata/d0e7d2fc-6fd2-43bc-bd68-8903470ad0ec) | 大きなdrug-like分子のGFN2/DFT電荷・双極子・構造 | DOI `10.3929/ethz-b-000482129`、データCC-BY-SA-3.0。論文CC-BY-4.0とは異なる | 出典・派生物条件を追加レビュー。多コンフォマーを分子例の水増しに使わない |
| [既存SolQuest調査](solvation-auxiliary.md) | 固定39溶媒の計算溶媒和、量子構造 | 既存の固定取得・権利・予約・失敗を保持 | GDB17からEGPへの転移は既に失敗。新しい規約や領域の検証を別計画にする |
| [既存ThermoML調査](thermoml-auxiliary-review.md) | 条件付き密度・蒸発熱候補 | 数表照合・状態・権利の未解決項目を保持 | 同状態の総δの補助検査候補。現在の密度と蒸発熱候補に共有graphがなく、3成分教師へ足さない |

これらのライセンスは2026-10-11に一次レコード／APIから読み取った候補の証拠です。
ファイル取得後の内容、checksum、個々の上流依存、用途・派生重みの許諾を確認する前に、
今回の取得済み件数・学習済み件数として数えません。
DESのようなダイマーでは、`smiles0.smiles1`全体のscaffoldを1つだけ比較しても隔離になりません。
多くの小さな鎖状分子が現在の保守的な予約に含まれるため、相互作用データが大きくても
適格なモノマーとダイマーの数は少なくなる可能性があります。
全2,798既存aliasと現在のHSP学習50構造の系列を保持し、採用数を実際に監査して報告します。

新しい直接量子計算を固定アルゴリズムで評価分子の構造から得ることと、
その分子・系列の量子正解を補助モデルの教師に使うことは異なります。
前者はラベルを使わない特徴抽出として計画可能ですが、後者は補助学習の予約除外が必要です。
同じ分子の100コンフォマー、50温度点、二体相互作用の多数姿勢は、100または50個の独立HSP分子ではありません。

## MSEに合わせた検証

回帰の損失をMSEに合わせる変更は、次の計画で学習側の系列分割・設定選択・
最終方式選択まで一貫させます。MAEで選んだ旧結果と今回のMSEで選ぶ結果を混在させません。
成分別方式の候補も選択規則を先に固定し、外側学習分割ごとに最初から選び直します。
残差モデルでは内側学習群だけでbaseを選び、外側検証ラベルをbaseの設定選択に戻しません。

量子特徴を使う方式、使わない方式、train平均、既存の化学／Embedding対照を同じ行で比較します。
δD/P/HのMSE・RMSE・MAE・R²、負予測、欠損、失敗、OOD／範囲内を全件保持します。
MSEはMPa、RMSEはMPa^0.5で、3成分の平均だけで達成を判定しません。
評価元に規約差・測定系譜不明・温度不明があっても、都合のよい行だけを残さず報告します。

独立実装で、JSON再読込み後の予測、全fold、選択規則、MSEと失敗集計を再計算します。
数値再現、特徴量の物理的意味、参照一致、実験的正しさは別の検証項目です。
合格したコードがあること、量子計算が正常終了すること、参照MSEが下がることを互いの証明にしません。

この調査だけでは、各成分のMSE目標は達成していません。
最初に実行すべきなのは、正確なローカルレビューを記録したGFN2気相特徴の置換実験と、
学習系列内でMSEを使う成分別方式選択です。

## 2026-10-12のxTB事前確認とparser

上記の公式Windows ZIPを取得した担当から、全299ファイルの固定ハッシュを受け取り、
実行前に照合しました。正確なローカルレビューを先に保存したうえで、
このプロジェクトで書いたエタノールの未最適化座標1つだけを気相GFN2で計算しました。
HSPデータ・評価構造・参照値は読み込まず、学習もしていません。
SCCは9反復で収束し、プロセスは正常終了しました。
これは入出力・実行方式の事前確認で、液体エタノールやHSPの検証ではありません。

実binaryの表示は`6.7.1pre (5071a88)`、build日は2024-07-23でした。
短いcommitは公式の`5071a8869720fb0ecd788af977159869b64a277c`へ解決できました。
元のv6.7.1タグとは別revisionとして、実際の
[換算定数](https://github.com/grimme-lab/xtb/blob/5071a8869720fb0ecd788af977159869b64a277c/src/mctc/convert.f90)と
[GFN2コード](https://github.com/grimme-lab/xtb/blob/5071a8869720fb0ecd788af977159869b64a277c/src/xtb/gfn2.f90)の出典を保持します。
Windows配布のDFT-D4 3.4.0とtagのwrapにある3.5.0の差、
`mctc-lib`の非固定`head`、Intel OpenMP／static依存の正確なbuild・許諾は未確認のまま記録しています。
source数表と依存全体について商用利用・再配布の許諾が完了したとは説明しません。

最初の計算では、JSONの`program call`にWindowsのbackslashがescapeされず、
厳密なJSON読み取りが失敗しました。生ファイルを修正せず保存しています。
exeを`D:/.../xtb.exe`のforward-slashで渡した別レビュー・別作業ディレクトリの再計算は、
科学量・metadataを修正せずJSONとして読み取れました。通常経路はこちらを使います。
任意の互換処理は、JSON例外が単一の`program call`行にあるときにそのmetadata行だけを外し、
元JSONと除いた行のハッシュ・理由を残します。既定は無効で、科学量の修正はありません。

もう1つ、初期readerはJSONのSCC-result双極子と最後のwavefunctionの原子multipoleを
同一stateと誤解していました。
[JSON出力コード](https://github.com/grimme-lab/xtb/blob/5071a8869720fb0ecd788af977159869b64a277c/src/main/json.F90)は
`dipole / a.u.`を`sccres%dipole`、電荷・原子双極子を`wfx`から出します。
[物性出力コード](https://github.com/grimme-lab/xtb/blob/5071a8869720fb0ecd788af977159869b64a277c/src/main/property.F90)の
stdoutの`full`は最後の`wfx`のmultipoleから再構成します。
このsourceの区別を確認してreaderを修正し、元の差を別診断に残しました。
閾値をHSP結果へ合わせたり、原子電荷・双極子を整合するよう書き換えたりしていません。
電子数も、全電子の原子番号総和ではなく、GFNの価電子数として検査します。

公開した`xtb_features`は、実行済みのJSON・stdout/stderrと原子順・座標を受け取る研究用parserです。
単一の正常SCC、GFN2・無溶媒・固定条件、完全原子の番号・順序・電荷、有限値、単位を確認します。
SCC-resultのfull双極子を主特徴とし、q-only双極子、最後のwavefunctionのfull双極子、state間差を区別します。
μはDebye、gapはeV、D4のαはa0³、C6/C8はHartree a0⁶／Hartree a0⁸です。
凝集エネルギー密度やHSP3成分へ変換しません。
3つの全コンフォマーが揃う場合だけ、20個の平均・ばらつき・幾何体積比・電荷／サイト特徴をまとめます。
体積は液体モル体積ではなく、統計量は等重みの記述量です。

37件の独自合成fixtureによるテストで、条件・権利scope・価電子・原子所属・単位・
失敗拒否・不正JSON・metadata限定処理・3体必須・欠損拒否・保存後の来歴を確認しました。
fixtureへ実際の量子計算値、外部数表、重みを同梱していません。
実行・取得機能も通常のHSPModelへの接続もありません。
大規模な参照MSEの比較結果は、別の凍結済み実験記録で報告します。

今回の5,253単点計算、757行への学習拡張、53候補の系列内MSE選択の実施記録は
[直接電子物性の比較](mse-reference-experiment.md)を参照してください。
本調査の提案と事前確認は、後続の数値比較・独立照合と区別して残します。
