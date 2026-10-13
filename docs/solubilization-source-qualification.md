# 活性剤の公称組成と、原試料の未確認事項

2026-10-14、[#110](https://github.com/myu65/hansenkit/issues/110)で、
[可溶化の原表75点](surfactant-solubilization-intake.md)に対応する6gradeの出典を確認しました。
原著Materialsは6gradeの公称化学名とEO数を記載しています。メーカーの説明もその対応を補強しますが、
2021年論文で使われた試料との同一性を確認したものではありません。
75点は同じ薬物の条件付き可溶化候補のままで、HSP3成分の教師や新しい学習結果には数えません。

## メーカーが示す公称組成

原著の購入先はTween20/80がBDH、Myrj52/59とBrij35/58がSigma-Aldrichです。
以下のメーカー資料は現在閲覧できる説明で、原報のカタログ番号、lot、製品仕様の版は特定できません。
別の製品や現在の仕様で、原試料の不明点を補いません。

| 原報grade | メーカーの化学分類・公称EO数 | 一次資料 |
| --- | --- | --- |
| Tween20 | polysorbate／sorbitan monolaurate系、20EO | [Sigma P6585](https://www.sigmaaldrich.com/CK/en/product/sial/p6585) |
| Tween80 | polysorbate／sorbitan monooleate系、20EO | [Sigma P1754](https://www.sigmaaldrich.com/US/en/product/sial/p1754) |
| Myrj52 | polyoxyethylene(40) stearate、40EO | [Sigma P3440](https://www.sigmaaldrich.com/US/en/product/sigma/p3440) |
| Myrj59 | Myrj S100／PEG-100 stearate、100EO | [Spectrum M1402の旧名対応](https://www.spectrumchemical.com/pdfdetails/pdf/index/productSku/M1402/)、[Croda S100](https://www.crodabeauty.com/en-gb/products/product/193-myrj_1_s100) |
| Brij35 | Brij L23、主成分はC12E23、23EO | [Sigma 16005](https://www.sigmaaldrich.com/US/en/product/sigma/16005) |
| Brij58 | polyoxyethylene(20) cetyl ether、20EO | [Sigma P5884](https://www.sigmaaldrich.com/US/en/product/sigma/p5884) |

[CrodaのMyrj説明](https://www.crodabeauty.com/en-gb/resources/technical-library/resource-finder/resource/85-myrj)も、
S40とS100をステアリン酸系の非イオン性ポリグリコールエステル、40/100EOとしています。
商品名の数字をEO数と読む規則はありません。Tween80の「80」は80EOを意味しません。

公称EO数は、各分子がその数のEOを持つという証明ではありません。
現行Tween80は複数脂肪酸を含み、Brij35は主成分の表示、Brij58は平均Mnの表示です。
平均値、仮定に基づく計算質量、ミセルの分子量を区別し、現在の製品Mnを原報試料へ移しません。
原報lotのEO分布、分散度、純度、エステル組成、添加剤、末端は未確認です。

商品ページの短いSMILESも代表鎖には使いません。
例えばP3440は40EOという名称と短い同族体のSMILESを併記しています。
掲載構造から平均鎖を作ったり、未知のEO分布やMnを埋めたりする操作は行っていません。

## 原報のモル分率を換算しない

[原著](https://doi.org/10.1371/journal.pone.0249485)のEq.1では、m2/M2の項を使います。
m2はg、M2はg/molの表記ですが、水・活性剤の混合媒体に対してM2をどう定めたかは未解決です。
公称組成や現行カタログ分子量をM2へ代入せず、元のx_eをモル濃度・質量濃度へ換算しません。
原表のmM、温度、相対不確かさとその定義の不明点も、そのまま保持します。
三反復の記載からセルごとのSDを推定せず、理想溶解度3点とHSPiP値は除外したままです。

## 系列をまとめる候補と薬物側の照合

メーカーの分類からは、Tween20/80のpolysorbate系、Myrj52/59のPEG-stearate系、
Brij35/58の脂肪アルコールethoxylate系という3つのまとめ方が考えられます。
これは似た材料を学習と評価へ分けないための保守的な候補で、認定済みの3系列ではありません。
同じ材料という判断や、6gradeを6独立系列と数える根拠には使いません。
分類名や周期キーで元の構造入力を置き換えず、部分的な照合で全体の分離を証明しません。

薬物側の公称simvastatinは、名前とCASの別々の照会が[PubChem CID54454](https://pubchem.ncbi.nlm.nih.gov/compound/54454)に一致しました。
原著の分子式と立体配置を含むIUPAC名も一致し、正規化で7つの立体中心を保持しました。
形式電荷はなく、有界互変異性の列挙は完了し、4つの別名を保持しています。
既存HSP予約との照合は骨格とトポロジーの2種類で一致し、分子の同一性を示すキーの一致は報告されていません。
公称薬物は予約済み系列に属し、新しい学習系列には数えません。
この結果も原報試料の純度・状態や、6gradeすべての構造と材料系列を認定するものではありません。

既存のHSP予約5,176別名は保持し、構造の資格確認と全材料系列の分離は未完了です。
`global_isolation_certificate_qualified=False`のまま、75点は候補・評価専用として保持します。
学習・HSP推算・物理精度の資格は追加しません。
メーカー資料の閲覧は、そのデータの学習利用や再配布の許諾確認とは別です。
原画像、転記、候補台帳は科学研究・私用・非商用のローカルに保持し、公開・配布物へ含めません。

今回の構造照会は、原画像とは別のローカル許可で実行しました。
許可書のSHA-256は`14c962cf72c910614cfd0ba0af267d230416c048f54624a0b9efc28d266ce79c`、
実行記録は`746e44955c20f6fea21c2fae5f9c6af80056da237e37cefa0620b76649f54a5f`、
確認結果の台帳は`a7ff30b50e5e97615cc60afb18ad21ea190eedb5bae742268e9504fd624b080b`です。
元の5,176予約を再構築せず、確認したソースコードの同じバイト列を実行して照合しました。
これらの記録でも`audit_basis="operator_assumption"`を保持し、学習と再配布の許可は追加していません。
