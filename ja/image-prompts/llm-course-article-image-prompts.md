言語: [中文](../../image-prompts/llm-course-article-image-prompts.md) | [English](../../en/image-prompts/llm-course-article-image-prompts.md) | 日本語

# LLM コース記事用画像プロンプト計画

参照元：`/Users/lienli/Documents/GitHub/learn-agent/images/workflows/blog-to-photo.md` と `blog-to-photo` スキル。

制約：各章につき記事内の技術説明図を3枚、全19章で合計57枚作成する。画像は後から中国語・英語・日本語へローカライズする前提なので、各 prompt では画像内に読める文字を生成しないことを明示する。アイコン、番号付きの小さな点、空白ラベル帯、ローカライズ用の余白だけを描く。

## 第1章：学習ループ、計算グラフ、再現可能実験

- 記事：`lessons/01_pytorch_training_intuition.md`
- 推奨アセットディレクトリ：`lessons/assets/01_pytorch_training_intuition`

### photo-01-training-loop：最小の学習ループ

- 目標パス：`lessons/assets/01_pytorch_training_intuition/photo-01-training-loop.png`
- 図解構造： データがモデルに入り、パラメータ更新を経て次の batch に戻る環状フローチャート。
- ハイライト： Loss と optimizer update

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 最小の学習ループ.

構図: データがモデルに入り、パラメータ更新を経て次の batch に戻る環状フローチャート。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: データ batch は小さな表カード、forward は小型ニューラルモジュール、loss はメーター、backprop は計算グラフを逆向きに通る矢印、optimizer update はパラメータつまみを回すレンチ、next batch は起点へ戻る循環矢印。.

ハイライト: 淡い黄色で Loss と optimizer update を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-computation-graph：計算グラフが Tensor 関係を記録する仕組み

- 目標パス：`lessons/assets/01_pytorch_training_intuition/photo-02-computation-graph.png`
- 図解構造： tensor が普通の数値から backprop 可能な対象へ変わる左から右への状態遷移図。
- ハイライト： `grad_fn` と勾配の流れ

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 計算グラフが Tensor 関係を記録する仕組み.

構図: tensor が普通の数値から backprop 可能な対象へ変わる左から右への状態遷移図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 入力 tensor グリッド、演算子ギアノード、中間 tensor カード、尾のある `grad_fn` マーカー、leaf parameter のアンカー、勾配の戻り矢印。.

ハイライト: 淡い黄色で `grad_fn` と勾配の流れ を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-reproducible-experiment：再現可能実験と失敗診断

- 目標パス：`lessons/assets/01_pytorch_training_intuition/photo-03-reproducible-experiment.png`
- 図解構造： 上段に学習設定、下段に検証シグナルを置く階層チェックリスト。
- ハイライト： overfit tiny と pytest gate

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 再現可能実験と失敗診断.

構図: 上段に学習設定、下段に検証シグナルを置く階層チェックリスト。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 固定されたサイコロの seed、分割されたデータフォルダ、train/eval スイッチ、小サンプルを見る虫眼鏡、下降する loss 曲線、pytest gate。.

ハイライト: 淡い黄色で overfit tiny と pytest gate を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第2章：言語モデルの確率目標

- 記事：`lessons/02_language_modeling.md`
- 推奨アセットディレクトリ：`lessons/assets/02_language_modeling`

### photo-01-next-token：Next-token prediction の学習目標

- 目標パス：`lessons/assets/02_language_modeling/photo-01-next-token.png`
- 図解構造： テキストを input と label にずらし、次 token を予測する水平パイプライン。
- ハイライト： input/label のずれと cross entropy

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Next-token prediction の学習目標.

構図: テキストを input と label にずらし、次 token を予測する水平パイプライン。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: テキスト紙帯、ずれた `input_ids`、ずれた `labels`、複数出口の logits 分布、cross entropy の的、sampling 用のサイコロと温度計。.

ハイライト: 淡い黄色で input/label のずれと cross entropy を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-shape-contract：言語モデルの Shape 契約

- 目標パス：`lessons/assets/02_language_modeling/photo-02-shape-contract.png`
- 図解構造： batch、time、vocab の3次元に展開する積層 tensor 図。
- ハイライト： 3D logits ブロック

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 言語モデルの Shape 契約.

構図: batch、time、vocab の3次元に展開する積層 tensor 図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: サンプルカードの束、時間軸目盛り、語彙引き出し、3D logits ブロック、label index の針、単一の loss scalar 点。.

ハイライト: 淡い黄色で 3D logits ブロック を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-generation-controls：Temperature、top-k、top-p が生成に与える影響

- 目標パス：`lessons/assets/02_language_modeling/photo-03-generation-controls.png`
- 図解構造： 確率分布から複数の sampling gate へ入る意思決定経路図。
- ハイライト： temperature つまみとフィルタ gate

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Temperature、top-k、top-p が生成に与える影響.

構図: 確率分布から複数の sampling gate へ入る意思決定経路図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 確率の山型曲線、temperature つまみ、top-k 候補箱、top-p 累積面積ふるい、サイコロ、出力 token カード。.

ハイライト: 淡い黄色で temperature つまみとフィルタ gate を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第3章：Tokenizer と Dataset 構築

- 記事：`lessons/03_tokenizer_and_dataset.md`
- 推奨アセットディレクトリ：`lessons/assets/03_tokenizer_and_dataset`

### photo-01-text-to-ids：Tokenizer の最小契約

- 目標パス：`lessons/assets/03_tokenizer_and_dataset/photo-01-text-to-ids.png`
- 図解構造： テキストが token id になり、再び decode される入力-処理-出力フロー。
- ハイライト： vocab と可逆チェック

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Tokenizer の最小契約.

構図: テキストが token id になり、再び decode される入力-処理-出力フロー。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 元テキストの紙片、切分けナイフ、番号付き vocab 辞書、token id のビーズ列、decode の戻り矢印、可逆チェックの鏡。.

ハイライト: 淡い黄色で vocab と可逆チェック を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-mask-boundary：attention mask と label mask の境界

- 目標パス：`lessons/assets/03_tokenizer_and_dataset/photo-02-mask-boundary.png`
- 図解構造： 「何を見るか」と「loss に入るか」を左右に分けた比較図。
- ハイライト： label mask と学習シグナル

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: attention mask と label mask の境界.

構図: 「何を見るか」と「loss に入るか」を左右に分けた比較図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 灰色 padding ブロック、attention の目隠し、prompt span、answer span、label mask の loss スイッチ、学習シグナルのハイライト経路。.

ハイライト: 淡い黄色で label mask と学習シグナル を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-subword-tradeoff：文字・単語・サブワードの取捨選択

- 目標パス：`lessons/assets/03_tokenizer_and_dataset/photo-03-subword-tradeoff.png`
- 図解構造： 長さ、脆さ、汎化を比較する三分岐の意思決定図。
- ハイライト： 組み立て可能なサブワードブロック

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 文字・単語・サブワードの取捨選択.

構図: 長さ、脆さ、汎化を比較する三分岐の意思決定図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 小さな文字粒、大きく壊れやすい単語ブロック、組み立て可能なサブワードブロック、未知語の破片、長いメジャー、汎化の橋。.

ハイライト: 淡い黄色で 組み立て可能なサブワードブロック を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第4章：Embedding とニューラル言語モデル

- 記事：`lessons/04_embedding_and_neural_lm.md`
- 推奨アセットディレクトリ：`lessons/assets/04_embedding_and_neural_lm`

### photo-01-id-to-vector：Embedding が token id を学習可能な vector に変える

- 目標パス：`lessons/assets/04_embedding_and_neural_lm/photo-01-id-to-vector.png`
- 図解構造： id が embedding table に入り vector を得る入力-検索-出力フロー。
- ハイライト： embedding table と vector

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Embedding が token id を学習可能な vector に変える.

構図: id が embedding table に入り vector を得る入力-検索-出力フロー。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 番号付き token バッジ、行列引き出し、lookup ロボットアーム、座標 vector 矢印、hidden dimension の目盛り、parameter update の小さなレンチ。.

ハイライト: 淡い黄色で embedding table と vector を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-causal-context：固定文脈モデルの causal 集約

- 目標パス：`lessons/assets/04_embedding_and_neural_lm/photo-02-causal-context.png`
- 図解構造： 現在位置が過去 token だけを集約する時系列図。
- ハイライト： causal boundary

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 固定文脈モデルの causal 集約.

構図: 現在位置が過去 token だけを集約する時系列図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 過去 token の紙帯、sliding context window、一方向の causal gate、context vector の漏斗、LM head 投影、next-token 目標カード。.

ハイライト: 淡い黄色で causal boundary を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-embedding-experiment：Embedding パラメータ変化の観察

- 目標パス：`lessons/assets/04_embedding_and_neural_lm/photo-03-embedding-experiment.png`
- 図解構造： 学習前後の比較から失敗診断へ進む実験台の図。
- ハイライト： vector 移動と近傍チェック

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Embedding パラメータ変化の観察.

構図: 学習前後の比較から失敗診断へ進む実験台の図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 初期 vector 散布、学習サンプルフォルダ、下降 loss グラフ、vector の移動軌跡、近傍チェック虫眼鏡、警告三角。.

ハイライト: 淡い黄色で vector 移動と近傍チェック を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第5章：Causal Self-Attention

- 記事：`lessons/05_attention.md`
- 推奨アセットディレクトリ：`lessons/assets/05_attention`

### photo-01-qkv-flow：Q/K/V の attention 計算経路

- 目標パス：`lessons/assets/05_attention/photo-01-qkv-flow.png`
- 図解構造： 入力 vector を Q/K/V に射影し、Value を集約する水平パイプライン。
- ハイライト： similarity と weighted values

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Q/K/V の attention 計算経路.

構図: 入力 vector を Q/K/V に射影し、Value を集約する水平パイプライン。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: token カード列、query probe、key tag、dot-product similarity メーター、softmax 漏斗、weighted value box の合流。.

ハイライト: 淡い黄色で similarity と weighted values を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-causal-mask：Causal mask が未来の覗き見を防ぐ

- 目標パス：`lessons/assets/05_attention/photo-02-causal-mask.png`
- 図解構造： 過去領域は見え、未来領域は遮断される三角行列図。
- ハイライト： mask gate

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Causal mask が未来の覗き見を防ぐ.

構図: 過去領域は見え、未来領域は遮断される三角行列図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: token 時間軸、attention grid、開いた過去セル、斜線の未来セル、shield 型の mask gate、合法 weight のハイライト経路。.

ハイライト: 淡い黄色で mask gate を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-attention-interpretation：Attention weights は観察できるが神格化しない

- 目標パス：`lessons/assets/05_attention/photo-03-attention-interpretation.png`
- 図解構造： 中央に attention heatmap、周囲に検証リマインダーを置いた観測ダッシュボード。
- ハイライト： 反例チェックと実験検証

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Attention weights は観察できるが神格化しない.

構図: 中央に attention heatmap、周囲に検証リマインダーを置いた観測ダッシュボード。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 淡い heatmap grid、attention arc、context dependency link、反例チェック虫眼鏡、テスト clipboard、誤読リスク警告灯。.

ハイライト: 淡い黄色で 反例チェックと実験検証 を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第6章：Transformer Block

- 記事：`lessons/06_transformer_block.md`
- 推奨アセットディレクトリ：`lessons/assets/06_transformer_block`

### photo-01-block-anatomy：Transformer Block の内部構造

- 目標パス：`lessons/assets/06_transformer_block/photo-01-block-anatomy.png`
- 図解構造： norm、attention、residual、FFN を示す縦方向の層状アーキテクチャ図。
- ハイライト： residual add と multi-head attention

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Transformer Block の内部構造.

構図: norm、attention、residual、FFN を示す縦方向の層状アーキテクチャ図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: residual trunk pipe、LayerNorm 調整つまみ、multi-head attention module、residual merge、FFN の2層 gear box、出力状態カード。.

ハイライト: 淡い黄色で residual add と multi-head attention を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-attention-not-enough：attention を積むだけでは足りない理由

- 目標パス：`lessons/assets/06_transformer_block/photo-02-attention-not-enough.png`
- 図解構造： 片側に関係モデリング、もう片側に非線形変換を置く比較図。
- ハイライト： 表現変換と residual stability

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: attention を積むだけでは足りない理由.

構図: 片側に関係モデリング、もう片側に非線形変換を置く比較図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 関係探索ネットワーク、context mixing node、表現変換 gear workshop、曲がった activation 関数、residual safety rope、完全な block module。.

ハイライト: 淡い黄色で 表現変換と residual stability を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-block-test-gate：Block の shape と安定性の受け入れ

- 目標パス：`lessons/assets/06_transformer_block/photo-03-block-test-gate.png`
- 図解構造： 入力が複数のテスト gate を通って pass 状態へ進む検査門番図。
- ハイライト： shape matching と mask tests

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Block の shape と安定性の受け入れ.

構図: 入力が複数のテスト gate を通って pass 状態へ進む検査門番図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: input shape のノギス、output matching card、mask shield、dropout mode switch、gradient return arrow、pytest gate stamp。.

ハイライト: 淡い黄色で shape matching と mask tests を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第7章：Mini GPT をゼロから実装する

- 記事：`lessons/07_mini_gpt.md`
- 推奨アセットディレクトリ：`lessons/assets/07_mini_gpt`

### photo-01-mini-gpt-stack：Mini GPT の decoder-only stack

- 目標パス：`lessons/assets/07_mini_gpt/photo-01-mini-gpt-stack.png`
- 図解構造： token ids から embedding、blocks、LM head へ進む層状アーキテクチャ。
- ハイライト： Transformer blocks と LM head

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Mini GPT の decoder-only stack.

構図: token ids から embedding、blocks、LM head へ進む層状アーキテクチャ。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 数字列、token embedding table、position scale、積層 Transformer blocks、output projection、vocab distribution。.

ハイライト: 淡い黄色で Transformer blocks と LM head を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-train-generate-loop：学習と生成は同じモデル骨格を共有する

- 目標パス：`lessons/assets/07_mini_gpt/photo-02-train-generate-loop.png`
- 図解構造： 上側は loss を計算する学習経路、下側は autoregressive generation を行う二経路フロー。
- ハイライト： training path と generation loop

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 学習と生成は同じモデル骨格を共有する.

構図: 上側は loss を計算する学習経路、下側は autoregressive generation を行う二経路フロー。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: batch input tape、label target card、loss target、optimizer wrench、prompt strip、生成 token を追加する loop arrow。.

ハイライト: 淡い黄色で training path と generation loop を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-checkpoint-boundary：Checkpoint の2つの復旧境界

- 目標パス：`lessons/assets/07_mini_gpt/photo-03-checkpoint-boundary.png`
- 図解構造： 左に inference recovery、右に training recovery を置く比較図。
- ハイライト： model weights と optimizer state

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Checkpoint の2つの復旧境界.

構図: 左に inference recovery、右に training recovery を置く比較図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: parameter safe、tokenizer key、inference recovery の play button、optimizer momentum gear、step odometer、training resume runway。.

ハイライト: 淡い黄色で model weights と optimizer state を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第8章：Hugging Face ワークフロー

- 記事：`lessons/08_huggingface_workflow.md`
- 推奨アセットディレクトリ：`lessons/assets/08_huggingface_workflow`

### photo-01-hf-inference：Hugging Face の最小推論ワークフロー

- 目標パス：`lessons/assets/08_huggingface_workflow/photo-01-hf-inference.png`
- 図解構造： model name から tokenizer、model、generate、decode へ進む水平パイプライン。
- ハイライト： tokenizer/model alignment

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Hugging Face の最小推論ワークフロー.

構図: model name から tokenizer、model、generate、decode へ進む水平パイプライン。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: repository tag、tokenizer encoding box、weight module、generation loop arrow、decode-back-to-text symbol、result card。.

ハイライト: 淡い黄色で tokenizer/model alignment を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-chat-template：Chat Template が対話をモデル形式へ変換する

- 目標パス：`lessons/assets/08_huggingface_workflow/photo-02-chat-template.png`
- 図解構造： messages が token sequence に整形される input-template-sequence flow。
- ハイライト： template と consistency check

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Chat Template が対話をモデル形式へ変換する.

構図: messages が token sequence に整形される input-template-sequence flow。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: dialogue bubbles、role flags、layout mold、tokenizer cutter、assistant span、alignment ruler。.

ハイライト: 淡い黄色で template と consistency check を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-trust-boundary：`trust_remote_code` と保存/読み込みの安全境界

- 目標パス：`lessons/assets/08_huggingface_workflow/photo-03-trust-boundary.png`
- 図解構造： model loading が trust、version、local save の gate を通る decision path。
- ハイライト： code boundary と version pinning

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: `trust_remote_code` と保存/読み込みの安全境界.

構図: model loading が trust、version、local save の gate を通る decision path。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: remote repository cloud box、code boundary warning line、trust switch、version pin、local disk box、reproducible reload arrow。.

ハイライト: 淡い黄色で code boundary と version pinning を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第9章：SFT 指示チューニング

- 記事：`lessons/09_sft_instruction_tuning.md`
- 推奨アセットディレクトリ：`lessons/assets/09_sft_instruction_tuning`

### photo-01-sft-format：SFT データ形式契約

- 目標パス：`lessons/assets/09_sft_instruction_tuning/photo-01-sft-format.png`
- 図解構造： instruction、input、output を training sample にまとめる構造化サンプル図。
- ハイライト： chat template と token spans

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: SFT データ形式契約.

構図: instruction、input、output を training sample にまとめる構造化サンプル図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: task card、optional material folder、expected-answer target card、chat-template mold、token-span strips、archived sample box。.

ハイライト: 淡い黄色で chat template と token spans を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-label-mask-span：Label mask は token span 単位で作る

- 目標パス：`lessons/assets/09_sft_instruction_tuning/photo-02-label-mask-span.png`
- 図解構造： prompt は loss に入らず answer だけが入る分割タイムライン。
- ハイライト： mask-on の answer span

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Label mask は token span 単位で作る.

構図: prompt は loss に入らず answer だけが入る分割タイムライン。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: prompt token tape、answer token tape、mask-off switch、mask-on switch、answer span だけに落ちる loss、boundary-check magnifier。.

ハイライト: 淡い黄色で mask-on の answer span を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-data-leakage：データ分割と漏洩診断

- 目標パス：`lessons/assets/09_sft_instruction_tuning/photo-03-data-leakage.png`
- 図解構造： サンプルが dedup、split、leakage detection、evaluation を通る gate-check flow。
- ハイライト： leakage detection

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: データ分割と漏洩診断.

構図: サンプルが dedup、split、leakage detection、evaluation を通る gate-check flow。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: raw sample box、dedup sieve、train drawer、eval drawer、leakage spotlight、trustworthiness stamp。.

ハイライト: 淡い黄色で leakage detection を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第10章：LoRA / QLoRA パラメータ効率微調整

- 記事：`lessons/10_lora_qlora.md`
- 推奨アセットディレクトリ：`lessons/assets/10_lora_qlora`

### photo-01-lora-math：LoRA の低ランク差分が元モデルに接続される仕組み

- 目標パス：`lessons/assets/10_lora_qlora/photo-01-lora-math.png`
- 図解構造： 凍結された大きな行列の横に2つの低ランク小行列を side path として接続する層状 overlay。
- ハイライト： low-rank bypass

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: LoRA の低ランク差分が元モデルに接続される仕組み.

構図: 凍結された大きな行列の横に2つの低ランク小行列を side path として接続する層状 overlay。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: locked large matrix、narrow low-rank A、narrow low-rank B、delta side-path arrow、scaling knob、merged output node。.

ハイライト: 淡い黄色で low-rank bypass を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-peft-workflow：PEFT の学習・保存・読み込みワークフロー

- 目標パス：`lessons/assets/10_lora_qlora/photo-02-peft-workflow.png`
- 図解構造： base model、adapter、training、saving、loading を示す水平パイプライン。
- ハイライト： 小さなパラメータだけを学習し adapter を保存すること

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: PEFT の学習・保存・読み込みワークフロー.

構図: base model、adapter、training、saving、loading を示す水平パイプライン。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: base model block、adapter slot card、小さな trainable gears、薄い adapter folder、assembled loading module、inference result card。.

ハイライト: 淡い黄色で 小さなパラメータだけを学習し adapter を保存すること を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-qlora-memory：QLoRA のメモリと merge 境界

- 目標パス：`lessons/assets/10_lora_qlora/photo-03-qlora-memory.png`
- 図解構造： 量子化によるメモリ節約と誤差/デプロイ選択の trade-off scale。
- ハイライト： memory budget と merge decision

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: QLoRA のメモリと merge 境界.

構図: 量子化によるメモリ節約と誤差/デプロイ選択の trade-off scale。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: 4-bit compressed block、LoRA adapter side card、memory budget cup、precision-risk crack magnifier、merge decision signpost、deployment package box。.

ハイライト: 淡い黄色で memory budget と merge decision を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第11章：ドメインデータエンジニアリング

- 記事：`lessons/11_domain_data_engineering.md`
- 推奨アセットディレクトリ：`lessons/assets/11_domain_data_engineering`

### photo-01-data-layers：ドメインデータ層と固定 Eval Set

- 目標パス：`lessons/assets/11_domain_data_engineering/photo-01-data-layers.png`
- 図解構造： raw material から training、evaluation、reporting layer へ進む層状アーキテクチャ。
- ハイライト： frozen eval set

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: ドメインデータ層と固定 Eval Set.

構図: raw material から training、evaluation、reporting layer へ進む層状アーキテクチャ。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: source files の山、cleaning sieve、labeling folder、training drawer、locked eval drawer、quality report page。.

ハイライト: 淡い黄色で frozen eval set を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-clean-dedup-risk：Cleaning、Deduplication、De-Identification のリスク制御チェーン

- 目標パス：`lessons/assets/11_domain_data_engineering/photo-02-clean-dedup-risk.png`
- 図解構造： data が normalization、near-duplicate detection、de-identification、audit を通る pipeline。
- ハイライト： de-identification と audit records

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Cleaning、Deduplication、De-Identification のリスク制御チェーン.

構図: data が normalization、near-duplicate detection、de-identification、audit を通る pipeline。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: normalization brush、overlapping-card duplicate check、redaction strips、risk sign、audit log scroll、traceable output tag。.

ハイライト: 淡い黄色で de-identification と audit records を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-dataset-builders：SFT、Distillation、Evaluation Data の異なる構築目標

- 目標パス：`lessons/assets/11_domain_data_engineering/photo-03-dataset-builders.png`
- 図解構造： 同じ domain materials から3種類の data products へ分かれる三分岐図。
- ハイライト： 3種類の data products

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: SFT、Distillation、Evaluation Data の異なる構築目標.

構図: 同じ domain materials から3種類の data products へ分かれる三分岐図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: domain material lake、SFT instruction card、teacher-to-student distillation arrow、evaluation exam sheet、filtering funnel、data-quality gate。.

ハイライト: 淡い黄色で 3種類の data products を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第12章：RAG 検索拡張生成

- 記事：`lessons/12_rag_baseline.md`
- 推奨アセットディレクトリ：`lessons/assets/12_rag_baseline`

### photo-01-rag-pipeline：最小 RAG Pipeline

- 目標パス：`lessons/assets/12_rag_baseline/photo-01-rag-pipeline.png`
- 図解構造： document chunking から retrieval、context assembly、answer generation へ進む水平 pipeline。
- ハイライト： retriever と context prompt

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 最小 RAG Pipeline.

構図: document chunking から retrieval、context assembly、answer generation へ進む水平 pipeline。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: document folder、slicing tool、embedding coordinate points、retriever magnet、context-prompt clipboard、answer card。.

ハイライト: 淡い黄色で retriever と context prompt を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-retrieval-debug：RAG のどの部分が壊れたかを特定する

- 目標パス：`lessons/assets/12_rag_baseline/photo-02-retrieval-debug.png`
- 図解構造： 失敗を chunking、vectors、recall、ranking、generation に分解する診断経路。
- ハイライト： citation check

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: RAG のどの部分が壊れたかを特定する.

構図: 失敗を chunking、vectors、recall、ranking、generation に分解する診断経路。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: misaligned chunks、shifted coordinates、missed-recall net、rerank staircase、prompt seam crack、citation evidence chain。.

ハイライト: 淡い黄色で citation check を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-citation-support：Citation の存在は Citation Support ではない

- 目標パス：`lessons/assets/12_rag_baseline/photo-03-citation-support.png`
- 図解構造： 単なる citation presence と evidence が本当に結論を支える状態を比較する evidence-court diagram。
- ハイライト： support relationship

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Citation の存在は Citation Support ではない.

構図: 単なる citation presence と evidence が本当に結論を支える状態を比較する evidence-court diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: answer claim card、小さな footnote、highlighted evidence strip、support bridge、broken unsupported bridge、verdict stamp。.

ハイライト: 淡い黄色で support relationship を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第13章：小型モデルの蒸留

- 記事：`lessons/13_distillation.md`
- 推奨アセットディレクトリ：`lessons/assets/13_distillation`

### photo-01-response-distill：Response Distillation の Teacher-to-Student 経路

- 目標パス：`lessons/assets/13_distillation/photo-01-response-distill.png`
- 図解構造： teacher model が回答を生成し、filter を通して student model を学習する水平 pipeline。
- ハイライト： filter と student training

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Response Distillation の Teacher-to-Student 経路.

構図: teacher model が回答を生成し、filter を通して student model を学習する水平 pipeline。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: prompt basket、large-model lighthouse、candidate answer cards、filtering sieve、small-model workshop、comparison balance。.

ハイライト: 淡い黄色で filter と student training を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-logit-distill：Logit Distillation が soft distribution signal を残す

- 目標パス：`lessons/assets/13_distillation/photo-02-logit-distill.png`
- 図解構造： teacher logits と student logits を揃える input-distribution-loss flow。
- ハイライト： soft distribution と KL loss

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Logit Distillation が soft distribution signal を残す.

構図: teacher logits と student logits を揃える input-distribution-loss flow。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: input batch stack、teacher soft-distribution peak、student distribution peak、temperature-scaling thermometer、KL loss ruler、update-student return arrow。.

ハイライト: 淡い黄色で soft distribution と KL loss を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-teacher-risk：Teacher を唯一のレビュアーにしない

- 目標パス：`lessons/assets/13_distillation/photo-03-teacher-risk.png`
- 図解構造： teacher output が rules、human checks、eval-set cross-checks を通る多者レビュー図。
- ハイライト： human spot check と eval set

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Teacher を唯一のレビュアーにしない.

構図: teacher output が rules、human checks、eval-set cross-checks を通る多者レビュー図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: teacher answer pile、rule checklist、human-review eye、eval-set exam、risk sample folder、usable-data lane。.

ハイライト: 淡い黄色で human spot check と eval set を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第14章：モデル評価

- 記事：`lessons/14_evaluation.md`
- 推奨アセットディレクトリ：`lessons/assets/14_evaluation`

### photo-01-eval-set-design：Eval Set 設計は能力境界から始める

- 目標パス：`lessons/assets/14_evaluation/photo-01-eval-set-design.png`
- 図解構造： task type、difficulty、risk、domain distribution を覆う層状 sample map。
- ハイライト： frozen version と risk labels

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Eval Set 設計は能力境界から始める.

構図: task type、difficulty、risk、domain distribution を覆う層状 sample map。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: capability legend、sample-card sea、difficulty stairs、risk flag、domain distribution map、locked frozen version。.

ハイライト: 淡い黄色で frozen version と risk labels を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-metrics-release-gate：Metrics は Release Gate になるべき

- 目標パス：`lessons/assets/14_evaluation/photo-02-metrics-release-gate.png`
- 図解構造： computed metrics が pass、rollback、continued repair を決める gate flow。
- ハイライト： threshold line と failure-case table

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Metrics は Release Gate になるべき.

構図: computed metrics が pass、rollback、continued repair を決める gate flow。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: prediction output pile、metric dashboard、threshold bar、open release gate、rollback arrow、failure-case table。.

ハイライト: 淡い黄色で threshold line と failure-case table を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-judge-calibration：Judge Model には Calibration が必要

- 目標パス：`lessons/assets/14_evaluation/photo-03-judge-calibration.png`
- 図解構造： automatic scoring と human scoring を揃え、sampled review する calibration bench。
- ハイライト： consistency check と calibration loop

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Judge Model には Calibration が必要.

構図: automatic scoring と human scoring を揃え、sampled review する calibration bench。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: judge model silhouette、scoring rubric ruler、human scoring board、dual-ruler consistency check、biased sample outlier、calibration loop。.

ハイライト: 淡い黄色で consistency check と calibration loop を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第15章：安全性、コンプライアンス、モデルカード

- 記事：`lessons/15_safety_and_model_card.md`
- 推奨アセットディレクトリ：`lessons/assets/15_safety_and_model_card`

### photo-01-risk-taxonomy：Risk Taxonomy が安全戦略を決める

- 目標パス：`lessons/assets/15_safety_and_model_card/photo-01-risk-taxonomy.png`
- 図解構造： input requests を domain、harm、certainty で分ける risk matrix。
- ハイライト： risk classification と policy routing

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Risk Taxonomy が安全戦略を決める.

構図: input requests を domain、harm、certainty で分ける risk matrix。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: user request card、risk matrix、high-risk flag、uncertainty fog、policy routing arrows、safe-response shield card。.

ハイライト: 淡い黄色で risk classification と policy routing を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-refusal-balance：Refusal と Over-Refusal のバランス

- 目標パス：`lessons/assets/15_safety_and_model_card/photo-02-refusal-balance.png`
- 図解構造： 片側に危険な通過、反対側に過度な保守、中間に説明可能な refusal を置く天秤図。
- ハイライト： middle lane

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Refusal と Over-Refusal のバランス.

構図: 片側に危険な通過、反対側に過度な保守、中間に説明可能な refusal を置く天秤図。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: dangerous request card、reasonable request card、refusal shield、blocked door for over-refusal、clarification signpost、safe useful middle lane。.

ハイライト: 淡い黄色で middle lane を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-model-card-report：Model Card と Risk Report のリリースチェーン

- 目標パス：`lessons/assets/15_safety_and_model_card/photo-03-model-card-report.png`
- 図解構造： evaluation results から model card、risk report、human review へつながる document-chain diagram。
- ハイライト： human review と release gate

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Model Card と Risk Report のリリースチェーン.

構図: evaluation results から model card、risk report、human review へつながる document-chain diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: evaluation dashboard page、failure-case table、model-card manual、risk report、human-review stamp、release gate。.

ハイライト: 淡い黄色で human review と release gate を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第16章：量子化とデプロイ

- 記事：`lessons/16_quantization_and_serving.md`
- 推奨アセットディレクトリ：`lessons/assets/16_quantization_and_serving`

### photo-01-memory-budget：推論メモリはモデル重みだけではない

- 目標パス：`lessons/assets/16_quantization_and_serving/photo-01-memory-budget.png`
- 図解構造： weights、KV cache、batch、context length を分けて示す stacked resource diagram。
- ハイライト： KV cache と memory budget

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 推論メモリはモデル重みだけではない.

構図: weights、KV cache、batch、context length を分けて示す stacked resource diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: model weights の大きな block、成長する KV-cache bar、multi-input batch、context length の長い measuring tape、temporary activation bubbles、memory budget の capacity jar。.

ハイライト: 淡い黄色で KV cache と memory budget を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-quantization-tradeoff：量子化実験の精度と速度の取捨選択

- 目標パス：`lessons/assets/16_quantization_and_serving/photo-02-quantization-tradeoff.png`
- 図解構造： fp16 から int8/4bit まで size、speed、quality を比較する trade-off scale。
- ハイライト： compression blocks と regression evaluation

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 量子化実験の精度と速度の取捨選択.

構図: fp16 から int8/4bit まで size、speed、quality を比較する trade-off scale。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: full FP16 weight、medium INT8 compressed block、small 4-bit compressed block、speed rocket、quality crack、regression-eval gate。.

ハイライト: 淡い黄色で compression blocks と regression evaluation を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-serving-release：Serving API、Monitoring、Rollback Loop

- 目標パス：`lessons/assets/16_quantization_and_serving/photo-03-serving-release.png`
- 図解構造： client request が serving engine に入り、monitoring が異常を検知して rollback する runtime loop。
- ハイライト： monitoring と rollback

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Serving API、Monitoring、Rollback Loop.

構図: client request が serving engine に入り、monitoring が異常を検知して rollback する runtime loop。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: client entry arrow、API contract card、serving engine box、benchmark stopwatch、monitoring radar、rollback button。.

ハイライト: 淡い黄色で monitoring と rollback を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第17章：法律ドメイン小型モデルプロジェクト

- 記事：`lessons/17_legal_domain_project.md`
- 推奨アセットディレクトリ：`lessons/assets/17_legal_domain_project`

### photo-01-legal-data-design：法律資料には管轄とバージョンを記録する

- 目標パス：`lessons/assets/17_legal_domain_project/photo-01-legal-data-design.png`
- 図解構造： material cards が jurisdiction、version、source、effective date を持つ evidence archive。
- ハイライト： jurisdiction と version time

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 法律資料には管轄とバージョンを記録する.

構図: material cards が jurisdiction、version、source、effective date を持つ evidence archive。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: legal dossier、jurisdiction の map pin、version time の calendar、source-link chain、highlighted clause slice、audit stamp。.

ハイライト: 淡い黄色で jurisdiction と version time を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-contract-risk-output：契約リスク識別の出力契約

- 目標パス：`lessons/assets/17_legal_domain_project/photo-02-contract-risk-output.png`
- 図解構造： contract clause が model に入り、risk category、evidence、suggestions を出力する structured-output diagram。
- ハイライト： evidence basis と risk level

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 契約リスク識別の出力契約.

構図: contract clause が model に入り、risk category、evidence、suggestions を出力する structured-output diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: contract clause strip、risk-label board、evidence citation folder、risk-level gauge、revision pencil、disclaimer boundary shield。.

ハイライト: 淡い黄色で evidence basis と risk level を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-legal-rag-safety-loop：法律 RAG、評価、安全境界のループ

- 目標パス：`lessons/assets/17_legal_domain_project/photo-03-legal-rag-safety-loop.png`
- 図解構造： retrieval、generation、citation checking、safety refusal、human review を接続する closed loop。
- ハイライト： citation check と human review

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 法律 RAG、評価、安全境界のループ.

構図: retrieval、generation、citation checking、safety refusal、human review を接続する closed loop。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: legal retrieval chain、answer card、citation evidence bridge、safety boundary gate、human-review eye、deploy-feedback arrow。.

ハイライト: 淡い黄色で citation check と human review を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第18章：医療ドメイン小型モデルプロジェクト

- 記事：`lessons/18_medical_domain_project.md`
- 推奨アセットディレクトリ：`lessons/assets/18_medical_domain_project`

### photo-01-medical-red-flags：Red Flags は普通の説明より優先される

- 目標パス：`lessons/assets/18_medical_domain_project/photo-01-medical-red-flags.png`
- 図解構造： symptom input がまず red-flag triage を通り、その後 ordinary explanation へ進む priority routing diagram。
- ハイライト： red flags と care-seeking boundary

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Red Flags は普通の説明より優先される.

構図: symptom input がまず red-flag triage を通り、その後 ordinary explanation へ進む priority routing diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: symptom record card、red flag、emergency alert bell、ordinary explanation card、uncertainty cloud、care boundary の hospital signpost。.

ハイライト: 淡い黄色で red flags と care-seeking boundary を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-medical-output-contract：医療 QA の出力契約

- 目標パス：`lessons/assets/18_medical_domain_project/photo-02-medical-output-contract.png`
- 図解構造： summary、risk、advice、contraindications、human review に分かれた structured answer template。
- ハイライト： non-diagnostic statement と contraindication reminder

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 医療 QA の出力契約.

構図: summary、risk、advice、contraindications、human review に分かれた structured answer template。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: consultation question bubble、non-diagnostic boundary shield、possible-cause branching tree、action checklist、contraindications の stop sign、stethoscope review stamp。.

ハイライト: 淡い黄色で non-diagnostic statement と contraindication reminder を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-false-reassurance：False reassurance は重大な失敗

- 目標パス：`lessons/assets/18_medical_domain_project/photo-03-false-reassurance.png`
- 図解構造： model answer が missed danger、insufficient evidence、safety evaluation で確認される risk diagnosis diagram。
- ハイライト： missed-risk detection と safety evaluation

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: False reassurance は重大な失敗.

構図: model answer が missed danger、insufficient evidence、safety evaluation で確認される risk diagnosis diagram。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: dangerous symptom card、weak reassurance bubble、missed-risk spotlight、broken evidence chain、safety-eval shield、failure-regression repair loop。.

ハイライト: 淡い黄色で missed-risk detection と safety evaluation を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

## 第19章：完全なドメインモデル工程テンプレート

- 記事：`lessons/19_domain_model_template.md`
- 推奨アセットディレクトリ：`lessons/assets/19_domain_model_template`

### photo-01-template-directory：ドメインモデル工程テンプレートのディレクトリ

- 目標パス：`lessons/assets/19_domain_model_template/photo-01-template-directory.png`
- 図解構造： configs、data、train、RAG、eval、reports、serving を示す project map。
- ハイライト： eval と reports

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: ドメインモデル工程テンプレートのディレクトリ.

構図: configs、data、train、RAG、eval、reports、serving を示す project map。 画面には 7 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: config gear、data box、training workshop、retrieval chain、evaluation room、report shelf、serving deployment box。.

ハイライト: 淡い黄色で eval と reports を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-02-unified-command：統一コマンド入口が工程チェーンをつなぐ

- 目標パス：`lessons/assets/19_domain_model_template/photo-02-unified-command.png`
- 図解構造： 1つの CLI entry が data、training、evaluation、serving、report tasks へ分配する command bus。
- ハイライト： CLI entry

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: 統一コマンド入口が工程チェーンをつなぐ.

構図: 1つの CLI entry が data、training、evaluation、serving、report tasks へ分配する command bus。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: console entry、data branch、training branch、eval branch、serving branch、report branch。.

ハイライト: 淡い黄色で CLI entry を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```

### photo-03-release-iteration：Release Gate と継続的改善ループ

- 目標パス：`lessons/assets/19_domain_model_template/photo-03-release-iteration.png`
- 図解構造： configuration version、tests、evaluation、release gate、新領域移植へつながる closed control loop。
- ハイライト： release gate と production feedback

正向プロンプト:

```text
記事内の技術説明図を作成する。画風：生成り色の紙背景、黒い手描きペン線、少し不均一な線幅、淡い黄色のハイライトを少量、editorial technical illustration、技術ブログ向けの手描きフローチャート、明確で控えめ、工程スケッチ感のある見た目。

テーマ: Release Gate と継続的改善ループ.

構図: configuration version、tests、evaluation、release gate、新領域移植へつながる closed control loop。 画面には 6 個の主要ノードを含め、手描き矢印で接続する。関係は明確にしつつ、混み合わないようにする。

ノードアイコン設計: version tag、testing mesh、evaluation dashboard page、release gate、online feedback arrow、新領域移植の migration box。.

ハイライト: 淡い黄色で release gate と production feedback を強調する。

多言語制約：画像内に中国語、英語、日本語、その他の読める長文を生成しない。各ノードにはアイコン、小さな番号点、空白ラベル帯だけを描き、後から多言語組版しやすい余白を残す。

背景：ごく薄い回路線、ノード接続線、工程スケッチ補助線、少量の紙テクスチャ。背景が主体より目立たないようにする。
```

負向プロンプト:

```text
写真写実にしない、3D にしない、複雑な UI スクリーンショットにしない、大量のコードを入れない、細かい文字を密集させない、複雑な表にしない、サイバーパンクにしない、ネオン色を使わない、暗い背景にしない、派手なカートゥーン調にしない、複雑な影をつけない、装飾を増やしすぎない、主要ノードは8個以内、背景の回路線を主体より目立たせない、中国語・英語・日本語の可読文字、長い数式、段落文を生成しない。
```
