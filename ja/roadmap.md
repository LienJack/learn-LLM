言語: [中文](../roadmap.md) | [English](../en/roadmap.md) | 日本語

# LLM をゼロからドメイン小型モデルへ進める Roadmap

これは、学習の直感から出発し、最終的に法律・医療などのドメイン小型モデル工程へ進むためのコースルートです。目的は、ばらばらにチュートリアルを読むことではありません。長期的に保守でき、実行でき、テストでき、再現できる学習プロジェクトを作ることです。

## 全体の主線

```text
学習ループ
  -> next-token 言語モデリング
    -> Tokenizer と Dataset
      -> Embedding と固定文脈モデル
        -> Causal Self-Attention
          -> Transformer Block
            -> Mini GPT
              -> オープンソースモデルのワークフロー
                -> SFT 指示チューニング
                  -> LoRA / QLoRA
                    -> ドメインデータエンジニアリング
                      -> RAG
                        -> 蒸留
                          -> 評価
                            -> 安全性とモデルカード
                              -> 量子化デプロイ
                                -> ドメインモデルプロジェクト
```

一言でまとめると、次のようになります。

> 検証可能な最小の学習ループから始め、言語モデルがなぜ next-token prediction なのかを理解する。そこから、テキストの数値化、文脈表現、動的 attention、Transformer を順に補う。最後に、データ、微調整、RAG、蒸留、評価、安全性、デプロイによって、モデルを実行可能・テスト可能・追跡可能なドメイン小型モデルにする。

## 通し例

コース後半が単なる工程チェックリストにならないよう、このコースでは3種類の契約リスク例を最初から最後まで使います。

```text
違約金が高すぎる：合同 违约金 过高 ， 它 可能 存在 风险
責任範囲が広すぎる：間接損害、逸失利益、弁護士費用を含む一切の損失を賠償する
情報不足：管轄、法令バージョン、契約類型、検索根拠が不足している
```

第1-7章では、これらの例を使って学習ループ、next-token、tokenizer、embedding、attention、Mini GPT を観察します。第8-13章では、同じ例を Hugging Face、SFT、LoRA、ドメインデータ、RAG、蒸留へ移します。第14-19章では、それらを評価項目、安全ゲート、デプロイログ、ドメインテンプレートへ発展させます。

各章は、次の問いに答えられるべきです。

```text
この章で追加された能力は、システムを「契約条項リスクを慎重に識別する」状態へどう近づけるのか？
```

## コンポーネント境界の早見表

第11-13章に入る前に、この表を先に覚えておくと、微調整、検索、蒸留を同じものとして混同しにくくなります。

| コンポーネント | 主な役割 | 代替できないもの |
| --- | --- | --- |
| SFT | 指示形式、出力構造、振る舞いの境界をモデルに学ばせる | 事実や引用の真正性は保証できない |
| LoRA / QLoRA | 微調整コストを下げ、adapter の保存と rollback をしやすくする | 汚いデータや曖昧なタスクは補えない |
| ドメインデータエンジニアリング | タスク、出典、ライセンス、脱識別、ラベル、eval を定義する | 外部の最新知識を自動的に使わせることはできない |
| RAG | 更新可能な根拠と citation 経路を提供する | モデルが必ず根拠を正しく使うとは限らない |
| 蒸留 | 強いモデルの検証可能な振る舞いを student に移す | teacher を事実ソースとして扱うことはできない |
| 評価 / 安全性 | 失敗を露出し、リリースゲートと人間レビューの境界を定義する | モデルの失敗を自動修復することはできない |

## 各章の最小実験ループ

各章には拡張実験を置けますが、主線に必要なのは最小ループ1つです。

| 章 | 主実験 |
| --- | --- |
| 1 | 契約 toy classifier：baseline / no step / no zero_grad / overfit tiny |
| 2 | bigram LM：正しい右シフト vs 誤った右シフト |
| 3 | tokenizer + collator：pad が loss に入るか |
| 4 | current token LM vs causal mean LM |
| 5 | causal mask あり attention vs causal mask なし attention |
| 6 | 1 / 2 / 4 層 block の loss と grad_norm 比較 |
| 7 | MiniGPT tiny corpus checkpoint round-trip |
| 8 | tiny HF model：load / generate / train one step / save_pretrained |
| 9 | 20件の SFT サンプルで JSON 出力形式を過学習 |
| 10 | LoRA rank と target_modules の比較 |
| 11 | 同じ契約条項から SFT / RAG / distill / eval の4種類のデータ形態を作る |
| 12 | 5 query の top-k 検索と citation support |
| 13 | teacher サンプルフィルタリング：通過率と reject_reason |
| 14 | 5件の eval item から eval_report と failure_cases を生成 |
| 15 | safety policy が高リスクの境界外出力を止める |
| 16 | 同一モデルの fp16 / int8 / int4 における品質と latency 比較 |
| 17 | 1つの契約条項から end-to-end でリスク JSON を出力 |
| 18 | 胸痛 + 呼吸困難の例で red flags を発火 |
| 19 | release gate がレポート不足・rollback target 不足のバージョンを止める |

## Stage 0：プロジェクト準備と学習方法

目標：ばらばらのチュートリアルではなく、長期的に保守できる学習工程を作る。

中心内容：

- プロジェクト構造
- 学習ルート
- 環境設定
- notebook 規約
- テスト規約
- 教材執筆規約

成果物：

- `README.md`
- `roadmap.md`
- `AGENTS.md`
- `pyproject.toml`
- `environment.yml`
- `lessons/`
- `notebooks/`
- `src/`
- `tests/`
- `projects/`

## Stage 1：PyTorch と深層学習の学習直感

中心問題：ニューラルネットワークは、そもそもどうやって「学ぶ」のか？

学習内容：

- Tensor
- shape
- パラメータ
- loss
- gradient
- backward
- optimizer
- `nn.Module`
- `Dataset`
- `DataLoader`
- 学習ループ

成果物：

- MLP 分類モデル
- 完全な学習ループ
- pytest テストファイル

対応章：`01_pytorch_training_intuition`

## Stage 2：言語モデルの基礎

中心問題：モデルが画像分類ではなく文の続きを書く場合、どう学習すればよいのか？

学習内容：

- next-token prediction
- `input_ids` / `labels` のずれ
- cross entropy
- logits
- softmax
- sampling
- temperature
- top-k / top-p
- bigram language model

成果物：

- bigram language model
- テキスト生成関数 `generate()`
- ごく小さな中国語コーパスでの学習実験

対応章：`02_language_modeling`

## Stage 3：Tokenizer と Dataset 構築

中心問題：モデルは中国語テキストを直接食べられない。では、テキストはどう数字になるのか？

学習内容：

- token
- vocab
- token id
- encode / decode
- padding
- truncation
- attention mask
- BPE の直感
- WordPiece の直感
- language modeling dataset
- chat / SFT データ形式

成果物：

- simple tokenizer
- LM Dataset
- SFT データ形式の例

対応章：`03_tokenizer_and_dataset`

## Stage 4：Embedding とニューラル言語モデル

中心問題：token id は単なる番号なのに、モデルはそこからどう意味を学ぶのか？

学習内容：

- embedding table
- embedding lookup
- hidden dimension
- context window
- neural language model
- parameter update

成果物：

- embedding-based language model
- embedding パラメータの変化を観察できる実験

対応章：`04_embedding_and_neural_lm`

## Stage 5：Attention 機構

中心問題：文中の各 token は、自分が誰を見るべきかをどう決めるのか？

学習内容：

- Q / K / V
- scaled dot-product attention
- attention weights
- causal mask
- self-attention
- attention 可視化

成果物：

- 手書き scaled dot-product attention
- causal mask の検証
- attention weights の可視化

対応章：`05_attention`

## Stage 6：Transformer Block

中心問題：attention は1つのモジュールにすぎない。完全な LLM block には、ほかに何が必要か？

学習内容：

- multi-head attention
- feed forward network
- residual connection
- LayerNorm / RMSNorm
- dropout
- pre-norm / post-norm
- Transformer block

成果物：

- 手書き Multi-Head Attention
- 手書き Transformer Block
- 入出力 shape テスト

対応章：`06_transformer_block`

## Stage 7：Mini GPT をゼロから実装する

中心問題：tokenizer、embedding、Transformer block、LM head を組み合わせれば GPT になるのか？

学習内容：

- decoder-only アーキテクチャ
- positional embedding / RoPE の直感
- LM head
- causal language modeling
- mini GPT の学習
- テキスト生成
- checkpoint の保存と読み込み

成果物：

- 学習可能な mini GPT
- 学習スクリプト
- テキスト生成スクリプト
- checkpoint の保存と読み込み

対応章：`07_mini_gpt`

## Stage 8：Hugging Face ワークフロー

中心問題：現実には毎回ゼロからモデルを書くことはない。オープンソースモデルをどう使うのか？

学習内容：

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `datasets`
- `Trainer`
- Accelerate
- `model.generate`
- モデル保存
- モデル読み込み
- chat template

成果物：

- 小さなオープンソースモデルを読み込む
- 1回の推論を行う
- 最小微調整を1回行う
- モデル結果を保存する

対応章：`08_huggingface_workflow`

## Stage 9：SFT 指示チューニング

中心問題：モデルを「文章の続きを書く」状態から「指示に従って答える」状態へどう変えるのか？

学習内容：

- instruction tuning
- SFT
- system / user / assistant messages
- chat dataset
- データクリーニング
- train / val / test 分割
- 形式一貫性
- 過学習の観察

成果物：

- SFT dataset
- SFT 学習スクリプト
- 学習前後の効果比較

対応章：`09_sft_instruction_tuning`

## Stage 10：LoRA / QLoRA によるパラメータ効率のよい微調整

中心問題：全量微調整は高価すぎる。少数のパラメータだけを学習できないか？

学習内容：

- PEFT
- LoRA
- rank
- alpha
- target_modules
- adapter
- merge adapter
- QLoRA
- 4-bit quantization
- VRAM 最適化

成果物：

- LoRA 微調整スクリプト
- QLoRA 微調整スクリプト
- adapter の保存・読み込みフロー

対応章：`10_lora_qlora`

## Stage 11：ドメインデータエンジニアリング

中心問題：ドメインモデルの能力は主にどこから来るのか。モデルか、データか？

学習内容：

- ドメインデータ収集
- データライセンスと利用境界
- データクリーニング
- 重複・近重複チェック
- 脱識別
- 品質フィルタリング
- 指示データ構築
- RAG chunk 構築
- 蒸留データ構築
- 評価データ構築と eval set の固定
- 法律/医療データのリスク

成果物：

- ドメイン SFT データ形式
- ドメイン評価セット形式
- データクリーニングスクリプト
- データ品質レポート
- データ manifest

対応章：`11_domain_data_engineering`

## Stage 12：RAG 検索拡張生成

中心問題：モデルパラメータはデータベースではない。回答前に資料を調べさせるにはどうするか？

学習内容：

- chunking
- embedding model
- vector store
- retriever
- top-k retrieval
- rerank の直感
- prompt with context
- citation
- RAG hallucination

成果物：

- ローカル RAG baseline
- 小さな知識ベース
- 検索 + 生成 pipeline
- 引用元付きの出力

対応章：`12_rag_baseline`

## Stage 13：小型モデルの蒸留

中心問題：大きなモデルは性能がよいが高価すぎる。その能力をどう小さなモデルへ移すのか？

学習内容：

- teacher model
- student model
- response distillation
- logit distillation の直感
- preference distillation
- 蒸留データ生成
- 蒸留データフィルタリング
- student 学習

成果物：

- teacher データ生成スクリプト
- student 学習スクリプト
- base / teacher / student 比較

対応章：`13_distillation`

## Stage 14：モデル評価

中心問題：モデルは話せるように見える。では、本当に良くなったとどう証明するのか？

学習内容：

- eval set
- 自動評価
- 人手評価
- 形式正確率
- 事実正確率
- 拒否能力
- hallucination test
- RAG 引用正確性
- 法律/医療安全評価

成果物：

- `eval_runner.py`
- `metrics.py`
- `eval_report.md`
- 高リスク failure cases 表

対応章：`14_evaluation`

## Stage 15：安全性、コンプライアンス、モデルカード

中心問題：法律/医療ドメインモデルは、答えがそれらしく見えるだけでは不十分。いつ答えてはいけないかも知る必要がある。

学習内容：

- データ脱識別
- プライバシー保護
- 拒否境界
- 不確実性の表現
- safety prompt
- 法律免責
- 医療免責
- model card
- risk report
- human review

成果物：

- `model_card_template.md`
- `risk_report.md`
- safety test set
- refusal test set

対応章：`15_safety_and_model_card`

## Stage 16：量子化とデプロイ

中心問題：モデルの学習が終わったあと、コスト、latency、throughput、品質、安全性の間で、検証可能な工程上のトレードオフをどう取るか？

学習内容：

- FP32 / FP16 / BF16
- INT8
- INT4
- bitsandbytes
- GGUF の直感
- vLLM の直感
- API server
- batching
- latency
- throughput
- quality / safety regression
- release gate
- rollback

成果物：

- 量子化推論スクリプト
- ローカル API server
- 簡単な benchmark
- release gate と rollback 設定

対応章：`16_quantization_and_serving`

## Stage 17：法律ドメイン小型モデルプロジェクト

中心問題：微調整、RAG、蒸留、評価をどう組み合わせて法律モデルにするか？

プロジェクト方向：

- 契約リスク識別
- 条項説明
- 契約修正提案
- 法律QA RAG
- 法令引用チェック

成果物：

```text
legal_contract_review/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

対応章：`17_legal_domain_project`

## Stage 18：医療ドメイン小型モデルプロジェクト

中心問題：慎重で、安全で、評価可能な医療啓発アシスタントをどう作るか？

プロジェクト方向：

- 医療啓発QA
- 症状説明
- 受診アドバイス
- 医療ガイドライン RAG
- 危険信号識別
- 医師の診断を代替しない

成果物：

```text
medical_qa_assistant/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

対応章：`18_medical_domain_project`

## Stage 19：完全なドメインモデル工程テンプレート

中心問題：ドメインモデルプロジェクトをどう再利用可能なテンプレートにするか？

学習内容：

- プロジェクトディレクトリ規約
- データバージョン管理
- 学習設定管理
- 実験記録
- 評価レポート
- モデルリリース
- 推論サービス
- 継続的改善

成果物：

```text
domain_model_template/
├── configs/
├── data/
├── scripts/
├── src/
├── tests/
├── reports/
└── README.md
```

対応章：`19_domain_model_template`

## 推奨学習順の短縮版

主線だけを見て、すべての拡張を追わない場合の順序は次のとおりです。

1. PyTorch の学習直感
2. 言語モデル基礎
3. Tokenizer と Dataset
4. Embedding とニューラル言語モデル
5. Attention
6. Transformer Block
7. Mini GPT
8. Hugging Face ワークフロー
9. SFT 指示チューニング
10. LoRA / QLoRA
11. ドメインデータエンジニアリング
12. RAG
13. 蒸留
14. 評価
15. 安全性とモデルカード
16. 量子化デプロイ
17. ドメインモデル完全プロジェクト

## 最初に作るべき章

最初のバッチでは、この5章だけを作ることを推奨します。

1. `01_pytorch_training_intuition`
2. `02_language_modeling`
3. `03_tokenizer_and_dataset`
4. `04_embedding_and_neural_lm`
5. `05_attention`

この5章で最小ループができます。

```text
モデルはどう学習するか
  -> テキスト生成はどう定義されるか
    -> テキストはどう数字になるか
      -> 数字はどうベクトルになるか
        -> token はどう互いを見るか
```

この5章を終えたら、次へ進みます。

- `06_transformer_block`
- `07_mini_gpt`
- `08_huggingface_workflow`
- `09_sft_instruction_tuning`
- `10_lora_qlora`

## 最終卒業プロジェクト

卒業プロジェクトは次の2つから選びます。

### 方向 A：法律契約レビュー小型モデル

入力：契約条項。

出力：

- リスクレベル
- リスク点
- 根拠
- 修正提案
- 不確実性の注意

技術構成：

- RAG
- LoRA
- 蒸留
- 評価
- モデルカード

### 方向 B：医療啓発QA小型モデル

入力：ユーザーの医療質問。

出力：

- わかりやすい説明
- 考えられる原因
- いつ受診すべきか
- リスク注意
- 医師の診断を代替しないこと

技術構成：

- RAG
- SFT
- safety refusal
- 蒸留
- 評価
