# 法律契約レビュー プロジェクト

言語: [中文](../../../projects/legal_contract_review/README.md) | [English](../../../en/projects/legal_contract_review/README.md) | 日本語

学習目標: SFT、LoRA、RAG、蒸留、評価、安全性、model card を組み合わせ、契約リスクを提示するシステムを作る。

本プロジェクトは法律助言を提供するものではなく、弁護士の代替にもなりません。出力はリスク提示と人間によるレビューの補助に限ります。

## 最小成果物

- `data/`: 匿名化された契約条項、レビュー規則、SFT サンプル、eval サンプル。
- `sft/`: 契約リスク出力フォーマットのデータ構築と LoRA 訓練。
- `rag/`: 条項ライブラリとレビュー基準の chunk、index、検索 QA。
- `distill/`: teacher データ生成、フィルタリング、student 訓練。
- `eval/`: JSON 形式、リスク検出、引用の正確性、拒否能力の評価。
- `reports/`: `eval_report.md`、`risk_report.md`、`model_card.md`。

## 出力契約

モデルは次を出力します。

- リスクレベル。
- リスクポイント。
- 根拠引用。
- 修正提案。
- 不確実性の説明。
- 人間によるレビューが必要か。

## 学習入口

対応する講義章: `lessons/17_legal_domain_project.md`。
