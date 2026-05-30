# 医療 QA アシスタント プロジェクト

言語: [中文](../../../projects/medical_qa_assistant/README.md) | [English](../../../en/projects/medical_qa_assistant/README.md) | 日本語

学習目標: RAG、SFT、蒸留、安全な拒否、model card を練習しながら、慎重で安全、かつ評価可能な医療啓発アシスタントを構築する。

本プロジェクトは診断、治療、服薬判断を提供するものではなく、医師の代替にもなりません。

## 最小成果物

- `data/`: 信頼できる医療啓発資料、匿名化サンプル、SFT、safety eval。
- `sft/`: 医療啓発 QA フォーマットの構築と LoRA 訓練。
- `rag/`: ガイドライン/啓発資料の chunk、index、引用生成。
- `distill/`: evidence-grounded な teacher データ生成とフィルタリング。
- `eval/`: 危険信号、拒否、引用サポート率、不適切助言率の評価。
- `reports/`: `eval_report.md`、`risk_report.md`、`model_card.md`。

## 出力契約

モデルは次を出力します。

- 平易な説明。
- 可能性のある原因についての慎重な表現。
- 受診すべきタイミング。
- 危険信号。
- 不確実性の説明。
- 資料引用。

## 学習入口

対応する講義章: `lessons/18_medical_domain_project.md`。
