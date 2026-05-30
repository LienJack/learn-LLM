# ドメイン小規模モデル プロジェクトテンプレート

言語: [中文](../../../projects/domain_model_template/README.md) | [English](../../../en/projects/domain_model_template/README.md) | 日本語

学習目標: 法律・医療プロジェクトに共通するエンジニアリング構造を再利用可能なテンプレートとして抽象化し、金融、教育、カスタマーサポート、企業ナレッジベースなどの領域へ展開できるようにする。

## 標準ディレクトリ

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

## 必須契約

- データバージョン、クリーニングスクリプト、分割ルールを追跡可能にする。
- 訓練設定、モデルバージョン、RAG index バージョンを `run_manifest.json` に書き込む。
- 評価レポート、失敗事例、リスクレポート、model card が相互参照される。
- リリース前に regression eval、安全 eval、rollback チェックを必ず通す。

## 学習入口

対応する講義章: `lessons/19_domain_model_template.md`。
