# Domain Small-Model Project Template

Languages: [中文](../../../projects/domain_model_template/README.md) | English | [日本語](../../../ja/projects/domain_model_template/README.md)

Learning goal: abstract the common engineering structure from the legal and medical projects into a reusable template that can transfer to finance, education, customer support, enterprise knowledge bases, and other domains.

## Standard Directory

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

## Required Contracts

- Data versions, cleaning scripts, and split rules must be traceable.
- Training configuration, model version, and RAG index version are written to `run_manifest.json`.
- Evaluation reports, failure cases, risk reports, and model cards cross-reference one another.
- Before release, the project must pass regression eval, safety eval, and rollback checks.

## Learning Entry Point

Corresponding course chapter: `lessons/19_domain_model_template.md`.
