# Legal Contract Review Project

Languages: [中文](../../../projects/legal_contract_review/README.md) | English | [日本語](../../../ja/projects/legal_contract_review/README.md)

Learning goal: combine SFT, LoRA, RAG, distillation, evaluation, safety, and model cards into a contract risk-warning system.

This project does not provide legal advice and does not replace a lawyer. Its output is only for risk flagging and human-review assistance.

## Minimum Deliverables

- `data/`: redacted contract clauses, review rules, SFT samples, and eval samples.
- `sft/`: data construction for the contract-risk output format and LoRA training.
- `rag/`: chunking, indexing, and retrieval QA for clause libraries and review guidelines.
- `distill/`: teacher data generation, filtering, and student training.
- `eval/`: evaluation for JSON format, risk detection, citation accuracy, and refusal ability.
- `reports/`: `eval_report.md`, `risk_report.md`, and `model_card.md`.

## Output Contract

The model should output:

- risk level.
- risk points.
- evidence citations.
- suggested revisions.
- uncertainty notes.
- whether human review is required.

## Learning Entry Point

Corresponding course chapter: `lessons/17_legal_domain_project.md`.
