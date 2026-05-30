# Medical QA Assistant Project

Languages: [中文](../../../projects/medical_qa_assistant/README.md) | English | [日本語](../../../ja/projects/medical_qa_assistant/README.md)

Learning goal: build a cautious, safe, and evaluable medical education assistant while practicing RAG, SFT, distillation, safe refusal, and model cards.

This project does not provide diagnosis, treatment, or medication decisions, and it does not replace a doctor.

## Minimum Deliverables

- `data/`: trusted medical education materials, redacted samples, SFT data, and safety evals.
- `sft/`: data construction for medical education QA format and LoRA training.
- `rag/`: chunking, indexing, and citation generation for guidelines and educational materials.
- `distill/`: evidence-grounded teacher data generation and filtering.
- `eval/`: evaluation for red flags, refusals, citation support rate, and inappropriate-advice rate.
- `reports/`: `eval_report.md`, `risk_report.md`, and `model_card.md`.

## Output Contract

The model should output:

- plain-language explanations.
- cautious wording for possible causes.
- when to seek medical care.
- red flags.
- uncertainty notes.
- source citations.

## Learning Entry Point

Corresponding course chapter: `lessons/18_medical_domain_project.md`.
