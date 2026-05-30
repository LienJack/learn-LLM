# Evaluation Report Template

Languages: [中文](../../reports/eval_report_template.md) | English | [日本語](../../ja/reports/eval_report_template.md)

## 1. Basic Information

- Project name:
- Model version:
- Base model:
- Adapter / checkpoint:
- Tokenizer / chat template:
- RAG index version:
- Eval set version:
- Evaluation date:

## 2. Task Definition

Describe the capabilities being evaluated:

- Task input:
- Expected output:
- Applicable scenarios:
- Out-of-scope scenarios:
- High-risk boundaries:

## 3. Data and Splits

- Training data version:
- Validation data version:
- Test data version:
- Data sources:
- De-identification strategy:
- train / val / test split rules:
- Whether duplicates and near-duplicates were checked:

## 4. Metrics

| Metric | Definition | Threshold | Current value | Conclusion |
| --- | --- | --- | --- | --- |
| Format accuracy | Whether the output is parseable | | | |
| Factual accuracy | Whether the output is supported by evidence | | | |
| Citation accuracy | Whether citations exist and are relevant | | | |
| Refusal accuracy | Whether no-answer/high-risk cases are refused | | | |
| Safety violation rate | Whether inappropriate advice is given | | | |
| p95 latency | Service response latency | | | |

## 5. Slice Results

| Slice | Sample count | Main metrics | Failures | Notes |
| --- | --- | --- | --- | --- |
| Regular samples | | | | |
| No-answer samples | | | | |
| High-risk samples | | | | |
| Long-context samples | | | | |
| Strict-format samples | | | | |

## 6. Failure Cases

| eval_id | Input summary | Problem in model output | Suspected cause | Next step |
| --- | --- | --- | --- | --- |
| | | | | |

## 7. Regression Comparison

| Version | Key changes | Metric improvements | Metric regressions | Releasable |
| --- | --- | --- | --- | --- |
| baseline | | | | |
| candidate | | | | |

## 8. Conclusion

- Whether the release gate was passed:
- Issues that must be fixed:
- Acceptable residual risks:
- Next round of data/model/evaluation improvements:
