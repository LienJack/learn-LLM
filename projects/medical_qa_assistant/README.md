# 医学问答助手项目

教学目标：构建一个谨慎、安全、可评测的医学科普助手，练习 RAG、SFT、蒸馏、安全拒答和 model card。

本项目不提供诊断、治疗或用药决策，不替代医生。

## 最小交付物

- `data/`: 可信医学科普资料、脱敏样本、SFT 和 safety eval。
- `sft/`: 医学科普问答格式构造与 LoRA 训练。
- `rag/`: 指南/科普资料 chunk、index、引用生成。
- `distill/`: evidence-grounded teacher 数据生成与过滤。
- `eval/`: 危险信号、拒答、引用支持率、不当建议率评测。
- `reports/`: `eval_report.md`、`risk_report.md`、`model_card.md`。

## 输出契约

模型应输出：

- 通俗解释。
- 可能原因的谨慎表述。
- 何时就医。
- 危险信号。
- 不确定性说明。
- 资料引用。

## 学习入口

对应课程章节：`lessons/18_medical_domain_project.md`。
