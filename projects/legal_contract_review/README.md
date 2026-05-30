# 法律合同审查项目

教学目标：把 SFT、LoRA、RAG、蒸馏、评测、安全和 model card 组合成一个合同风险提示系统。

本项目不提供法律意见，不替代律师。输出只能作为风险提示和人工复核辅助。

## 最小交付物

- `data/`: 脱敏合同条款、审查规则、SFT 和 eval 样本。
- `sft/`: 合同风险输出格式数据构造与 LoRA 训练。
- `rag/`: 条款库和审查规范的 chunk、index、检索问答。
- `distill/`: teacher 数据生成、过滤和 student 训练。
- `eval/`: JSON 格式、风险识别、引用准确性、拒答能力评测。
- `reports/`: `eval_report.md`、`risk_report.md`、`model_card.md`。

## 输出契约

模型应输出：

- 风险等级。
- 风险点。
- 证据引用。
- 修改建议。
- 不确定性说明。
- 是否需要人工复核。

## 学习入口

对应课程章节：`lessons/17_legal_domain_project.md`。
