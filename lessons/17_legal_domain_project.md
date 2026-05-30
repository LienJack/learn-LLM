# 第 17 章：法律领域小模型项目

## 1. 本章真正要解决的问题

前 16 章分别学了训练、语言模型、Tokenizer、Transformer、Hugging Face、SFT、LoRA、数据工程、RAG、蒸馏、评测、安全和部署。本章把它们合成一个法律合同审查项目。

本项目不是法律意见系统，也不替代律师。它是一个教学工程：输入合同条款，输出风险提示、依据、修改建议和不确定性说明，并把高风险场景交给人工复核。

核心问题：

```text
如何把微调、RAG、蒸馏、评测组合成法律合同审查小模型？
```

## 2. 问题链

1. 合同审查需要识别条款风险，而不是泛泛聊天。
2. 合同语料需要脱敏、来源记录和风险标签。
3. SFT 让模型学会合同风险输出格式。
4. RAG 提供条款库、模板和内部审查规范作为依据。
5. LoRA 降低领域微调成本。
6. 蒸馏把强模型审查样例迁移给小模型。
7. 评测、安全和模型卡决定项目是否可演示或可上线。
8. 下一章问题：同样的工程闭环如何迁移到医学科普助手？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| contract clause | 合同条款文本 | text | `clause` | 风险识别 |
| risk point | 风险结构 | JSON object | `risk_points` | issue / evidence |
| review guideline | 审查依据 | chunks | RAG knowledge base | citation |
| SFT sample | 指令样本 | messages | `contract_sft.jsonl` | 输出格式 |
| human review | 人工复核 | status | `needs_human_review` | 高风险门禁 |
| model card | 发布说明 | markdown | `model_card.md` | 用途限制 |

## 4. 项目目录

```text
legal_contract_review/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── sft/
│   └── eval/
├── sft/
│   ├── build_dataset.py
│   └── train_lora.py
├── rag/
│   ├── chunk_documents.py
│   ├── build_index.py
│   └── rag_pipeline.py
├── distill/
│   ├── generate_teacher_data.py
│   └── filter_distill_data.py
├── eval/
│   ├── evaluate.py
│   ├── metrics.py
│   └── failure_cases.csv
├── reports/
│   ├── eval_report.md
│   ├── risk_report.md
│   └── model_card.md
└── README.md
```

目录本身就是学习成果：每个文件都对应前面章节的一项能力。

## 5. 数据设计

合同审查数据至少分三类：

```text
contract clauses: 脱敏合同条款
review guidelines: 内部审查规则或公开模板
risk examples: 风险等级、风险点、依据、建议
```

SFT 样本格式：

```json
{
  "id": "contract_sft_0001",
  "source_id": "contract_doc_001",
  "risk_tags": ["liability", "needs_human_review"],
  "messages": [
    {"role": "system", "content": "你是谨慎的合同风险分析助手，不提供最终法律意见。"},
    {"role": "user", "content": "分析以下条款：<CLAUSE>..."},
    {"role": "assistant", "content": "{\"risk_level\":\"medium\",\"risk_points\":[...],\"basis\":[...],\"suggestion\":\"...\",\"uncertainty\":\"需律师复核\"}"}
  ]
}
```

所有真实合同必须脱敏。金额、日期、主体角色可以保留为占位符，便于模型学习合同结构。

合同数据最重要的不是数量，而是来源、标签和边界清楚。一个可训练样本至少要能回答：

```text
这段条款来自哪里？
它是否已经脱敏？
风险标签是谁标的？
答案依据是什么？
是否需要人工复核？
是否允许进入训练集、评测集或只允许做内部示例？
```

脱敏不能只靠把公司名替换成“某公司”。合同中还可能包含金额、账号、地址、联系人、项目名、交易结构和履约时间表。教学项目可以保留结构，把敏感值替换成占位符：

```text
甲方 -> PARTY_A
乙方 -> PARTY_B
人民币 120 万元 -> AMOUNT_1
2026 年 5 月 28 日 -> DATE_1
北京市朝阳区... -> ADDRESS_1
```

这样模型仍能学习合同语言和风险结构，但不会记住真实主体信息。

### 法律资料必须记录管辖区和版本

合同风险不是脱离地区和时间存在的。一个条款在不同管辖区、不同法规版本、不同合同类型下，风险判断可能不同。

因此法律知识库和样本至少要记录：

```text
jurisdiction
source_name
source_version
published_at / effective_at
document_type
license_or_usage_note
```

如果资料版本不明确，模型不应给出确定法律结论。正确行为是：

```text
risk_level = "unknown"
needs_human_review = true
uncertainty = "缺少适用管辖区或资料版本，无法给出确定判断"
```

这不是保守过头，而是法律领域模型的基本边界。

## 6. 输出契约

合同审查模型不应该自由发挥。建议固定 JSON 输出：

```json
{
  "risk_level": "low|medium|high|unknown",
  "jurisdiction": "unknown|CN|other",
  "risk_points": [
    {
      "issue": "...",
      "why_it_matters": "...",
      "evidence": ["source_id#chunk_id"],
      "suggested_revision": "...",
      "confidence": "low|medium|high"
    }
  ],
  "uncertainty": "...",
  "legal_advice_boundary": true,
  "needs_human_review": true
}
```

格式固定后，评测和人工复核才能稳定进行。

这里的输出契约要同时服务三件事：

1. 给用户可读的风险提示。
2. 给系统可解析的结构化字段。
3. 给审计人员可追踪的证据链。

因此 `risk_points` 不能只写一句“存在违约风险”，而要拆成 issue、原因、证据和建议：

```json
{
  "issue": "违约责任范围过宽",
  "why_it_matters": "条款要求 PARTY_A 对所有间接损失负责，可能超出常见责任边界",
  "evidence": ["guideline_002#chunk_04"],
  "suggested_revision": "建议限定为直接损失，并增加责任上限",
  "needs_human_review": true
}
```

如果模型无法找到证据，正确行为不是编造理由，而是输出 `risk_level="unknown"`，并把 `needs_human_review` 设为 `true`。

这里的 `suggested_revision` 不是最终法律意见，而是供人工复核的修改方向。模型不能承诺“这样改一定有效”，也不能基于单条条款判断案件输赢。

## 7. RAG 设计

知识库可以包含：

- 合同模板和条款库。
- 内部审查规范。
- 公开法律科普材料。
- 已批准的示例解释。

RAG pipeline：

```text
clause query
  -> retrieve similar clauses / guidelines
  -> build context with source ids
  -> ask model to analyze only with evidence
  -> output JSON + citations
```

如果没有相关证据，模型应输出 `risk_level="unknown"` 并说明需要人工复核。

## 8. 微调与蒸馏

训练路线：

```text
base instruct model
  -> LoRA SFT on approved contract examples
  -> RAG teacher generates hard cases
  -> filter distilled examples
  -> train student adapter
```

不要让 teacher 直接生成不可审查的法律结论。teacher 输出必须保留证据、prompt 版本、过滤状态和人工抽检结果。

这个项目中，SFT、RAG 和蒸馏各自解决不同问题：

| 组件 | 主要作用 | 不能替代什么 |
| --- | --- | --- |
| SFT | 学会合同审查输出格式和基本表达 | 不能保证引用真实 |
| RAG | 提供条款库和审查规范依据 | 不能保证模型正确使用证据 |
| LoRA | 降低领域格式训练成本 | 不能弥补坏数据 |
| 蒸馏 | 扩充高质量审查样例 | 不能把 teacher 输出直接当真 |
| Eval | 暴露风险、格式和引用失败 | 不能自动解决失败 |

毕业项目的关键，是把这些组件串成闭环，而不是把每个组件单独跑通。

## 9. 评测设计

Eval set 至少覆盖：

- 风险识别：是否找出关键风险。
- 条款解释：是否解释清楚风险原因。
- 修改建议：是否具体但不过度承诺。
- 引用检查：依据是否来自检索材料。
- 拒答能力：证据不足时是否输出 unknown。
- 高风险复核：是否标记 `needs_human_review`。
- 格式准确率：JSON 是否可解析。

报告必须比较：

```text
base model
SFT LoRA
RAG pipeline
distilled student
```

法律评测不能只问“风险等级是否一致”。一个模型可能正确判断 high risk，却给出错误理由；也可能引用存在，但引用不支持结论。因此建议指标拆开：

```text
json_valid_rate: 输出能否解析
risk_level_accuracy: 风险等级是否符合标注
risk_point_recall: 是否找出关键风险点
citation_support_rate: 引用是否支持风险点
unknown_when_no_evidence_rate: 无证据时是否拒绝判断
human_review_recall: 高风险是否触发人工复核
```

失败案例要按 root cause 分类：数据缺口、检索失败、输出格式失败、过度法律结论、安全边界失败。这样下一轮才知道该补数据、改 RAG、调 prompt，还是修改产品边界。

## 10. 安全边界

必须明确：

- 输出是风险提示，不是最终法律意见。
- 高风险条款必须人工复核。
- 资料不足时不能编造依据。
- 不处理未脱敏个人或商业秘密数据。
- 不根据单条条款给出完整法律结论。

安全边界要进入 system prompt、SFT 样本、安全 eval、model card 和 README。

## 11. 部署闭环

最小可演示 API：

```text
POST /review-contract-clause
input: clause text + optional document metadata
output: risk JSON + citations + audit versions + latency
```

上线前必须能回滚：

```text
model_version: legal-lora-v1
adapter_version: legal-adapter-v1
rag_index_version: legal-guidelines-2026-05
prompt_template_version: legal-rag-prompt-v3
safety_policy_version: legal-safety-v2
quantization: int8
rollback_target: legal-baseline-v0
```

法律项目尤其需要审计日志，但日志本身也可能包含敏感信息。教学版可以记录脱敏后的字段：

```text
request_id
model_version
adapter_version
rag_index_version
prompt_template_version
safety_policy_version
quantization
input_hash
retrieved_chunk_ids
output_json
safety_flags
needs_human_review
latency_ms
finish_reason
parse_status
```

发布包还应保留：

```text
benchmark_report.md
deployment_manifest.json
rollback_config.json
```

这样第 16 章的 release gate 才能判断：法律模型不是“能返回风险 JSON”就可以演示，而是报告、版本、引用、安全和回滚都能被检查。

如果输入包含真实合同全文，生产系统还要明确日志保留期限、访问控制和删除机制。课程项目不要求实现完整合规系统，但必须让学习者知道：模型部署不是只开一个 `/predict`。

## 12. 贯穿样例：违约责任条款

本章可以围绕一条简化条款跑完整闭环：

```text
若 PARTY_A 未按期交付，应赔偿 PARTY_B 因此产生的一切损失，包括间接损失、可得利益损失及律师费。
```

系统应完成：

1. 脱敏并生成 SFT 样本。
2. 从条款库检索“责任范围”“间接损失”“责任上限”等相关规范。
3. 输出 JSON 风险提示。
4. 引用检索到的 chunk。
5. 标记 `needs_human_review=true`。
6. 在 eval report 中记录风险识别、引用支持和格式指标。

这条样例把前面章节串起来：Tokenizer 和 SFT 处理文本格式，RAG 提供依据，蒸馏扩充相似案例，评测验证 JSON 与 citation，安全章节要求不把输出包装成最终法律意见，部署章节记录版本和延迟。

对应的 `expected_output.json` fixture 可以先写成：

```json
{
  "risk_level": "high",
  "jurisdiction": "unknown",
  "risk_points": [
    {
      "issue": "违约责任范围过宽",
      "why_it_matters": "条款要求赔偿一切损失，并包含间接损失、可得利益损失及律师费，可能扩大责任承担范围。",
      "evidence": ["guideline_002#chunk_04"],
      "suggested_revision": "建议限定为直接损失，并明确责任上限和除外情形。",
      "confidence": "medium"
    }
  ],
  "uncertainty": "缺少适用管辖区、合同类型和资料版本，不能给出最终法律判断。",
  "legal_advice_boundary": true,
  "needs_human_review": true
}
```

测试不要求模型逐字一致，但必须检查字段存在、JSON 可解析、citation 指向真实 chunk、缺少管辖区时不能输出确定法律结论。

## 13. 必写实验

- 构造 30 条脱敏合同条款 SFT 样本。
- 构建一个小型条款知识库和 RAG index。
- 用 LoRA 训练一个合同风险输出格式模型。
- 评测 JSON 格式准确率、风险识别、引用准确性和拒答能力。
- 写 model card，说明不替代律师和人工复核边界。

## 14. 失败模式

- 输出像法律意见，但没有依据。
- RAG 引用了相似但无关条款。
- 模型把 `medium` 和 `high` 风险混用。
- 修改建议过度具体，超出证据。
- 未脱敏合同进入训练或日志。
- 人工复核标记缺失。
- 输出最终法律结论，但没有管辖区、资料版本或 citation 支持。

## 15. 测试验收

本章 tests 至少验证：

1. 合同 SFT 样本 schema 合法且已脱敏。
2. 输出 JSON 包含 `risk_level`、`risk_points`、`evidence`、`needs_human_review`。
3. RAG citation 指向存在的合同条款或审查规范 chunk。
4. 无证据样本触发 `risk_level="unknown"` 或拒答路径。
5. 输出包含 `jurisdiction` 和 `legal_advice_boundary`。
6. Model card 明确用途限制和人工复核要求。

## 16. 本章记忆锚点与边界

本章最重要的一句话是：

> 法律领域模型的核心不是答得像律师，而是让风险点、依据、边界和人工复核可追踪。

你需要记住：

1. 合同数据必须脱敏。
2. 风险输出必须结构化。
3. citation 必须支持风险点。
4. 缺少证据、管辖区或版本时输出 unknown。
5. 高风险条款必须进入 human review。

本章项目不能替代律师意见。下一章把同一工程闭环迁移到医学科普场景。

## 17. 下一章

法律合同审查强调证据和人工复核。医学科普助手更强调危险信号、就医建议和不替代诊断。下一章把同一工程闭环迁移到医学领域。
