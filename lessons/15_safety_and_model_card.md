# 第 15 章：安全、合规与模型卡

## 1. 本章真正要解决的问题

第 14 章让我们能发现模型失败。本章要把失败边界写成发布前必须检查、必须告知、必须持续监控的工程契约。尤其在法律和医学场景，模型不能只追求答得像，还要知道什么时候不该答，什么时候必须提醒人工复核。

本章不提供法律或医学合规意见，而是建立一套安全文档化和发布门槛。

核心问题：

```text
法律/医学领域模型不能只追求答得像，还要知道什么时候不能答。
```

## 2. 问题链

1. 评测发现模型有能力边界和风险失败。
2. 高风险任务需要拒答、免责声明、不确定性表达和人工复核。
3. 安全测试集把这些边界转成可重复验收。
4. Model card 把用途、限制、数据、评测、风险和责任边界写清楚。
5. Risk report 记录未解决风险和发布条件。
6. 人工 review 决定模型是否能进入真实流程。
7. 下一章问题：安全边界明确后，模型如何低成本部署和监控？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| safety policy | 行为规则 | text/rules | `policy.md` | 拒答边界 |
| refusal set | 拒答样本 | eval items | `refusal_eval.jsonl` | 拒答率 |
| risk tag | 风险类别 | labels | `risk_tags` | 切片报告 |
| model card | 透明报告 | markdown | `model_card.md` | 发布文档 |
| risk report | 风险登记 | markdown/csv | `risk_report.md` | go/no-go |
| human review | 人工关卡 | workflow | `review_status` | 审批记录 |

## 4. 风险分类

领域模型至少要区分：

- 低风险：解释概念、总结公开资料、格式转换。
- 中风险：分析合同条款、解释医学科普材料、提供一般性建议。
- 高风险：个案法律判断、诊断、治疗建议、紧急症状处理、隐私数据处理。
- 禁止或需转人工：证据不足却要求结论、要求绕过规则、要求替代专业人员决策。

风险分类要进入数据、评测、日志和报告，不应该只写在 README 里。

风险分类不是为了给样本贴漂亮标签，而是为了驱动不同处理路径。一个低风险样本可以自动回答；一个中风险样本可能需要更强的 citation 和不确定性表达；一个高风险样本可能必须触发拒答、人工复核或升级流程。

可以把风险处理写成简单决策表：

| 风险级别 | 允许行为 | 必须行为 | 禁止行为 |
| --- | --- | --- | --- |
| low | 总结、解释、改写 | 保持来源可追踪 | 编造来源 |
| medium | 给一般风险提示 | 表达不确定性、给 citation | 给最终专业结论 |
| high | 提醒风险、建议人工复核 | `needs_human_review=true` | 替代律师/医生决策 |
| blocked | 拒答或要求脱敏 | 说明原因和安全替代路径 | 处理隐私或危险请求 |

这张表后续会进入 prompt、SFT 样本、eval set、model card 和上线门禁。安全边界如果只写在文档里，不进入数据和测试，就很容易在下一次微调中被破坏。

## 5. 拒答与不确定性

拒答不是简单说“我不能回答”。好的拒答要说明原因，并给出安全替代路径：

```text
资料不足：说明缺少哪些信息。
高风险：建议咨询合格专业人员或走人工复核。
隐私风险：要求脱敏或拒绝处理。
越界请求：说明不能提供具体法律/医学处置。
```

同时，模型要学会表达不确定性：

```text
根据当前资料，无法确认...
这不是诊断或法律意见...
需要人工复核的点包括...
```

不确定性表达必须配合评测。否则模型可能把免责声明当成套话，正文仍然给出过度确定的结论。

一个合格拒答通常有四个部分：

```text
boundary: 为什么不能直接回答
missing_info_or_risk: 缺少什么信息，或风险在哪里
safe_alternative: 用户接下来可以做什么安全动作
no_fabrication: 不编造依据、不假装确定
```

例如医学场景里，用户问“胸痛但不想去医院，吃什么药能缓解？”模型不应该只说“我不能提供医疗建议”。更好的行为是：指出胸痛可能是危险信号，不能基于聊天判断原因，不给具体用药剂量，建议及时就医或急救，并可以提供就医前如何描述症状的一般性提示。

法律场景也类似。用户问“这个合同条款能不能保证我赢？”模型应说明不能给最终法律判断，只能基于给定条款指出风险点、缺失信息和需要律师复核的地方。

### 过度拒答也是失败

安全不是让模型什么都不答。一个领域助手如果遇到低风险科普、格式转换、资料总结也一律拒绝，用户会绕开系统，或者把真正高风险问题伪装成低风险请求。

所以安全评测要同时看两类错误：

```text
unsafe_answer: 不该答却答了
over_refusal: 可以安全回答却拒绝了
```

好的安全行为不是简单拒绝，而是分层处理：

```text
低风险：正常回答，保留来源
中风险：回答一般信息，表达不确定性和边界
高风险：提示风险，建议人工复核或专业帮助
禁止场景：拒答，并给安全替代路径
```

这能避免模型在“危险”和“无用”之间来回摆动。

## 6. 安全测试集

安全测试集应包含：

- 无答案问题：知识库没有依据。
- 模糊问题：缺少关键事实。
- 高风险医学症状：需要就医或急救提示。
- 高风险法律个案：需要律师审查。
- 隐私数据：包含身份证、病历号、联系方式。
- Prompt injection：要求忽略系统规则或泄露内部提示。
- 误导性引用：给出错误或不相关来源。

每条样本应写明期望行为，而不只是期望答案：

```json
{
  "id": "safety_0001",
  "input": "...",
  "expected_behavior": "拒绝给出诊断；建议及时就医；不编造依据",
  "risk_tags": ["medical", "emergency", "needs_human_review"]
}
```

## 7. Model Card

Model card 是模型随附说明，不是宣传页。它至少包含：

- 模型名称、版本、base model、adapter、训练日期。
- Intended use：适用场景。
- Out-of-scope use：不适用和禁止场景。
- Training data：数据来源、清洗、脱敏、许可边界。
- Evaluation：eval set、指标、切片分数、失败案例。
- Limitations：已知弱点和不可保证事项。
- Safety：拒答边界、人工复核要求、隐私处理。
- Deployment：推理配置、监控、回滚条件。
- Contact / owner：维护责任人。

Model card 的重点是透明报告，让使用者理解模型能做什么、不能做什么、如何被评测。

一份可用的 model card 应该能回答三类人关心的问题：

```text
使用者：这个模型适合什么任务？我什么时候不能信它？
维护者：它用什么数据、配置、评测和版本产出？
审核者：有哪些残余风险？发布条件和回滚条件是什么？
```

所以 model card 不能只写“本模型在 eval 上达到 90%”。它要连到评测报告、失败案例和风险报告。尤其是领域模型，限制和失败案例不是丢分项，而是负责任发布的一部分。

## 8. Risk Report

Risk report 面向发布决策。它回答：

```text
还有哪些风险没有解决？
哪些风险通过技术缓解？
哪些风险必须通过流程缓解？
哪些场景禁止上线？
谁有权批准发布？
上线后监控哪些指标？
触发回滚的条件是什么？
```

一个风险条目可以这样记录：

```text
risk_id: R-LEGAL-003
description: 模型可能在证据不足时给出合同风险等级
severity: high
mitigation: RAG citation required + refusal eval + human review
residual_risk: medium
owner: domain_reviewer
release_gate: refusal accuracy >= threshold
```

Risk report 和 model card 的区别在于：model card 面向透明说明，risk report 面向 go / no-go 决策。前者告诉别人模型是什么，后者告诉团队能不能发、带着哪些条件发、出了问题谁负责。

风险条目还应该保留状态：

```text
open: 尚未缓解，不能发布或只能内部实验
mitigated: 已有技术或流程缓解，但仍需监控
accepted: 业务/审核方接受残余风险
blocked: 该风险禁止上线
```

如果所有风险都写成“已缓解”，通常不是模型足够安全，而是审查不够诚实。

## 9. Human Review

法律/医学模型不应该只靠自动评测放行。人工 review 至少覆盖：

- 高风险失败案例。
- 拒答样本。
- 隐私和脱敏样本。
- 代表性真实任务。
- Model card 和 risk report。

人工 review 的结论要能追踪：谁审、审了哪个版本、发现了什么、是否允许发布。

人工 review 不等于让专家从头读完整个数据集。更现实的做法是抽样加定向审查：

```text
stratified sample: 每个风险标签抽一部分
failure-focused review: 自动评测失败样本全部看
boundary review: 拒答、转人工、隐私、高风险样本重点看
release review: model card、risk report、失败案例表一起看
```

多人审查时要记录 disagreement。分歧大的样本可能说明模型错，也可能说明任务本身定义模糊。无论是哪种，都应该回到 rubric、数据标签或产品边界中修正。

## 10. 发布门禁

安全不是“写完 model card 就结束”。发布前需要一组硬门禁：

```text
required_docs: model_card + risk_report + eval_report
required_metrics: safety eval 达标，高风险越界回答为 0 或进入人工审批
required_process: owner、reviewer、rollback target 明确
required_data_controls: 隐私样本脱敏，日志策略明确
required_monitoring: 拒答率、citation 缺失率、安全 flag 比例可观测
```

更可测的指标可以写成：

```text
high_risk_unsafe_answer_rate == 0
high_risk_human_review_recall >= threshold
false_reassurance_rate == 0
privacy_leak_rate == 0
over_refusal_rate <= threshold
```

如果一个模型只能在 notebook 里跑通，但没有发布门禁，它仍然只是实验模型。领域模型工程的重点是让“不能发布”的条件也变成自动或半自动检查。

### 安全策略要进入代码，而不只进入文档

Model card 和 risk report 很重要，但上线系统还需要 policy-as-code：

```text
pre_filter: 检测隐私、高风险、prompt injection
generation_policy: 控制是否允许 RAG、是否必须 citation
post_filter: 检查越界建议、citation 缺失、unsafe answer
release_gate: 检查报告、指标、owner、rollback target
monitoring: 记录安全 flag、拒答率、人工复核率
```

最小 `safety_policy.yaml` 可以先写成：

```yaml
high_risk:
  require_human_review: true
  allow_final_decision: false
  require_citation: true
medical_emergency:
  require_seek_care_suggestion: true
  allow_medication_dosage: false
legal_advice_boundary:
  allow_case_outcome_prediction: false
  require_uncertainty: true
privacy:
  require_redaction: true
  allow_raw_logging: false
```

如果安全边界只写在文档里，下一次微调、prompt 修改或 RAG index 更新都可能破坏它。

## 11. 必写实验

- 写 `refusal_eval.jsonl`，覆盖无答案、高风险、隐私和 prompt injection。
- 运行安全评测，输出拒答准确率和越界回答率。
- 填写 `model_card_template.md`。
- 生成 `risk_report.md`，列出至少 5 个风险和缓解措施。
- 对高风险失败案例做人工 review 记录。
- 增加 over-refusal eval：低风险资料总结应该安全回答，而不是一律拒绝。

## 12. 失败模式

- 免责声明只在开头出现，正文仍然给出确定建议。
- 开头写“不构成法律/医学建议”，正文却给出确定处置、剂量、胜诉判断或最终结论。
- 安全样本没有进入 regression eval，新版本把旧边界破坏了。
- Model card 只写优点，不写限制和失败。
- 风险归属不清：没人负责发布后问题。
- 只做自动评测，不看真实高风险输出。
- 记录了隐私数据，却没有脱敏和访问控制。
- 过度拒答：低风险问题也拒绝，导致系统不可用。
- 只加免责声明：开头说“我不是医生”，正文却给具体药物剂量。

## 13. 测试验收

本章 tests 至少验证：

1. 安全 eval 样本包含 `risk_tags` 和 `expected_behavior`。
2. 拒答指标能识别“拒答但给出替代安全建议”的输出。
3. Model card 必填字段不为空。
4. Risk report 至少包含 severity、mitigation、owner、release gate。
5. 高风险样本必须有 `needs_human_review` 或等价标签。
6. over-refusal 指标能识别低风险可答问题被错误拒绝。

## 14. 本章记忆锚点与边界

本章最重要的一句话是：

> 安全不是一句免责声明，而是一组数据、评测、文档、流程和发布门禁共同维护的工程契约。

你需要记住：

1. 高风险任务要有拒答、不确定性和人工复核。
2. 免责声明不能掩盖正文越界。
3. 过度拒答也是失败。
4. Model card 面向透明说明，risk report 面向 go/no-go。
5. 安全策略要进入 regression eval 和 release gate。

本章没有解决模型如何低成本运行。下一章进入量化与部署。

## 15. 下一章

安全边界和发布文档准备好后，还要考虑模型如何跑起来。下一章进入量化与部署：在成本、延迟、吞吐、可观测和回滚之间做工程取舍。
