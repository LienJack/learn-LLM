# 第 13 章：蒸馏小模型

## 1. 本章真正要解决的问题

RAG 可以让强模型基于外部证据回答，但每次调用强模型都可能昂贵、慢、不可控。领域项目常常希望把强模型在某些任务上的行为迁移给更小、更便宜、更容易部署的 student model。

蒸馏不是“复制大模型全部能力”。它是在明确任务分布上，把 teacher 的输出、偏好或概率信息转化成 student 的训练信号。

核心问题：

```text
大模型效果好但太贵，怎么把可验证的领域能力迁移给小模型？
```

## 2. 问题链

1. Teacher model 能回答复杂问题，但调用成本高。
2. Student model 便宜，但原始能力不足。
3. Response distillation 用 teacher 生成答案训练 student。
4. Logit distillation 用 teacher 概率分布提供更细的监督。
5. Preference distillation 用成对偏好教 student 选择更好回答。
6. 蒸馏数据必须过滤 hallucination、格式错误和越界建议。
7. 下一章问题：蒸馏后怎么证明 student 真的变好了？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| teacher | 强模型 | function | `teacher.generate` | 生成数据 |
| student | 小模型 | parameters | `student_model` | 微调 |
| response | 文本监督 | messages | `distill.jsonl` | 质量过滤 |
| logits | 概率分布 | `(B, T, V)` | `teacher_logits` | KL loss |
| preference | 偏好对 | pair | `chosen/rejected` | 排序能力 |
| filter | 质量门 | rules/model/human | `filter.py` | 通过率 |

## 4. Response Distillation

最常见蒸馏方式是让 teacher 生成答案，再把答案当作 SFT 数据训练 student：

```text
prompt + retrieved context
  -> teacher answer
  -> filter / edit / approve
  -> SFT example
  -> student fine-tune
```

样本必须记录生成来源：

```json
{
  "id": "distill_0001",
  "teacher_model": "teacher-model-id",
  "teacher_prompt_version": "rag_prompt_v3",
  "generation_config": {"temperature": 0.2, "top_p": 0.9},
  "filter_status": "approved",
  "messages": [...]
}
```

如果不记录 teacher 和 prompt 版本，未来无法解释 student 为什么学到某种回答风格。

Response distillation 的质量取决于 teacher 输出是否适合被 student 学习。Teacher 答得长、流畅、像专家，不代表适合训练。训练样本应该稳定、可验证、符合目标格式，并覆盖拒答和边界场景。否则 student 学到的是 teacher 的语气，而不是可部署的领域能力。

一个蒸馏样本最好同时保留：

```text
prompt
retrieved_context / citations
teacher_response
teacher_model
teacher_prompt_version
generation_config
filter_status
review_notes
source_group
```

这些字段后续会进入过滤、切分和评测。

## 5. Logit Distillation

Response distillation 只给 student 一个目标答案。Logit distillation 还试图让 student 学习 teacher 对词表的软分布。

```text
teacher_logits: FloatTensor[B, T, V]
student_logits: FloatTensor[B, T, V]

teacher_probs_T = softmax(teacher_logits / temperature)
student_log_probs_T = log_softmax(student_logits / temperature)

loss = CE(student_logits, hard_labels)
     + lambda * temperature^2 * KL(teacher_probs_T || student_probs_T)
```

在 PyTorch 里常见形式是：

```python
kl = F.kl_div(
    student_log_probs_T,
    teacher_probs_T,
    reduction="batchmean",
)
```

这里的 KL 方向很重要：我们希望 student 的分布靠近 teacher 的分布。

软分布能表达“哪些错误更接近正确答案”。但它昂贵得多：需要保存或在线计算大词表 logits，也会引入 teacher 的偏见和错误信心。

Logit distillation 的边界也很明显：

1. 如果 teacher 和 student tokenizer 不同，词表分布很难直接对齐。
2. 完整保存 `(B, T, V)` logits 成本很高。
3. 可以只保存 top-k logits，或训练时在线请求 teacher。
4. teacher 的软分布也可能包含错误信心。
5. 高风险领域不能因为 teacher 概率高，就把输出当成事实。

教学项目可以先实现 response distillation，再把 logit distillation 当作进阶实验。

## 6. Preference Distillation

有时 teacher 不直接给标准答案，而是比较两个答案：

```json
{
  "prompt": "...",
  "chosen": "更好、更安全、更有依据的答案",
  "rejected": "更差、幻觉或越界的答案",
  "reason": "chosen 引用了证据，rejected 编造了来源"
}
```

偏好数据适合训练模型避开坏回答，尤其适合安全、拒答和格式稳定性。但它不是本课程主线的第一步，因为它需要更复杂的训练目标。

## 7. 蒸馏数据过滤

Teacher 输出不能直接信任。至少过滤：

- 是否回答了问题，而不是泛泛解释。
- 是否被给定资料支持。
- 是否引用真实来源。
- 是否遵守输出格式。
- 是否包含隐私、越界法律/医学建议或危险建议。
- 是否表达必要的不确定性。

过滤可以分三层：

```text
rule filter: schema、长度、敏感词、citation 存在性
model filter: 让审稿模型判断支持性和风险
human review: 高风险样本人工抽检或全检
```

过滤器不应该只保留“看起来漂亮”的答案。领域 student 还需要学会：

```text
资料不足时拒答
高风险时转人工
引用缺失时不下结论
格式不完整时修正或拒绝
```

如果过滤过程把所有拒答、失败、边界样本都删掉，student 会变得过度自信。好的蒸馏数据集应该同时包含正向回答和安全边界。

蒸馏样本生命周期可以固定为：

```text
eval gap
  -> teacher prompt
  -> teacher response
  -> rule/model/human filter
  -> approved distill jsonl
  -> student SFT / LoRA
  -> same eval set comparison
```

过滤器输出不要只写 passed/failed，至少记录拒绝原因：

| reject_reason | 示例 |
| --- | --- |
| `no_citation` | 答案完整但没有来源 |
| `unsupported_claim` | 引用不支持结论 |
| `unsafe_advice` | 法律/医学越界建议 |
| `bad_format` | JSON 不可解析或字段缺失 |
| `privacy_risk` | 复述了未脱敏个人信息 |
| `over_confident` | 资料不足却给确定结论 |

### 不要用 teacher 自己当唯一审稿人

一个常见错误是：

```text
teacher 生成答案
-> teacher 判断答案好不好
-> 通过的样本训练 student
```

这会把 teacher 的盲点完整传给 student。更稳的过滤应该混合：

```text
规则检查：schema、citation、长度、敏感字段
证据检查：答案是否被 context 支持
模型辅助：另一个 judge model 辅助评分
人工抽检：高风险样本必须人工看
```

尤其是法律和医学场景，teacher 输出越像专家，越容易让人忽视它可能在编造依据或越界建议。

## 8. Student 训练

Student 训练本质上回到 SFT / LoRA：

```text
distilled dataset
  -> train/val/test split by source
  -> SFT or LoRA
  -> compare base / teacher / student
```

必须保留 base student 作为对照。否则 student 变好还是原模型本来就会，无法判断。

## 9. 对比评测

蒸馏报告至少比较三者：

```text
base student: 未蒸馏的小模型
teacher: 生成蒸馏数据的大模型
student: 蒸馏后小模型
```

指标不要只看平均分，还要看切片：

- 常规任务。
- 长上下文任务。
- 无答案/拒答任务。
- 高风险法律/医学任务。
- 格式严格任务。

优秀的 student 不一定超过 teacher，但应在目标成本下接近 teacher，并明显超过 base student。

蒸馏评测还要记录成本和延迟。Student 的目标通常不是绝对超过 teacher，而是在更低成本下达到足够质量：

```text
quality: eval score / human score / citation support
cost: 每 1000 次请求成本
latency: p50 / p95
safety: 高风险拒答和越界回答
```

如果 student 质量略低但成本降低很多，并且安全指标不退化，它可能是更适合部署的选择。反过来，如果 student 平均分接近 teacher，却在高风险切片明显退化，就不能上线。

## 10. 必写实验

- 用 RAG + teacher 生成一小批蒸馏样本。
- 写过滤脚本，统计通过率和拒绝原因。
- 用同一 student base 训练 SFT baseline 与 distill dataset。
- 比较 base / teacher / student 在同一 eval set 上的表现。
- 构造 teacher 错误样本，验证过滤器能拦截一部分。

## 11. 失败模式

- Teacher hallucination 被 student 学会。
- 蒸馏数据太同质：student 只学到模板，没学到能力。
- 只保留 teacher 好看的答案：缺少拒答和失败边界。
- 用 teacher 自己评 teacher 数据：质量过滤过度乐观。
- Student 容量太小：目标能力迁移不过去。
- 对比不含 base student：无法证明蒸馏贡献。

## 12. 测试验收

本章 tests 至少验证：

1. 蒸馏样本必须记录 teacher model、prompt version 和 filter status。
2. 过滤器能拒绝无 citation、空答案、格式错误样本。
3. train / val / test split 不按蒸馏后样本随机泄漏 source。
4. student 训练前后在 tiny eval 上有可观察差异。
5. eval report 同时包含 base、teacher、student 三列。

## 13. 本章记忆锚点与边界

本章最重要的一句话是：

> 蒸馏不是复制大模型全部能力，而是在明确任务分布上压缩一种可验证行为。

你需要记住：

1. response distillation 最简单，但强依赖 teacher 答案质量。
2. logit distillation 更细，但要求 vocab 对齐且成本高。
3. preference distillation 适合学习“哪个回答更好”，但训练目标更复杂。
4. 蒸馏数据必须过滤 hallucination、越界建议和格式错误。
5. student 必须和 base student、teacher 同题对比。

本章没有证明 student 真的变好。下一章进入评测。

## 14. 下一章

蒸馏让小模型更便宜，但“看起来会回答”仍然不是证据。下一章进入模型评测：如何证明模型真的变好、哪里仍然失败。
