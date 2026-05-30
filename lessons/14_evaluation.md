# 第 14 章：模型评测

## 1. 本章真正要解决的问题

第 13 章得到一个蒸馏后的 student，但模型“能说话”不等于模型“可靠”。LLM 项目最危险的地方，是用几个看起来不错的样例替代评测，用平均分掩盖高风险失败。

本章补上的能力是：把模型质量拆成可重复运行、可解释、可追踪失败案例的评测系统。

核心问题：

```text
模型看起来会说话，怎么证明它真的变好了？
```

## 2. 问题链

1. 训练 loss 下降只能证明模型更拟合训练目标。
2. Eval set 定义要测试的真实能力和风险边界。
3. 指标把输出转成可比较数字，但数字必须能追溯样本。
4. 自动评测适合格式、引用、检索和部分事实检查。
5. 人工评分适合安全性、完整性、专业性和边界判断。
6. 失败案例表比平均分更能指导下一轮数据和训练。
7. 下一章问题：评测发现高风险边界后，如何写进安全、合规和 model card？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| eval item | 测试样本 | dict | `EvalExample` | gold / rubric |
| prediction | 模型输出 | text / JSON | `Prediction` | 输出解析 |
| metric | 打分函数 | scalar | `metrics.py` | 切片分数 |
| rubric | 评分标准 | levels | `rubric.md` | 人工一致性 |
| slice | 样本子集 | tags | `risk_tags` | 高风险表现 |
| report | 证据汇总 | markdown/json | `eval_report.md` | 回归对比 |

## 4. Eval Set 设计

评测集不是从训练集中随便抽一些样本。它应该覆盖能力、风险和失败边界：

```text
capability: 模型应该会什么
format: 输出是否能被程序解析
grounding: 答案是否被证据支持
refusal: 资料不足时是否拒答
safety: 是否避免越界建议
robustness: 输入有噪声时是否稳定
```

每条 eval 样本至少包含：

```json
{
  "id": "eval_0001",
  "input": "...",
  "expected_behavior": "指出风险并引用来源；资料不足则拒答",
  "gold_facts": ["..."],
  "required_citations": ["doc_001#chunk_03"],
  "risk_tags": ["contract", "needs_human_review"],
  "rubric": "legal_contract_risk_v1"
}
```

这里最容易犯的错误，是把 eval set 当成“训练集的一个留出比例”。对领域模型来说，eval set 更像是一份产品验收清单：它不只要问模型会不会回答，还要问模型在证据不足、高风险、格式严格、用户诱导和长上下文压力下会不会失控。

因此，eval set 的样本来源最好显式分层：

```text
normal capability cases: 目标场景中的常规问题
hard capability cases: 需要多步推理或长上下文的任务
negative cases: 知识库没有答案或证据不足
format cases: 必须输出 JSON / citations / fields
safety cases: 法律、医学、隐私、越界建议
regression cases: 历史上失败过、修过、不能再坏的样本
```

不要追求一开始就做一个很大的 eval set。教学项目可以先从 20-50 条高质量样本开始，但每条样本都要有 `id`、来源、预期行为、风险标签和评分标准。小而可审计的 eval set，比大而来历不清的表格更有价值。

评测样本还要避免训练泄漏。第 9 章和第 13 章已经强调按 `source_group` 切分，评测也一样：如果同一份合同、同一篇医学资料或同一个 teacher 生成批次同时出现在训练和评测中，指标会高得不真实。

## 5. 指标层

不同任务需要不同指标：

- 格式准确率：JSON 是否可解析、字段是否齐全。
- 引用准确率：引用是否存在、是否支持答案。
- 事实准确率：答案事实是否与 gold 或证据一致。
- 拒答准确率：无答案/高风险问题是否拒答。
- 召回率：RAG 是否检索到包含答案的 chunk。
- 人工评分：专业性、完整性、风险表达、可用性。

平均分必须配合切片分数：

```text
overall_score
by_domain
by_risk_tag
by_prompt_type
by_document_source
by_answerability
```

否则模型可能在普通样本上很好，在高风险样本上很差。

一个实用的指标表可以分成三层：

| 层级 | 例子 | 作用 |
| --- | --- | --- |
| 结构指标 | JSON parse rate、字段完整率 | 判断输出能不能进入下游系统 |
| 证据指标 | citation existence、citation support、retrieval recall | 判断回答是否可追溯 |
| 行为指标 | refusal accuracy、risk flag recall、human score | 判断模型是否符合任务边界 |

指标设计要避免“看起来精确但其实不对”的数字。例如，citation existence 只能证明引用字符串存在，不能证明引用真的支持答案；JSON parse rate 只能证明格式可解析，不能证明内容正确。因此报告里应该同时写明每个指标测到了什么、没测到什么。

### 指标要变成 release gate

评测不是只生成漂亮表格。领域模型至少要有发布门槛：

```text
json_valid_rate >= 0.98
citation_support_rate >= 0.90
unknown_when_no_evidence_rate >= 0.95
high_risk_unsafe_answer_rate == 0
privacy_leak_rate == 0
p95_latency_ms <= target
```

阈值可以根据项目阶段调整，但必须提前写清楚。否则团队很容易在看到平均分提升时忽略高风险退化。

release gate 的作用是把“不能上线”的条件写成程序和报告，而不是靠最后开会凭感觉决定。

## 6. 自动评测

自动评测适合可程序验证的目标：

```text
parse_json(output)
check_required_fields(output)
check_citation_exists(output, knowledge_base)
check_answer_contains_refusal(output)
check_retrieved_gold_chunk(top_k)
```

对于事实判断，可以用规则、gold facts、检索证据或 judge model 辅助，但 judge model 不能成为唯一证据。高风险场景需要人工抽检或人工全检。

### Judge model 需要校准

LLM-as-judge 可以辅助判断事实支持、回答完整性和安全边界，但不能直接当真理。

至少要做一个小型校准集：

```text
人工标注 20-50 条样本
judge model 打分
比较 judge 与人工的一致性
记录 judge 容易误判的类型
```

如果 judge 喜欢更长、更礼貌、更像专家的回答，它可能会高估“流畅但无依据”的输出。高风险法律/医学样本必须保留人工抽检或人工全检路径。

## 7. 人工评分

人工评分要有 rubric，不能靠“感觉好不好”：

```text
5: 完全满足任务，事实被证据支持，表达边界清楚
4: 小问题，不影响使用
3: 部分正确，但缺少关键依据或边界
2: 有明显错误，需要人工修正
1: 危险、幻觉、越界或格式不可用
```

多人评分时要记录 disagreement。分歧大的样本往往说明任务定义或 rubric 不清楚。

## 8. Eval Runner

最小评测 runner：

```text
load eval set
for each example:
    build prompt
    run model / RAG pipeline
    parse output
    compute automatic metrics
    save prediction
aggregate metrics
write eval_report.md
write failure_cases.csv
```

每次评测都要保存：

- model id / adapter id / checkpoint。
- tokenizer 和 chat template 版本。
- RAG index 版本。
- generation config。
- eval set 版本。
- predictions 原文。

没有 predictions 原文的 report 不可审计。

一个可审计 prediction 记录应接近这样：

```json
{
  "eval_id": "eval_0001",
  "model_id": "legal-student-v2",
  "input": "...",
  "raw_output": "...",
  "parsed_output": {"risk_level": "medium"},
  "metrics": {
    "json_valid": true,
    "citation_exists": true,
    "refusal_correct": false
  },
  "latency_ms": 842,
  "generation_config": {"temperature": 0.2, "max_new_tokens": 512}
}
```

注意 `raw_output` 和 `parsed_output` 都要保存。只保存解析后的 JSON，会丢掉模型绕过格式、夹带解释、输出多段文本等重要失败线索；只保存原文，又不方便聚合指标。

教学项目中的 runner 可以先用一个本地 fake model 或规则函数代替真实 LLM。关键不是调用多强的模型，而是把 `load -> predict -> parse -> score -> aggregate -> report` 的评测骨架跑通。

## 9. 失败案例表

失败案例表至少包含：

```text
eval_id
input
expected_behavior
model_output
metric_failures
risk_tags
suspected_root_cause
next_action
```

常见 root cause：

- 数据缺口：训练集中没有类似任务。
- 检索失败：RAG 没找到正确资料。
- Prompt 约束弱：模型自由发挥。
- 模型容量不足：student 学不会复杂推理。
- 安全样本不足：拒答边界模糊。

失败案例表不是报告附件，而是下一轮工作的入口。每个失败案例都应该被归到一个行动：

```text
data_gap -> 补训练或蒸馏样本
retrieval_gap -> 调 chunk / embedding / top_k / query rewrite
prompt_gap -> 强化输出契约或拒答条件
metric_gap -> 修改评测逻辑，避免漏判
safety_gap -> 加入安全 eval 和人工复核
product_gap -> 明确该场景不支持
```

如果一个失败案例无法归因，说明你还没有足够证据复盘它；这时应该补日志、保存中间检索结果或增加人工 review，而不是直接“再训练一次看看”。

## 10. 回归评测

每次改数据、prompt、adapter、RAG index 或 decoding 参数，都要跑同一套 regression eval。报告要能回答：

```text
哪些指标变好了？
哪些指标变差了？
哪些高风险样本仍然失败？
是否引入了新的格式错误？
是否存在成本、延迟、拒答率的 trade-off？
```

不要只发布最高平均分版本。领域模型通常需要在准确率、拒答率、延迟和安全之间取舍。

回归评测报告最好包含“变化方向”，而不只是新版本分数：

| metric | old | new | delta | gate |
| --- | ---: | ---: | ---: | --- |
| json_valid_rate | 0.96 | 0.99 | +0.03 | pass |
| citation_support_rate | 0.84 | 0.81 | -0.03 | review |
| high_risk_refusal_rate | 0.92 | 0.88 | -0.04 | fail |

这样才能看见 trade-off。一个版本可能平均分更高，却把高风险拒答做坏了；这种版本在领域项目里应该失败，而不是因为 leaderboard 数字漂亮就发布。

## 11. 贯穿实验：从 5 条样本开始

本章的最小实验可以只包含 5 条 eval item：

1. 一个普通可回答问题。
2. 一个要求 JSON 格式的合同风险问题。
3. 一个必须引用指定资料的问题。
4. 一个知识库没有答案的问题。
5. 一个高风险医学或法律问题。

对这 5 条样本分别保存 prediction、自动指标和失败原因。然后手动制造两个模型版本：

```text
base: 输出自由文本，格式和 citation 经常失败
student: 输出结构更稳定，但仍可能在高风险样本上过度回答
```

即使不训练真实模型，也可以观察评测系统的价值：它会告诉你模型具体坏在哪里，而不是只给一句“效果还可以”。

## 12. 必写实验

- 写 `eval_runner.py` 跑固定 eval set。
- 写 `metrics.py` 计算格式准确率、引用存在率、拒答准确率。
- 生成 `eval_report.md` 和 `failure_cases.csv`。
- 比较 base / SFT / LoRA / RAG / distilled student。
- 构造高风险切片，单独报告法律/医学拒答和人工复核提示。
- 写 release gate 阈值文件，验证高风险失败或 p95 延迟超标时发布检查失败。

## 13. 失败模式

- 用训练样本当 eval：指标虚高。
- 只看平均分：高风险失败被淹没。
- judge model 无校准：自动评分看似客观，实际偏向某种写法。
- 不保存 predictions：无法复盘。
- eval set 太小：单个样本波动改变结论。
- 忽视成本和延迟：效果好但不可部署。

## 14. 测试验收

本章 tests 至少验证：

1. eval item schema 合法且 id 唯一。
2. `eval_runner.py` 输出 predictions、metrics 和 report。
3. JSON 格式指标能正确识别合法/非法输出。
4. citation 指标能发现不存在或不支持答案的引用。
5. regression report 能比较两个模型版本的指标差异。

## 15. 本章记忆锚点与边界

本章最重要的一句话是：

> 评测不是证明模型“看起来不错”，而是把能力、失败、风险和回归变成可重复检查的证据链。

你需要记住：

1. eval set 是产品验收清单，不是训练集留出比例。
2. 平均分必须配合切片分数。
3. 自动指标能测结构和部分证据，不能替代人工风险判断。
4. failure cases 是下一轮数据、RAG、prompt 和安全策略的入口。
5. regression eval 防止新版本修一个问题、弄坏另一个问题。

本章没有定义完整发布边界。下一章进入安全、合规与模型卡。

## 16. 下一章

评测会暴露模型在哪些场景不该答、该提醒、该交给人。下一章进入安全、合规与模型卡，把这些边界写成发布前必须交付的说明和测试。
