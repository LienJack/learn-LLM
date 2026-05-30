# 第 19 章：完整领域模型工程模板

## 1. 本章真正要解决的问题

第 17 章和第 18 章分别做了法律、医学项目。两个领域差异很大，但工程骨架相似：数据治理、SFT、RAG、蒸馏、评测、安全、部署和持续迭代。

本章把这些共同部分抽象成可复用模板。目标不是再写一个 demo，而是建立一个新领域项目也能照着启动、审查、训练、评测和发布的工程结构。

核心问题：

```text
如何把一个领域模型项目做成可复用模板？
```

## 2. 问题链

1. 单个领域项目可以手工拼，但难以复用。
2. 可复用模板必须固定目录、配置、数据契约和报告。
3. 数据版本、模型版本、RAG index 版本要能互相追踪。
4. 训练、评测、部署要有统一命令入口。
5. 风险、安全和人工复核要成为模板的一部分。
6. 持续迭代依赖 regression eval 和 failure cases。
7. 课程收束：从 tensor 训练闭环走到领域模型工程闭环。

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| domain template | 项目骨架 | directory tree | `domain_model_template/` | 新领域迁移 |
| config | 实验参数 | YAML / JSON | `configs/*.yaml` | 可复现 |
| manifest | 运行证据 | JSON | `run_manifest.json` | 版本追踪 |
| report chain | 发布证据 | markdown/csv | `reports/` | go/no-go |
| release gate | 发布门槛 | rules | check script | 阻止半成品 |
| failure loop | 迭代闭环 | cases -> actions | `failure_cases.csv` | 持续改进 |

## 4. 模板目录

```text
domain_model_template/
├── configs/
│   ├── data.yaml
│   ├── train_lora.yaml
│   ├── rag.yaml
│   ├── eval.yaml
│   └── serving.yaml
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── sft/
│   ├── distill/
│   └── eval/
├── scripts/
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── build_rag_index.py
│   ├── generate_distill_data.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── serve.py
│   ├── check_release_gate.py
│   └── rollback.py
├── src/
│   ├── data/
│   ├── training/
│   ├── rag/
│   ├── evaluation/
│   ├── safety/
│   └── serving/
├── tests/
│   ├── test_data_schema.py
│   ├── test_rag_pipeline.py
│   ├── test_metrics.py
│   ├── test_safety_policy.py
│   └── test_release_gate.py
├── reports/
│   ├── data_quality_report.md
│   ├── eval_report.md
│   ├── failure_cases.csv
│   ├── risk_report.md
│   ├── model_card.md
│   └── run_manifest.json
└── README.md
```

模板不是目录展示。每个目录都要对应一个可运行命令、一个可测试契约或一个发布证据。

## 5. 配置管理

不要把关键实验参数散落在脚本里。至少配置：

```yaml
project:
  name: domain_model_template
  domain: legal|medical|custom

base_model:
  model_id: ...
  revision: ...

data:
  train_path: data/sft/train.jsonl
  val_path: data/sft/val.jsonl
  eval_path: data/eval/eval.jsonl
  split_seed: 42

training:
  method: lora
  learning_rate: 0.0002
  batch_size: 4
  gradient_accumulation_steps: 8
  max_seq_length: 2048

rag:
  index_version: ...
  chunk_size: 512
  top_k: 5

serving:
  model_version: ...
  adapter_version: ...
  rag_index_version: ...
  prompt_template_version: ...
  safety_policy_version: ...
  quantization: ...
  rollback_target: ...
```

配置文件是实验复现的入口，也是报告生成的依据。

配置管理的关键不是 YAML 语法，而是把“会影响结果的选择”从脚本中拿出来。凡是改变后会影响训练、检索、评测或部署的参数，都应该可追踪：

```text
模型：base model、revision、adapter、quantization
数据：路径、版本、split seed、过滤规则
训练：学习率、batch、max length、LoRA rank
RAG：chunk size、overlap、embedding model、top_k
评测：eval set、metrics、thresholds、slices
服务：max_new_tokens、timeout、model / adapter / RAG / prompt / safety policy version、rollback target
```

当报告中出现指标变化时，你才能回到配置，判断到底是哪一项改变导致了结果变化。

## 6. 数据版本管理

每次训练都要能追踪：

```text
raw data version
cleaning script version
SFT dataset version
distill dataset version
eval dataset version
RAG index version
```

推荐在训练输出中保存 `run_manifest.json`：

```json
{
  "run_id": "2026-05-28_lora_v3",
  "base_model": "model-id@revision",
  "model_version": "domain-model-v3",
  "adapter_version": "domain-adapter-v3",
  "dataset_version": "sft_v3",
  "rag_index_version": "kb_v5",
  "prompt_template_version": "rag_prompt_v4",
  "safety_policy_version": "safety_v2",
  "quantization": "int8",
  "config_files": ["configs/train_lora.yaml", "configs/eval.yaml"],
  "benchmark_report": "reports/benchmark_report.md",
  "rollback_target": "domain-model-v2",
  "git_commit": "..."
}
```

`run_manifest.json` 是整套工程的证据索引。它不替代报告，但它告诉你报告来自哪次运行。一个领域模型版本如果没有 manifest，后续很难回答：

```text
这个 adapter 用的是哪份 SFT 数据？
评测时用的是哪个 RAG index？
model card 里的分数对应哪个 checkpoint？
线上这个输出来自哪个 prompt 版本？
```

manifest 的字段不需要一开始完美，但必须覆盖模型、数据、配置、代码和评测产物。

## 7. 统一命令入口

模板应提供一组稳定命令：

```bash
python scripts/prepare_data.py --config configs/data.yaml
python scripts/train_lora.py --config configs/train_lora.yaml
python scripts/build_rag_index.py --config configs/rag.yaml
python scripts/evaluate.py --config configs/eval.yaml
python scripts/serve.py --config configs/serving.yaml
```

命令稳定后，CI、文档、教学和生产迁移都更容易。

统一命令入口也让课程从 notebook 走向工程。Notebook 适合探索和教学，脚本适合复现和自动化。一个成熟项目可以两者并存：

```text
notebooks/: 解释机制、可视化、手动观察
scripts/: 固定流程、可复现运行、CI 调用
src/: 可测试的核心逻辑
tests/: 工程契约和回归保护
reports/: 运行结果和发布证据
```

如果某个关键流程只能在 notebook 里靠手工点运行，它就还没有进入工程闭环。

## 8. 报告链路

每个 run 至少输出：

- `data_quality_report.md`
- `eval_report.md`
- `failure_cases.csv`
- `risk_report.md`
- `model_card.md`
- `run_manifest.json`
- `benchmark_report.md`
- `deployment_manifest.json`

报告之间要能互相引用：eval report 引用数据版本，model card 引用 eval report，risk report 引用 failure cases。

## 9. 测试体系

模板 tests 不只测代码能不能跑，还要测工程契约：

- 数据 schema。
- 脱敏规则。
- train/eval 不泄漏。
- RAG citation 存在。
- 输出格式可解析。
- 安全样本触发拒答或人工复核。
- model card 必填项完整。

这些 tests 是领域项目的“护栏”。每加一个领域，都要先补对应护栏。

模板测试可以分成四类：

| 类型 | 例子 | 防止什么问题 |
| --- | --- | --- |
| schema tests | JSONL 字段、config 必填项 | 数据/配置变形 |
| split tests | source_group 不泄漏 | 指标虚高 |
| behavior tests | 拒答、citation、格式解析 | 模型输出越界 |
| release tests | report、model card、rollback target | 半成品上线 |

这些测试不要求训练真实大模型。很多测试可以用小样本、fake model 或规则输出完成。重点是把工程契约固定下来。

## 10. 发布门槛

一个领域模型版本发布前至少满足：

```text
data quality report 已生成
eval report 无关键回归
safety eval 达到阈值
model card 完整
risk report 已审查
rollback target 可用
owner 已确认
```

如果任一项缺失，模型只能停留在实验阶段。

### release gate 要写成脚本

发布门槛不能只写在 README 里。模板应提供：

```bash
python scripts/check_release_gate.py --manifest reports/run_manifest.json
```

至少检查：

```text
eval_report 存在
risk_report 存在
model_card 存在
run_manifest 存在
rollback_target 非空
benchmark_report 存在
safety eval 达标
高风险失败没有新增
model / tokenizer / adapter / RAG index / prompt / safety policy 版本齐全
```

缺任一项，脚本应返回非零退出码。这样 CI、教学作业和真实项目都能用同一套门禁。

## 11. 持续迭代

上线后迭代循环：

```text
collect failures
  -> label root causes
  -> update data / prompt / RAG / adapter
  -> run regression eval
  -> update model card and risk report
  -> release or rollback
```

每个失败案例都要进入某个行动：

- 补数据。
- 改 prompt。
- 改检索。
- 调整安全策略。
- 标记产品不支持。

失败案例如果不进入迭代系统，就只会在下个版本重复出现。

持续迭代也要避免“看到失败就补一条数据”的短视做法。每个失败案例先做 root cause，再决定行动：

```text
retrieval_failure -> 改 chunk / embedding / query / index
format_failure -> 改 prompt / SFT 格式样本 / parser
safety_failure -> 补 safety eval / 拒答样本 / policy
knowledge_gap -> 补知识库或训练数据
capacity_gap -> 换模型、调 LoRA、减少任务复杂度
product_gap -> 明确不支持该场景
```

这样课程的终点才不是“跑通一次”，而是形成能持续改进的领域模型系统。

## 12. 新领域迁移步骤

把模板迁移到一个新领域，可以按这个顺序：

1. 写清楚 intended use 和 out-of-scope use。
2. 定义输出契约和安全边界。
3. 收集 20-50 条高质量种子样本。
4. 建立最小 RAG 知识库和 citation 规则。
5. 写 eval set，先覆盖失败边界而不是追求数量。
6. 跑 base model，生成第一版 failure cases。
7. 决定是先改 prompt、补 RAG，还是做 SFT / LoRA。
8. 生成 model card、risk report 和 run manifest。
9. 写 release gate，阻止缺报告、缺回滚、缺安全评测的版本发布。

这个顺序刻意把评测和安全提前。因为领域模型最常见的失败不是“模型不会说话”，而是“模型说得太像真的，但边界和证据不可靠”。

一个 90 分钟迁移作业可以不训练模型，只做企业客服或教育问答的最小工程壳：

| 步骤 | 交付物 |
| --- | --- |
| 1 | `intended_use.md`：能做什么、不能做什么 |
| 2 | `output_schema.json`：固定回答字段和 citation 字段 |
| 3 | `eval.jsonl`：5 条成功样本 + 5 条失败边界 |
| 4 | `run_manifest.json`：base model、RAG index、prompt、安全策略版本 |
| 5 | `check_release_gate.py`：缺 eval/model card/rollback target 时失败 |

这个作业刻意不训练模型。目的不是追效果，而是验证学习者能把法律模板迁移成“可评测、可审查、可回滚”的新领域项目。

## 13. 必写实验

- 复制模板目录，创建一个新领域项目骨架。
- 填写 `configs/data.yaml`、`configs/eval.yaml` 和 `configs/serving.yaml`。
- 生成一个包含模型、数据、RAG index 和配置版本的 `run_manifest.json`。
- 写一个最小发布检查，缺少 eval report、model card 或 rollback target 时失败。
- 用 5 条 failure cases 跑一次 root cause 分类，输出下一轮行动清单。

## 14. 毕业验收

完成本课程后，学习者应能交付：

1. 一个可运行的最小训练闭环。
2. 一个可解释的 mini GPT 主干。
3. 一个 Hugging Face SFT / LoRA 工作流。
4. 一个带 citation 的 RAG baseline。
5. 一个蒸馏数据生成和过滤流程。
6. 一个 eval runner 和 failure cases 报告。
7. 一个 model card 和 risk report。
8. 一个可部署、可回滚的领域模型项目模板。

这些交付物之间应该能互相连接：训练闭环产生模型，SFT/LoRA 调整行为，RAG 提供证据，蒸馏扩展能力，评测发现失败，安全文档定义边界，部署配置保留版本和回滚路径。

最终仓库结构可以收束成：

```text
mini_gpt/
hf_sft_lora/
domain_project_legal/
domain_project_medical/
domain_template/
reports/
```

评分也应按工程闭环拆开，而不是只看模型输出是否好看：

| 模块 | 权重 |
| --- | ---: |
| 训练闭环与 Mini GPT | 20% |
| HF / SFT / LoRA 工作流 | 20% |
| RAG 与 citation support | 20% |
| Eval / safety / model card | 25% |
| Serving / manifest / release gate | 15% |

## 15. 失败模式

- 模板只剩目录，没有命令和报告。
- 配置散落在脚本里，实验不可复现。
- eval set 没有版本，回归无法比较。
- RAG index 更新后没有同步 model card。
- 风险报告滞后于模型发布。
- 所有领域共用同一安全策略，忽略领域差异。
- release gate 只写在文档里，没有脚本或 CI 入口。

## 16. 测试验收

本章 tests 至少验证：

1. 模板目录包含 configs、data、scripts、src、tests、reports。
2. 每个 config 可解析，且包含必填字段。
3. `run_manifest.json` 能记录模型、数据、RAG index 和配置版本。
4. 报告必填文件存在且互相引用版本。
5. 发布检查脚本能阻止缺少 eval report 或 rollback target 的版本。

## 17. 课程收束

这条路线从第 1 章的：

```text
forward -> loss -> backward -> optimizer.step
```

开始，到第 19 章的：

```text
data -> train -> RAG -> distill -> eval -> safety -> deploy -> monitor -> rollback
```

结束。

中间每一章都在补一个真实工程能力：训练、建模、表示、上下文、复用、微调、检索、蒸馏、评测、安全、部署和持续迭代。

本课程的最终记忆锚点是：

> 不是把 LLM 做成一个能聊天的 demo，而是把模型行为做成可训练、可检索、可评测、可审查、可部署、可回滚的工程系统。

如果你能把这个模板迁移到一个新领域，并留下数据、代码、测试、报告和回滚路径，这门课就不再只是“学过 LLM”，而是已经能开始做可维护的领域小模型工程。
