# 第 16 章：量化与部署

## 1. 本章真正要解决的问题

前面章节得到一个经过评测和安全审查的模型，但模型还没有真正进入使用流程。部署时会遇到新的约束：显存、延迟、吞吐、并发、冷启动、监控、回滚和成本。

量化不是为了炫技，而是在质量、显存、延迟和吞吐之间做取舍。它通常能降低权重显存，但不保证在所有硬件、模型结构和并发配置下都更快；是否值得使用，必须用同一套 eval set、同一套解码参数和同一套 benchmark 验证。

服务化也不是“开一个 API”。真正的服务化，是把模型变成一个可观测、可限流、可审计、可回滚的系统组件。

核心问题：

```text
模型训练好了，怎么在成本、延迟、吞吐、质量和安全之间做可验证的工程取舍？
```

本章可以按两个半章学习：

```text
16A 量化实验：同一 eval set 下比较 fp16 / int8 / int4 的质量、显存和延迟
16B 服务化契约：固定 API、日志、manifest、release gate 和 rollback
```

这样主线更清楚：量化回答“用什么格式跑”，服务化回答“如何可观测、可审计、可回滚地跑”。

## 2. 问题链

1. 原始状态：模型在 notebook 里能生成，但这不等于能服务真实请求。
2. 新问题一：权重、KV cache、runtime buffer 和并发请求可能超过显存预算。
3. 新机制一：使用 FP16 / BF16 / INT8 / INT4 或 GGUF 等推理格式降低资源压力。
4. 新边界一：量化可能让格式、引用、安全拒答或长文本生成退化，必须绑定 eval。
5. 新问题二：单请求能跑，不代表并发下延迟、吞吐和错误率可接受。
6. 新机制二：benchmark、batching、KV cache、timeout、限流和 serving engine。
7. 新边界二：batching 可能提高吞吐，也可能提高单请求等待时间。
8. 新问题三：线上回答出错后，如果没有版本、日志和中间状态，很难复盘。
9. 新机制三：API 契约、监控、审计字段、deployment manifest 和结构化错误。
10. 新边界三：新版本可能让安全或引用退化，必须有 release gate 和 rollback。
11. 下一章问题：如何把训练、RAG、评测、安全和部署组合成法律领域完整项目？

## 3. Concept Card

| 概念 | 数学对象 | Shape / 单位 | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| precision | 数值格式 | bits/value | `torch_dtype` | 显存和速度 |
| quantization | 低精度权重 + scale | int weights + scale/zero point | `quantization_config` | 质量回归 |
| KV cache | 历史 key/value 缓存 | `[L, B, H, T, Dh] * 2` | `past_key_values` / server cache | 长上下文显存 |
| prefill | prompt 前向计算 | prompt tokens | benchmark timer | TTFT |
| decode | 逐 token 生成 | output tokens | generation loop | tokens/s |
| batching | 多请求合批 | dynamic batch | serving queue | 吞吐 / p95 |
| API contract | 请求/响应协议 | JSON schema | FastAPI / client | schema test |
| observability | 运行证据 | logs / metrics | logger / monitor | 故障复盘 |
| rollback | 成组版本恢复 | model + adapter + RAG + prompt | deployment config | 回滚演练 |

## 4. 数值格式

常见格式：

```text
FP32: 训练稳定，但显存大。
FP16: 常见推理/训练格式，显存约为 FP32 一半。
BF16: 指数范围更大，现代硬件常用。
INT8: 权重更小，推理常用。
INT4: 更省显存，质量和兼容性更需要验证。
```

不要只看权重大小。推理显存还包括 KV cache、激活、batch、runtime buffer 和碎片。

同一个模型在不同阶段会遇到不同瓶颈：

```text
加载模型时：权重大小决定基础显存门槛
处理长输入时：prefill 计算和 KV cache 变大
生成长回答时：decode token-by-token 成为瓶颈
并发请求时：batching、队列和缓存管理决定吞吐
```

所以“这个模型 4bit 只有几 GB”并不等于“它可以稳定服务 20 个并发长上下文请求”。部署前必须测真实 prompt 长度、输出长度和并发模式。

### 推理显存不只是模型权重

很多初学者看到“4bit 模型只有几 GB”，就以为它一定能稳定部署。这个判断不够。

推理显存至少包括：

```text
weights: 模型权重
KV cache: 每层每个 head 保存历史 key/value
activations / temporary buffers: 当前 forward 的中间结果
batching overhead: 多请求合批带来的额外占用
runtime fragmentation: 推理框架和显存碎片
```

KV cache 可以粗略理解为：

```text
num_layers * batch_size * num_heads * seq_len * head_dim * 2
```

最后的 `2` 对应 key 和 value。

这解释了为什么同一个模型：

```text
短 prompt + 单请求：能跑
长 prompt + 长输出：变慢
长 prompt + 多并发：可能 OOM
```

所以部署前不能只问“模型权重多大”，还要问：

```text
真实输入多长？
平均输出多长？
p95 输出多长？
并发是多少？
是否开启 streaming？
是否包含 RAG 检索和后处理？
```

## 5. 量化实验

量化必须和评测绑定：

```text
baseline FP16/BF16
  -> INT8
  -> INT4
  -> compare quality + latency + memory
```

最小实验表不应只看平均分：

| 版本 | peak memory | p50 latency | p95 latency | TTFT | tokens/s | json valid | citation support | safety regression | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp16/bf16 | | | | | | | | | baseline |
| int8 | | | | | | | | | |
| int4 | | | | | | | | | |

其中：

```text
json valid: 严格格式是否退化
citation support: 引用是否仍然支持答案
safety regression: 高风险拒答、unknown、human_review 是否退化
```

如果 int4 让高风险拒答退化，即使平均质量变化很小，也不能直接上线。

量化评估需要保留同一套输入、同一套解码参数和同一套 eval set。否则你无法判断差异来自量化，还是来自 prompt、temperature 或模型版本变化。

最小实验流程：

```text
load fp16 model -> run eval + benchmark -> save report
load int8 model -> run same eval + benchmark -> save report
load int4 model -> run same eval + benchmark -> save report
compare quality / latency / memory / safety
```

量化不是单向收益。它可能降低显存，却让某些硬件上速度变慢；也可能让普通问答几乎不变，却让严格 JSON 格式更容易出错。因此必须把格式指标和安全指标放进对比表。

一个 int4 版本即使显存更低、吞吐更高，只要 `high_risk_unsafe_answer_rate` 退化或 `citation_support_rate` 明显下降，就不能只因为成本低而发布。

## 6. GGUF 与本地推理

GGUF 常用于本地 CPU/GPU 混合推理生态。学习重点不是记命令，而是理解：

- 权重被转换成推理引擎支持的格式。
- 不同 quant level 在体积、速度和质量上取舍不同。
- tokenizer、chat template 和 special tokens 仍然必须一致。
- 本地推理也要跑同一套 eval，而不是只看能不能输出。

本地推理最常见的隐藏问题，是“模型能输出中文”被误认为“模型和训练时一致”。但 tokenizer、chat template、system prompt、stop tokens 任一不一致，都会导致模型行为变化。部署检查应至少保存：

```text
base model id
adapter id
quantization format
tokenizer version
chat template hash
generation config
eval report id
```

这些字段后续会进入 model card、serving config 和 run manifest。

如果 LoRA adapter 需要 merge 或转换格式，merge 前后都要跑同一套 eval。不能假设转换格式后行为完全一致。

## 7. Serving Engine

最小 API server 可以自己写，但生产推理通常需要专门 serving engine，例如支持：

- continuous batching / dynamic batching。
- KV cache 管理。
- tensor parallel。
- streaming output。
- OpenAI-compatible API。
- 请求队列、超时和取消。

这些能力解决的是并发下 GPU 利用率和用户等待时间。单请求 demo 很快，不代表并发服务可用。

### Serving engine 不会修复模型行为错误

Serving engine 解决的是性能和并发问题：

```text
batching
KV cache
streaming
timeout
queue
parallelism
```

它不能解决：

```text
citation 编造
JSON 格式不稳定
高风险问题越界回答
RAG 检索错误
prompt injection
```

所以部署优化必须和第 14、15 章的评测、安全一起看。一个服务可以很快地输出错误答案；这不是成功部署，而是更快地放大风险。

## 8. API 契约

服务接口要固定：

```json
{
  "request_id": "req_001",
  "messages": [
    {"role": "user", "content": "分析这段合同风险..."}
  ],
  "generation_config": {
    "temperature": 0.2,
    "max_new_tokens": 512
  }
}
```

响应应同时包含用户可见字段和审计字段：

```json
{
  "request_id": "req_001",
  "answer": "...",
  "citations": [],
  "safety_flags": [],
  "needs_human_review": false,
  "model_version": "legal-sft-v3",
  "adapter_version": "legal-lora-v2",
  "rag_index_version": "legal-guidelines-2026-05",
  "prompt_template_version": "rag_prompt_v4",
  "safety_policy_version": "safety_v2",
  "quantization": "int4",
  "generation_config_id": "gen_low_temp_v1",
  "token_usage": {
    "prompt_tokens": 512,
    "completion_tokens": 128
  },
  "finish_reason": "stop",
  "parse_status": "valid_json",
  "latency_ms": 1234
}
```

领域系统不要只返回一段字符串。引用、安全标记、版本和延迟都是排查问题的证据。

这些字段不是为了好看，而是为了回答线上事故里的关键问题：

```text
这次回答来自哪个模型？
挂的是哪个 adapter？
查的是哪个 RAG index？
用的是哪个 prompt 模板？
量化版本是什么？
是否因为 max_new_tokens 截断？
JSON 是否解析成功？
是否触发安全策略？
```

错误响应也要结构化：

```json
{
  "request_id": "req_001",
  "error": {
    "code": "generation_timeout",
    "message": "request exceeded max latency budget"
  },
  "model_version": "legal-sft-v3",
  "retryable": true
}
```

没有错误契约，调用方只能把所有失败都当成 500 或空答案，后续监控和回滚会非常困难。

这些字段让第 14、15 章的评测和安全门禁能延伸到线上。

## 9. Benchmark

Benchmark 至少分三类：

- 单请求延迟：p50、p95、p99。
- 吞吐：tokens/s、requests/s、并发数。
- 质量回归：同一 eval set 上的指标变化。

还要拆分：

```text
prefill time: 处理输入 prompt
decode time: 逐 token 生成
time to first token
total latency
output tokens per second
```

长 prompt 的瓶颈常在 prefill，长回答的瓶颈常在 decode。

### Benchmark 要先定义输入分布

Benchmark 不可比，通常不是因为计时代码错，而是输入条件不同。

报告必须记录：

```text
warmup 次数
并发数
prompt 长度分布，例如 p50=512, p95=2048
输出长度分布，例如 p50=128, p95=512
max_new_tokens
temperature / top_p / top_k
是否 streaming
是否包含 RAG 检索
是否包含 safety filter
硬件和 dtype / quantization
```

一个教学版 benchmark 输入分布可以先固定为：

```text
prompt_len: p50=512, p95=2048
max_new_tokens: 512
concurrency: 1 / 4 / 16
with_rag: true
temperature: 0.2
```

领域模型建议把 latency 拆成：

```text
retrieval_latency_ms
generation_latency_ms
postprocess_latency_ms
total_latency_ms
```

否则你只知道“慢”，但不知道慢在检索、生成、JSON 解析，还是安全后处理。

## 10. 监控与回滚

上线后至少监控：

- 请求量、错误率、超时率。
- p50/p95/p99 延迟。
- 输入/输出 token 分布。
- 拒答率、安全 flag 比例。
- citation 缺失率。
- 用户反馈和人工复核结果。

回滚条件要提前写好：

```text
错误率超过阈值
延迟超过阈值
高风险越界回答出现
RAG citation 大量缺失
新版本 eval regression 失败
```

监控分两类：系统健康和模型行为。系统健康包括延迟、错误、吞吐、资源占用；模型行为包括拒答率、citation 缺失率、安全 flag、人工复核比例和用户反馈。

回滚也要提前演练。一个可回滚部署至少知道：

```text
current_model_version
current_adapter_version
rollback_model_version
current_rag_index_version
rollback_rag_index_version
prompt_template_version
generation_config_id
safety_policy_version
config compatibility
rollback command
owner
```

如果 RAG index、adapter 和 prompt 模板是一起升级的，回滚也要成组回滚。只回滚模型、不回滚 index 或 prompt，可能得到一个从未评测过的组合。

回滚前还应做 compatibility check，确认：

```text
model_version
adapter_version
tokenizer_version
rag_index_version
prompt_template_version
generation_config_id
safety_policy_version
```

这些对象在当前版本和回滚版本中是成组匹配的。

## 11. Release Gate：什么版本不能发布

部署章节最终要落到发布门禁。一个模型版本不能只因为“API 能返回答案”就发布。

最小 release gate：

```text
eval_report_exists == true
risk_report_exists == true
model_card_exists == true
benchmark_report_exists == true
rollback_target_exists == true
json_valid_rate >= threshold
citation_support_rate >= threshold
high_risk_unsafe_answer_rate == 0
p95_latency_ms <= threshold
error_rate <= threshold
```

如果某项失败，版本只能停留在实验环境。

release gate 的价值是把“不能上线”的条件写成脚本，而不是靠最后人工凭感觉判断。

## 12. 贯穿实验：同一模型四种服务配置

本章可以用一个 fake generator 或极小本地模型做教学实验，重点是部署指标而不是模型能力：

```text
config_a: baseline，正常输出，低并发
config_b: quantized，低显存，但可能格式退化
config_c: strict serving，短 max_new_tokens + timeout，延迟低但可能截断
config_d: unsafe new version，高风险样本失败，用于验证 rollback
```

对三种配置运行同一批请求，记录：

```text
latency_ms
tokens_per_second
error_rate
json_valid_rate
model_version
rollback_target
deployment_manifest
benchmark_report
```

这样可以观察一个真实部署取舍：更短的 `max_new_tokens` 可能降低延迟，但也可能让回答截断；量化可能降低显存，但必须确认评测指标没有明显回退。

## 13. 必写实验

- 对同一模型跑 fp16、int8、int4 推理质量和显存对比。
- 写一个本地 API server，返回 answer、citations、model_version、latency。
- 写 benchmark 脚本，报告 p50/p95、tokens/s、并发下错误率。
- 压测不同 batch size / max_new_tokens。
- 演练版本回滚：旧模型和新模型在同一 eval set 上可切换。
- 故意发布缺少 rollback target 的配置，验证 release gate 会失败。
- 故意让量化版本 `high_risk_unsafe_answer_rate` 退化，验证 release gate 会失败。

## 14. 失败模式

- 只看模型文件大小，不看 KV cache 和并发显存。
- 量化后不跑安全 eval。
- API 没有 model version，无法追踪线上输出来自哪个模型。
- benchmark 只测单请求，不测并发和长上下文。
- 没有限流：突发请求拖垮服务。
- 没有回滚：新版本出问题只能临时手工修。
- 只看吞吐：batching 后总 tokens/s 变高，但 p95 latency 已经不可接受。
- benchmark 没有 warmup 或输入分布：报告数字不可比较。
- 日志保存未脱敏原始敏感输入。
- 只回滚模型，不回滚 adapter、RAG index、prompt 和 safety policy。

## 15. 测试验收

本章 tests 至少验证：

1. API success response 包含 `answer`、`model_version`、`latency_ms`、`finish_reason`。
2. API error response 包含 `request_id`、`error.code`、`model_version` 和 `retryable`。
3. generation config 必须包含 `max_new_tokens`，防止无限生成。
4. benchmark report 必须包含 p50、p95、tokens/s、error_rate 和输入长度分布。
5. 量化版本必须跑同一 eval set，并生成 fp16/int8/int4 对比报告。
6. serving config 必须包含 `rollback_target`。
7. release gate 会阻止缺少 eval report、model card、risk report 或 rollback target 的版本。
8. 高风险安全指标退化时，release gate 必须失败。
9. 日志不能保存未脱敏原始敏感输入；至少应支持 hash 或脱敏记录。
10. rollback config 必须成组记录 model、adapter、RAG index、prompt template 和 safety policy。

## 16. 本章记忆锚点与边界

本章最重要的一句话是：

> 部署不是让模型跑起来，而是在质量、成本、延迟、安全、观测和回滚之间建立可验证的工程契约。

你需要记住：

1. 量化通常能降低权重显存，但必须重新评测质量、格式、引用和安全。
2. 推理显存不只有权重，KV cache 和并发会改变显存预算。
3. Benchmark 要同时看 p50/p95、TTFT、tokens/s、错误率和质量回归。
4. API 不能只返回字符串，必须返回版本、引用、安全标记、延迟和错误结构。
5. 回滚必须成组回滚模型、adapter、RAG index、prompt 和 safety policy。
6. release gate 要阻止缺报告、缺回滚、质量退化或安全退化的版本发布。

本章没有解决具体领域任务如何组合。下一章进入法律合同审查项目，把训练、RAG、蒸馏、评测、安全和部署合成一个完整小模型工程。

## 17. 下一章

我们已经有训练、微调、RAG、蒸馏、评测、安全和部署的组件。下一章开始毕业项目：把这些组件组合成法律合同审查小模型。
