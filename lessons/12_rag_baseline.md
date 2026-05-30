# 第 12 章：RAG 检索增强生成

## 1. 本章真正要解决的问题

第 11 章把领域数据工程讲清楚后，一个新问题出现：并不是所有知识都应该写进模型参数。法律条文、医学指南、公司制度和产品文档会更新；很多答案还需要可追溯依据。

RAG 的核心不是“加一个向量数据库”，而是把回答过程拆成两个可检查阶段：先找证据，再基于证据回答。

核心问题：

```text
模型参数不是数据库，怎么让模型回答前先查资料，并把依据暴露出来？
```

## 2. 问题链

1. SFT / LoRA 会改变模型行为，但不能保证知识新鲜和可追溯。
2. 外部知识库可以保存可更新文档。
3. 文档必须切成 chunk，才能被检索和拼进上下文。
4. Embedding model 把 query 和 chunk 映射到同一向量空间。
5. Retriever 找 top-k 候选，reranker 可进一步重排。
6. Generator 只基于检索上下文回答，并输出 citation。
7. 下一章问题：如果大模型调用成本高，能否用 teacher 生成数据训练 student？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| document | 原始资料 | text | `Document` | 来源元数据 |
| chunk | 检索单元 | text span | `Chunk` | chunk size |
| embedding | 稠密向量 | `(D,)` | `embed(text)` | 相似度 |
| vector store | 向量索引 | `(N, D)` | `VectorStore` | top-k |
| retriever | 候选召回 | list[chunk] | `retrieve(query)` | recall |
| context prompt | 带证据提示 | text | `build_prompt` | 引用 |
| citation | 来源指针 | doc id/span | `sources` | 可追踪性 |

## 4. RAG 最小管线

```text
documents
  -> parse
  -> clean
  -> chunk
  -> embed chunks
  -> build index
query
  -> embed query
  -> retrieve top-k chunks
  -> optional rerank
  -> build prompt with context
  -> generate answer
  -> return answer + citations
```

每一步都要能单独测试。否则模型答错时不知道是切块错、召回错、排序错、prompt 错，还是生成器幻觉。

RAG 的工程价值在于可归因。模型回答错了，你可以沿着链路追问：

```text
知识库里有没有答案？
chunk 是否保留了答案所在上下文？
retriever 是否召回正确 chunk？
prompt 是否把证据放进上下文？
generator 是否遵守只基于证据回答？
citation 是否指向真实来源？
```

如果这几个问题都没有日志和中间结果，RAG 就退化成“把文档塞进 prompt”，并没有真正提高可控性。

### RAG 出错时要能定位是哪一段坏了

RAG 的价值不是“加了向量库”，而是错误可以分段诊断：

| 追问 | 可能问题 | 需要查看的对象 |
| --- | --- | --- |
| 知识库里有答案吗？ | 数据缺口 | raw / cleaned documents |
| chunk 保留了答案吗？ | 切块错误 | chunk text / metadata |
| retriever 找到了吗？ | 召回失败 | top-k chunks / scores |
| reranker 排前了吗？ | 排序失败 | rerank scores |
| prompt 放进证据了吗？ | context packing 失败 | final prompt |
| generator 遵守证据了吗？ | 生成幻觉 | answer vs context |
| citation 支持结论吗？ | 引用不真实 | cited chunk span |

如果这些中间结果没有保存，RAG 就无法复盘。

最小 RAG 失败复盘日志可以长这样：

```json
{
  "query_id": "contract_q_001",
  "query": "这段条款是否缺少责任上限？",
  "top_k": [
    {"chunk_id": "guideline_002#chunk_04", "score": 0.82, "reason": "责任范围"},
    {"chunk_id": "guideline_008#chunk_01", "score": 0.77, "reason": "违约责任"}
  ],
  "final_prompt_id": "legal_rag_prompt_v3",
  "answer_parse_status": "valid_json",
  "citations": ["guideline_002#chunk_04"],
  "citation_support": false,
  "failure_root_cause": "retrieved_relevant_but_not_supporting"
}
```

这类日志让你能区分“没检索到”“检索到了但 prompt 没放进去”“引用存在但不支持结论”。没有它，第 14 章的 failure cases 很难落到可修复行动。

## 5. Chunking

Chunk 太小会丢上下文，太大会稀释 embedding 并挤占 prompt。常见策略：

- 固定 token 长度切块。
- 按标题、段落、条款编号切块。
- 使用 overlap 保留跨边界信息。
- 保留 metadata：`doc_id`、标题、页码、段落号、字符范围。

领域文档优先保持语义结构。例如合同条款和法规条文的编号不能随意丢掉；医学指南的适应症、禁忌症、危险信号最好不要切散。

## 6. Embedding 与相似度

Embedding model 决定 query 和 chunk 是否能被放到同一向量空间比较。检索时通常计算 cosine similarity 或 dot product：

```text
query_embedding: FloatTensor[D]
chunk_embeddings: FloatTensor[N, D]
scores: FloatTensor[N]
topk = torch.topk(scores, k=k)
top_k_indices = topk.indices
top_k_scores = topk.values
```

要注意：embedding 相似不等于事实支持。一个 chunk 语义接近问题，但可能没有答案所需的关键证据。

例如用户问“合同是否有责任上限”，一个包含“违约责任”的 chunk 可能相似度很高，但它未必包含“责任上限”条款。评测时要区分：

```text
retrieval relevance: 主题是否相关
answer support: 是否真的支持答案
```

很多 RAG 系统失败，不是因为完全检索不到相关文档，而是检索到了“看起来相关但不够支持结论”的文档。

## 7. Retriever 与 Reranker

Retriever 负责快速召回，reranker 负责精排。最小 baseline 可以先只做 top-k dense retrieval，然后再加入：

- keyword / BM25 召回，补充专有名词和编号。
- hybrid retrieval，合并 dense 和 sparse 结果。
- reranker，对 query-chunk pair 做相关性排序。
- metadata filter，例如只查某个法规版本或文档类型。

每次升级检索策略都要用同一 eval set 比较，不要凭单个样例感觉变好。

## 8. Prompt With Context

RAG prompt 必须明确约束模型：

```text
你只能基于给定资料回答。
如果资料不足，请说“资料不足，无法判断”。
回答中必须引用来源编号。

[资料 1] doc_id=...
...
[资料 2] doc_id=...
...

问题：...
```

这不能完全消除幻觉，但能把行为目标说清楚，并为评测提供依据。

## 9. Citation

Citation 不是在答案后面随便加链接。每个引用应至少包含：

```text
doc_id
chunk_id
title
page_or_section
span_start / span_end
```

评测时要检查两件事：

1. 答案中的事实是否被引用 chunk 支持。
2. 引用 chunk 是否真的来自允许使用的知识库版本。

Citation 还要避免“批量引用”。如果答案包含三个事实，却只在末尾放一个来源，评测很难判断每个事实是否被支持。更好的做法是让风险点、结论或段落分别携带 citation。

在法律/医学场景中，citation 不是学术规范问题，而是安全机制：当模型输出被质疑时，人可以回到来源材料判断它是否越界。

### citation existence 不等于 citation support

评测时要区分两件事：

```text
citation_exists: 引用 id 是否存在
citation_supports_answer: 引用内容是否真的支持答案中的事实
```

例如答案说：

```text
该条款约定了责任上限。
```

但引用 chunk 只出现了“违约责任”，没有出现“责任上限”，那么 citation exists 是 true，citation support 应该是 false。

领域 RAG 的核心验收指标应该包含 citation support rate，而不是只检查答案末尾有没有来源编号。

## 10. 必写实验

- 改变 chunk size 和 overlap，比较 top-k recall。
- 对同一问题比较 dense retrieval、keyword retrieval、hybrid retrieval。
- 构造无答案问题，验证模型是否拒答。
- 构造相似但错误的干扰文档，测试 reranker 和 prompt 约束。
- 输出答案、引用和检索分数，生成一份 RAG 失败案例表。

## 11. 失败模式

- 检索不到：知识库有答案，但 chunk 或 embedding 召回失败。
- 检索到但不用：context 有证据，模型仍靠参数记忆回答。
- 检索到错误相似文档：答案被相近主题误导。
- citation 不真实：引用了来源，但答案事实不在来源中。
- chunk 缺 metadata：无法追溯答案依据。
- prompt 太长：关键证据被截断或排在模型注意力弱的位置。

## 12. 测试验收

本章 tests 至少验证：

1. chunker 输出保留 `doc_id`、`chunk_id` 和文本范围。
2. embedding index 的 top-k 返回稳定且数量正确。
3. retriever 对已知 query 能找回包含答案的 chunk。
4. 无答案 query 返回空证据或触发拒答路径。
5. RAG 输出包含 answer 和 citations，citation 指向存在的 chunk。
6. citation support 指标能发现“引用存在但不支持答案”的样本。

## 13. 本章记忆锚点与边界

本章最重要的一句话是：

> RAG 不是把文档塞给模型，而是把回答拆成可检查的检索、证据、生成和引用链路。

你需要记住：

1. chunk 要保留语义结构和 metadata。
2. embedding 相似不等于事实支持。
3. retriever 负责召回，reranker 负责精排。
4. prompt 要明确资料不足时拒答。
5. citation 要能追溯，并且要支持答案。

本章没有解决强模型调用成本。下一章进入蒸馏。

## 14. 下一章

RAG 让模型回答前查资料，但每次调用强模型生成仍然可能昂贵。下一章进入蒸馏：用 teacher 产生高质量训练信号，让 student 学到更便宜的领域能力。
