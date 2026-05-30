# 第 5 章：Causal Self-Attention

## 1. 本章真正要解决的问题

第 4 章的 causal mean model 已经能看历史，但它有一个明显问题：

> 它把历史位置按固定规则混在一起，不知道当前 token 真正应该看谁。

看这个句子：

```text
合同 违约金 过高 ， 它 可能 存在 风险
```

当模型预测“可能”后面的 token 时，“它”更应该回看“违约金”，而不是平均看“合同”“过高”和标点。

所以本章真正要解决的问题是：

```text
一句话中每个 token 如何动态决定自己应该看哪些历史 token？
```

这就是 causal self-attention 出现的原因。

## 2. 问题链

1. 原始状态：固定 pooling 能看历史，但不能按上下文动态选择。
2. 问题：不同 token 在不同句子里需要关注不同历史位置。
3. 新机制：每个位置生成 query、key、value。
4. query 与 key 点积，得到“当前位置应该看哪些位置”的分数。
5. softmax 把分数变成 attention weights。
6. value 按权重加权求和，得到上下文表示。
7. causal mask 禁止当前位置看到未来 token。
8. 新边界：单头 attention 只是一次信息混合，完整 LLM block 还需要多头、残差、归一化和 FFN。

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| Q | 查询向量 | `(B, T, H)` | `q_proj(x)` | 点积分数 |
| K | 被查询向量 | `(B, T, H)` | `k_proj(x)` | mask 前 logits |
| V | 被汇聚内容 | `(B, T, H)` | `v_proj(x)` | 加权求和 |
| weights | 注意力分布 | `(B, T, T)` | `softmax(scores)` | 可视化 |
| causal mask | 下三角约束 | `(T, T)` | `torch.tril` | 未来权重为 0 |

## 4. Shape 契约

```text
x:       FloatTensor[B, T, D]
q,k,v:   FloatTensor[B, T, H]
scores:  FloatTensor[B, T, T] = q @ k.transpose(-2, -1) / sqrt(H)
mask:    BoolTensor[T, T]
weights: FloatTensor[B, T, T]
out:     FloatTensor[B, T, H]
```

对 causal LM，`weights[:, i, j]` 在 `j > i` 的位置必须为 0。否则训练时模型偷看答案，loss 会虚低，生成时崩掉。

Attention 的关键不是“所有 token 互相看”，而是“每个位置根据当前表示动态选择信息来源”。同一个 token 在不同句子里可以关注不同位置：

```text
这份合同中的违约金过高，它可能...
这份报告中的指标过高，它可能...
```

两个“它”需要回看不同名词。固定 pooling 很难表达这种条件选择，而 query-key 点积可以让每个位置产生自己的检索分布。

缩放因子 `sqrt(H)` 也不是数学装饰。head dimension 越大，点积方差越大；不缩放时 softmax 容易变得过尖，模型过早只看一个位置，梯度也更不稳定。

## 5. 最小实现

```python
def scaled_dot_product_attention(q, k, v, causal: bool = True):
    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1) / head_dim**0.5
    if causal:
        t = q.size(-2)
        mask = torch.tril(torch.ones(t, t, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```

这段代码是后续 multi-head attention 的核心。先把单头写对，再引入 batch、head 和 projection。

Causal mask 是语言模型和普通序列编码器的重要分界。普通 self-attention 可以让每个位置看完整句；causal self-attention 只能看当前和历史位置。训练时如果忘记 mask，模型会直接看到答案 token，loss 会异常低，但生成时未来 token 不存在，模型会突然变差。

教学实验可以故意跑两版：

```text
with mask: loss 更真实，生成较稳定
without mask: 训练 loss 虚低，生成暴露问题
```

这比单纯说“不能偷看未来”更有说服力。后续所有 decoder-only 模型都建立在这个约束上。

### causal mask 和 padding mask 要分清

本章的最小实现只处理 causal mask：

```text
位置 i 不能看 j > i 的未来 token
```

但真实 batch 里还会有 padding mask：

```text
pad 位置不应该被任何有效 token 当作上下文
```

两种 mask 解决的问题不同：

```text
causal mask: 防止偷看未来
padding mask: 防止看见补齐符号
```

后面写完整 Transformer 时，需要把两者组合起来。组合时还要注意一个数值边界：如果某一行所有位置都被 mask 成 `-inf`，softmax 会产生 NaN。纯 causal mask 不会出现这个问题，因为每个位置至少能看自己；但 padding query 行可能触发这个边界。

组合 mask 的最小 shape 可以这样记：

```text
causal_mask:  BoolTensor[1, 1, T, T]   # query i 不能看 future key j
padding_mask: BoolTensor[B, 1, 1, T]   # key j 是不是有效 token
combined:     BoolTensor[B, 1, T, T]
scores:       FloatTensor[B, H, T, T]
```

也就是说，padding mask 通常先 mask key 维度；如果 pad query 行还会被后续使用，再额外把对应输出清零。不要把 causal mask 和 padding mask 合成一个没有维度说明的二维矩阵，否则 multi-head 版本很容易广播错。

### attention weights 能看，但不能神化

Attention weights 很适合教学可视化，因为它能显示某个位置把多少权重分给历史 token。

但它不是完整解释：

- 权重大，不一定代表最终答案真的由这个 token 决定。
- 多层、多头、FFN 和 residual 会继续改变信息。
- 真正可靠的诊断要结合任务 loss、输出变化和干预实验。

所以本章看 attention weights，是为了检查机制是否工作，而不是宣布“模型已经具备可解释性”。

## 6. 必写实验

- 构造递增 token 序列，验证第 `i` 个位置不能关注 `i+1`。
- 可视化 attention weights，观察每行权重和为 1。
- 去掉缩放因子 `sqrt(H)`，观察 softmax 过尖导致梯度不稳定。
- 去掉 causal mask，观察训练 loss 虚低但生成不可靠。

## 7. 失败模式

- mask dtype 或 device 不一致：运行时报错。
- 使用 `0` 而不是 `-inf` mask：未来 token 仍可能获得权重。
- softmax 维度写错：每列归一而不是每个 query 对所有 key 归一。
- attention weights 只看起来漂亮，但没有和任务 loss 关联。

## 8. 测试验收

本章 tests 至少验证：

1. attention 输出 shape 正确。
2. weights 最后一维求和约等于 1。
3. causal mask 后所有未来位置权重为 0。
4. 禁用 mask 时未来位置可见，用于对照。
5. 修改未来 token 不改变过去位置输出。
6. softmax 维度是 key 维度，而不是 query 维度。
7. attention 在 `float32` 下没有 NaN。
8. 使用 `0` mask 而不是 `-inf` mask 的错误实现应被测试抓住。

## 9. 本章记忆锚点与边界

本章最重要的一句话是：

> Attention 不是让 token “随便互相看”，而是让每个位置根据 query-key 匹配，动态选择应该汇聚哪些历史 value。

你需要记住：

1. `scores = Q @ K^T / sqrt(d_k)`
2. `weights = softmax(scores)`
3. `out = weights @ V`
4. causal LM 中，未来位置必须被 mask 掉。
5. attention weights 可观察，但不是完整解释。

本章没有解决深层训练稳定性，也没有解决多种关系同时建模的问题。

## 10. 下一章

Attention 解决了“看谁”，但 LLM block 还需要多头、残差、归一化和 FFN 才能稳定堆叠。下一章进入 Transformer Block。
