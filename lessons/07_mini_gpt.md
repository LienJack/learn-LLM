# 第 7 章：Mini GPT 从零实现

## 1. 本章真正要解决的问题

前面几章分别实现了 LM 目标、tokenizer、embedding、attention 和 block。本章把它们组合成一个 decoder-only language model，并让它完成训练、保存、加载和生成。

但把 block 串起来能 forward，不等于拥有一个可复现 GPT。一个真正可用的 Mini GPT 还要能说明：输入文本怎样变成 id，位置怎样编码，生成时怎样裁剪上下文，checkpoint 是否足以恢复推理或继续训练。

核心问题：

```text
把所有局部机制连接起来，最小 GPT 还需要哪些工程契约？
```

## 2. 问题链

1. LM 目标定义了监督信号。
2. Tokenizer 把文本变成 id。
3. Embedding 和 position embedding 提供输入表示。
4. Transformer blocks 做 causal context mixing。
5. LM head 输出 next-token logits。
6. Checkpoint 不只是保存权重，还要保存配置、tokenizer、生成配置和必要训练状态。
7. 下一章问题：现实中从零训练太贵，如何复用开源模型工作流？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| token embedding | token 表示 | `(V, D)` | `tok_emb` | 参数量 |
| position embedding | 位置表示 | `(T, D)` | `pos_emb` | 上下文长度 |
| blocks | 堆叠变换 | `(B, T, D)` | `nn.ModuleList` | 层数对 loss |
| lm head | 词表投影 | `(D, V)` | `lm_head` | logits |
| checkpoint | 状态快照 | 文件 | `save/load` | 复现生成 |

本章是前 1-6 章的拼装闭环：

| 来自章节 | 零件 | 在 Mini GPT 中的位置 |
| --- | --- | --- |
| 第 1 章 | 训练闭环 | `loss.backward()` / `optimizer.step()` |
| 第 2 章 | next-token loss | `labels` / cross entropy |
| 第 3 章 | tokenizer / dataset | `input_ids` / `attention_mask` |
| 第 4 章 | token embedding | `tok_emb(input_ids)` |
| 第 5 章 | causal attention | block 内部 mask |
| 第 6 章 | transformer block | `nn.ModuleList(blocks)` |

## 4. Shape 契约

```text
input_ids: LongTensor[B, T], T <= block_size
positions: LongTensor[T]
hidden: FloatTensor[B, T, D]
logits: FloatTensor[B, T, V]
labels: LongTensor[B, T]
loss: scalar
```

生成时每一步只取最后一个位置的 logits：

```text
next_logits = logits[:, -1, :]
next_id = sample(next_logits)
input_ids = cat(input_ids, next_id)
```

Mini GPT 的 forward 一次性处理整段训练序列，但 generate 是循环调用：

```text
prompt ids
-> forward
-> 取最后一个位置 logits
-> 采样 next token
-> append
-> 如果超过 block_size，裁剪左侧历史
-> 重复
```

这就是自回归生成。它慢但通用，因为每个新 token 都依赖之前生成的全部上下文。后续部署章节里的 prefill、decode、KV cache，本质上都是围绕这个循环做性能优化。

## 5. 最小实现结构

本章代码至少包含：

- `MiniGPTConfig`
- `MiniGPT`
- `train_mini_gpt.py`
- `generate_text.py`
- `save_checkpoint(path, model, config, tokenizer)`
- `load_checkpoint(path)`

配置必须保存 `vocab_size`、`block_size`、`hidden_dim`、`num_layers`、`num_heads`、`dropout`，否则 checkpoint 无法可靠加载。

Checkpoint 不是只保存 `state_dict`。一个可复现 checkpoint 至少需要：

```text
model_config
model_state_dict
tokenizer vocab / special tokens
training step
random seed or generation config
```

一个教学版 `checkpoint.json` 可以长这样：

```json
{
  "model_config": {
    "vocab_size": 128,
    "block_size": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.1
  },
  "tokenizer": {
    "type": "simple_char",
    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"]
  },
  "training": {
    "global_step": 1200,
    "seed": 42,
    "best_val_loss": 1.73
  },
  "generation_config": {
    "temperature": 0.8,
    "top_k": 20,
    "max_new_tokens": 64
  }
}
```

如果只保存权重，加载时你可能用错 vocab size、block size 或 tokenizer，得到一个看似能运行但行为不一致的模型。第 7 章开始，模型工程从“写模块”进入“保存和复现运行”的阶段。

### checkpoint 分两种：推理恢复和训练恢复

如果 checkpoint 只是为了推理，至少要保存：

```text
model_config
model_state_dict
tokenizer vocab / special tokens
generation_config
```

但如果 checkpoint 还要支持恢复训练，仅保存模型权重不够。还需要保存：

```text
optimizer_state_dict
scheduler_state_dict
global_step / epoch
random seed
torch / cuda / numpy / python RNG state
best validation metric
training config
```

否则你能“加载模型”，但不能恢复同一条训练轨迹。这也是 Mini GPT 从玩具模型进入工程模型的分界线：一个模型文件如果不能说明它由什么数据、什么配置、什么 tokenizer、什么随机状态得到，就很难被复现和审计。

Position embedding 也要特别注意。Token embedding 告诉模型“这个 token 是什么”，position embedding 告诉模型“它在序列哪里”。如果 prompt 长度超过 `block_size`，position id 会越界；生成时必须裁剪上下文或使用支持更长上下文的位置机制。

## 6. 必写实验

- tiny corpus overfit：证明整个 GPT 管线能记住很小语料。
- checkpoint round-trip：保存后加载，给同样 prompt 应输出同样 logits。
- context crop：prompt 超过 `block_size` 时只保留最近上下文。
- temperature / top-k：生成质量和多样性对比。
- 去掉 position embedding 对照：观察模型是否难以区分相同 token 的不同位置。
- train/eval 生成对照：含 dropout 的模型在 `eval()` 下 greedy 生成应可复现。
- resume training：保存 optimizer / scheduler / RNG 后继续训练，对比未保存这些状态的差异。

一个好的 tiny corpus 实验不是为了得到优美文本，而是为了验证整条管线没有断：

```text
tokenizer -> dataset -> model -> loss -> backward -> optimizer -> checkpoint -> generate
```

如果 tiny corpus 都无法 overfit，优先怀疑数据错位、mask、学习率、模型容量或训练循环，而不是怀疑“模型不够大”。这个诊断习惯会贯穿后面 SFT、LoRA 和领域项目。

## 7. 失败模式

- position id 超过 `block_size`：embedding 越界。
- 保存了权重没保存 config：加载时结构不一致。
- tokenizer 版本变了：同一 prompt 的 id 不一致。
- 训练时 teacher forcing，生成时自回归，二者分布不同。
- 只保存 `state_dict`：能加载推理，但无法恢复同一条训练轨迹。
- generate 前忘记 `model.eval()`：dropout 让 greedy 生成也不稳定。

## 8. 测试验收

本章 tests 至少验证：

1. `MiniGPT(input_ids, labels)` 返回 logits 和 loss。
2. logits shape 是 `(B, T, V)`。
3. causal mask 防止未来 token 泄漏。
4. checkpoint 加载后参数逐项相同。
5. 恢复训练 checkpoint 包含 optimizer、scheduler、global step 和 RNG state。
6. `generate()` 输出不会超过指定长度并能遇到 `<eos>` 停止。

## 9. 本章记忆锚点与边界

本章最重要的一句话是：

> Mini GPT 不只是 block 的串联，而是一套从 tokenizer 到 checkpoint 再到 generate 的完整语言模型契约。

你需要记住：

1. token embedding 说明“是什么 token”。
2. position embedding 说明“在什么位置”。
3. blocks 做 causal context mixing。
4. lm head 把 hidden state 投影回 vocab。
5. generate 是自回归循环，不是一次 forward 输出整句。
6. checkpoint 必须保存模型、tokenizer、配置和必要训练状态。

本章没有解决现实项目中的模型复用和生态工具链。下一章进入 Hugging Face 工作流。

## 10. 下一章

从零实现让我们理解结构，但现实项目通常从 Hugging Face 模型开始。下一章学习如何加载、推理、微调和保存开源模型。
