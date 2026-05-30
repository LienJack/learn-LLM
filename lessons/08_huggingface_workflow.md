# 第 8 章：Hugging Face 工作流

## 1. 本章真正要解决的问题

第 7 章从零实现 mini GPT，是为了理解语言模型内部结构。真实项目通常不会从随机初始化开始训练，而是复用开源模型、tokenizer、配置、权重格式和训练工具链。

本章补上的能力是：把“我理解了 GPT 结构”升级成“我能可靠加载、推理、最小微调、保存和复现实验”。

从零实现让我们看清结构；Hugging Face 让我们复用生态，但也把错误藏进配置、tokenizer、revision 和 checkpoint 里。

核心问题：

```text
现实中不可能每次从零训练模型，如何使用开源模型而不丢掉前面建立的工程判断？
```

## 2. 问题链

1. 从零训练证明了结构可行，但数据、算力和时间都不现实。
2. Hugging Face Hub 提供模型权重、config、tokenizer 和 processor。
3. `AutoTokenizer` 和 `AutoModelForCausalLM` 让代码从具体架构中解耦。
4. `model.generate()` 复用标准自回归生成流程，但仍需要控制 prompt、采样和停止条件。
5. `datasets` 和 `Trainer` 把数据处理、训练参数、评估、保存组织成可复现工作流。
6. Accelerate 处理设备、混合精度和分布式训练入口，但不能替代实验设计。
7. 下一章问题：加载模型后，如何让它从“续写文本”变成“按指令回答”？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| pretrained config | 架构超参 | JSON | `AutoConfig` | hidden size / layers |
| tokenizer | 文本到 id | `(B, T)` | `AutoTokenizer` | chat template |
| causal LM | next-token 模型 | logits `(B, T, V)` | `AutoModelForCausalLM` | prompt 推理 |
| dataset row | 训练样本 | dict | `datasets.Dataset` | map / split |
| trainer state | 训练过程 | checkpoint | `Trainer` | save / resume |
| generated ids | 输出 token | `(B, T+N)` | `model.generate` | decode |

### MiniGPT 到 Hugging Face 的映射

Hugging Face 没有改变第 7 章的模型契约，只是把对象标准化：

| Mini GPT 对象 | Hugging Face 对象 | 检查点 |
| --- | --- | --- |
| `MiniGPTConfig` | `AutoConfig` | hidden size、layers、vocab size 是否一致 |
| simple tokenizer | `AutoTokenizer` | special tokens、chat template、pad token |
| `MiniGPT.forward` | `AutoModelForCausalLM.forward` | `input_ids`、`attention_mask`、`labels` |
| 手写 `generate()` | `model.generate()` | max length、sampling、stop tokens |
| `save_checkpoint()` | `save_pretrained()` | model、tokenizer、config 是否同目录保存 |
| 训练 history | `TrainerState` / logs | seed、step、eval report 是否可复现 |

这张表是第 7 章到第 8 章的桥：前面从零实现不是玩具练习，而是让你知道开源工具链背后每个对象应该承担什么契约。

## 4. 最小推理工作流

加载模型时要把 tokenizer、model、device、dtype 和 trust policy 写清楚：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Large language models learn to"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

教学阶段优先使用小模型验证流程。不要一开始就下载大模型，否则错误会被显存、网络和权限问题淹没。

生产或可复现实验中不要只写一个浮动的 model id。应尽量固定 revision：

```python
revision = "commit_hash_or_tag"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
```

否则未来同一个 model id 指向的权重、tokenizer 或配置可能变化，旧实验无法复现。

Hugging Face 工作流最重要的变化，是很多前面手写的对象变成了标准接口：

```text
config: 模型结构和超参数
tokenizer: 文本协议
model: 权重和 forward
generation_config: 生成策略
trainer_state: 训练过程记录
```

这会让上手变快，也会让错误更隐蔽。比如 tokenizer 和 model 来自不同目录时，代码可能仍能运行，但 token id 与 embedding row 对不上，模型输出就不可解释。因此本章的重点不是“记住 API”，而是把前面建立的 shape、mask、tokenizer、checkpoint 判断迁移到开源模型生态里。

### tokenizer 和 model vocab 必须对齐

Hugging Face 里最隐蔽的错误之一，是 tokenizer 和 model 看起来都能加载，但 vocab 实际不一致。

检查方式：

```python
num_tokenizer_tokens = len(tokenizer)
num_embedding_rows = model.get_input_embeddings().weight.size(0)

assert num_tokenizer_tokens <= num_embedding_rows
assert model.get_output_embeddings().weight.size(0) == model.config.vocab_size
```

如果你新增了 special tokens，例如：

```python
tokenizer.add_special_tokens({"pad_token": "<pad>"})
```

就必须同步调整模型 embedding：

```python
model.resize_token_embeddings(len(tokenizer))
```

否则新增 token 没有对应 embedding row，训练和推理都会变得不可解释。

对很多 causal LM，如果原本没有 `pad_token`，教学阶段可以临时设置：

```python
tokenizer.pad_token = tokenizer.eos_token
```

但要知道这只是工程折中。它解决 batch padding 报错，不代表 `<pad>` 和 `<eos>` 在语义上相同。真正训练时仍要确保 pad 位置不参与 loss。

### `trust_remote_code` 是安全边界

有些模型需要：

```python
trust_remote_code=True
```

这意味着加载模型时会执行仓库中的自定义 Python 代码。教学项目默认不要开启，除非你明确知道模型来源、代码内容和风险。

## 5. Chat Template

指令模型不是直接吃“随便拼接的字符串”。不同模型有不同对话格式，例如 system、user、assistant 的边界 token 可能不同。应优先使用 tokenizer 自带的 chat template：

```python
messages = [
    {"role": "system", "content": "你是谨慎的中文技术助教。"},
    {"role": "user", "content": "解释什么是 causal mask。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

如果先 `apply_chat_template(tokenize=False)` 再调用 tokenizer，必须避免重复添加特殊 token。模板、special tokens 和 label mask 是 SFT 章节的关键边界。

## 6. 最小微调工作流

最小微调不是“跑一个 Trainer demo”，而是要固定以下契约：

```text
raw examples
  -> format text / messages
  -> tokenize
  -> build labels
  -> train / val split
  -> TrainingArguments
  -> Trainer.train()
  -> evaluate
  -> save_pretrained()
```

训练 causal LM 时，常见数据字段是：

```text
input_ids: LongTensor[B, T]
attention_mask: LongTensor[B, T]
labels: LongTensor[B, T]
```

对普通续写任务，`labels` 通常是 `input_ids` 的拷贝，并把 padding 位置改成 `-100`。对 SFT，user/system 位置也通常应改成 `-100`，否则模型会被训练去复述用户问题。

本章只要求你能检查普通 causal LM 的 `labels` 和 pad mask。assistant-only label mask、chat template span 和指令样本质量会在第 9 章专门展开，避免把 HF 对象学习和 SFT 目标混成一团。

Trainer 能帮你组织训练循环，但它不会自动判断数据目标是否正确。你仍然要人工检查一个 batch：

```text
decode input_ids: 模型实际看到了什么
decode labels != -100: 模型实际被要求学什么
attention_mask: padding 是否被屏蔽
```

这个检查非常朴素，但能提前发现大部分微调事故：模板重复、答案被截断、padding 进入 loss、user 内容参与 loss、特殊 token 添加两次。

## 7. 保存与加载

一个可复现的 Hugging Face 实验至少保存：

- model weights：`model.save_pretrained(output_dir)` 或 `trainer.save_model(output_dir)`。
- tokenizer：`tokenizer.save_pretrained(output_dir)`。
- training args：学习率、batch size、epoch、gradient accumulation、seed。
- dataset 版本：原始数据路径、清洗脚本 hash、split seed。
- eval report：训练前后同一 prompt / eval set 的对比。

加载时必须从同一个目录恢复 model 和 tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir)
```

只保存权重不保存 tokenizer，会导致同一文本得到不同 token ids，评测不可复现。

## 8. Accelerate 的位置

Accelerate 不是一套新的模型理论，而是设备与分布式训练抽象。它能帮助 Trainer 或自定义训练循环处理多 GPU、mixed precision、FSDP / DeepSpeed 等工程问题。

教学阶段先把单机 CPU / 单卡流程写正确，再引入：

- `accelerate config`
- `accelerate launch`
- mixed precision
- gradient accumulation
- checkpoint resume

不要用 Accelerate 掩盖 shape 错误、label 错误或数据泄漏。分布式只会把小错误放大。

实际学习顺序应该是：

```text
CPU / tiny model 跑通数据和 shape
-> 单卡跑通最小训练
-> 保存、加载、评测可复现
-> 再引入 mixed precision / accelerate / 多卡
```

这样遇到 OOM、device mismatch 或分布式 checkpoint 问题时，你知道基础训练目标已经正确，不会同时排查十类问题。

## 9. 必写实验

- 加载 tiny causal LM，验证 prompt 到生成文本的完整推理链路。
- 对同一个 prompt 比较 greedy、temperature、top-k 输出。
- 构造 20-100 条 tiny text dataset，跑一次最小 Trainer 微调。
- 保存模型和 tokenizer，再重新加载，验证同一 prompt 的 logits shape 和生成流程可用。
- 人工记录训练前后同一组 prompt 的输出变化，不用“loss 下降”替代行为观察。
- 新增 special token 后执行 `resize_token_embeddings(len(tokenizer))`，验证 logits 词表维度与 embedding 行数一致。
- 固定 revision 与 generation config，验证同一模型版本和 greedy 配置可复现。

## 10. 失败模式

- model 和 tokenizer 来自不同目录：token id 与 embedding 不匹配。
- `pad_token` 未设置：batch padding 或 data collator 报错。
- chat template 手写错误：模型看到的角色边界与预训练格式不一致。
- `max_length` 截断了答案关键部分：训练样本看似正常，实际 label 不完整。
- 只看 train loss：模型可能记住格式，却没有提升目标能力。
- 保存 checkpoint 但没有保存数据版本：实验无法复现。
- 新增 special token 后忘记 resize embedding：新增 token 无法正确训练，甚至查表越界。
- 默认开启 `trust_remote_code=True`：把模型加载变成未审查代码执行。

## 11. 测试验收

本章 tests 至少验证：

1. tokenizer 输出包含 `input_ids` 和 `attention_mask`，且 shape 一致。
2. causal LM 前向输出 logits，并验证 `logits.size(-1) == model.get_output_embeddings().weight.size(0)`。
3. `len(tokenizer) <= model.get_input_embeddings().weight.size(0)`；如果新增 special tokens，测试应验证已执行 `resize_token_embeddings(len(tokenizer))`。
4. data collator 把 pad 位置 label 改成 `-100`。
5. `save_pretrained()` 后可从本地目录重新加载 model 和 tokenizer。
6. 同一 seed、同一 greedy 生成配置下，短 prompt 输出可复现。

## 12. 本章记忆锚点与边界

本章最重要的一句话是：

> Hugging Face 工作流不是替你思考工程契约，而是把模型、tokenizer、配置、训练状态和生成策略放进标准接口。

你需要记住：

1. model、tokenizer、config 和 revision 要成组固定。
2. tokenizer 新增 token 后必须 resize model embeddings。
3. pad token 可以临时复用 eos，但 pad 位置不能进 loss。
4. `trust_remote_code=True` 是代码执行边界。
5. Trainer 能组织训练，不能替你检查 label、泄漏和评测。

本章没有解决“按指令回答”。下一章进入 SFT。

## 13. 下一章

本章解决“如何复用开源模型”。但普通 causal LM 的目标仍是续写。下一章进入 SFT：如何把模型训练成遵循 system / user / assistant 指令格式的助手。
