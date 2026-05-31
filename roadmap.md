# LLM 从 0 到领域小模型 Roadmap

这份路线图的目标不是做一张“LLM 名词全景图”，而是带初学者稳定走完一条可执行路线：先把数学基础放进真实模型问题里，再从零实现一个 MiniGPT，然后进入开源模型、评测、数据工程、RAG、SFT、LoRA、蒸馏、安全部署，最后做出法律或医学领域小模型。

参考文档里最值得继承的写作方法是：

```text
先从真实困惑出发
  -> 说明这个概念是什么
    -> 解释它为什么会出现
      -> 用最小公式、最小矩阵或最小概率例子推一遍
        -> 给出生活类比
          -> 建立直观感受
            -> 说明它和前后知识的关系
              -> 落到机器学习 / LLM 中的作用
                -> 用最小实验验证它
```

新版 roadmap 的主线调整为：

```text
风险边界与学习仓库
  -> 函数、参数、梯度与训练闭环
    -> 张量、shape、线性变换与向量空间
      -> 概率、Softmax、交叉熵与 next-token
        -> Tokenizer / Embedding / Bigram / 神经语言模型
          -> Attention / Transformer / MiniGPT
            -> 现代 LLaMA 架构
              -> Hugging Face 工作流
                -> 最小评测系统
                  -> 领域数据工程
                    -> RAG baseline
                      -> SFT
                        -> LoRA / QLoRA
                          -> 证据约束蒸馏
                            -> 安全、合规与模型卡
                              -> 量化、服务与发布门禁
                                -> 法律 / 医学领域项目
```

进阶主题单独放入选修：

```text
DPO / RLHF / PPO / GRPO
DeepSeek-R1 推理训练复盘
Agent 与受控工作流
多模态基础
长上下文与推理效率
```

一句话概括：

> 主线只保留“做出可靠领域小模型”必须掌握的内容；数学不再孤立成篇，而是在第一次真正派上用场的章节里出现；高级训练和扩展方向作为选修，避免初学者在还没有评测、数据和安全边界之前就被复杂概念压垮。

## 写作要求

后续每一章都要避免只列概念。每个重要知识点尽量回答下面 8 个问题：

| 问题 | 写作目标 |
| --- | --- |
| 它是什么 | 用一句朴素的话说清定义 |
| 它怎么来的 | 说明它要解决什么现实问题 |
| 它怎么推导 | 用最小公式、最小矩阵或最小概率例子走一遍 |
| 生活类比 | 找一个日常经验帮助理解 |
| 直观感受 | 说明数值变大、变小、相近、相远分别意味着什么 |
| 关联知识 | 说明它和前置/后置概念怎么连起来 |
| 在机器学习中的作用 | 说明它在训练、推理、评测或部署中干什么 |
| 最小实验 | 用代码、表格或测试验证这个概念 |

每个工程章节额外使用这个模板：

| 模块 | 要回答的问题 |
| --- | --- |
| 本章核心困惑 | 学习者为什么会卡在这里 |
| 前置知识 | 学这一章之前必须会什么 |
| 本章新增能力 | 学完以后系统多了什么能力 |
| 手算例子 | 是否能用小数字走一遍 |
| 最小代码 | 是否能跑通最小功能 |
| 测试用例 | shape、mask、loss、保存加载、输出格式是否可测 |
| 常见错误 | 初学者最容易误解什么 |
| 和领域项目的关系 | 它如何服务法律/医学小模型 |
| 验收标准 | 什么结果才算学会 |

## 每章交付物

每一章至少留下 5 样东西：

| 交付物 | 要求 |
| --- | --- |
| 概念文章 | 按“是什么、怎么来的、怎么推导、类比、直觉、作用”写 |
| 最小代码 | 不追求大而全，只验证本章核心概念 |
| 测试文件 | 至少验证 shape、loss、mask、保存加载或输出格式 |
| 验收标准 | 说明什么结果算通过 |
| 复盘问题 | 列出 3-5 个面试或项目迁移问题 |

## 数学融入原则

`lessons/math_foundations_deep_dive.md` 不再作为主线前置文章使用。它保留为教师备课、补充阅读和公式深挖素材库；真正面向学习者的主线文章，要把数学放回它第一次产生作用的章节里。

新的写作原则是：

```text
先遇到工程困惑
  -> 引出必须用到的数学对象
    -> 用最小数字例子算一遍
      -> 马上落到模型组件或训练现象
        -> 用代码 / 测试 / 评测验证
```

也就是说，学习者不需要先读完一篇完整数学专题，才开始学模型。每一章只引入解决本章问题所必需的数学，并把它和后续模型知识接上。

## 渐进式主线架构

主线按“数学基础 -> 最小语言模型 -> Transformer -> 现代开源模型 -> 领域可靠性”五个阶段推进：

| 阶段 | 章节 | 目标 | 数学进入方式 |
| --- | --- | --- | --- |
| 0. 风险与仓库 | 第 0 章 | 先知道为什么要可复现、可评测、可审计 | 暂不展开公式，只建立实验变量、样本、指标的概念 |
| 1. 可学习函数地基 | 第 1-3 章 | 理解模型为什么能学、张量如何流动、语言模型为什么是概率分布 | 函数、参数、非线性、张量 shape、梯度、概率、Softmax、交叉熵 |
| 2. 文本变成模型 | 第 4-6 章 | 从文本、token、embedding 走到最小神经语言模型 | token id、向量空间、矩阵乘法、相似度、最大似然、困惑度 |
| 3. Transformer 从零实现 | 第 7-10 章 | 从 attention 手算到 MiniGPT，再映射到 LLaMA | 点积 attention、mask、残差、归一化、位置编码、RoPE、KV cache |
| 4. 开源模型与领域能力 | 第 11-17 章 | 使用 HF、建立评测、做 RAG/SFT/LoRA/蒸馏 | 泛化、过拟合、低秩分解、熵、KL、检索相似度、数据泄漏 |
| 5. 可靠发布 | 第 18-22 章 | 安全、合规、服务、发布审计和领域项目 | 置信区间、切片评测、不确定性、量化误差、发布门禁 |

一句话版本：

```text
先学一个函数怎样被训练
  -> 再学文本怎样变成张量和概率目标
    -> 再学 token 之间怎样通过 attention 交换信息
      -> 再学现代大模型如何工程化
        -> 最后学领域模型怎样被证据、评测和安全边界约束
```

## 数学嵌入矩阵

下面这张表替代原来的“数学 1-15”独立章节。写每篇文章时，直接把对应数学写进本章的“核心困惑、手算例子、最小实验、常见错误和验收标准”里。

| 主线章节 | 必须融入的数学 | 本章要解决的模型问题 | 最小数学例子 |
| --- | --- | --- | --- |
| 第 1 章：训练闭环 | 函数、参数、非线性、loss、梯度、链式法则 | 神经网络为什么能从错误中调整参数 | `y = wx + b`，手算一次 MSE loss 和梯度方向 |
| 第 2 章：张量与 shape | 向量、矩阵、张量、矩阵乘法、broadcasting | 文本和模型内部状态为什么都是高维数组 | `[B,T] -> [B,T,C] -> [B,T,V]` 的 shape trace |
| 第 3 章：next-token | 条件概率、链式分解、logits、Softmax、交叉熵 | 语言模型为什么训练“下一个 token” | 长度为 4 的序列右移，手算一个 token 的 cross entropy |
| 第 4 章：Tokenizer / Dataset | 离散化、mask、有效样本、label masking | 文本怎样变成可训练数据，哪些 token 不该算 loss | padding label 设为 `-100` 前后的 loss 对比 |
| 第 5 章：Embedding | 向量空间、点积、范数、距离、余弦相似度 | token id 如何获得语义位置，RAG 为什么能检索 | 比较 `dot(a,b)` 和 `cos(a,b)`，解释长度影响 |
| 第 6 章：神经语言模型 | 最大似然、上下文窗口、困惑度、采样 | 从统计 bigram 走向可学习上下文表示 | bigram 概率表 vs embedding+MLP 的 loss 下降 |
| 第 7 章：Attention | 线性投影、点积、缩放、Softmax、causal mask | 每个 token 如何决定看谁 | 用 3 个 token 手算 `QK^T / sqrt(d)` 和 mask 后权重 |
| 第 8 章：Transformer Block | 多头分解、残差、LayerNorm/RMSNorm、激活函数 | 为什么 attention 外还需要 FFN、残差和归一化 | 对比有无 residual 的梯度范数和输出 shape |
| 第 9 章：MiniGPT | 位置编码、生成采样、perplexity、checkpoint 一致性 | 如何把组件组成完整 decoder-only LM | 同一 prompt 下 greedy、temperature、top-k 的输出差异 |
| 第 10 章：LLaMA | RoPE 旋转、SwiGLU、GQA、KV cache 复杂度 | 现代开源模型相对教学 GPT 改了什么 | 手算二维 RoPE 旋转，比较 MHA/GQA 的 cache shape |
| 第 11 章：HF 工作流 | 参数量、dtype、显存估算、版本可复现 | 如何使用开源模型但不把它当黑盒 | 用参数量和 dtype 估算 tiny model 显存 |
| 第 12 章：评测系统 | 指标、切片、置信区间、数据泄漏 | 怎么知道模型真的变好 | `18/20` 的通过率为什么仍有不确定性 |
| 第 13 章：数据工程 | 分布、采样偏差、train/eval split、去重 | 领域数据如何决定模型学到什么 | 重复样本如何让 eval 虚高 |
| 第 14 章：RAG | 向量检索、Recall@k、rerank、相似度不等于支持 | 模型如何先查证据再回答 | top-1 相似但证据不支持的反例 |
| 第 15 章：SFT | assistant-only loss、格式约束、过拟合 | 模型如何从续写变成按指令回答 | system/user token 参与 loss 后的错误学习目标 |
| 第 16 章：LoRA / QLoRA | 低秩分解、rank、量化误差、泛化 | 为什么只训练少量参数也能适配任务 | 计算 rank=4/8/16 的可训练参数量 |
| 第 17 章：蒸馏 | 熵、KL 散度、teacher/student 分布差异 | 如何迁移可验证行为，而不是复制幻觉 | 比较 hard label 和 soft distribution 的信息差 |
| 第 18 章：安全与模型卡 | 不确定性、风险分层、拒答阈值 | 领域模型什么时候必须拒答或转人工 | 高风险样本的 false negative 成本 |
| 第 19 章：量化与服务 | 数值精度、延迟/吞吐、质量回归 | 部署优化如何影响质量与安全 | fp16/int8/int4 输出差异和 eval regression |
| 第 20-22 章：领域项目与审计 | 统计证据、发布门禁、失败分类、可追溯性 | 什么材料能证明模型可以发布 | 平均分通过但红旗切片失败的 release block |

## 深挖文章的定位

`lessons/math_foundations_deep_dive.md` 可以保留，但它的定位要从“主线前置章节”改成“深挖索引”。每个主线章节在文末用一小段链接回对应深挖段落即可，例如：

```text
想继续深挖：见《数学深挖篇》第 3 节“相似度”和第 4 节“概率”。
```

这样读者的主路径仍然是文章本身：先把本章模型问题跑通，再按需要回到数学深挖篇补推导。

## 贯穿样例线

课程固定使用三类样例贯穿前后：

```text
合同风险：违约金过高、责任范围过宽、管辖区缺失、法规版本缺失
医学科普：胸痛、呼吸困难、儿童发热、孕妇用药、药物禁忌
结构化输出：必须输出 JSON，字段完整，不能编造依据
```

每章都要回答：

```text
本章新增了什么能力，让系统离“谨慎、可解释、可评测的领域小模型”更近一步？
```

从第 0 章开始就要建立风险边界：

```text
法律项目：只做信息辅助，不构成法律意见，高风险结论需要律师复核
医学项目：只做科普和分流提醒，不做诊断，红旗症状必须建议及时就医
```

## 主线必修目录

主线目录按阶段阅读，不建议跳过第 0-10 章。第 0-10 章共 11 篇，负责把数学基础和模型结构缝在一起；第 11 章之后再进入开源生态、评测和领域可靠性。

```text
第 0 章：建立边界和仓库
第 1-3 章：数学地基进入训练、shape 和概率目标
第 4-6 章：文本进入 token、embedding 和最小语言模型
第 7-10 章：attention、Transformer、MiniGPT 和 LLaMA
第 11-17 章：HF、eval、data、RAG、SFT、LoRA、蒸馏
第 18-22 章：安全、部署、法律/医学项目和发布审计
```

## 第 0 章：课程目标、风险边界与学习仓库

核心问题：怎样把学习变成一个长期可维护、可复现、可评测的工程？

讲法：先说明为什么零散看课容易断掉，再搭一个能持续添加 lesson、notebook、test、project 的学习仓库，并从一开始写清法律/医学边界。

学习内容：

- 项目结构
- 三档环境：CPU、单卡 GPU、云端
- seed 固定
- `course_config.yaml`
- 实验日志
- 数据 manifest
- 最小 eval runner 雏形
- 法律/医学风险边界
- notebook、src、tests、projects 规范

全局配置示例：

```yaml
seed: 42
device: cpu
project_domain: legal_demo
eval_set_version: v0.1
risk_policy_version: v0.1
```

最小实验：

- 创建 repo 结构
- 跑通一个 pytest
- 生成一份空的 `eval_report.md`
- 记录一次实验配置

验收标准：

- CPU 路线可以跑 tiny model 教学实验
- 单卡 GPU 路线可以跑 LoRA / RAG / 小模型推理
- 云端路线有训练和部署的配置说明

对应目录：`00_course_goal_and_risk_boundary`

## 第 1 章：从函数到 PyTorch 训练闭环

核心问题：神经网络到底怎么“学会”？

讲法：从 `y = f(x)` 开始，把参数、loss、梯度、optimizer 串成一个训练循环。合同风险 toy classifier 只是为了理解训练闭环，不代表最终 LLM 形态。

数学融入：函数、参数、非线性、loss、梯度、链式法则。不要单独讲微积分，直接用一次 toy classifier 的错误、反传和参数更新解释“模型为什么能学”。

学习内容：

- `nn.Module`
- loss
- backward
- optimizer
- train / eval loop
- `zero_grad()` 为什么必须有
- train loss / val loss 分叉意味着什么

主实验：

- 合同风险 toy classifier
- 用 `y = wx + b` 手算一次 loss、梯度方向和参数更新
- 对比 baseline / no backward / no step / no zero_grad
- 在 tiny dataset 上观察过拟合

验收标准：

- 能用“预测错了 -> loss 变大 -> 梯度指出方向 -> optimizer 更新参数”解释训练闭环
- 能解释不调用 `backward()` 会怎样
- 能解释不调用 `step()` 会怎样
- 能解释不调用 `zero_grad()` 会怎样
- 能画出 train loss 和 val loss 分叉时的过拟合现象

对应目录：`01_pytorch_training_loop`

## 第 2 章：张量、shape 与 PyTorch 基础

核心问题：为什么 LLM 学习里 shape 比公式更容易卡人？

讲法：用 `[B,T]`、`[B,T,C]`、`[B,H,T,D]` 和 `[B,T,V]` 贯穿 tokenizer、embedding、attention、logits 和 labels。

数学融入：向量、矩阵、张量、矩阵乘法、broadcasting。每个公式都要同时写出 shape，让学生先能看懂数据流。

学习内容：

- broadcasting
- matrix multiplication
- reshape / view / transpose
- batch / seq_len / hidden / vocab
- `Dataset` / `DataLoader`
- device 与 dtype

主实验：

- 构造一批 token ids
- 经过 embedding 和 linear head
- 打印每一步 shape
- 写 shape pytest

验收标准：

- 能解释 `[B,T,C]` 中每一维含义
- 能解释 attention score 为什么是 `[B,H,T,T]`
- 能定位一次 shape mismatch

对应目录：`02_tensor_shape_pytorch`

## 第 3 章：概率、Softmax、交叉熵与 next-token

核心问题：如果模型不是分类图片，而是继续写一句话，它该怎么训练？

讲法：先用“我喜欢学大模 -> 型”解释自回归，再推 `input_ids` 和 `labels` 为什么要右移。

数学融入：条件概率、链式分解、logits、Softmax、交叉熵、困惑度。用一个 4-token 序列手算，而不是先讲抽象概率论。

学习内容：

- next-token prediction
- teacher forcing
- logits / Softmax
- cross entropy
- `input_ids` / `labels` 错位
- greedy / sampling
- temperature / top-k / top-p
- perplexity

主实验：

- bigram LM
- 正确右移 vs 错误右移
- 同一 prompt 的不同采样策略

验收标准：

- 能手算长度为 4 的序列：

```text
input_ids = [A, B, C, D]
labels    = [B, C, D, EOS]
```

- 能解释训练时一次性预测多个位置，推理时一个 token 一个 token 生成

对应目录：`03_next_token_language_modeling`

## 第 4 章：Tokenizer、LM Dataset 与 padding label masking

核心问题：文本如何变成模型能计算、能训练、能正确忽略 padding 的数字？

讲法：先解释 one-hot 为什么不够，再讲 token id、vocab、BPE、attention mask、LM dataset 和 `ignore_index`。

边界说明：本章只处理 causal LM 数据中的 padding label masking。chat template 和 assistant-only loss 不在本章正式展开，它们属于第 15 章 SFT 的训练目标问题。

数学融入：离散化、有效样本、mask、loss averaging。重点说明“哪些位置参与训练”本身就是数学目标的一部分。

学习内容：

- token / vocab / token id
- encode / decode
- padding / truncation
- attention mask
- BPE / WordPiece 直觉
- LM dataset
- `ignore_index=-100`

主实验：

- simple tokenizer
- LM Dataset
- 验证 pad token 是否进入 loss
- 对比 padding 参与 loss 与 padding label 设为 `-100` 后的平均 loss
- 观察中文法律/医学文本 tokenization

领域 tokenizer 观察：

```text
《中华人民共和国民法典》第五百八十五条
对乙酰氨基酚 500mg q6h
违约金不得超过实际损失的30%
```

验收标准：

- 能输出 `input_ids`、`attention_mask`、`labels`
- padding 的 labels 必须是 `-100`
- 能解释本章的 label masking 与第 15 章 SFT assistant-only loss 的区别

对应目录：`04_tokenizer_dataset_label_masking`

## 第 5 章：Embedding 与相似度

核心问题：token id 只是编号，模型怎么学出语义？RAG 的 embedding 又和 token embedding 有什么不同？

讲法：把 embedding 表讲成“可学习的词典”，再用距离和相似度解释语义空间，并明确 token embedding 不等于 retrieval embedding。

数学融入：向量空间、点积、范数、欧氏距离、余弦相似度。先把 token embedding、相似度计算和 retrieval embedding 切开，attention 只预告一句：第 7 章会再次使用点积，但那里的对象是 `Q/K`，不是 RAG 文档向量。

学习内容：

- token embedding
- token embedding 不是什么：不是一句话的语义向量
- embedding lookup
- hidden dimension
- context window
- dot product / cosine similarity
- sentence / document embedding
- retrieval embedding 不是什么：不是 LLM 内部 token embedding
- embedding 可视化

主实验：

- embedding-based language model
- 观察 token embedding 训练前后变化
- 比较 `a=[10,0]`、`b=[1,0]`、`c=[0,1]` 的 dot product 与 cosine similarity
- 用 sentence embedding 做合同条款检索

验收标准：

- 能区分 LLM 内部 token embedding 和 RAG 使用的 retrieval embedding
- 能比较 dot product 与 cosine similarity 的差异

对应目录：`05_embedding_and_similarity`

## 第 6 章：从 Bigram 到神经语言模型

核心问题：最小语言模型如何从统计表走向神经网络？

讲法：先做 bigram，再加入 embedding 和 MLP，让学生看到“上下文表示”如何从查表变成可学习函数。

数学融入：最大似然、统计计数、上下文窗口、采样和 perplexity。让学生看到传统统计语言模型怎样自然过渡到神经语言模型。

学习内容：

- bigram count
- neural LM
- context window
- hidden dimension
- causal training
- perplexity
- generate loop

主实验：

- bigram baseline
- embedding + MLP LM
- greedy / sampling generate

验收标准：

- loss 能下降
- perplexity 可计算
- generate 支持 `max_new_tokens` 和 `eos_token`

对应目录：`06_bigram_to_neural_lm`

## 第 7 章：Attention 手算与实现

核心问题：一句话中每个 token 怎么决定自己应该看谁？

讲法：按照参考文档的方式，用 Excel 或小矩阵把 `Q`、`K`、`V` 每一步算出来。

数学融入：线性投影、点积、缩放、Softmax、causal mask。attention 的公式必须和小矩阵手算绑定，避免只背 `QK^T`。

学习内容：

- Q / K / V
- dot-product attention
- `sqrt(d_k)` 的原因
- Softmax attention weights
- causal mask
- self-attention
- attention 可视化

shape 主线：

```text
x: [B, T, C]
Wq: [C, C]
q: [B, T, C]
q after split heads: [B, H, T, D]
score: [B, H, T, T]
mask: [T, T]
output: [B, T, C]
```

主实验：

- 手写 scaled dot-product attention
- 有 mask vs 无 mask
- 可视化合同条款中的 token attention

验收标准：

- 未来 token 的 attention weight 必须为 0
- score shape 必须是 `[B,H,T,T]`
- 能解释为什么除以 `sqrt(d_k)`

对应目录：`07_attention_from_scratch`

## 第 8 章：Transformer Block

核心问题：attention 只是一个模块，完整 block 还缺什么？

讲法：先讲 attention 解决“看谁”，再讲 FFN 解决“如何变换”，残差和归一化解决“深层稳定”。

数学融入：多头分解、非线性激活、残差连接、LayerNorm/RMSNorm、训练稳定性。把“为什么能堆深”讲成尺度和梯度问题。

学习内容：

- multi-head attention
- FFN
- activation
- residual connection
- LayerNorm / RMSNorm
- dropout
- pre-norm / post-norm
- attention complexity `O(T^2)`
- 参数量与显存估算
- KV cache 直觉

主实验：

- 手写 Multi-Head Attention
- 手写 Transformer Block
- 对比 1 / 2 / 4 层 block 的 loss 与 grad_norm
- 计算 tiny block 参数量

验收标准：

- 输入输出 shape 一致
- residual 不改变主干 shape
- 能解释 attention 为什么随序列长度平方增长

对应目录：`08_transformer_block`

## 第 9 章：MiniGPT 从零实现

核心问题：把 tokenizer、embedding、Transformer block 和 LM head 组合起来，是不是就得到 GPT？

讲法：不急着上大模型，先做一个能训练、能保存、能生成、能测试的极小 GPT。

数学融入：绝对位置 embedding、perplexity、生成采样、temperature/top-k/top-p。把前面所有数学对象收束到一个完整 decoder-only LM。本章只使用最简单的位置表示完成 MiniGPT 闭环，RoPE 放到第 10 章作为现代 LLaMA 架构改进。

学习内容：

- decoder-only GPT
- learned position embedding
- causal language modeling
- train loop
- generate loop
- checkpoint 保存和加载
- perplexity
- pytest

主实验：

- MiniGPT tiny corpus
- checkpoint round-trip
- 生成合同风险描述样例

验收标准：

- 能训练一个 tiny corpus
- loss 能下降
- checkpoint 保存加载后输出一致
- generate 支持 temperature / top-k / top-p
- perplexity 可计算
- 至少 5 个 pytest：shape、mask、loss、save/load、generate 长度

对应目录：`09_mini_gpt`

## 第 10 章：现代 LLaMA 架构

核心问题：主流开源大模型在教学版 GPT 上改了什么，为什么改？

讲法：先有 MiniGPT 的完整骨架，再理解 LLaMA 是现代化 decoder-only Transformer：RoPE、RMSNorm、SwiGLU、GQA、KV cache。

数学融入：RoPE 旋转、SwiGLU 非线性、GQA 的分组共享、KV cache 的 shape 和复杂度。现代架构改动都要回答“它解决了哪一个数学/工程压力”。

从 MiniGPT 到 LLaMA 的替换表：

| MiniGPT 教学组件 | LLaMA 现代组件 | 解决的问题 |
| --- | --- | --- |
| learned position embedding | RoPE | 更好处理相对位置信息和长度泛化 |
| LayerNorm | RMSNorm | 简化归一化，降低计算开销 |
| ReLU / GELU FFN | SwiGLU FFN | 提升表达能力 |
| MHA | GQA | 降低推理时 KV cache 成本 |
| 每步重复算历史 token | KV cache | 加速自回归生成 |

学习内容：

- decoder-only 架构
- RoPE
- RMSNorm
- SwiGLU
- MHA / MQA / GQA
- KV cache
- causal LM head
- logits 到 token

主实验：

- 手写 RoPE 最小实现
- 对比 MHA 和 GQA 的 KV cache shape
- 画出 LLaMA block 数据流

验收标准：

- 能说明 LLaMA 相比 MiniGPT 改了什么
- 能解释 `Q/K` 加 RoPE、`V` 不加 RoPE
- 能解释 GQA 为什么降低推理阶段 KV cache 开销

对应目录：`10_llama_modern_block`

## 第 11 章：Hugging Face 工作流

核心问题：现实中不可能每次从零写模型，如何使用开源模型且不把它当黑盒？

讲法：把从零实现的组件映射到 `AutoTokenizer`、`AutoModelForCausalLM`、`Trainer` 和 `generate`。

数学融入：参数量、dtype、显存估算、logits 到 token 的映射。开源工具要回扣前面手写过的数学对象，避免黑盒化。

学习内容：

- 固定 `transformers` / `datasets` / `accelerate` / `peft` 版本
- tiny model 选择
- CPU fallback
- 显存估算
- `AutoTokenizer`
- `AutoModelForCausalLM`
- `datasets`
- `Trainer` / custom loop
- `model.generate`
- `save_pretrained`
- chat template

映射表：

| 自己实现 | Hugging Face |
| --- | --- |
| tokenizer | `AutoTokenizer` |
| model | `AutoModelForCausalLM` |
| train loop | `Trainer` / custom loop |
| generate | `model.generate` |
| checkpoint | `save_pretrained` |

主实验：

- 加载一个 tiny HF model
- 完成一次推理
- train one step
- 保存和重新加载

验收标准：

- 同一个 prompt 保存加载前后输出一致或可解释
- 版本写入实验日志
- CPU fallback 可运行 tiny demo

对应目录：`11_huggingface_workflow`

## 第 12 章：最小评测系统

核心问题：在开始 RAG、SFT、LoRA 之前，怎么先知道模型有没有变好？

讲法：评测前移。先搭 eval harness，再做训练和检索。否则后面所有优化都只能靠感觉。

数学融入：指标、切片、置信区间、数据泄漏和回归测试。评测章节要明确：demo 是样例，统计才是证据。

学习内容：

- eval set
- metrics
- JSON schema validation
- format accuracy
- retrieval recall
- citation support
- refusal accuracy
- failure cases
- regression tests
- data leakage check
- human review rubric

主实验：

- 20 条合同风险 eval item
- 20 条医学科普 eval item
- `eval_runner.py`
- `metrics.py`
- `eval_report.md`
- `failure_cases.csv`

人工评审 rubric：

- 答案是否越界
- 不确定性是否充分
- 引用是否真正支持结论
- 是否遗漏高风险提醒
- 是否把信息辅助写成确定建议

全局评测指标字典：

从第 12 章开始，后续 RAG / SFT / LoRA / 蒸馏 / 量化 / 发布都必须复用同一套 eval runner。

| 指标 | 用在哪些章节 | 失败意味着什么 |
| --- | --- | --- |
| `format_accuracy` | SFT、LoRA、项目 | 输出结构不可靠 |
| `citation_support` | RAG、蒸馏、项目 | 引用不能支持结论 |
| `refusal_accuracy` | RAG、安全、项目 | 无依据或高风险时仍强答 |
| `red_flag_recall` | 医学项目、安全 | 漏掉高风险症状 |
| `leakage_check` | 数据工程、评测 | train/eval 混入重复样本 |
| `regression_delta` | LoRA、量化、发布 | 优化后质量或安全倒退 |

验收标准：

- 每次模型或 prompt 改动都能跑同一套 eval
- 输出格式错误会被自动标记
- 检索无依据时必须输出 `unknown` 或触发拒答

对应目录：`12_minimum_eval_harness`

## 第 13 章：领域任务定义与数据工程

核心问题：领域模型的能力主要来自模型，还是任务定义和数据？

讲法：先定义任务边界、数据来源、许可、脱敏、标注规范和 eval set，再决定要不要 RAG、SFT、LoRA 或蒸馏。

数学融入：样本分布、采样偏差、train/eval split、去重和 leakage。数据工程要让学生理解“模型学到什么”首先由数据分布决定。

学习内容：

- 任务边界
- 数据来源和许可矩阵
- PII / PHI 脱敏
- 清洗去重
- 质量过滤
- train / eval leakage 检查
- 标注规范
- SFT 数据
- RAG chunk
- 蒸馏数据
- eval set 冻结
- data card
- manifest
- 样本审计日志

主实验：

- 同一条合同条款生成 SFT / RAG / distill / eval 四种数据形态
- 输出数据质量报告
- 输出 data card

验收标准：

- 每条数据有来源、许可、版本和用途
- train/eval 无泄漏
- 高风险字段已脱敏

对应目录：`13_domain_task_and_data_engineering`

## 第 14 章：RAG Baseline

核心问题：模型参数不是数据库，怎么让模型回答前先查资料？

讲法：按“离线入库、在线检索、带引用生成、无依据拒答”拆开讲，避免把 RAG 讲成一个黑盒。

数学融入：embedding 相似度、Recall@k、rerank、相似度不等于证据支持。必须用反例说明“检索到了相似文本”不等于“答案有依据”。

学习内容：

- document parsing
- chunking
- chunk overlap
- embedding model
- vector store
- hybrid search
- top-k
- rerank
- query rewrite
- prompt with context
- citation span
- answer grounding
- 无依据拒答
- prompt injection 防御

主实验：

- 输入：5 个固定问题、20 个固定 chunk
- 输出：top-k 文档、answer.json、citations
- 指标：Recall@3、引用命中率、无依据拒答率

输出格式：

```json
{
  "answer": "",
  "citations": [
    {
      "source_id": "",
      "span_id": "",
      "quote": "",
      "support_level": "full | partial | none"
    }
  ],
  "needs_human_review": true
}
```

验收标准：

- Recall@3 >= 0.8
- 所有回答必须包含 `source_id` 和 `span_id`
- 检索不到依据时必须输出 `unknown`，不能编造
- 必须复用第 12 章固定 eval set；如果检索指标提升但安全切片下降，本章实验不能算通过

对应目录：`14_rag_baseline`

## 第 15 章：SFT 指令微调

核心问题：怎么让模型从“续写文本”变成“按指令回答”？

讲法：先区分预训练和后训练，再说明 SFT 是让模型学习指令格式、回答结构和行为边界。

数学融入：assistant-only loss、格式约束、过拟合、训练/验证分叉。SFT 不是让模型“懂事实”，而是改变条件生成分布。

学习内容：

- instruction tuning
- SFT
- system / user / assistant
- chat dataset
- assistant-only loss
- JSON schema validation
- 数据清洗
- train / val / test
- 格式一致性
- 过拟合观察

主实验：

- 教学实验：20 条合同风险 SFT 样本，只验证 chat template 和 JSON 格式
- 课程项目：数百条结构化样本，必须 train / val / test 分离
- 训练前后 eval 对比

验收标准：

- system/user token 不参与 loss
- assistant 输出符合 JSON schema
- 训练后格式准确率提升，但必须报告事实/引用风险
- 必须复用第 12 章固定 eval set
- SFT 成功不等于模型事实更可靠；如果格式准确率提升，但 `citation_support`、`refusal_accuracy` 或 `red_flag_recall` 下降，应判定为行为对齐失败

对应目录：`15_sft_instruction_tuning`

## 第 16 章：LoRA / QLoRA 参数高效微调

核心问题：全量微调太贵，能不能只训练少量参数？

讲法：先用低秩分解直觉解释 LoRA，再讲 rank、target_modules、adapter 保存回滚和 QLoRA。

数学融入：低秩分解、rank、可训练参数量、量化误差和泛化。LoRA 章节要把“省参数”讲成矩阵更新空间的限制。

低秩直觉：LoRA 到底限制了什么？

```text
全量微调允许模型学习任意形状的参数更新 ΔW。
LoRA 假设：领域适配需要的 ΔW 不一定是满秩的，可以用两个小矩阵 A 和 B 近似。

ΔW ≈ A @ B
```

本章不要先背公式。先做一个 `4 x 4` toy matrix，看 rank=1、rank=2、rank=4 的重构误差和参数量差异。

学习内容：

- PEFT
- LoRA
- rank / alpha
- target_modules
- adapter 保存和加载
- merge adapter
- QLoRA
- 4-bit quantization
- 显存优化

实验矩阵：

| 实验 | 变量 | 固定项 | 指标 |
| --- | --- | --- | --- |
| rank 对比 | r=4/8/16 | 数据、lr、epoch | eval F1、格式准确率 |
| target module 对比 | q/v vs q/k/v/o vs all linear | rank、数据 | 质量、显存、速度 |
| merge 对比 | merge 前/后 | prompt | 输出一致性 |

验收标准：

- adapter 可保存、加载、回滚
- merge 前后输出差异可解释
- 报告显存、速度和质量指标
- 必须复用第 12 章固定 eval set；如果主指标提升但安全切片下降，本章实验不能算通过

对应目录：`16_lora_qlora`

## 第 17 章：证据约束蒸馏

核心问题：大模型效果好但太贵，怎么把可验证行为迁移给小模型？

讲法：teacher 不是事实来源，而是基于证据生成训练样本的工具。法律/医学蒸馏必须是证据约束蒸馏。

数学融入：熵、KL 散度、hard label 与 soft distribution、teacher/student 分布差异。蒸馏要强调迁移的是可验证行为，不是 teacher 的权威性。

soft label 直觉：

```text
teacher 输出：
A: 0.70, B: 0.20, C: 0.10

hard label 只告诉 student：A 是正确答案。
soft distribution 还告诉 student：B 比 C 更像正确答案。
KL 散度衡量的是 student 的概率分布和 teacher 的概率分布有多不像。
```

学习内容：

- teacher model
- student model
- response distillation
- logit distillation 直觉
- preference distillation 选读直觉
- evidence-grounded generation
- 蒸馏数据生成
- 数据过滤
- unsupported answer reject
- student 训练
- base / teacher / student 对比

蒸馏流程：

```text
RAG 取证据
  -> teacher 基于证据回答
    -> 自动检查引用
      -> 人工抽检
        -> 过滤 unsupported answer
          -> student 训练
            -> 固定 eval set 对比
```

验收标准：

- 每条 teacher 样本必须有证据来源
- unsupported answer 必须被过滤
- student 必须在固定 eval set 上和 base / teacher 对比
- 必须复用第 12 章固定 eval set；如果蒸馏提升平均分但降低引用支持、拒答或高风险人工复核，本章实验不能算通过

对应目录：`17_evidence_constrained_distillation`

## 第 18 章：安全、合规与模型卡

核心问题：法律/医学领域模型不能只追求答得像，还要知道什么时候不能答。

讲法：安全不是最后补一页免责声明，而是贯穿任务定义、数据、RAG、SFT、蒸馏、部署的边界系统。

数学融入：风险分层、拒答阈值、不确定性、false negative 成本。安全章节要把“不能答”写成可评测、可触发的行为。

风险分类：

- 法律建议越界
- 医学诊断越界
- 紧急症状处理失败
- 引用不存在
- 隐私泄露
- prompt injection
- 不确定时仍强答

学习内容：

- 数据脱敏
- 隐私保护
- 拒答边界
- 不确定性表达
- 法律免责声明
- 医学免责声明
- model card
- risk report
- human review
- red-team set

发布门禁：

- 高风险问题拒答率
- 红旗症状召回率
- 引用支持率
- 人工复核触发率
- 隐私泄漏测试

验收标准：

- 高风险样例必须触发拒答或人工复核
- 引用不存在时不能输出确定性结论
- model card 和 risk report 必须随模型版本发布

对应目录：`18_safety_model_card`

## 第 19 章：量化、服务与发布门禁

核心问题：模型训练好了，怎么在成本、延迟、吞吐、质量和安全之间做可验证取舍？

讲法：先解释数值精度，再比较 fp16、int8、int4 对质量、速度和显存的影响，最后建立线上日志、监控、灰度和回滚。

数学融入：数值精度、量化误差、吞吐/延迟统计、quality regression。部署优化必须和固定 eval set 绑定，不能只看速度。

学习内容：

- FP32 / FP16 / BF16
- INT8 / INT4
- bitsandbytes
- GGUF
- vLLM
- API server
- batching
- latency
- throughput
- quality regression
- observability
- request / response logging
- PII masking in logs
- rate limiting
- prompt injection detection
- online eval
- canary release
- rollback playbook

主实验：

- fp16 / int8 / int4 对比
- 本地 API server
- benchmark
- 发布门禁和回滚配置

验收标准：

- 每个部署版本都有 eval report
- 日志中 PII/PHI 已 mask
- 质量或安全回归时 release gate 阻止发布

对应目录：`19_quantization_serving_release_gate`

## 第 20 章：法律合同审查项目

核心问题：如何把 RAG、LoRA、蒸馏、评测和安全组合成法律合同审查小模型？

项目方向：

- 合同风险识别
- 条款解释
- 修改建议
- 法条引用检查
- 不确定性提示
- 人工律师复核

硬约束：

- 必须标明司法辖区
- 必须标明法律来源版本和生效日期
- 必须引用具体 span
- 必须区分法律信息和法律建议

输出格式：

```json
{
  "jurisdiction": "CN | US-CA | ...",
  "risk_level": "high | medium | low | unknown",
  "risk_points": [],
  "evidence": [
    {
      "source_id": "",
      "title": "",
      "effective_date": "",
      "span_id": "",
      "support_level": "full | partial | none"
    }
  ],
  "revision_suggestion": "",
  "uncertainty": "",
  "legal_advice_boundary": "仅供信息参考，不构成法律意见",
  "needs_human_review": true
}
```

项目结构：

```text
legal_contract_review/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

验收标准：

- 固定 eval set 通过 release gate
- 引用支持率达标
- 高风险和不确定样例触发人工复核
- 输出不得构成确定性法律意见

对应目录：`20_legal_contract_review_project`

## 第 21 章：医学科普助手项目

核心问题：如何做一个谨慎、安全、可评测的医学科普助手？

项目方向：

- 医学科普问答
- 症状解释
- 就医建议
- 指南 RAG
- 危险信号识别
- 药物禁忌提示
- 特殊人群提醒：儿童、孕妇、老人、慢病患者

硬约束：

- 不替代医生诊断
- 红旗症状必须建议及时就医
- 必须标明指南来源和版本
- 必须保护 PHI / 隐私

输出格式：

```json
{
  "summary": "",
  "possible_explanations": [],
  "red_flags": [],
  "seek_care_level": "emergency | urgent | routine | self-care | unknown",
  "self_care_general_info": "",
  "medication_warning": "",
  "evidence": [
    {
      "source_id": "",
      "guideline_version": "",
      "span_id": ""
    }
  ],
  "uncertainty": "",
  "not_diagnosis": true,
  "needs_clinician_review": true
}
```

项目结构：

```text
medical_qa_assistant/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

验收标准：

- 红旗症状召回率达标
- 不确定或高风险问题触发人工复核
- 输出不得替代医生诊断
- 药物、儿童、孕妇等高风险场景必须谨慎处理

对应目录：`21_medical_qa_assistant_project`

## 第 22 章：毕业发布审计

核心问题：一个领域小模型能不能发布，应该由什么材料证明？

讲法：把课程中积累的代码、数据、评测、安全和部署材料汇总成一次 release review。

交付物：

- `eval_report.md`
- `failure_taxonomy.md`
- `model_card.md`
- `data_card.md`
- `risk_report.md`
- `release_decision.md`
- `rollback_playbook.md`

验收标准：

- 有明确发布结论：release / no release / internal only
- 每个高风险失败都有 owner 和 follow-up
- 发布版本可回滚
- 模型、数据、评测集和配置可追溯

对应目录：`22_graduation_release_audit`

## 进阶选修目录

进阶选修不是主线必修。它们适合在完成 MiniGPT、HF、eval、data、RAG、SFT、LoRA 和安全项目后再学。

## 选修 A1：偏好学习、DPO 与 RLHF 概念

核心目标：理解 SFT 后为什么还需要偏好优化，但不急着实现复杂 RL。

学习内容：

- preference data
- chosen / rejected
- reward model 直觉
- DPO 直觉
- RLHF 的基本流程
- KL 约束

最小实验：

- 构造 20 条 chosen / rejected
- 训练或模拟一个偏好打分器
- 比较 SFT 输出和偏好优化目标

## 选修 A2：PPO / GRPO toy 实验

核心目标：在可验证任务上理解 reward、advantage 和 KL 约束。

学习内容：

- policy
- reference model
- reward
- advantage
- PPO 直觉
- GRPO 直觉
- rule-based reward

适合任务：

- JSON 是否合法
- 引用是否存在
- 答案是否拒答
- 是否命中红旗症状
- 数学答案是否正确

## 选修 A3：DeepSeek-R1 推理训练复盘

核心目标：理解 R1-Zero、cold start、多阶段训练和蒸馏思路，不要求复现大规模训练。

学习内容：

- R1-Zero
- cold start SFT
- reasoning RL
- rejection sampling
- 多阶段训练
- CoT 数据
- 推理蒸馏

安全提醒：

- 法律/医学项目中，推理过程必须受证据约束
- teacher 的推理不能当事实来源
- 长 CoT 不是安全和正确性的保证

## 选修 A4：工作流优先：从确定性 pipeline 到受控 Agent

核心目标：知道什么时候需要 Agent，什么时候普通 workflow 更好。

原则：

- 能用普通 workflow 就不要用 Agent
- 能用固定检索流程就不要让模型自由规划
- 高风险动作必须 human-in-the-loop

学习内容：

- workflow vs agent
- tool calling
- planner
- memory
- reflection
- multi-step retrieval
- search + RAG
- 失败恢复

## 选修 A5：多模态基础

核心目标：理解图片和文本如何进入同一个模型系统，但不做医学影像诊断。

适用范围：

- 图片说明
- 图文检索
- 报告理解
- 普通图文问答

不适用范围：

- 医学影像诊断
- 高风险自动决策

学习内容：

- CNN 直觉
- ViT patch
- image token
- CLIP 对比学习
- BLIP
- LLaVA
- projection layer
- visual instruction tuning

## 选修 A6：长上下文与推理效率

核心目标：理解长文本不是只改上下文长度，还涉及位置编码、KV cache、显存和评测。

学习内容：

- RoPE scaling
- KV cache
- paged attention
- batching
- 长文本检索
- long-context eval
- lost-in-the-middle

## 第一阶段最小闭环：第 0-10 章，共 11 篇

第一批最应该顺序完成的章节：

1. `00_course_goal_and_risk_boundary`
2. `01_pytorch_training_loop`
3. `02_tensor_shape_pytorch`
4. `03_next_token_language_modeling`
5. `04_tokenizer_dataset_label_masking`
6. `05_embedding_and_similarity`
7. `06_bigram_to_neural_lm`
8. `07_attention_from_scratch`
9. `08_transformer_block`
10. `09_mini_gpt`
11. `10_llama_modern_block`

第 0-10 章不建议跳过。它们形成“从数学基础到模型知识”的最小闭环：

```text
知道风险边界
  -> 模型如何训练
    -> shape 如何流动
      -> 文本生成如何定义
        -> 文本如何变数字
          -> 数字如何变向量
            -> 统计语言模型如何过渡到神经语言模型
              -> token 如何互相看
                -> block 如何堆成 Transformer
                  -> 从零训练一个 MiniGPT
                    -> 映射到现代 LLaMA 架构
```

学完这条线，再进入：

1. `11_huggingface_workflow`
2. `12_minimum_eval_harness`
3. `13_domain_task_and_data_engineering`
4. `14_rag_baseline`
5. `15_sft_instruction_tuning`
6. `16_lora_qlora`
7. `17_evidence_constrained_distillation`

## 组件边界速记

为了避免把微调、检索、蒸馏和 Agent 混成一件事，先记住这张表：

| 组件 | 主要作用 | 不能替代什么 |
| --- | --- | --- |
| Eval | 判断模型是否真的变好，暴露失败 | 不能自动修复模型 |
| 数据工程 | 定义任务、来源、许可、脱敏、标签和 eval | 不能自动提升模型能力 |
| RAG | 提供可更新证据和 citation 链路 | 不能保证模型一定正确使用证据 |
| SFT | 学指令格式、输出结构、行为边界 | 不能保证事实和引用真实 |
| LoRA / QLoRA | 降低微调成本，便于 adapter 保存回滚 | 不能弥补脏数据或模糊任务 |
| 蒸馏 | 把强模型的可验证行为迁移给 student | 不能把 teacher 当事实来源 |
| Safety | 定义拒答、人工复核和发布门禁 | 不能让模型天然正确 |
| Agent | 组织工具、检索、规划和多步执行 | 不能把不清楚的任务自动变清楚 |

## 最终毕业项目

毕业项目二选一，不建议同时做法律和医学。课程验收看完整闭环，不看领域数量。

方向 A：法律合同审查小模型。

输入：合同条款、司法辖区、合同类型、可检索法律来源。

输出：

- 风险等级
- 风险点
- 引用证据
- 修改建议
- 不确定性提示
- 法律建议边界
- 是否需要人工律师复核

技术组合：

- RAG
- Eval
- LoRA
- 证据约束蒸馏
- 安全拒答
- 模型卡
- release gate

方向 B：医学科普问答小模型。

输入：用户医学问题、年龄/孕产/慢病等必要背景、可检索指南来源。

输出：

- 通俗解释
- 可能原因
- 红旗症状
- 何时就医
- 药物和特殊人群提醒
- 不确定性表达
- 不替代医生诊断
- 是否需要临床人员复核

技术组合：

- RAG
- Eval
- SFT / LoRA
- 安全拒答
- 证据约束蒸馏
- 人工复核边界
