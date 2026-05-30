# LLM 从 0 到领域小模型 Roadmap

这是一套从训练直觉出发，最终走向法律、医学等领域小模型工程的课程路线。目标不是零散看教程，而是建立一个长期可维护、可运行、可测试、可复现的学习工程。

## 整体主线

```text
训练闭环
  -> next-token 语言建模
    -> Tokenizer 与 Dataset
      -> Embedding 与固定上下文模型
        -> Causal Self-Attention
          -> Transformer Block
            -> Mini GPT
              -> 开源模型工作流
                -> SFT 指令微调
                  -> LoRA / QLoRA
                    -> 领域数据工程
                      -> RAG
                        -> 蒸馏
                          -> 评测
                            -> 安全与模型卡
                              -> 量化部署
                                -> 领域模型项目
```

一句话概括：

> 先从一个可验证的最小训练闭环出发，理解语言模型为什么是 next-token prediction；再逐步补上文本数字化、上下文表示、动态注意力和 Transformer；最后用数据、微调、RAG、蒸馏、评测、安全和部署，把模型做成可运行、可测试、可追溯的领域小模型。

## 贯穿样例线

为了避免课程后半段变成工程清单，本课程固定使用三类合同风险样例贯穿前后：

```text
违约金过高：合同 违约金 过高 ， 它 可能 存在 风险
责任范围过宽：赔偿一切损失，包括间接损失、可得利益损失及律师费
资料不足：缺少管辖区、法规版本、合同类型或检索依据
```

前 1-7 章用这些样例观察训练闭环、next-token、tokenizer、embedding、attention 和 Mini GPT。第 8-13 章把同一批样例迁移到 Hugging Face、SFT、LoRA、领域数据、RAG 和蒸馏。第 14-19 章再把它们变成评测项、安全门禁、部署日志和领域模板。

每章都应该能回答一个问题：

```text
本章新增了什么能力，让系统离“谨慎识别合同条款风险”更近一步？
```

## 组件边界速记

这张表应在学习第 11-13 章前先记住，避免把微调、检索和蒸馏混成一件事：

| 组件 | 主要作用 | 不能替代什么 |
| --- | --- | --- |
| SFT | 让模型学会指令格式、输出结构和行为边界 | 不能保证事实或引用真实 |
| LoRA / QLoRA | 降低微调成本，便于 adapter 保存和回滚 | 不能弥补脏数据或模糊任务 |
| 领域数据工程 | 定义任务、来源、许可、脱敏、标签和 eval | 不能自动让模型使用外部最新知识 |
| RAG | 提供可更新证据和 citation 链路 | 不能保证模型一定正确使用证据 |
| 蒸馏 | 把强模型的可验证行为迁移给 student | 不能把 teacher 当事实来源 |
| 评测 / 安全 | 暴露失败、定义发布门禁和人工复核边界 | 不能自动修复模型失败 |

## 每章最小实验闭环

每章可以有扩展实验，但主线只需要一个最小闭环：

| 章节 | 主实验 |
| --- | --- |
| 1 | 合同 toy classifier：baseline / no step / no zero_grad / overfit tiny |
| 2 | bigram LM：正确右移 vs 错误右移 |
| 3 | tokenizer + collator：pad 是否进入 loss |
| 4 | current token LM vs causal mean LM |
| 5 | attention with causal mask vs without causal mask |
| 6 | 1 / 2 / 4 层 block 的 loss 与 grad_norm 对比 |
| 7 | MiniGPT tiny corpus checkpoint round-trip |
| 8 | tiny HF model：load / generate / train one step / save_pretrained |
| 9 | 20 条 SFT 样本过拟合 JSON 输出格式 |
| 10 | LoRA rank 与 target_modules 对比 |
| 11 | 同一条合同条款生成 SFT / RAG / distill / eval 四种数据形态 |
| 12 | 5 个 query 的 top-k 检索与 citation support |
| 13 | teacher 样本过滤：通过率与 reject_reason |
| 14 | 5 条 eval item 生成 eval_report 和 failure_cases |
| 15 | safety policy 阻止高风险越界输出 |
| 16 | 同一模型 fp16 / int8 / int4 质量与延迟对比 |
| 17 | 一条合同条款端到端输出风险 JSON |
| 18 | 胸痛 + 呼吸困难样例触发 red flags |
| 19 | release gate 阻止缺报告、缺回滚目标的版本 |

## 阶段 0：项目准备与学习方法

目标：建立一个长期可维护的学习工程，而不是零散看教程。

核心内容：

- 项目结构
- 学习路线
- 环境配置
- notebook 规范
- 测试规范
- 教材写作规范

产出：

- `README.md`
- `roadmap.md`
- `AGENTS.md`
- `pyproject.toml`
- `environment.yml`
- `lessons/`
- `notebooks/`
- `src/`
- `tests/`
- `projects/`

## 阶段 1：PyTorch 与深度学习训练直觉

核心问题：神经网络到底是怎么“学会”的？

学习内容：

- Tensor
- shape
- 参数
- loss
- gradient
- backward
- optimizer
- `nn.Module`
- `Dataset`
- `DataLoader`
- 训练循环

产出：

- 一个 MLP 分类模型
- 一个完整训练循环
- 一个 pytest 测试文件

对应章节：`01_pytorch_training_intuition`

## 阶段 2：语言模型基础

核心问题：如果模型不是分类图片，而是继续写一句话，它该怎么训练？

学习内容：

- next token prediction
- `input_ids` / `labels` 错位
- cross entropy
- logits
- softmax
- sampling
- temperature
- top-k / top-p
- bigram language model

产出：

- 一个 bigram language model
- 一个文本生成函数 `generate()`
- 一个极小中文语料训练实验

对应章节：`02_language_modeling`

## 阶段 3：Tokenizer 与数据集构造

核心问题：模型不能直接吃中文文本，文本如何变成数字？

学习内容：

- token
- vocab
- token id
- encode / decode
- padding
- truncation
- attention mask
- BPE 直觉
- WordPiece 直觉
- language modeling dataset
- chat / SFT 数据格式

产出：

- 一个 simple tokenizer
- 一个 LM Dataset
- 一个 SFT 数据格式样例

对应章节：`03_tokenizer_and_dataset`

## 阶段 4：Embedding 与神经语言模型

核心问题：token id 只是编号，模型怎么从编号学出语义？

学习内容：

- embedding table
- embedding lookup
- hidden dimension
- context window
- neural language model
- parameter update

产出：

- 一个 embedding-based language model
- 可观察 embedding 参数变化的实验

对应章节：`04_embedding_and_neural_lm`

## 阶段 5：Attention 机制

核心问题：一句话中每个 token 如何决定自己应该看谁？

学习内容：

- Q / K / V
- scaled dot-product attention
- attention weights
- causal mask
- self-attention
- attention 可视化

产出：

- 手写 scaled dot-product attention
- 验证 causal mask
- 可视化 attention weights

对应章节：`05_attention`

## 阶段 6：Transformer Block

核心问题：attention 只是一个模块，完整的 LLM block 还缺什么？

学习内容：

- multi-head attention
- feed forward network
- residual connection
- LayerNorm / RMSNorm
- dropout
- pre-norm / post-norm
- Transformer block

产出：

- 手写 Multi-Head Attention
- 手写 Transformer Block
- 测试输入输出 shape

对应章节：`06_transformer_block`

## 阶段 7：Mini GPT 从零实现

核心问题：把 tokenizer、embedding、Transformer block 和 LM head 组合起来，是不是就得到 GPT？

学习内容：

- decoder-only 架构
- positional embedding / RoPE 直觉
- LM head
- causal language modeling
- 训练 mini GPT
- 生成文本
- checkpoint 保存和加载

产出：

- 一个可训练的 mini GPT
- 一个训练脚本
- 一个文本生成脚本
- checkpoint 保存与加载

对应章节：`07_mini_gpt`

## 阶段 8：Hugging Face 工作流

核心问题：现实中不可能每次从零写模型，如何使用开源模型？

学习内容：

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `datasets`
- `Trainer`
- Accelerate
- `model.generate`
- 模型保存
- 模型加载
- chat template

产出：

- 加载一个开源小模型
- 完成一次推理
- 完成一次最小微调
- 保存模型结果

对应章节：`08_huggingface_workflow`

## 阶段 9：SFT 指令微调

核心问题：怎么让模型从“续写文本”变成“按指令回答”？

学习内容：

- instruction tuning
- SFT
- system / user / assistant messages
- chat dataset
- 数据清洗
- train / val / test 划分
- 格式一致性
- 过拟合观察

产出：

- 一个 SFT 数据集
- 一个 SFT 训练脚本
- 一个训练前后效果对比

对应章节：`09_sft_instruction_tuning`

## 阶段 10：LoRA / QLoRA 参数高效微调

核心问题：全量微调太贵，能不能只训练少量参数？

学习内容：

- PEFT
- LoRA
- rank
- alpha
- target_modules
- adapter
- merge adapter
- QLoRA
- 4-bit quantization
- 显存优化

产出：

- 一个 LoRA 微调脚本
- 一个 QLoRA 微调脚本
- 一个 adapter 保存和加载流程

对应章节：`10_lora_qlora`

## 阶段 11：领域数据工程

核心问题：领域模型的能力主要来自哪里？模型，还是数据？

学习内容：

- 领域数据收集
- 数据许可与使用边界
- 数据清洗
- 去重和近重复检查
- 脱敏
- 质量过滤
- 指令数据构造
- RAG chunk 构造
- 蒸馏数据构造
- 评测数据构造与 eval set 冻结
- 法律/医学数据风险

产出：

- 领域 SFT 数据格式
- 领域评测集格式
- 数据清洗脚本
- 数据质量报告
- 数据 manifest

对应章节：`11_domain_data_engineering`

## 阶段 12：RAG 检索增强生成

核心问题：模型参数不是数据库，怎么让模型回答前先查资料？

学习内容：

- chunking
- embedding model
- vector store
- retriever
- top-k retrieval
- rerank 直觉
- prompt with context
- citation
- RAG hallucination

产出：

- 一个本地 RAG baseline
- 一个小型知识库
- 一个检索 + 生成 pipeline
- 输出带引用来源

对应章节：`12_rag_baseline`

## 阶段 13：蒸馏小模型

核心问题：大模型效果好但太贵，怎么把能力迁移给小模型？

学习内容：

- teacher model
- student model
- response distillation
- logit distillation 直觉
- preference distillation
- 蒸馏数据生成
- 蒸馏数据过滤
- student 训练

产出：

- teacher 数据生成脚本
- student 训练脚本
- base / teacher / student 对比

对应章节：`13_distillation`

## 阶段 14：模型评测

核心问题：模型看起来会说话，怎么证明它真的变好了？

学习内容：

- eval set
- 自动评测
- 人工评分
- 格式准确率
- 事实准确率
- 拒答能力
- 幻觉测试
- RAG 引用准确性
- 法律/医学安全评测

产出：

- `eval_runner.py`
- `metrics.py`
- `eval_report.md`
- 高风险失败案例表

对应章节：`14_evaluation`

## 阶段 15：安全、合规与模型卡

核心问题：法律/医学领域模型不能只追求答得像，还要知道什么时候不能答。

学习内容：

- 数据脱敏
- 隐私保护
- 拒答边界
- 不确定性表达
- 安全提示
- 法律免责声明
- 医学免责声明
- model card
- risk report
- human review

产出：

- `model_card_template.md`
- `risk_report.md`
- 安全测试集
- 拒答测试集

对应章节：`15_safety_and_model_card`

## 阶段 16：量化与部署

核心问题：模型训练好了，怎么在成本、延迟、吞吐、质量和安全之间做可验证的工程取舍？

学习内容：

- FP32 / FP16 / BF16
- INT8
- INT4
- bitsandbytes
- GGUF 直觉
- vLLM 直觉
- API server
- batching
- latency
- throughput
- quality / safety regression
- release gate
- rollback

产出：

- 量化推理脚本
- 本地 API server
- 简单 benchmark
- 发布门禁和回滚配置

对应章节：`16_quantization_and_serving`

## 阶段 17：法律领域小模型项目

核心问题：如何把微调、RAG、蒸馏、评测组合成法律模型？

项目方向：

- 合同风险识别
- 条款解释
- 合同修改建议
- 法律问答 RAG
- 法条引用检查

产出：

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

对应章节：`17_legal_domain_project`

## 阶段 18：医学领域小模型项目

核心问题：如何做一个谨慎、安全、可评测的医学科普助手？

项目方向：

- 医学科普问答
- 症状解释
- 就医建议
- 医学指南 RAG
- 危险信号识别
- 不替代医生诊断

产出：

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

对应章节：`18_medical_domain_project`

## 阶段 19：完整领域模型工程模板

核心问题：如何把一个领域模型项目做成可复用模板？

学习内容：

- 项目目录规范
- 数据版本管理
- 训练配置管理
- 实验记录
- 评测报告
- 模型发布
- 推理服务
- 持续迭代

产出：

```text
domain_model_template/
├── configs/
├── data/
├── scripts/
├── src/
├── tests/
├── reports/
└── README.md
```

对应章节：`19_domain_model_template`

## 推荐学习顺序精简版

如果只看主干，不看所有扩展，顺序是：

1. PyTorch 训练直觉
2. 语言模型基础
3. Tokenizer 与 Dataset
4. Embedding 与神经语言模型
5. Attention
6. Transformer Block
7. Mini GPT
8. Hugging Face 工作流
9. SFT 指令微调
10. LoRA / QLoRA
11. 领域数据工程
12. RAG
13. 蒸馏
14. 评测
15. 安全与模型卡
16. 量化部署
17. 领域模型完整项目

## 第一批最应该先做的章节

建议第一批只做这 5 章：

1. `01_pytorch_training_intuition`
2. `02_language_modeling`
3. `03_tokenizer_and_dataset`
4. `04_embedding_and_neural_lm`
5. `05_attention`

这 5 章形成最小闭环：

```text
模型如何训练
  -> 文本生成如何定义
    -> 文本如何变数字
      -> 数字如何变向量
        -> token 如何互相看
```

学完这 5 章，再进入：

- `06_transformer_block`
- `07_mini_gpt`
- `08_huggingface_workflow`
- `09_sft_instruction_tuning`
- `10_lora_qlora`

## 最终毕业项目

毕业项目二选一。

### 方向 A：法律合同审查小模型

输入：合同条款。

输出：

- 风险等级
- 风险点
- 依据
- 修改建议
- 不确定性提示

技术组合：

- RAG
- LoRA
- 蒸馏
- 评测
- 模型卡

### 方向 B：医学科普问答小模型

输入：用户医学问题。

输出：

- 通俗解释
- 可能原因
- 何时就医
- 风险提醒
- 不替代医生诊断

技术组合：

- RAG
- SFT
- 安全拒答
- 蒸馏
- 评测
