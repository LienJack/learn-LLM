# LLM 从 0 到领域小模型

这是一套面向中文读者的 LLM 系统课程。它不把大模型学习简化成 prompt 技巧，也不只停留在“知道 Transformer 很重要”的概念层面，而是带你从学习仓库、训练闭环和最小 GPT 开始，逐步走到评测、数据工程、RAG、SFT、LoRA、证据约束蒸馏、安全部署和领域项目发布审计。

课程主线很明确：如果你想让一个模型在法律、医学这类高风险领域中谨慎回答，它需要依次具备哪些能力？这些能力如何训练出来？如何被实验看见？又如何被测试、评测、模型卡和发布门禁约束住？

## 这套课讲什么

新版目录从第 0 章开始建立风险边界与可复现仓库，然后补齐数学直觉、PyTorch 训练闭环、张量 shape、next-token 语言模型、Tokenizer、Embedding、Attention、Transformer Block 和 MiniGPT。完成从零实现后，课程转入现代 LLaMA 架构、Hugging Face 工作流、最小评测系统、领域数据工程、RAG、SFT、LoRA/QLoRA、证据约束蒸馏、安全合规、量化服务与发布门禁，最后落到法律合同审查、医学科普助手和毕业发布审计。

贯穿样例包括合同风险、医学科普和结构化 JSON 输出。法律项目只做信息辅助，不构成法律意见；医学项目只做科普和分流提醒，不替代医生诊断。课程会反复强调：模型输出不是“看起来对”就结束，而是要能被数据、实验、失败案例、引用证据和可追溯报告共同支撑。

## 核心概念地图

```text
风险边界与学习仓库
  -> 函数、张量、概率与训练闭环
    -> Tokenizer / Embedding / Attention / Transformer
      -> MiniGPT 从零实现
        -> 现代 LLaMA 与 Hugging Face
          -> 最小评测系统
            -> 数据工程 / RAG / SFT / LoRA / 蒸馏
              -> 安全、模型卡、量化服务、发布门禁
                -> 法律项目 / 医学项目 / 毕业发布审计
```

## 课程设计思路

每章都围绕一个真实能力缺口展开，而不是罗列术语。你会先看到“本章核心困惑”，再进入前置知识、新增能力、最小推导或最小代码、常见错误、测试验收，以及它和法律/医学领域项目的关系。课程刻意保留失败模式，例如错误右移 label、pad token 进入 loss、attention mask 泄漏未来信息、检索无依据却强答、量化后不跑安全回归，因为这些错误比顺滑 demo 更能帮助你建立工程判断。

## 你能学到什么

学完这套课程，你应该能够从零实现一个小型 GPT 训练闭环，读懂现代开源模型工作流，构造 SFT、RAG、蒸馏和评测数据，使用 LoRA/QLoRA 做参数高效微调，搭建带引用的本地 RAG baseline，并为法律或医学这类高风险应用设计安全边界、模型卡、评测报告、发布门禁和回滚方案。

## 推荐读法

如果你还没有系统做过深度学习训练，建议从第 0 章顺序读到第 10 章，先把学习仓库、MiniGPT 骨架和现代 LLaMA 组件映射搭起来。如果你已经熟悉 Transformer，可以从第 10 章进入现代开源模型工作流，再重点读第 12-19 章，补齐评测、数据、RAG、微调、蒸馏、安全和部署。第 20-22 章适合在你准备做自己的领域模型项目时反复参考。

主线文章会在每章就地展开本章必须用到的公式、手算例子和工程含义；[数学深挖篇：从可学习函数到可靠领域小模型](lessons/math_foundations_deep_dive.md) 用来把这些分散在各章里的数学对象再串成一条完整问题链。读完第 1-7 章后，可以用它复盘函数、张量、线性代数、概率信息论、梯度优化、RoPE、LoRA 和评测统计之间的关系。

## 课程目录

| 章节 | 文章 | 概述 |
| --- | --- | --- |
| 00 | [课程目标、风险边界与学习仓库](lessons/00_course_goal_and_risk_boundary.md) | 建立可复现学习仓库、实验日志、数据 manifest、空 eval report 和法律/医学风险边界。 |
| 01 | [从函数到 PyTorch 训练闭环](lessons/01_pytorch_training_loop.md) | 从 `y=f(x)`、loss、梯度和优化器理解神经网络如何学习，并用 toy classifier 验证训练闭环。 |
| 02 | [张量、shape 与 PyTorch 基础](lessons/02_tensor_shape_pytorch.md) | 用 `[B,T]`、`[B,T,C]`、`[B,H,T,D]`、`[B,T,V]` 贯穿 LLM 的 shape 契约。 |
| 03 | [概率、Softmax、交叉熵与 next-token](lessons/03_next_token_language_modeling.md) | 把续写文本转成 next-token 概率建模，理解 labels 右移、采样和 perplexity。 |
| 04 | [Tokenizer、LM Dataset 与 padding label masking](lessons/04_tokenizer_dataset_label_masking.md) | 讲清文本到 token id、padding、attention mask、LM dataset 和 padding label masking。 |
| 05 | [Embedding 与相似度](lessons/05_embedding_and_similarity.md) | 区分 token embedding 与 retrieval embedding，用点积和余弦相似度理解语义空间。 |
| 06 | [从 Bigram 到神经语言模型](lessons/06_bigram_to_neural_lm.md) | 从 bigram 统计表走向 embedding + MLP，建立最小神经语言模型和生成循环。 |
| 07 | [Attention 手算与实现](lessons/07_attention_from_scratch.md) | 手写 scaled dot-product attention，理解 Q/K/V、`sqrt(d_k)`、causal mask 和 `[B,H,T,T]`。 |
| 08 | [Transformer Block](lessons/08_transformer_block.md) | 把 multi-head attention、FFN、残差、归一化和复杂度估算合成一个可训练 block。 |
| 09 | [MiniGPT 从零实现](lessons/09_mini_gpt.md) | 组合 tokenizer、embedding、Transformer block 和 LM head，完成训练、生成、保存加载和 pytest。 |
| 10 | [现代 LLaMA 架构](lessons/10_llama_modern_block.md) | 从 MiniGPT 迁移到 RoPE、RMSNorm、SwiGLU、GQA 和 KV cache 等现代 decoder-only 组件。 |
| 11 | [Hugging Face 工作流](lessons/11_huggingface_workflow.md) | 使用 `AutoTokenizer`、`AutoModelForCausalLM`、`Trainer`、`generate()` 和 `save_pretrained()`。 |
| 12 | [最小评测系统](lessons/12_minimum_eval_harness.md) | 在 RAG/SFT/LoRA 前搭建 eval harness、schema validation、failure cases 和回归评测。 |
| 13 | [领域任务定义与数据工程](lessons/13_domain_task_and_data_engineering.md) | 定义任务边界、数据来源、许可、脱敏、标注规范、manifest、data card 和 eval 冻结。 |
| 14 | [RAG Baseline](lessons/14_rag_baseline.md) | 按离线入库、在线检索、带引用生成、无依据拒答实现最小 RAG。 |
| 15 | [SFT 指令微调](lessons/15_sft_instruction_tuning.md) | 让模型学习指令格式、JSON 输出契约和行为边界，同时报告事实与引用风险。 |
| 16 | [LoRA / QLoRA 参数高效微调](lessons/16_lora_qlora.md) | 用低秩 adapter 和量化底座降低微调成本，并评估 rank、target_modules 和 merge 差异。 |
| 17 | [证据约束蒸馏](lessons/17_evidence_constrained_distillation.md) | 让 teacher 基于 RAG 证据生成训练样本，过滤 unsupported answer，再训练 student。 |
| 18 | [安全、合规与模型卡](lessons/18_safety_model_card.md) | 建立拒答边界、不确定性表达、red-team set、model card、risk report 和 human review。 |
| 19 | [量化、服务与发布门禁](lessons/19_quantization_serving_release_gate.md) | 比较 fp16/int8/int4，搭建服务、日志、监控、canary、release gate 和 rollback。 |
| 20 | [法律合同审查项目](lessons/20_legal_contract_review_project.md) | 把 RAG、LoRA、蒸馏、评测、安全和部署组合成法律信息辅助系统。 |
| 21 | [医学科普助手项目](lessons/21_medical_qa_assistant_project.md) | 构建谨慎、安全、可评测的医学科普助手，覆盖红旗症状、指南引用和 clinician review。 |
| 22 | [毕业发布审计](lessons/22_graduation_release_audit.md) | 汇总 eval、failure taxonomy、model card、data card、risk report、release decision 和 rollback。 |

## 数学专题

| 专题 | 文章 | 概述 |
| --- | --- | --- |
| M01 | [数学深挖篇：从可学习函数到可靠领域小模型](lessons/math_foundations_deep_dive.md) | 从“为什么不能手写规则”一路推到函数、张量、相似度、概率信息论、梯度优化、深层稳定性、RoPE、LoRA 和评测统计。 |

## 快速开始

建议先创建虚拟环境并安装项目依赖，然后运行最小测试：

```bash
python -m pytest tests -q
```

如果你只想阅读课程文章，可以从第 0 章开始；如果你想跟着做实验，请把每章的最小代码、测试验收和报告产物一起完成。
