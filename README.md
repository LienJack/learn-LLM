# LLM 系统学习教材工程

语言版本：中文 | [English](en/README.md) | [日本語](ja/README.md)

这不是 API 教程，也不是只讲概念的科普材料。

本项目是一套可运行、可测试、可修改、可复现的中文 LLM 系统学习教材。目标是让学习者从最小训练闭环出发，逐步掌握语言模型、Tokenizer、Attention、Transformer、微调、RAG、蒸馏、评测与领域小模型工程。

本教材默认你已经具备 Python 基础，但不默认你已经系统学过深度学习、概率建模、信息论、GPU 训练或分布式系统。

## 本教材的主线

我们围绕一个核心问题推进：

> 如果要让模型根据上下文生成可靠答案，需要依次补上哪些能力？这些能力又如何被实验、测试和失败案例验证？

这条主线分成五层：

1. **优化层**：模型如何通过 loss、计算图、梯度和 optimizer 从错误中更新参数。
2. **序列建模层**：语言模型如何把“继续写一句话”转化为 next-token probability。
3. **表示与结构层**：Tokenizer、Embedding、Attention、Transformer 如何把文本变成可计算、可训练、可组合的上下文表示。
4. **适配与增强层**：如何用 SFT、LoRA/QLoRA、RAG、蒸馏把通用模型变成可用的任务模型。
5. **工程治理层**：如何用评测、安全、模型卡、部署和领域项目模板，让模型行为可验证、可追溯、可迭代。

本教材会尽量复用一个贯穿例子：从“合同条款风险识别”出发，逐步走到法律问答 RAG、领域微调、评测、安全拒答和模型卡。医学问答作为第二个高风险领域案例，用于强调安全边界和人类审核。

## 本教材不做什么

- 不从 Python 语法讲起。
- 不把 LLM 简化成 prompt 技巧。
- 不只给可以跑通的 demo。
- 不用“loss 下降了”替代严肃评测。
- 不把 RAG、LoRA、蒸馏混成一团。

## 每章固定交付物

每一章至少包含：

- `lessons/xx_*.md`：问题追问式正文。
- `notebooks/xx_*.ipynb`：可交互实验。
- `src/`：可复用实现。
- `tests/`：行为验收测试。
- 本章失败实验：展示常见错误和诊断方法。

每章都必须回答：

1. 当前能力缺口是什么？
2. 关键数学对象是什么？
3. 输入、输出、参数、loss 的 shape 是什么？
4. 最小实现是什么？
5. 如何设计实验观察它？
6. 常见失败模式是什么？
7. tests 如何证明实现没有坏？

## 第 1 章的学习契约

第 1 章不是“跑通一个 MLP”。它要建立后续所有章节都会复用的训练习惯：

- 训练闭环：`forward -> loss -> backward -> optimizer.step`。
- 计算图：理解 autograd 记录什么、反传什么。
- train/val：训练集表现不等于泛化能力。
- 可复现实验：固定 seed、数据划分、shuffle、初始化。
- overfit tiny：如果小数据都过拟合不了，训练管线大概率有 bug。
- tests 不是 smoke test：测试应该验证参数更新、loss 下降、eval 不产生梯度、train/eval 模式差异。

## 快速开始

使用 `venv`：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

使用 Conda：

```bash
conda env create -f environment.yml
conda activate llm-from-zero
pip install -e ".[dev]"
```

运行全部测试：

```bash
pytest -q
```

运行第 1 章基础训练实验：

```bash
python -m src.training.simple_mlp --experiment baseline
```

运行第 1 章小数据过拟合实验：

```bash
python -m src.training.simple_mlp --experiment overfit_tiny
```

运行 notebook：

```bash
jupyter lab
```

然后打开对应章节的 notebook，例如 `notebooks/01_tensor_autograd.ipynb`。

## 目录说明

- `lessons/`: 中文教材正文，采用问题追问式写法。
- `notebooks/`: 边读边运行的教学 notebook。
- `src/`: 可复用代码，从最小训练模块逐步演化到语言模型、微调、RAG、蒸馏和评测。
- `tests/`: 用 pytest 验证 shape、数值行为、训练行为、复现性和失败模式。
- `projects/`: 垂直领域项目模板，例如法律合同审查、医学问答助手。
- `reports/`: 模型评测报告和 model card 模板。
- `workflows/`: Pro + Codex 协作写作流程。

## 当前进度

- 已创建教材工程骨架，并建立 Pro + Codex 教材写作 workflow。
- `roadmap.md` 对应的 19 篇课程文章已经补齐，正文采用问题追问式结构。
- 第 1-19 章正文、notebook、src 与 tests 骨架已补齐，覆盖从训练闭环、语言建模、Tokenizer、Attention、Transformer、微调、LoRA、领域数据、RAG、蒸馏、评测、安全治理到部署和领域项目模板。
- 当前重点是把每章的正确实验、故意失败实验、pytest gate、notebook 观察项和命令行验收继续做实，并用 `pytest -q` 与报告产物证明交付状态。
- `projects/` 与 `reports/` 已补齐法律合同审查、医学问答助手、领域小模型模板、评测报告和 model card 示例。
- 当前交付物以本地测试和静态检查验收；全部文章暂不提交 Pro 单章审核，后续可由用户统一打包给 Pro 做总审。
