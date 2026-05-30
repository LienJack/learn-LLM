# 第 1 章：训练闭环、计算图与可复现实验

## 1. 本章真正要解决的问题

你已经会 Python，所以本章不讲语法。我们直接面对深度学习最核心的问题：

> 一个模型为什么能从错误里变好？我们又怎样证明这个“变好”不是错觉？

只会写下面这段代码还不够：

```python
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

这只是训练的表面。专业训练还要继续追问：

- 如果 `backward()` 算错了，怎么发现？
- 如果 loss 下降但验证集变差，模型是真的变好了吗？
- 如果 seed 不固定，实验结论可信吗？
- 如果参数根本没有更新，测试能抓出来吗？
- 如果小数据都过拟合不了，训练管线是不是有 bug？

本章要建立的是后面训练 mini GPT、SFT、LoRA、蒸馏都会复用的最小专业训练系统。

## 2. 问题链

1. 原始问题：代码能跑，不代表模型真的在学习。
2. Tensor shape 是训练系统的第一层契约。
3. 前向计算产生 loss，计算图记录局部依赖。
4. `backward()` 把 loss 的影响传回参数，`step()` 真正更新参数。
5. train/val split、overfit tiny、seed 和 history 让训练结论可验证。
6. tests 必须证明参数更新、loss 下降、评估无梯度和实验可复现。
7. 下一章问题：分类训练闭环建立后，如何把目标改成序列 next-token prediction？

## 3. Concept Card

| 概念 | 数学对象 | Shape | 代码对象 | 实验对象 |
| --- | --- | --- | --- | --- |
| tensor | 数值数组 | `(B, D)` | `torch.Tensor` | shape 检查 |
| model | 参数化函数 | `x -> logits` | `SimpleMLP` | forward |
| loss | 标量目标 | `()` | `CrossEntropyLoss` | loss 曲线 |
| gradient | 参数导数 | 与参数同 shape | `.grad` | grad norm |
| optimizer | 更新规则 | 参数集合 | `SGD/Adam` | update norm |
| split | 泛化估计 | train / val | `split_dataset` | val loss |
| seed | 随机控制 | scalar | `TrainingConfig.seed` | 可复现 |

## 4. 从数字容器到训练对象：为什么 tensor 的 shape 是第一语言

Tensor 可以先理解为“带 shape 的数字容器”。但在训练里，shape 不是注释，而是契约。

为了让这个契约和后面的领域项目接上，本章使用一个极小的合同风险 toy task：

```text
输入 x: [违约金比例, 逾期天数]
输出 y: 0 = 低风险, 1 = 高风险
```

例如：

```text
x = [0.01, 3]   -> 低风险
x = [0.30, 60]  -> 高风险
```

这当然不是一个真实法律模型。它的作用只是让我们用二维数字先看清训练闭环：

```text
x:      [batch_size, 2]
logits: [batch_size, 2]
y:      [batch_size]
loss:   scalar
```

这个 shape 契约比变量名可靠。后面语言模型会变成：

```text
input_ids: [batch_size, seq_len]
logits:    [batch_size, seq_len, vocab_size]
labels:    [batch_size, seq_len]
```

如果你现在不习惯追 shape，到了 Attention 的 `[B, H, T, T]` 时会很容易迷路。

本课程后面会反复回到三类合同风险：

```text
违约金过高：金额或比例明显偏高，需要提示风险
责任范围过宽：赔偿一切损失、间接损失、可得利益损失
资料不足：缺少管辖区、法规版本、合同类型或证据来源
```

本章只把它们压缩成二维 toy features，用来回答一个更基础的问题：loss 是否真的通过梯度改变了参数。等到第 17 章，这三类风险会重新展开成脱敏条款、RAG citation、JSON 输出和人工复核门禁。

## 5. 计算图：PyTorch 到底记录了什么

训练不是“模型犯错后自动变聪明”。更准确地说：

1. 前向计算把输入变成 loss。
2. PyTorch 在前向过程中记录计算图。
3. 反向传播沿着计算图，把 loss 对每个参数的影响传回去。
4. optimizer 根据梯度更新参数。

例如一个两层分类器可以写成：

```python
h = torch.relu(x @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, y)
```

这里要注意：`cross_entropy` 接收的是 raw logits，而不是 softmax 之后的概率，也不建议把最后输出先过 ReLU 再交给它。隐藏层可以有 ReLU，最后一层输出要保留为未归一化的类别分数。

计算图里的每个节点都只需要回答一个局部问题：

> 上游传来了 `dL / dout`，我根据自己的局部公式，应该把多少梯度传给输入和参数？

反向传播不是一次性对整个模型施法，而是很多局部链式法则连起来。

### `requires_grad`、`grad_fn` 和 leaf tensor

- `requires_grad=True` 表示 PyTorch 需要跟踪这个 tensor 参与的计算。
- `grad_fn` 指向产生这个 tensor 的计算节点。
- `nn.Parameter` 通常是 leaf tensor，训练后梯度会累积到它的 `.grad` 上。

这也是为什么你不能随便在训练中插入 `.detach()` 或把中间结果变成 `.item()`。它们可能切断计算图，让梯度回不到参数。

## 6. `backward()` 和 `step()` 的分工

一句话：

> `loss.backward()` 计算“该怎么改”，`optimizer.step()` 执行“真的去改”。

更具体地说：

```python
optimizer.zero_grad()
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

- `zero_grad()`：清掉上一批留下的梯度。
- `model(x)`：前向计算，建立计算图。
- `loss_fn(logits, y)`：把预测错误压成一个标量。
- `backward()`：沿计算图计算每个参数的 `.grad`。
- `step()`：用 `.grad` 更新参数。

如果忘记 `step()`，loss 可以被算出来，但参数不会变。

如果忘记 `zero_grad()`，梯度会跨 batch 累加，初学阶段常常导致训练现象难以解释。

## 7. 梯度检查：不要盲目信任你刚写的模块

PyTorch 的内置算子通常可靠，但当你后面自己写 attention、mask、loss 或自定义模块时，需要知道一种 sanity check：

```text
数值梯度 ≈ [L(theta + eps) - L(theta - eps)] / (2 * eps)
```

这叫有限差分梯度检查。

它不是训练时使用的方法，因为太慢；它是调试时用来回答：

> autograd 算出来的梯度，和数值近似梯度方向一致吗？

本章代码保持 MLP 简洁，但测试和正文会建立这个意识。后面写 attention 时，这个意识会变得非常重要。

## 8. train/val split：训练集变好不等于模型变好

第一版代码的问题是：训练和评估使用同一个 dataloader。这只能说明模型在训练集上表现变好了，不能说明它学到了可泛化规律。

专业训练必须至少拆成：

- train set：给 optimizer 更新参数。
- validation set：不更新参数，只观察泛化表现。

于是每轮训练要记录：

```text
train_loss, train_acc
val_loss, val_acc
grad_norm, update_norm
```

如果出现：

```text
train_loss 持续下降
val_loss 开始上升
```

这通常意味着过拟合：模型越来越记住训练数据，但对没见过的数据不一定更好。

## 9. overfit tiny：小数据都记不住，训练管线大概率有问题

一个非常有用的训练 sanity check 是：

> 取很小一批无噪声数据，让模型反复训练，它应该能几乎 100% 记住。

如果小数据都过拟合不了，可能的问题包括：

- loss 和 labels 对不上。
- optimizer 没有更新参数。
- 学习率太小或太大。
- 模型容量不够。
- 数据和标签被打乱错配。
- 训练模式、梯度或 device 处理有 bug。

本章提供：

```bash
python -m src.training.simple_mlp --experiment overfit_tiny
```

这不是为了追求真实泛化，而是为了验证训练管线本身有学习能力。

## 10. 初始化、学习率和 batch size

### 初始化

模型参数一开始不是“空白”，而是随机初始化。初始化影响：

- 初始 logits 的尺度。
- 梯度的尺度。
- 不同 seed 下训练曲线的差异。

### 学习率

学习率控制每次更新参数走多大一步：

- 太小：loss 下降很慢。
- 合适：loss 稳定下降。
- 太大：loss 震荡甚至发散。

### batch size

batch size 控制每次用多少样本估计梯度：

- 小 batch：梯度噪声大，但更新频繁。
- 大 batch：梯度更稳定，但每次更新成本更高。

这些不是调参玄学，而是训练系统的观测对象。第 1 章代码会返回 history，让你能比较不同配置下的曲线。

## 11. 随机种子与可复现实验

“我跑了一次 loss 降了”不是可靠结论。专业训练至少要控制：

- dataset 生成 seed。
- train/val split seed。
- model 初始化 seed。
- DataLoader shuffle generator。

本章代码用统一的 `TrainingConfig(seed=...)` 控制这些随机源。测试里会检查固定 seed 下训练 history 和最终参数可复现。

这不是形式主义。等你做 LoRA、DPO 或 RAG 评测时，如果实验不可复现，错误分析会非常痛苦。

## 12. `model.train()`、`model.eval()` 与 `torch.no_grad()`

这三个东西经常被混在一起，但它们不是一回事。

### `model.train()`

告诉模型进入训练模式。Dropout 会随机丢弃部分激活，BatchNorm 会更新统计量。

### `model.eval()`

告诉模型进入评估模式。Dropout 关闭随机性，BatchNorm 使用已有统计量。

### `torch.no_grad()`

告诉 PyTorch 不要记录计算图。它节省内存，也避免验证阶段意外产生梯度。

所以评估函数通常同时需要：

```python
@torch.no_grad()
def evaluate(...):
    model.eval()
```

本章的 `SimpleMLP` 支持可选 dropout，就是为了让你实际看到 train/eval 模式的差异。

## 13. 本章代码结构

核心代码在 `src/training/simple_mlp.py`。

它提供：

- `TrainingConfig`：集中管理 seed、lr、batch size、epochs 等配置。
- `TrainingHistory`：结构化记录每个 epoch 的 train/val 指标。
- `set_seed`：统一控制随机性。
- `split_dataset`：生成无重叠 train/val split。
- `make_dataloaders`：构造固定 shuffle generator 的 dataloader。
- `compute_grad_norm`：观察梯度是否存在、是否爆炸。
- `compute_update_norm`：观察参数是否真的更新。
- `run_training`：运行基础训练实验。
- `run_overfit_tiny_experiment`：运行小数据过拟合实验。

运行基础实验：

```bash
python -m src.training.simple_mlp --experiment baseline
```

运行过拟合小数据实验：

```bash
python -m src.training.simple_mlp --experiment overfit_tiny
```

## 14. 必写实验

- baseline 训练：记录 train/val loss、accuracy、grad norm 和 update norm。
- overfit tiny：验证小数据可被模型记住。
- seed 复现实验：同一配置下 history 和参数可复现。
- train/eval 对照：观察 dropout 在两种模式下的行为差异。
- 有限差分梯度检查：对小模块验证 autograd 梯度方向。

本章主实验可以收束为四组对照：

| 实验 | 改动 | 观察指标 | 预期现象 |
| --- | --- | --- | --- |
| baseline | 正常训练 | train/val loss、accuracy、update_norm | loss 下降，参数更新 |
| no step | 跳过 `optimizer.step()` | update_norm | loss 可计算，但参数不变 |
| no zero_grad | 跳过 `zero_grad()` | grad_norm、loss 曲线 | 梯度累积，曲线更难解释 |
| overfit tiny | 很小无噪声数据反复训练 | train accuracy | 应接近 100% |

这四组实验比单独看一条 loss 曲线可靠：它们分别证明“能学习”“没更新会被发现”“梯度累积会被发现”“管线容量足以记住简单样本”。

## 15. 失败模式

- 忘记 `optimizer.step()`：loss 被计算出来，但参数不更新。
- 忘记 `optimizer.zero_grad()`：梯度跨 batch 累加，训练现象混乱。
- 训练集和验证集混用：泛化能力被高估。
- 小数据都无法过拟合：数据、label、学习率或更新链路大概率有 bug。
- 评估阶段没有 `torch.no_grad()`：验证过程会记录不必要的计算图，浪费显存并变慢；如果后续错误地对验证 loss 调用 `backward()`，还会把不该参与训练的梯度混入调试过程。
- seed 不固定：一次运行的结论无法复查。

## 16. 测试验收

本章测试不只检查“代码能跑”。它要证明训练管线的关键行为成立：

- dataset 输出 shape 正确。
- model forward 输出 shape 正确。
- train/val split 没有重叠。
- loss 随 epoch 明确下降。
- 参数在一个 epoch 后确实更新。
- evaluate 不产生梯度。
- 固定 seed 可复现。
- 小样本可以过拟合。
- train/eval 模式下 dropout 行为不同。
- grad norm 和 update norm 可观测且大于 0。

运行：

```bash
pytest -q tests/test_training_loop.py
```

## 17. 本章验收标准

你学完本章后，应该能回答：

- 为什么训练集 loss 下降不等于模型泛化能力变好？
- `loss.backward()` 计算了什么？结果放在哪里？
- `optimizer.step()` 怎么证明真的更新了参数？
- 为什么要做有限差分梯度检查？
- 为什么 overfit tiny 是训练管线 sanity check？
- 为什么固定 seed 不只是“为了结果好看”？
- `model.eval()` 和 `torch.no_grad()` 的区别是什么？

## 18. 本章记忆锚点与边界

本章解决了一个最基本的问题：

> 模型不是因为“看见答案”就自动变聪明，而是因为 loss 通过计算图产生梯度，optimizer 用梯度更新参数。

你需要记住三件事：

1. **shape 是训练系统的第一层契约**：shape 不对，后面的解释都不可靠。
2. **loss 下降不等于模型变好**：必须看 validation、overfit tiny、seed 复现和失败实验。
3. **tests 不是 smoke test**：测试要证明参数真的更新、评估不建图、小数据能过拟合、随机性可复现。

本章没有解决语言模型问题。

## 19. 下一章预告

分类器预测的是：

```text
P(y | x)
```

语言模型要预测的是：

```text
P(x_t | x_<t)
```

也就是说，它不是在一个固定类别里选答案，而是在每个位置预测下一个 token。

下一章会把训练闭环迁移到序列概率建模：cross entropy、perplexity、input/label shift 和 bigram language model。
