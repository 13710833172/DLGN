# DLGN：可微逻辑门网络

[English](README.md) | **简体中文**

**以连续门优化学习，以离散逻辑执行推理。**

本项目是 [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) 的独立 PyTorch 实现与复现研究。下述贡献指本仓库的实现与评估内容，底层学习范式归属于原论文。

## 摘要

构建由布尔运算组成的分类器，需要连接连续优化与离散执行。DLGN 通过稀疏连接的双输入逻辑门实现这一连接：训练时优化 16 种布尔运算的可微混合，导出时为每个门选择一种确定运算。本仓库将 soft 训练、离散导出与配对评估纳入统一的 PyTorch 工作流。代表性 MNIST 模型的测试准确率达到 **98.15%（soft）与 97.86%（hard）**；CIFAR-10 large 配置的 **hard 准确率达到 55.44%**，soft 与 hard 相差 **0.14 个百分点**。这些配对结果刻画了从可学习运算混合到离散逻辑门执行的转换表现。

## 贡献

- **明确的“训练—逻辑执行”路径。** 固定稀疏连接与可学习运算 logits，将梯度优化连接到可检查的离散逻辑门选择。
- **面向导出模型的配对评估。** 同时报告 soft 与 hard 模式，直接检验离散门选择对应的分类表现。
- **模块化研究实现。** 数据嵌入、逻辑运算、网络层、训练、评估与导出相互分离，为 [Conv-DLGN](https://github.com/13710833172/Conv-DLGN) 和 [Logic-SNN](https://github.com/13710833172/Logic-SNN) 提供基础。

## 方法概述

DLGN 学习的是**每个节点使用哪种逻辑运算**；节点的两个输入连接在初始化后保持固定。

![DLGN 方法概览：soft 网络的反向传播训练、保留相同连接的 argmax 离散导出与 hard 推理，以及全部 16 种逻辑算子。](docs/figures/DLGN_overview_redesign.svg)

*图 1：DLGN 的训练与推理。(a) soft 节点通过反向传播学习 16 种实值逻辑算子的混合。(b) 对每个节点取 argmax，选择一种布尔算子进行 hard 推理，同时保留连接与类别分组读出。(c) 完整的算子库、实值扩展及按输入 00、01、10、11 排列的真值输出。图中的网络规模与预测结果均为示意。*

[查看完整 SVG](docs/figures/DLGN_overview_redesign.svg)

对于输入为 $a,b$ 的逻辑门，可微输出为：

$
p_k = \frac{e^{\theta_k}}{\sum_{j=0}^{15} e^{\theta_j}},
\qquad
g_{\theta}(a,b) = \sum_{k=0}^{15} p_k f_k(a,b).
$

$
k^{*} = \arg\max_{0 \leq k \leq 15} \theta_k.
$

其中 $p_k$ 为布尔算子 $f_k$ 的归一化权重；离散导出时选择 logit 最大的算子 $k^{*}$。

$f_k$ 表示双输入布尔运算的连续扩展。每个 soft 门有 16 个 logits，导出后保留运算编号与输入连接。图像经展平或阈值嵌入后进入稀疏逻辑层，再通过类别分组求和生成分类分数。对应实现见[逻辑层](src/models/logic_layer.py)、[布尔运算](src/ops/logic_ops.py)与[离散导出](src/discretize.py)。

## 实验

**研究问题：将运算混合转换为确定逻辑门后，分类器能否保持其准确率？** 表 1 对每个配置给出两种执行模式的配对结果。

| 数据集 / 配置 | Soft 测试准确率 | Hard 测试准确率 | Soft − hard（百分点） |
| --- | ---: | ---: | ---: |
| MNIST / normal | 98.15% | **97.86%** | 0.29 |
| CIFAR-10 / small | 48.80% | 47.91% | 0.89 |
| CIFAR-10 / large | 55.58% | **55.44%** | **0.14** |

*表 1：已完成的代表性本地配置。每行对应同一个选定模型的 soft 评估与 hard 导出；不同行采用不同配置及训练设置。*

**核心发现。** MNIST 结果展示了离散门选择下 97.86% 的分类准确率；CIFAR-10 large 进一步展示了 soft 与 hard 执行的接近程度，二者相差 0.14 个百分点。表格用于评估单个模型的转换表现，不将跨配置差异归因于某个独立架构因素。

**实验口径。** 按验证集 soft 准确率选择检查点，再在测试集评估。表中为各个已完成配置的结果，不是多个独立随机种子的均值。MNIST 加载器在训练时使用连续展平像素，在验证和测试时使用阈值化像素；CIFAR-10 使用配置对应的阈值嵌入。该输入处理方式属于本实现的评估协议。这里测量的是 PyTorch 中的分类与离散转换表现；硬件时延和能耗属于独立评估维度。数据集、检查点与原始实验历史保留在仓库之外。

## 快速使用

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 完整训练配置。
python src/train.py --config configs/mnist_normal_paper.yaml
```

该示例启动新的训练，结果取决于配置与运行环境；表格汇总的是此前完成的配置。评估与导出时使用实际生成的检查点路径：

```bash
python src/evaluate.py --checkpoint path/to/checkpoint.pt
python src/discretize.py --checkpoint path/to/checkpoint.pt
python -m pytest tests/test_logic_ops.py tests/test_connectivity.py tests/test_output_grouping.py -v
```

## 仓库导航

| 入口 | 内容 |
| --- | --- |
| [src/data/](src/data/) | 数据加载与嵌入；下载的数据集放在根目录 `/data/` |
| [src/models/](src/models/)、[src/ops/](src/ops/) | Soft/hard 网络与布尔运算 |
| [configs/](configs/) | MNIST 与 CIFAR-10 配置 |
| [scripts/](scripts/) | 资源探测与绘图工具 |
| [REFERENCE_NOTES.md](REFERENCE_NOTES.md) | 方法来源与实现范围 |

## 结论

DLGN 将可微学习到离散逻辑的转换具体化为固定稀疏图、可学习布尔运算选择，以及对最终分类器的配对评估。
