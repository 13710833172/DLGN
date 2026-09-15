# 项目总览（给代码小白）

这份文档用大白话解释当前项目在做什么、做到哪一步了、每个核心文件是干嘛的。

---

## 1. 这个项目在做什么？

一句话：
**用“可微分逻辑门网络（Diff Logic Net）”做图像分类**，先训练“软逻辑门”，再导出“硬逻辑门”电路做推理。

和常见 CNN 不一样，这里不是学卷积核权重，而是：

- 每层连接关系固定随机生成（可复现）
- 每个神经元在 16 种逻辑门里学习“选哪一个门”
- 训练结束后把软选择离散化成硬门（argmax）

目前支持：

- MNIST（debug / small / normal）
- CIFAR-10（small / large）

---

## 2. 当前进展（截至现在）

已完成：

- 16 个逻辑算子（软/硬）完整实现
- 固定随机连接与可复现机制
- DiffLogicNet / HardLogicNet 全链路
- MNIST 与 CIFAR-10 数据管线
- CIFAR-10 两种论文向二值嵌入：
  - small：每通道 3 个阈值（0.25/0.5/0.75），输入维 9216
  - large：每通道 31 个阈值（i/32, i=1..31），输入维 95232
- 训练、评估、离散化脚本打通
- 日志、最终测试、Table-4 风格汇总打通
- 可行性/冒烟脚本（含大模型 OOM 明确提示）
- 测试用例完备（当前已通过）

风险点（你要知道）：

- `cifar10_large_paper.yaml`（5×256000, batch 100）非常重，很多常见 GPU 会 OOM。
- 项目不会偷偷改小参数；OOM 会直接报错并提示你手动复制配置修改。

---

## 3. 目录与文件逐项说明（核心）

> 说明：`logs/`、`figures/`、`checkpoints/` 里有很多运行产物，会越来越多；这里先讲“模板”和“核心代码文件”。

### 3.1 根目录

- `README.md`：项目主说明文档（环境、配置、命令、注意事项）
- `requirements.txt`：依赖版本（torch/torchvision/pyyaml/pytest/matplotlib）
- `PROJECT_SUMMARY_BEGINNER.md`：这份小白版总览

### 3.2 `configs/`（训练配置）

- `configs/mnist_debug.yaml`：快速调试用的小网络
- `configs/mnist_small_paper.yaml`：MNIST 论文向 small 配置（6×8000）
- `configs/mnist_normal_paper.yaml`：MNIST 论文向 normal 配置（6×64000）
- `configs/cifar10_small_paper.yaml`：CIFAR-10 论文向 small（4×12000, `paper_small` 嵌入）
- `configs/cifar10_large_paper.yaml`：CIFAR-10 论文向 large（5×256000, `paper_large` 嵌入）

### 3.3 `src/`（核心源码）

#### 入口脚本

- `src/train.py`：训练主程序
  - 读取 YAML 配置
  - 构建数据加载器（MNIST/CIFAR）
  - 训练软模型、验证软/硬精度
  - 保存最佳 checkpoint
  - 写 `train_log.csv`、`layer_stats.csv`、`final_test.json`、`run_summary.json`
  - 生成 Table4 汇总和离散 bundle
  - CUDA 时记录 `cuda_memory.csv`
  - OOM 时给明确提示（不静默改配置）
- `src/evaluate.py`：加载 checkpoint，在测试集评估 soft/hard 精度与 gap
- `src/discretize.py`：把训练好的 soft 模型导出为离散硬门 bundle（gate_ids + connectivity + meta）

#### 数据模块

- `src/data/mnist.py`：MNIST 下载、二值化、展平、train/val/test 划分
- `src/data/cifar10.py`：CIFAR-10 下载与二值嵌入
  - `paper_small`：3 阈值/通道 -> 9216 维
  - `paper_large`：31 阈值/通道 -> 95232 维
- `src/data/loaders.py`：数据分发与配置校验
  - 根据 `data.dataset` 选择 MNIST/CIFAR
  - 根据 `data.cifar10_embedding` 选择 small/large 嵌入
  - 校验 `input_dim`、`neurons_per_layer % num_classes == 0`

#### 模型模块

- `src/models/logic_layer.py`：可微逻辑层（软门）
- `src/models/logic_net.py`：多层可微网络（DiffLogicNet）
- `src/models/hard_logic_layer.py`：离散硬逻辑层
- `src/models/hard_logic_net.py`：硬网络（HardLogicNet），可由 soft 模型离散化得到

#### 算子与工具

- `src/ops/logic_ops.py`：16 种逻辑算子的软实现 + 真值表 + 硬查表
- `src/utils/connectivity.py`：固定随机二输入连接生成（可复现）
- `src/utils/output_grouping.py`：把最后一层输出按类别分组，得到 logits（tau/beta）
- `src/utils/metrics.py`：门熵、激活统计、soft-hard 一致性、gap 等指标
- `src/utils/table4_summary.py`：Table-4 风格汇总（Model/#Param/OPs/T[GPU]/Train Time）

#### 包初始化

- `src/__init__.py`
- `src/data/__init__.py`
- `src/models/__init__.py`
- `src/ops/__init__.py`
- `src/utils/__init__.py`

这些 `__init__.py` 主要是让 Python 把目录当包使用，本身通常不放业务逻辑。

### 3.4 `scripts/`（辅助脚本）

- `scripts/plot_training_log.py`：把 `logs/run_xxx` 的 CSV 画成图（train/val/layer/gates）
- `scripts/check_mnist_normal_feasibility.py`：MNIST normal 一步可行性检查（前向+反向）
- `scripts/check_cifar10_small_feasibility.py`：CIFAR small 一步可行性检查
- `scripts/check_cifar10_large_feasibility.py`：CIFAR large 一步可行性检查（高概率 OOM，会明确提示）

### 3.5 `tests/`（测试）

- `tests/conftest.py`：pytest 环境准备
- `tests/test_logic_ops.py`：16 逻辑算子正确性
- `tests/test_connectivity.py`：连接可复现性
- `tests/test_output_grouping.py`：分组与 tau/beta 逻辑
- `tests/test_discretization.py`：离散化正确性
- `tests/test_soft_hard_consistency.py`：soft/hard 一致性
- `tests/test_table4_summary.py`：Table4 参数展示与输出文件
- `tests/test_cifar10.py`：CIFAR 嵌入维度、阈值、配置校验
- `tests/__init__.py`：包标识

---

## 4. 运行后会生成什么文件（你最常接触）

每次训练会生成一个新目录：`logs/run_<时间戳>/`

常见文件：

- `train_log.csv`：每个 epoch 的训练/验证指标
- `layer_stats.csv`：每层门熵、激活统计
- `final_test.json`：最佳模型在测试集的 soft/hard 结果
- `run_summary.json`：本次运行产物索引
- `mnist_table4_summary.csv` / `.md`：Table4 风格单行总结（名字沿用 mnist）
- `discrete_bundle_table4.pt`：用于空间统计的离散 bundle
- `cuda_memory.csv`：CUDA 每 epoch 峰值显存（如果启用 CUDA）

另外：

- `checkpoints/.../best_model.pt`：最佳 checkpoint
- `figures/.../*.png`：画图脚本产出的可视化

---

## 5. 小白最常用命令（直接可抄）

### 安装依赖

```bash
cd /data/linan/MPhil_1/differentiable_logic_gate
pip install -r requirements.txt
```

### 先跑测试（确认代码健康）

```bash
python -m pytest tests/ -q
```

### 训练（例：CIFAR-10 small）

```bash
python scripts/check_cifar10_small_feasibility.py
python src/train.py --config configs/cifar10_small_paper.yaml
```

### 训练（例：CIFAR-10 large，务必先检查）

```bash
python scripts/check_cifar10_large_feasibility.py
python src/train.py --config configs/cifar10_large_paper.yaml
```

### 评估与离散化

```bash
python src/evaluate.py --checkpoint checkpoints/cifar10_small_paper/best_model.pt
python src/discretize.py --checkpoint checkpoints/cifar10_small_paper/best_model.pt
```

### 画训练图

```bash
python scripts/plot_training_log.py logs/run_<时间戳> -o figures/my_run
```

---

## 6. 给零基础同学的理解路线（建议）

建议按这个顺序看：

1. 先读 `README.md`（知道怎么跑）
2. 再看 `configs/*.yaml`（知道参数在什么地方改）
3. 看 `src/train.py`（知道完整流程）
4. 看 `src/data/*.py`（知道输入怎么变成二值向量）
5. 看 `src/models/*.py` + `src/ops/logic_ops.py`（知道模型核心）
6. 最后看 `tests/`（知道每块怎么被验证）

---

## 7. 当前状态一句话总结

**项目已经从“可跑 MNIST”扩展到“可跑 CIFAR-10 small/large、并带可行性与汇总体系”的完整实验框架；核心数学逻辑未改，工程流程已打通，主要瓶颈在大配置的显存需求。**

---

## 8. 运行结果分析：`figures/normal_mnist`（对应 `logs/run_20260415_163820`）

这一组图（`run_20260415_163820_train_val.png` / `layers.png` / `gates.png`）对应的是
**MNIST normal 配置**（6 层 × 64000，paper 风格）。

### 8.1 结论

- 这次训练是**成功且稳定**的。
- 模型精度很高：测试集硬模型准确率 **97.85%**，软模型 **97.88%**。
- 软硬差距（gap）只有 **0.03%**，说明离散化后几乎不掉点。
- 训练时长约 **7.52 小时**（200 epoch）。
- GPU 峰值显存约 **3.55 GiB**（在本机这次运行下是可承受的）。

### 8.2 关键指标（来自日志原始文件）

来自 `logs/run_20260415_163820/final_test.json`：

- `best_epoch`: 119
- `best_val_soft_acc`: 0.9766
- `best_val_hard_acc`: 0.9764
- `test_soft_acc`: 0.9788
- `test_hard_acc`: 0.9785
- `test_gap`: 0.0003

来自 `logs/run_20260415_163820/mnist_table4_summary.md`：

- Model: `Diff Logic Net`
- Param.: `384000`
- OPs: `384K`
- T. [GPU]: `0.1013 ms/img`（纯 PyTorch 硬推理计时）
- Train Time: `27087.4 s (7.5243 h)`

### 8.3 训练过程解读（看图时你该关注什么）

结合 `train_log.csv`：

- 第 1 轮：
  - `train_loss=1.0738`，`train_acc=64.74%`
  - `val_soft_acc=89.82%`，说明模型很快学到有效模式
- 最佳验证点在第 119 轮：
  - `val_soft_acc=97.66%`，`val_hard_acc=97.64%`
- 第 200 轮结束时：
  - `val_soft_acc=97.60%`，`val_hard_acc=97.56%`
  - 说明后期基本进入平台期（继续训练提升很有限）

### 8.4 `layers.png` 的含义（门控是否“学明白了”）

从 `layer_stats.csv` 的分层平均趋势看：

- 平均门熵（`mean_gate_entropy`）从 **1.9287**（epoch 1）降到 **0.6266**（epoch 200）
  - 含义：门选择从“犹豫”变得更“确定”
- 平均最大门概率（`mean_max_gate_probability`）从 **0.3945** 升到 **0.8031**
  - 含义：每个神经元更偏向某几个固定逻辑门
- 激活均值稳定在约 **0.497** 附近
  - 含义：整体激活分布稳定，没有明显塌缩

### 8.5 显存与可运行性

来自 `cuda_memory.csv`（200 轮统计）：

- `peak_allocated_gib`: 约 **3.44 GiB**
- `peak_reserved_gib`: 约 **3.55 GiB**

这说明这次 `mnist_normal_paper` 在当前环境是可稳定跑完的。