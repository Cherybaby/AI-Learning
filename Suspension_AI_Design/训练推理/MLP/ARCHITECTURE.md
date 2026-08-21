# MLP (PointMLP) 网络架构说明

> 对应代码：`Suspension_AI_Design/训练推理/MLP/train_nn.py`
> 本文档描述的所有维度、参数量、指标均来自**已训练模型的真实 checkpoint**
> （`nn_out/model.pt` 的 `ckpt['arch']` + 重建模型后 `named_parameters()` 的真实 shape，
> 以及 `nn_out/metrics.json` 的真实训练结果），非凭空推算。

## 1. 任务定义

用 33 维硬点坐标 + 1 维行程（`wc_stroke`）预测 8 个悬架运动学量，逐行独立回归
（不是序列模型，每个 `(硬点, stroke)` 组合当作一个独立样本）。

| 项目 | 值 |
|---|---|
| 输入维度 | 34（33个硬点坐标 + 1个 `wc_stroke`） |
| 输出维度 | 8（运动学量） |
| 训练样本 | 969 组几何 × 101 个 stroke 点（长格式，每行独立） |
| 模型类型 | `PointMLP`（逐点三维编码器，本次训练实际使用的 `model_kind='point'`） |
| 集成成员数 | 5（`Ensemble`，5个独立训练的 PointMLP 取预测均值） |

## 2. 输入构成（34维）

33维硬点坐标可拆分为 **11 个硬点 × 3 坐标(x,y,z)**，加 1 维 `wc_stroke`：

| 硬点分组 | 坐标名 |
|---|---|
| O1 | O1_x, O1_y, O1_z |
| O2 | O2_x, O2_y, O2_z |
| O3 | O3_x, O3_y, O3_z |
| O4 | O4_x, O4_y, O4_z |
| S1 | S1_x, S1_y, S1_z |
| S2 | S2_x, S2_y, S2_z |
| U1 | U1_x, U1_y, U1_z |
| U2 | U2_x, U2_y, U2_z |
| U3 | U3_x, U3_y, U3_z |
| U4 | U4_x, U4_y, U4_z |
| R0 | R0_x, R0_y, R0_z |
| stroke | wc_stroke（行程，范围 [-50, 50] mm） |

`n_points=11, coord=3` → `11×3+1=34`，与代码中 `PointMLP.__init__` 的断言
`assert n_points * coord + 1 == n_in` 一致。

## 3. 输出（8维）

| 输出量 | 说明 |
|---|---|
| toe | 前束角 (°) |
| camber | 外倾角 (°) |
| caster | 后倾角 (°) |
| caster_arm | 后倾拖距 (mm) |
| scrub_radius | 主销偏移距 (mm) |
| tire_con_point_x | 接地点 X (mm) |
| tire_con_point_y | 接地点 Y (mm) |
| tire_con_point_z | 接地点 Z (mm) |

## 4. 网络结构（单个集成成员，`PointMLP`）

真实训练超参（来自 `model.pt` 的 `ckpt['arch']`）：

```
model_kind = "point"
n_in       = 34
n_out      = 8
n_points   = 11
coord      = 3
emb        = 32          # 每个硬点独立编码维度
trunk      = [512, 256, 128]
p_drop     = 0.0
residual   = False        # 主干为普通 Sequential（非残差块堆叠）
hard_idx   = None          # 未启用独立输出头
```

### 4.1 逐点编码层（Encoder，每个硬点独立权重）

33维硬点先 reshape 成 `(11, 3)`，即11个点各自的(x,y,z)。**每个点用自己独立的一组
权重**（不共享），经过两层小 MLP 编码成32维：

| 层 | 参数 | 形状 | 参数量 | 说明 |
|---|---|---|---|---|
| 输入 | — | `(B, 11, 3)` | — | 11个硬点，每点3坐标 |
| W1 | `nn.Parameter` | `(11, 3, 32)` | 1,056 | 11个点各自的 3→32 投影权重 |
| b1 | `nn.Parameter` | `(11, 32)` | 352 | 对应偏置 |
| ReLU | — | `(B, 11, 32)` | 0 | 非线性激活 |
| W2 | `nn.Parameter` | `(11, 32, 32)` | 11,264 | 11个点各自的 32→32 投影权重 |
| b2 | `nn.Parameter` | `(11, 32)` | 352 | 对应偏置 |
| ReLU + Dropout | — | `(B, 11, 32)` | 0 | 非线性激活 + Dropout(p=0.0) |
| Flatten | — | `(B, 352)` | 0 | `11×32=352`，拼接所有点的编码 |

通过 `torch.einsum` 实现"分组线性变换"，让"同一个硬点的三个坐标是一个整体"
这一先验被编码进网络结构，而不是把33维当作无结构的扁平向量直接送入MLP。

拼接 `wc_stroke`（1维）后，主干网络输入为 `352 + 1 = 353` 维。

### 4.2 主干网络（Trunk，`ResTrunk(residual=False)` → 普通 Sequential）

| 层 | 维度变换 | 参数量 |
|---|---|---|
| Linear + ReLU + Dropout | 353 → 512 | weight 180,736 + bias 512 = 181,248 |
| Linear + ReLU + Dropout | 512 → 256 | weight 131,072 + bias 256 = 131,328 |
| Linear + ReLU + Dropout | 256 → 128 | weight 32,768 + bias 128 = 32,896 |

主干输出维度（`trunk_dim`）= 128。

### 4.3 输出头（`MultiHead`，`hard_idx=None` 时退化为单一 Linear）

| 层 | 维度变换 | 参数量 |
|---|---|---|
| Linear (`head.shared`) | 128 → 8 | weight 1,024 + bias 8 = 1,032 |

本次训练未启用独立输出头（`hard_idx=None`），8个输出量共享同一个线性层。

### 4.4 单成员参数量汇总

| 模块 | 参数量 |
|---|---|
| 逐点编码器（W1/b1/W2/b2） | 13,024 |
| 主干网络（3层Linear） | 345,472 |
| 输出头 | 1,032 |
| **单成员总计** | **359,528** |
| **5个集成成员总计** | **1,797,640** |

### 4.5 完整前向数据流（单成员，batch维省略）

```
输入 (34,)
  ├─ 硬点部分 (33,) → reshape (11, 3)
  │     │
  │     ├─ einsum(W1) + b1 → ReLU        (11, 32)
  │     ├─ einsum(W2) + b2 → ReLU+Drop   (11, 32)
  │     └─ flatten                        (352,)
  │
  └─ stroke部分 (1,) ──────────────────────┘
                    │
              concat → (353,)
                    │
        Linear+ReLU+Drop (353→512)
                    │
        Linear+ReLU+Drop (512→256)
                    │
        Linear+ReLU+Drop (256→128)
                    │
        Linear (128→8)  [head.shared]
                    │
                输出 (8,)
```

集成推理：5个独立训练的 `PointMLP` 各自前向一次，取输出的算术平均
（`Ensemble.forward`: `torch.stack([...]).mean(0)`），并裁剪到训练集各输出量的
`[min-2%range, max+2%range]` 值域范围（抑制外推异常值）。

## 5. 训练配置

| 项 | 值 |
|---|---|
| 优化器 | Adam，lr=2e-3（默认），weight_decay=2e-4 |
| 学习率调度 | `ReduceLROnPlateau`（验证集R²不提升时衰减0.5，patience=5） |
| 损失函数 | MSE（可选按输出量加权，本次未启用 `target_weights`） |
| 数据切分 | 按几何编号（曲线）切分 train/val/test = 8:1:1，避免同一曲线的行程点同时出现在训练和测试集（防止数据泄漏） |
| 标准化 | 输入/输出各自做 z-score 标准化，仅用训练集统计量 |
| 早停 | 按验证集R²早停，patience=40 |
| 随机种子 | 42（5个集成成员分别为 42, 43, 44, 45, 46） |

## 6. 真实训练结果（来自 `nn_out/metrics.json`）

| 数据集切分 | 平均 R² |
|---|---|
| 训练集 | 0.9968 |
| 验证集 | 0.9857 |
| 测试集 | 0.9828 |

判据：测试集平均 R² ≥ 0.98 → **✅ 达标**（`passed: true`）。

## 7. 与 Attention 版本（`train_nn_attention.py`）的关键差异

| 对比项 | MLP（本文档） | Attention（Seq2Seq） |
|---|---|---|
| 样本粒度 | 逐行独立（每个stroke点是独立样本） | 整条曲线（101个stroke点一起并行预测） |
| 硬点利用方式 | 每个点独立小MLP编码后拼接 | 编码成少量memory token供cross-attention查询 |
| 序列结构建模 | 无（stroke当作普通特征输入） | 有（self-attention让曲线内部各点互相感知） |
| 集成 | 5个PointMLP取平均 | 单模型（无集成） |
| 测试集平均R² | 0.9828 | 参考 Attention 训练结果（因架构和数据利用方式不同，不直接可比） |
