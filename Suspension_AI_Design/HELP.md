# 悬架硬点 → 运动学曲线 神经网络代理模型 · 使用帮助

本文档说明 `Suspension_AI_Design/` 下这套 AI 流程的**完整使用过程**：从环境安装、数据解析、
训练验证、结果可视化，到推理与排错。目标：给定五连杆后悬架的 33 个硬点坐标，
用神经网络推理出随轮心行程 `wc_stroke` 变化的 8 条运动学曲线
（toe、camber、caster、caster_arm、scrub_radius、tire_con_point_x/y/z）。

---

## 0. 一句话流程

```
AI_HP.xlsx ──parse──▶ nn_data/(dataset.npz+meta.json) ──train──▶ nn_out/(model.pt+metrics.json+曲线图)
                                                        └─inspect/visualize──▶ 数据格式图 / 训练结果图
```

---

## 1. 环境与依赖

脚本运行需要 Python 3 + 以下库，统一装到工作区内、且被 `.gitignore` 忽略的 `/workspace/.pylibs`，
脚本会自动把该目录加入 `sys.path`（也可用环境变量 `PPTX_LIBS` 指定其它目录）。

**必需**：`numpy`、`matplotlib`、`openpyxl`、`torch`(CPU)

```bash
# 1) 常规库（若本环境 pip 共享缓存损坏而报 No space/截断/连接中断，用全新缓存绕开）
PIP_CACHE_DIR=/workspace/.pipcache TMPDIR=/workspace/.piptmp \
  python3 -m pip install --target /workspace/.pylibs --break-system-packages \
  numpy matplotlib openpyxl

# 2) PyTorch(CPU 版)：务必用官方 CPU 索引，否则会拉到超大的 CUDA 包
PIP_CACHE_DIR=/workspace/.pipcache TMPDIR=/workspace/.piptmp \
  python3 -m pip install --target /workspace/.pylibs --break-system-packages \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple "torch==2.5.1+cpu"
```

> 运行所有脚本时都加前缀 `PYTHONPATH=/workspace/.pylibs`。
> 例：`PYTHONPATH=/workspace/.pylibs python3 train_nn.py`

---

## 2. 文件清单

| 文件 | 作用 |
|------|------|
| `suspension_config.py` | **共享配置**：预设常量 + 列定义（前束/外倾初值、轮胎静力半径、目标名等） |
| `parse_excel_to_dataset.py` | **第 1 步**：解析 `AI_HP.xlsx` → NN 数据集 `nn_data/dataset.npz` + `meta.json` |
| `train_nn.py` | **第 2 步**：PyTorch MLP 训练 / 验证 / 评估 / 保存 / 示例推理绘图 |
| `inspect_dataset.py` | 可视化 `dataset.npz` 的**数据格式**（结构、取值范围、样例曲线） |
| `visualize_results.py` | 可视化**训练结果**（指标柱状、parity 散点、曲线对照、残差分布） |
| `AI_HP.xlsx` | 原始数据（Sheet1） |
| `RR_FIVE_LINK_SUSP.m/.py` | 精确运动学求解器（可用于生成更多样本） |

产物目录：`nn_data/`（数据集）、`nn_out/`（模型与图）。

---

## 3. 数据格式

### 3.1 原始 Excel（`AI_HP.xlsx / Sheet1`）
- A 列 `Run #`：样本序号。
- **B–AH 列（33 个）**：硬点坐标，命名 `<点>_x/_y/_z`，点包括
  O1–O4（上控制臂）、S1/S2（前束拉杆）、U1–U4（下控制臂）、R0（轮心）。
- **AI 列起**：9 组随编号 `[50..150]` 变化的量，每组 101 个值。其中
  **`wc_stroke[50..150]` 是自变量**（轮心行程，实测范围约 −50~+50 mm），
  其余 8 组是随行程变化的运动学曲线。
- 每一行是一组几何（一套硬点）对应的完整曲线数据。

### 3.2 解析后的数据集（`nn_data/dataset.npz`，长格式）
NPZ 内含 3 个 numpy 数组：

| 数组 | 形状 | 含义 |
|------|------|------|
| `X` | (97869, 34) | 每行 = 33 个硬点坐标 + `wc_stroke`（第 34 列，自变量） |
| `Y` | (97869, 8)  | 8 个运动学量的值 |
| `groups` | (97869,) | 每行所属的几何编号（同一编号的 101 行按 `wc_stroke` 排列即成曲线） |

- 行数 = 几何数 × 行程点数 = 969 × 101 = 97869。
- `nn_data/meta.json` 记录：特征名、目标名、stroke 列表、样本数、`wc_stroke` 范围、
  **预设常量**、以及设计位实测统计。
- 快速查看：`PYTHONPATH=/workspace/.pylibs python3 inspect_dataset.py`
  会打印结构+样例，并生成 `nn_out/dataset_format.png` 与 `nn_out/dataset_sample_head.csv`。

---

## 4. 预设常量（`suspension_config.py`）

| 常量 | 值 | 说明 |
|------|----|------|
| `TIRE_STATIC_RADIUS_MM` | 352.0 | 模型轮胎静力半径 |
| `INIT_TOE_DEG` | 0.2 | 初始前束角（设计位 `wc_stroke≈0`） |
| `INIT_CAMBER_DEG` | -1.4 | 初始外倾角（设计位 `wc_stroke≈0`） |

用途：随数据集/模型一起保存，并在**设计位校验**中把「预设 vs 实测 vs 模型」的
toe/camber 对照打印出来；曲线图上以绿色 ★ 标出设计位预设点。
（实测已验证：设计位 toe≈0.20°、camber≈−1.44°，与预设一致。）

---

## 5. 完整流程（命令）

```bash
cd /workspace/Suspension_AI_Design
export PYTHONPATH=/workspace/.pylibs        # 之后各命令可省略前缀

# 第 1 步：Excel → 数据集
python3 parse_excel_to_dataset.py           # 生成 nn_data/dataset.npz + meta.json

# 第 2 步：训练 + 评估 + 保存 + 示例推理绘图
python3 train_nn.py                         # 生成 nn_out/model.pt + metrics.json + 示例曲线
python3 train_nn.py --time-budget 240       # 可选：调训练时间预算(秒)
python3 train_nn.py --infer-only            # 可选：仅用已存模型做示例推理

# 第 3 步（可选）：可视化数据格式
python3 inspect_dataset.py

# 第 4 步（可选）：可视化训练结果
python3 visualize_results.py
```

---

## 6. 各脚本详解

### 6.1 `parse_excel_to_dataset.py`
- 参数：`--xlsx`（默认 `./AI_HP.xlsx`）、`--out`（默认 `./nn_data`）。
- 逻辑：读 Sheet1 → 取 33 硬点为输入、按 `<量>[编号]` 解析输出 → 按「几何×行程点」展开成长格式 → 存 npz + meta。
- 会打印「预设 vs 设计位实测」的 toe/camber 对照做一致性检查。

### 6.2 `train_nn.py`（PyTorch）
- 参数：`--epochs`(默认 600)、`--time-budget`(默认 220s)、`--infer-only`、`--model {point,flat}`（默认 `point`：逐点三维编码器，把每个硬点的 x/y/z 作为一个三维单元编码；`flat`：打平 33 维标准 MLP）。
- 按量加权（改善 caster/caster_arm/scrub_radius 等薄弱量，详见 §10.5）：`--target-weights "量名:权重,..."`、
  快捷方式 `--caster-weight`（同时设 caster+caster_arm）、`--scrub-weight`（设 scrub_radius）。
  均可选，不传时等权重，与之前训练完全一致。
- 独立输出头（比单纯加权更根本，详见 §10.6）：`--hard-targets "量名,量名"` 给指定量额外配
  独立小 MLP 头，`--head-hidden` 调其隐藏层宽度（默认64）。不传时与之前完全一致（共享输出层）。
- 模型：`point`=逐点编码器(11 点各自 3→16→16 → 拼接 + wc_stroke → 主干 256→128 → 8)；`flat`=`MLP(34→256→256→128→8)`。均用 ReLU + Dropout + Adam(weight_decay)。
  > 注：两种结构在本数据上测试集平均 R² 相当(≈0.43~0.44)——瓶颈是**数据密度**而非网络结构，见 §9。
- 训练：按**整条曲线(几何)** 划分 train/val/test=775/96/98（避免同一几何的点跨集泄漏）；
  X、Y 各自标准化（训练集统计）；带**时间预算 + 早停 + 保留最优权重**的循环。
- 线程：已固定 `TORCH_THREADS=8`（本机 192 核，若用默认 96 线程会因超订极慢）。
- 输出：`nn_out/model.pt`（权重+结构+标准化参数+元数据+预设）、`metrics.json`、
  `curves_test_sample_*.png/.csv`（某测试几何 预测 vs 真实，含预设 ★）。
- 推理 API：`predict_curves(model, xs, ys, 硬点33, stroke网格) -> (n_stroke, 8)`。

### 6.3 `inspect_dataset.py`
- 参数：`--data`(默认 `nn_data`)、`--out`(默认 `nn_out`)。
- 产物：`nn_out/dataset_format.png`（①样例数据表 ②X各列范围 ③Y各量范围 ④一条几何的8曲线）、
  `nn_out/dataset_sample_head.csv`（前 20 行完整列）。

### 6.4 `visualize_results.py`
- 复用 `train_nn.py` 的模型加载与推理，读 `nn_out/model.pt`+`metrics.json`。
- 产物：`results_metrics.png`（R² 柱状+指标表）、`results_parity.png`（预测vs真实散点）、
  `results_curves.png`（4 条测试几何曲线对照）、`results_residual.png`（残差分布）。

---

## 7. 产物清单

```
nn_data/
  dataset.npz              # X/Y/groups（可由 parse 重建，已 gitignore）
  meta.json                # 元数据 + 预设
nn_out/
  model.pt                 # 训练好的 PyTorch 模型（含标准化参数、元数据）
  metrics.json             # train/val/test 各量 R²/RMSE/MAE + 设计位校验
  curves_test_sample_*.png/.csv   # 示例几何 预测 vs 真实曲线
  dataset_format.png / dataset_sample_head.csv   # 数据格式可视化
  results_metrics.png / results_parity.png / results_curves.png / results_residual.png  # 结果可视化
```

---

## 8. 指标解读

- **R²**：越接近 1 越好（解释了多少方差）；可能为负（比预测均值还差）。
- **RMSE / MAE**：与该量同量纲的误差（如 toe/camber 为度，接地点为 mm）。MAE 更贴近“典型误差”。
- **达标判据**：脚本设「测试集平均 R² ≥ 0.98」为目标（可在 `train_nn.py` 的 `TARGET_MEAN_R2` 改）。
- **设计位校验**：在 `|wc_stroke|<2.5mm` 处对比 toe/camber 的 预设/实测/模型 三者。

---

## 9. 当前结果与局限（如实）

- 训练集平均 R²≈**0.99**，但**测试集(未见过的几何)平均 R²≈0.44**，未达 0.98。
- 已用三法交叉验证同一「天花板」：线性回归≈0.30、几何 k-NN≈0.29、MLP≈0.44。
- 根因：**969 组几何分布在约 30 维硬点空间过于稀疏（维数灾难）**，对全新几何外推困难；
  `results_parity.png` 显示——绝大多数测试点贴合对角线，仅**少数“分布外”几何**预测失准并拉低 R²。
- 因此：模型对**分布内**（接近样本）的设计可用（MAE 小：toe≈0.21°、camber≈0.40°、接地点≈2mm），
  对**远离样本**的设计不可靠。仅 `tire_con_point_z`（行程主导）泛化好（R²≈0.95）。

---

## 10. 如何提升精度

1. **扩充 DOE 样本（最有效）**：用精确求解器 `RR_FIVE_LINK_SUSP.m/.py` 批量生成几千~上万组几何-曲线，覆盖更广硬点空间后重训。
2. **降维**：对 33 个硬点做敏感性分析，只保留影响大的坐标作输入，缓解维数灾难。
3. **残差/物理引导建模**：先用解析近似给基线，NN 只学残差。
4. **直接用求解器**：运动学是确定性的，新设计可直接精确计算；NN 代理用于超大规模快速评估/优化内循环。
5. **按量加权训练，改善个别薄弱量（`--target-weights` / `--caster-weight` / `--scrub-weight`）**：
   `caster`（后倾角）、`caster_arm`（后倾拖距）、`scrub_radius`（主销偏移距）这三个量的 MAE/R²
   通常明显弱于其它 5 个量，**根因是物理计算方式本身对硬点误差更敏感**，不是模型学习能力不足：
   求解器里 `caster = atan2(-d[0], d[2])`（`d` = 主销轴向量 `LUP-LLWR`，随刚体位姿变换），
   `caster_arm`/`scrub_radius` 则要再算主销轴与地面的交点 `G = LLWR + t*d`（`t` 是一次除法，
   除数是主销轴的垂直分量 `d[2]`）。33 维硬点的预测误差先传导到刚体位姿，再经过这次求交点的
   除法运算被放大，因此这三个量对同等大小的硬点误差，输出误差会明显大于 toe/camber 等直接由
   位姿算出的量。而训练时损失函数和早停判据默认对 8 个量**等权重平均**，容易被表现好的量
   （如 `tire_con_point_z`，几乎线性、R²≈0.999）掩盖这三个量的真实短板。

   `train_nn.py` 支持按量加权损失 + 早停判据，让训练过程更关注这几个薄弱量：
   ```bash
   # 简单方式：caster/caster_arm/scrub_radius 权重设为 2.5(其余量默认1.0)
   python3 train_nn.py --caster-weight 2.5 --scrub-weight 2.5

   # 或用 --target-weights 逐个指定任意量的权重
   python3 train_nn.py --target-weights "caster:2.5,caster_arm:2.5,scrub_radius:2.5"
   ```
   不传这些参数时行为与之前完全一致（等权重），`model.pt`/`metrics.json` 都会记录本次训练
   实际用的权重，方便复现。实测（`--epochs 300 --time-budget 120 --ensemble 3`，
   `--caster-weight 2.5 --scrub-weight 2.5` vs 不加权 baseline）：

   | 量 | 加权前 MAE | 加权后 MAE | 加权前 R² | 加权后 R² |
   |---|---|---|---|---|
   | caster | 0.1117 | 0.0975 (↓12.7%) | 0.9627 | 0.9640 |
   | caster_arm | 0.6008 | 0.5293 (↓11.9%) | 0.9594 | 0.9623 |
   | scrub_radius | 1.2881 | 1.0722 (↓16.8%) | 0.9814 | 0.9861 |

   其它 5 个量（toe/camber/接地点 x/y/z）基本不受影响，测试集整体平均 R² 两次训练都是 0.9808。
   权重不是越大越好——过大会让被加权的量收敛更快但让其它量的相对学习预算变少，
   建议从 2.0~3.0 开始尝试，再根据 `metrics.json` 里各量的 R²/MAE 调整。

   **权重加大到一定程度后收益递减，甚至会让其它量变差**（实测把 `--caster-weight`/
   `--scrub-weight` 从 2.5 提到 4.0 后，`caster_arm` MAE 反而从 0.5293 升到 0.5660，
   整体测试集 R² 从 0.9808 掉到 0.9798、跌破 0.98 目标）。原因是共享的最后一层
   `nn.Linear(hidden_dim, 8)` 权重矩阵，容量并没有随着权重变大而变化——继续加大权重
   只是让梯度更偏向这些量，但输出层要同时兼顾 8 个量的容量没变，加过头后反而挤占了
   其它量的优化预算。

6. **给薄弱量配独立输出头（`--hard-targets`），比单纯加大权重更根本**：
   `--hard-targets "量名,量名"` 给指定的量各自额外配一个独立的 2 层小 MLP 头
   （不与其它量共享最后一层权重），只多花一点点参数量，就能让这些量获得不被其它量
   稀释的独立非线性变换能力。推荐与 `--target-weights`/`--caster-weight`/`--scrub-weight`
   搭配使用（权重决定"训练多关注"，独立头决定"关注后有没有额外容量去响应"）：
   ```bash
   python3 train_nn.py --caster-weight 2.5 --scrub-weight 2.5 \
       --hard-targets "caster_arm,scrub_radius"
   # --head-hidden 可调独立头的隐藏层宽度(默认64)
   ```
   不传 `--hard-targets` 时行为与之前完全一致（共享单一输出层）。实测（同样
   `--epochs 300 --time-budget 120 --ensemble 3 --caster-weight 2.5 --scrub-weight 2.5`，
   仅加/不加 `--hard-targets "caster_arm,scrub_radius"`）：

   | 量 | 仅加权 MAE | 加权+独立头 MAE | R²(仅加权) | R²(加权+独立头) |
   |---|---|---|---|---|
   | caster | 0.0975 | **0.0929** | 0.9640 | 0.9659 |
   | caster_arm | 0.5293 | **0.5059** | 0.9623 | 0.9640 |
   | scrub_radius | 1.0722 | 1.0797(基本持平) | 0.9861 | 0.9852 |
   | 测试集整体平均 R² | 0.9808 | **0.9815** | — | — |

   加独立头后 `caster`/`caster_arm` 进一步改善，整体 R² 也比单纯加权更高（0.9815 vs
   0.9808），且没有像"权重加到4.0"那样出现其它量变差的副作用——独立头是当前
   （数据量固定的前提下）比继续调大权重更值得优先尝试的手段。

---

## 11. 常见问题 / 排错

| 现象 | 原因 / 解决 |
|------|------------|
| `ModuleNotFoundError: torch/numpy` | 未加 `PYTHONPATH=/workspace/.pylibs`，或依赖被工作区重置清掉 → 重装（见 §1） |
| pip 报 `No space left` / JSON 截断 / `Connection broken` | 共享缓存 `/dep-cache/pip` 损坏 → 用全新缓存：`PIP_CACHE_DIR=/workspace/.pipcache TMPDIR=/workspace/.piptmp` |
| `pip install torch` 拉到超大包/超时 | 必须用 `--index-url https://download.pytorch.org/whl/cpu` 装 CPU 版 |
| 训练极慢、迟迟不出 epoch | 多核机默认线程超订 → 已设 `TORCH_THREADS=8`；可按需 `export TORCH_THREADS=4` |
| `ndarray has no attribute 'ptp'` | numpy 2.x 移除该方法 → 用 `np.ptp(x)`（脚本已修） |
| 图中中文变方框 | matplotlib 缺中文字体 → 脚本已改用英文标签 |
| `nn_data/` 或 `.pylibs` 不见了 | 属可重建/被 gitignore 的目录，被工作区重置清掉是正常的 → 重跑 parse / 重装依赖 |

---

## 12. 推理用法示例（Python）

```python
import sys; sys.path.insert(0, "/workspace/.pylibs")
import numpy as np, train_nn as tn

model, xs, ys, meta = tn.load_bundle()          # 加载 nn_out/model.pt
hardpoints = np.zeros(33)                         # 换成你的 33 个硬点坐标（顺序见 meta['feat_names'][:33]）
stroke = np.linspace(-50, 50, 101)                # 轮心行程扫掠
curves = tn.predict_curves(model, xs, ys, hardpoints, stroke)  # (101, 8)
# curves 的 8 列依次为 meta['target_names']
for j, name in enumerate(meta["target_names"]):
    print(name, curves[:, j][:3], "...")
```

---

## 13. 数据增广：用求解器批量造样本（把测试 R² 拉到 0.99）

前面已证明测试偏差是**数据密度**问题。用精确求解器批量生成更多样本后，同样的网络测试 R² 从 ~0.44 跃升到 **0.99**。

```bash
# 1) 用求解器 DOE 生成 N 组样本（±mm 扰动），存到 nn_data_gen/
PYTHONPATH=/workspace/.pylibs python3 generate_dataset.py --n 5000 --pert 5 --out nn_data_gen
# 2) 在生成数据上训练（大数据集用大 batch）
PYTHONPATH=/workspace/.pylibs python3 train_nn.py --data nn_data_gen --model point --ensemble 2 --batch 2048 --time-budget 200
# 3) 可视化（指向同一数据集）
PYTHONPATH=/workspace/.pylibs python3 visualize_results.py --data nn_data_gen
```

**实测**：5000 组 → 测试集平均 R²=**0.99**（全 8 个量均 ≥0.98：toe0.991/camber0.984/caster0.986/caster_arm0.989/scrub_radius0.985/接地点0.985~0.998）。

⚠️ 注意：
- 现已扩展为**全 8 个量**：caster/caster_arm/scrub_radius 由主销轴(LUP-LLWR)与地面(接地点z)交点相对接地点按标准定义计算(符号约定可能与原Excel不同)。
- 硬点参数化仍用求解器的 A1..B5,C(与 Excel 的 O/U/S/R 命名不同)，故 nn_data_gen 是**自洽数据集**；如需严格匹配 Excel 硬点命名，需再做 O/U/S/R↔A/B/C 的点映射标定。
- 切回原 Excel 数据模型：`python3 train_nn.py`（默认读 nn_data，8 输出）。

---

## 14. 达到"图片以上水平"的最终配置与结果

在生成数据(≥5000 组)上，用以下配置把 8 项曲线的 **MAE/Median/P90/MaxError/D2 全部做到优于参考表**：
- 逐点编码 `emb=32`、主干 `trunk=(512,256,128)`、**dropout=0**（去掉输出噪声，显著降 MAE）；
- 优化器 Adam + **ReduceLROnPlateau**（验证 R² 停滞则 lr×0.5，精调收敛）；
- `--batch 8192`（大 batch 换更多轮次）、单模型即可。
- 训练时用 `torchinfo.summary` 打印结构（Total params ≈ 359,528）。

命令：
```bash
PYTHONPATH=/workspace/.pylibs python3 train_nn.py --data nn_data_gen --model point \
    --ensemble 1 --batch 8192 --time-budget 250 --epochs 5000
```

**结果（测试集平均 R²=0.9994）——全 8 项 MAE 均优于参考表：**

| 曲线 | 本模型 MAE | 参考表 MAE |
|------|-----------|-----------|
| 前束角 Toe (°) | 0.035 | 0.097 |
| 外倾角 Camber (°) | 0.030 | 0.066 |
| 后倾角 Caster (°) | 0.026 | 0.266 |
| 后倾拖距 Caster Arm (mm) | 0.163 | 1.371 |
| 主销偏移距 Scrub Radius (mm) | 0.289 | 2.793 |
| 接地点 X (mm) | 0.099 | 0.416 |
| 接地点 Y (mm) | 0.219 | 0.345 |
| 接地点 Z (mm) | 0.335 | 0.503 |

Max Error、Median、平滑度 D2 亦全部优于参考表。

---

## 15. 推理网页（仪表盘 UI）

把训练好的代理模型包成一个**本地推理网页**，仿参考仪表盘：左侧模型设置滑块、Setup 预设下拉、
设计位目标值表、3×3 运动学曲线图。纯标准库后端 + 无 CDN 依赖的前端（SVG 自绘曲线）。

### 15.1 文件

| 文件 | 作用 |
|------|------|
| `infer.py` | 自包含推理模块（模型结构 + `load_bundle` + `predict_curves` + `compute`），**只依赖 numpy+torch**，不引 matplotlib/scipy。 |
| `web_app.py` | `http.server` 后端：`GET /`（页面）、`GET /api/setups`（预设几何）、`POST /api/infer`（推理）。 |
| `web/index.html` | 前端仪表盘（侧栏滑块 / Setup 下拉 / Jounce+Steering 目标值表 / 3×3 曲线网格）。 |

### 15.2 启动

```bash
cd /workspace/Suspension_AI_Design
# 依赖若被工作区重置清掉，先按 §1 重装 numpy + torch(CPU) 到 /workspace/.pylibs
PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app.py            # 默认 http://127.0.0.1:5173
PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app.py --port 8080
PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app.py --self-test  # 线程内启动+请求自测后退出
```

浏览器打开 `http://127.0.0.1:5173`。

### 15.3 页面说明

- **模型设置（侧栏）**：
  - `轮跳扫描范围 [±mm]`、`扫描分辨率 [点数]`：直接决定推理的行程栅格 `linspace(-range, range, N)`（范围夹在模型有效值域 ±50mm 内）。
  - `假设的轮胎负载半径 [mm]`：模型在**固定轮胎静力半径（预设 352mm）**下训练，此滑块仅作工况标注、**不改变**神经网络输出（点击 “?” 有说明）。
- **Setup 下拉**：`Setup 1` 为求解器标称几何（分布中心，最可靠）；其余为 ±4mm 可复现小扰动的分布内几何。
- **设计位一阶目标值（Jounce）**：设计位（WCH=0）处各量沿行程的梯度（每米）——Bump steer、Bump camber、Caster/Caster-trail/Scrub-radius 变化率。
- **设计位转向目标值（Steering）**：WCH=0 处的 Toe / Camber / Caster / Caster trail / Scrub radius。
- **3×3 曲线**：8 个模型输出 + 派生的前束变化率曲线，横轴为轮心行程 WCH [mm]。

> ⚠️ 与参考图差异（如实）：参考图含 Anti-squat / Anti-lift / Roll center height 等需力线/侧倾中心几何才能得到的量，**当前代理模型不产出**这些，故未虚构；本页展示的是模型真实输出的 8 条曲线及其设计位值/梯度。
> 安全：后端默认仅监听 `127.0.0.1`、无鉴权，仅供本地/单机演示，勿在无访问控制下绑定 `0.0.0.0` 暴露公网。
