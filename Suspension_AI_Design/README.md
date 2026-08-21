# 五连杆后悬架运动学分析 (RR Five-Link Suspension)

以悬架硬点为输入，对**五连杆独立后悬架**进行运动学分析：沿轮心 Z 向逐步扫描轮跳行程（回弹 → 压缩），在每个行程点求解刚体姿态，计算 camber（外倾）、toe（前束）、轮胎接地点轨迹，并基于接地点轨迹求解侧倾中心高度（RCH）随轮跳的变化。提供 **MATLAB** 与 **Python** 两套等价实现。

## 目录结构

```
Suspension_AI_Design/
├── RR_FIVE_LINK_SUSP.m     # 主脚本（MATLAB）：硬点定义 + 轮跳扫描 + 姿态求解 + 输出
├── XP_algorithm.m          # 约束函数（MATLAB）：供 fsolve 调用的五连杆长度约束方程
├── AngAgError.m.m          # 位置传感器测角误差谐波分析（FFT 数值方法）
├── 15-16-02-0007.pdf / 15-16-02-0007_detailed_summary.md  # 参考资料
├── HELP.md / README.md     # 说明文档
│
├── 数据处理/                # Excel/求解器数据 → NN 训练数据集
│   ├── AI_HP.xlsx              # 原始硬点+曲线数据（969 组几何）
│   ├── parse_excel_to_dataset.py  # 第1步：Excel → nn_data/（长格式数据集）
│   ├── generate_dataset.py        # 用求解器(RR_FIVE_LINK_SUSP)批量生成 DOE 数据 → nn_data_gen/
│   ├── merge_datasets.py          # 合并 nn_data + nn_data_gen → nn_data_merged/
│   ├── analyze_data_anomalies.py  # 分析 AI_HP.xlsx 异常行 → ../训练推理/nn_out/anomaly_report.*
│   ├── inspect_dataset.py         # 可视化数据集格式 → ../训练推理/nn_out/dataset_format.png
│   ├── nn_data/                   # parse_excel_to_dataset.py 的输出
│   ├── nn_data_gen/                # generate_dataset.py 的输出
│   ├── nn_data_gen_s/               # generate_dataset.py 的另一批输出
│   └── nn_data_merged/             # merge_datasets.py 的输出
│
├── 训练推理/                # 模型训练、推理、评估
│   ├── suspension_config.py    # 共享配置：预设常量（轮胎静力半径/初始前束/初始外倾）
│   ├── RR_FIVE_LINK_SUSP.py    # 精确求解器（Python 版），被 数据处理/generate_dataset.py 复用
│   ├── train_nn.py             # 第2步：PyTorch MLP 训练/验证/评估/保存 → nn_out/
│   ├── train_nn2.py            # 备选架构：Attention Seq2Seq 训练 → nn_out2/
│   ├── infer.py                 # 自包含推理模块（供 ../网页部署/web_app.py 调用）
│   ├── eval_merged_by_source.py # 在合并数据集上按来源(Excel/求解器)分别评估
│   ├── visualize_results.py     # 可视化训练结果（metrics/parity/curves/residual 图）
│   ├── export_test_samples.py   # 导出测试集样本预测vs真值（供 ../网页部署/web_app2.py 使用）
│   ├── nn_out/                  # train_nn.py 的输出（model.pt/metrics.json/曲线图）
│   └── nn_out2/                 # train_nn2.py 的输出
│
└── 网页部署/                # 推理/对照演示网页（标准库 http.server，无需 Flask）
    ├── web_app.py               # 仪表盘 + 手动/批量硬点推理网页后端
    ├── web_app2.py              # 测试集「预测 vs 真值」对照网页后端
    └── web/                     # 前端页面（index.html/predict.html/compare.html）
```

⚠️ 跨目录依赖说明：各脚本仍用 `_HERE = os.path.dirname(os.path.abspath(__file__))` 定位自身数据目录，
移动到子文件夹后已同步修正为通过 `os.path.dirname(_HERE)` 拼接兄弟文件夹路径（如 `../训练推理/`、`../数据处理/`），
无需额外设置 `PYTHONPATH` 即可跨文件夹 import（脚本内部已自动把依赖的兄弟文件夹插入 `sys.path`）。


## 文件说明

| 文件 | 作用 |
| --- | --- |
| [RR_FIVE_LINK_SUSP.m](RR_FIVE_LINK_SUSP.m) | MATLAB 主脚本。定义悬架硬点（上/下控制臂、转向拉杆、轮心、稳定杆等），按轮跳步长扫描行程，每步通过 `fsolve` 求解姿态参数满足几何约束，进而计算各外点坐标、外倾/前束、接地点轨迹与侧倾中心高度。 |
| [XP_algorithm.m](XP_algorithm.m) | 供 `fsolve` 调用的约束方程函数。由姿态角构造旋转矩阵，将初始硬点映射到当前工况，返回五根连杆“当前长度 − 标称长度”的残差，收敛时趋近于 0。 |
| [AngAgError.m.m](AngAgError.m.m) | 独立工具脚本。对位置传感器（sin/cos 增益误差、偏置、正交误差）的测角误差做 FFT 谐波分析，提取前若干阶谐波系数。 |
| [RR_FIVE_LINK_SUSP.py](RR_FIVE_LINK_SUSP.py) | 主脚本的 Python 等价实现。用 `dataclass` 组织硬点，`scipy.optimize.fsolve` 求解姿态，与 MATLAB 版保持一致的旋转矩阵与分支选择逻辑。 |

## 环境依赖

### MATLAB
- MATLAB（需 Optimization Toolbox，提供 `fsolve`）

### Python
```bash
pip install numpy scipy
```

## 使用方法

### MATLAB
在 MATLAB 中打开本目录，运行主脚本：

```matlab
RR_FIVE_LINK_SUSP
```

`XP_algorithm.m` 作为约束函数被主脚本通过 `fsolve` 自动调用，无需单独运行。
测角误差分析可单独运行：

```matlab
AngAgError
```

### Python
```bash
python 训练推理/RR_FIVE_LINK_SUSP.py
```

## 关键参数（主脚本）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `BOUstroke` | 100 | 轮胎上跳（压缩）行程 mm |
| `REBstroke` | -100 | 轮胎下跳（回弹）行程 mm |
| `tire_radius` | 352 | 轮胎静力半径 mm |
| `step` | 5 | 轮跳扫描步长 mm |

硬点坐标（`A1/B1 … LUP/LLWR/SCLP`）在脚本顶部以 `[x, y, z]` 形式定义，对应上/下控制臂内外点、转向拉杆、轮心、稳定杆连接点等。

## 版本控制说明

仓库已配置 [.gitignore](.gitignore)，默认忽略：

- Python 虚拟环境 `.venv/` 与缓存 `__pycache__/`
- 编辑器配置 `.vscode/`（常含本地绝对路径）
- MATLAB 临时/输出文件（`*.asv`、`*.mat`、`slprj/` 等）
- 图像与数据导出（`*.png`、`*.fig`、`*.csv` 等，可由脚本重新生成）

源代码脚本（`*.m`、`*.py`）均保留上传。

---

## 🧠 AI 神经网络代理模型（硬点 → 运动学曲线，PyTorch）

用 **PyTorch MLP** 学习 `AI_HP.xlsx` 中「33 个硬点坐标 → 8 条运动学曲线」的映射，
训练后对给定硬点扫掠 `wc_stroke` 直接推理出曲线。流程拆成**两个脚本** + 一个共享配置：

| 文件 | 作用 |
|------|------|
| `suspension_config.py` | 共享配置：**预设常量**（轮胎静力半径 352mm、初始前束 0.2°、初始外倾 −1.4°）与列定义 |
| `parse_excel_to_dataset.py` | **第 1 步**：Excel → NN 可直接用的数据集（`nn_data/dataset.npz` + `meta.json`） |
| `train_nn.py` | **第 2 步**：PyTorch 训练/验证/评估/保存 + 推理绘图 |

### 数据与建模
- 输入 = B–AH 列 33 个硬点坐标（O1–O4、S1/S2、U1–U4、R0 的 x/y/z）；
  输出 = 随编号 [50..150] 变化的 9 组量，其中 **`wc_stroke` 是自变量**（轮心行程 −50~+50mm），
  其余 8 个是曲线：`toe, camber, caster, caster_arm, scrub_radius, tire_con_point_x/y/z`。
- **长格式**：每「几何 × 行程点」展开为一条样本 → 输入 `[33 硬点 + wc_stroke]`(34 维)，
  输出 8 个量（969×101≈9.79 万行）。推理时扫掠 `wc_stroke` 即得曲线。
- 划分：按**整条曲线(几何)** 切 train/val/test=775/96/98，避免跨集泄漏。
- 预处理：X、Y 各自标准化（numpy，训练集统计）。
- 模型：PyTorch `MLP(34→256→256→128→8)`，ReLU + Dropout + Adam(weight_decay)；
  训练带**时间预算 + 早停 + 保留最优权重**（受限环境稳定完成；已修复 192 核下线程超订导致的极慢问题，默认 `TORCH_THREADS=8`）。
- 指标：每量 **R²/RMSE/MAE**，达标判据「测试集平均 R²≥0.98」。

### 预设常量的作用
预设写入 `suspension_config.py` 并随 `meta.json`/`model.pt` 保存，并用于**设计位校验**：
在 `|wc_stroke|<2.5mm`（设计位）处对比 toe/camber 与预设——实测 **toe≈0.20°/camber≈−1.44°**、
模型 **toe≈0.19°/camber≈−1.40°**，与预设（0.2°/−1.4°）高度一致，佐证列映射与模型在设计位的正确性；
曲线图上也用绿色 ★ 标出设计位预设点。

### 运行（两步）
```bash
cd Suspension_AI_Design
# 依赖装在 /workspace/.pylibs（见根 requirements.txt），脚本会自动加入 sys.path
PYTHONPATH=/workspace/.pylibs python3 数据处理/parse_excel_to_dataset.py   # 第1步：生成 数据处理/nn_data/
PYTHONPATH=/workspace/.pylibs python3 训练推理/train_nn.py                 # 第2步：训练+评估+推理绘图
PYTHONPATH=/workspace/.pylibs python3 训练推理/train_nn.py --infer-only    # 用已存 model.pt 推理
```
产物：`训练推理/nn_out/model.pt`（权重+结构+标准化参数+元数据+预设）、`metrics.json`、
`curves_test_sample_*.png/csv`。推理 API：`predict_curves(model, xs, ys, 硬点33, stroke网格) -> (n,8)`。

### 结果与重要结论（如实）
- 训练集平均 R²≈**0.99**，**测试集(未见几何)平均 R²≈0.44**，未达 0.98。
- 这是**数据密度限制、非框架/代码问题**（sklearn 与 PyTorch 得到同一天花板）：
  | 方法 | 测试集平均 R² |
  |------|---------------|
  | 线性回归 | ~0.30 |
  | 几何 k-NN(k=10) | ~0.29 |
  | MLP(sklearn/PyTorch) | **~0.44** |
  根因：**969 组几何分布在约 30 维硬点空间过于稀疏**（维数灾难），对全新几何外推困难；
  仅 `tire_con_point_z`（行程主导）泛化好（R²≈0.95）。
- 绝对误差不大（MAE：toe≈0.21°、camber≈0.40°、caster≈0.65°、接地点≈2mm），
  R² 主要被少数分布外几何拖低。**对分布内设计可用，远离样本不可靠**。

### 如何达到更高精度
1. **增加 DOE 样本**：用精确求解器 `RR_FIVE_LINK_SUSP.m/.py` 批量生成几千~上万组，泛化会显著改善（最有效）。
2. **降维**：对 33 硬点做敏感性分析，只留影响大的坐标作输入。
3. **残差/物理引导**：先解析基线，NN 只学残差。
4. **直接用求解器**：运动学确定性，新设计可直接精确求解；NN 代理用于超大规模快速评估。
