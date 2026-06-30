# 五连杆后悬架运动学分析 (RR Five-Link Suspension)

以悬架硬点为输入，对**五连杆独立后悬架**进行运动学分析：沿轮心 Z 向逐步扫描轮跳行程（回弹 → 压缩），在每个行程点求解刚体姿态，计算 camber（外倾）、toe（前束）、轮胎接地点轨迹，并基于接地点轨迹求解侧倾中心高度（RCH）随轮跳的变化。提供 **MATLAB** 与 **Python** 两套等价实现。

## 目录结构

```
Suspension_AI_Design/
├── RR_FIVE_LINK_SUSP.m     # 主脚本（MATLAB）：硬点定义 + 轮跳扫描 + 姿态求解 + 输出
├── XP_algorithm.m          # 约束函数（MATLAB）：供 fsolve 调用的五连杆长度约束方程
├── AngAgError.m.m          # 位置传感器测角误差谐波分析（FFT 数值方法）
├── RR_FIVE_LINK_SUSP.py    # 主脚本的 Python 等价实现（numpy + scipy.fsolve）
└── .venv/                  # Python 虚拟环境（不纳入版本控制）
```

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
python RR_FIVE_LINK_SUSP.py
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
