# 李导数 · CLF · CBF · HOCBF · QP — 控制理论进阶笔记

> 从数学基础到安全关键控制的完整递进，每个概念配合手算案例，附 PMSM 电机控制实例。

---

## 目录

1. [数学基础：李导数与控制仿射系统](#1)
2. [控制李雅普诺夫函数（CLF）](#2)
3. [控制屏障函数（CBF）](#3)
4. [CLF-CBF QP 统一框架](#4)
5. [高阶 CBF（HOCBF）](#5)
6. [PMSM 控制实例](#6)
7. [脉络总结](#7)

---

<h2 id="1">1. 数学基础</h2>

### 1.1 李导数 (Lie Derivative)

李导数度量一个标量函数沿向量场方向的变化率。在非线性控制中，它是把系统动力学"代入"到标量函数导数的核心工具。

> **定义：** 设 $h: \mathbb{R}^n \to \mathbb{R}$ 是光滑标量函数，$f: \mathbb{R}^n \to \mathbb{R}^n$ 是光滑向量场。则 $h$ 沿 $f$ 的李导数为：
>
> $$L_f h(x) = \frac{\partial h}{\partial x} f(x) = \nabla h(x) \cdot f(x)$$

**高阶李导数（递归定义）：**

$$L_f^0 h(x) = h(x), \qquad L_f^k h(x) = \frac{\partial}{\partial x}\left(L_f^{k-1} h(x)\right) f(x), \quad k \ge 1$$

**混合李导数**（先沿 $f$ 再沿 $g$）：

$$L_g L_f h(x) = \frac{\partial}{\partial x}\left(L_f h(x)\right) g(x)$$

> ⚠️ 一般情况 $L_g L_f h \neq L_f L_g h$，不可交换。

#### 案例 1：二维系统的李导数手算

> 考虑一个简单的 2D 非线性系统：
>
> $$f(x) = \begin{bmatrix} x_1 + x_2 \\ -x_1 + x_2 \end{bmatrix}, \qquad h(x) = x_1^2 + x_2^2$$
>
> **Step 1** — 计算 $\nabla h$：
>
> $$\nabla h(x) = \begin{bmatrix} \frac{\partial h}{\partial x_1} & \frac{\partial h}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 2x_1 & 2x_2 \end{bmatrix}$$
>
> **Step 2** — 计算 $L_f h$（标量乘向量）：
>
> $$L_f h(x) = \nabla h \cdot f = 2x_1(x_1 + x_2) + 2x_2(-x_1 + x_2)$$
>
> $$= 2x_1^2 + 2x_1 x_2 - 2x_1 x_2 + 2x_2^2 = 2(x_1^2 + x_2^2) = 2h(x)$$
>
> 这个结果说明：$h$ 沿 $f$ 方向的变化率恰好是 $2h$ 本身——如果 $h$ 代表"能量"，那能量在沿 $f$ 流动时指数增长。

#### 案例 2：混合李导数手算

> 在上面的系统上增加一个控制方向 $g(x) = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$：
>
> **Step 1** — 先算 $L_f h = 2(x_1^2 + x_2^2)$（已知）
>
> **Step 2** — 再算 $L_g L_f h$：
>
> $$\frac{\partial}{\partial x}(L_f h) = \begin{bmatrix} \frac{\partial}{\partial x_1}(2x_1^2 + 2x_2^2) & \frac{\partial}{\partial x_2}(2x_1^2 + 2x_2^2) \end{bmatrix} = \begin{bmatrix} 4x_1 & 4x_2 \end{bmatrix}$$
>
> $$L_g L_f h(x) = \begin{bmatrix} 4x_1 & 4x_2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 4x_1$$
>
> **Step 3** — 反过来算 $L_f L_g h$ 做对比：
>
> $$L_g h = \begin{bmatrix} 2x_1 & 2x_2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 2x_1$$
>
> $$L_f L_g h = \frac{\partial}{\partial x}(2x_1) \cdot f = \begin{bmatrix} 2 & 0 \end{bmatrix} \begin{bmatrix} x_1 + x_2 \\ -x_1 + x_2 \end{bmatrix} = 2(x_1 + x_2)$$
>
> 显然 $L_g L_f h = 4x_1 \neq 2(x_1 + x_2) = L_f L_g h$，验证了不可交换性。

> **直观理解：** 把 $f(x)$ 想成"水流方向"，$h(x)$ 是"水温"。$L_f h$ 就是沿水流方向走时水温变化的快慢——本质上就是方向导数。高阶李导数 $L_f^2 h$ 就是"水温变化率的变化率"。

---

### 1.2 控制仿射系统 (Control-Affine System)

> **定义：** 一个非线性系统能写成 $\dot{x} = f(x) + \sum_{i=1}^{m} g_i(x) u_i = f(x) + G(x)u$，就称为**控制仿射系统**。
>
> - $x \in \mathbb{R}^n$ — 状态向量
> - $u \in \mathbb{R}^m$ — 控制输入
> - $f(x)$ — **漂移向量场**（drift，不控也会有的部分）
> - $G(x) = [g_1(x) \cdots g_m(x)] \in \mathbb{R}^{n \times m}$ — 控制矩阵

"仿射"的含义：系统关于控制输入 $u$ 是**线性的**（有常数偏移项 $f(x)$）。

**为什么要写成仿射形式？** CLF/CBF 要求控制输入以线性方式进入系统，这样约束才能转化为关于 $u$ 的**线性不等式**——即 QP 可解。

#### 案例 3：倒立摆写成控制仿射形式

> 倒立摆动力学：$\ddot{\theta} = \frac{g}{l}\sin\theta + \frac{1}{ml^2} u$
>
> **Step 1** — 定义状态 $x = [\theta,\; \dot{\theta}]^\top$，得 $\dot{x}_1 = x_2$，$\dot{x}_2 = \frac{g}{l}\sin x_1 + \frac{1}{ml^2} u$
>
> **Step 2** — 写成仿射形式：
>
> $$f(x) = \begin{bmatrix} x_2 \\ \frac{g}{l}\sin x_1 \end{bmatrix}, \qquad G(x) = \begin{bmatrix} 0 \\ \frac{1}{ml^2} \end{bmatrix}$$
>
> 这是单输入 ($m=1$) 仿射系统。注意非线性 $\sin x_1$ 被包在 $f(x)$ 里。

#### 案例 4：PMSM 写成控制仿射形式

> 状态 $x = [i_d, i_q, \omega_m]^\top$，输入 $u = [u_d, u_q]^\top$：
>
> $$f(x) = \begin{bmatrix} -\frac{R_s}{L_s}i_d + p\omega_m i_q \\ -\frac{R_s}{L_s}i_q - p\omega_m i_d - \frac{\psi_f}{L_s}p\omega_m \\ \frac{1}{J}(K_t i_q - T_L - B\omega_m) \end{bmatrix}, \quad G(x) = \begin{bmatrix} \frac{1}{L_s} & 0 \\ 0 & \frac{1}{L_s} \\ 0 & 0 \end{bmatrix}$$
>
> 这是双输入 ($m=2$) 仿射系统。关键观察：$G$ 的第三行全为 0——电压不能**直接**改变转速（$r=2$）。

> **关键性质：** 对控制仿射系统，标量函数 $h(x)$ 的全导数为：
>
> $$\dot{h}(x) = L_f h(x) + \underbrace{L_G h(x)}_{\text{行向量}} \cdot u$$
>
> $\dot{h}$ 关于 $u$ 是**仿射**的（线性 + 常数项），这是后续所有 QP 约束为线性的保证。

---

<h2 id="2">2. 控制李雅普诺夫函数（CLF）—— 稳定性</h2>

### 2.1 从经典李雅普诺夫到 CLF

经典：找 $V(x) \succ 0$，使得 $\dot{V}(x) \prec 0$。但对控制系统，我们可以**选** $u$ 来强制 $\dot{V} \prec 0$。

> **定义（CLF）：** 对 $\dot{x} = f(x) + G(x)u$，连续可微、正定、径向无界的 $V$ 是 CLF 当：
>
> $$\inf_{u \in \mathbb{R}^m} \left[ L_f V(x) + L_G V(x) \cdot u \right] < 0, \quad \forall x \neq 0$$
>
> 即：**存在**某个 $u$ 能让 $V$ 下降。

#### 案例 5：一维标量系统的 CLF 手算

> 考虑最简单的标量系统 $\dot{x} = -x + u$（注意这里 $f(x) = -x$, $g(x) = 1$）：
>
> 选候选 $V(x) = \frac{1}{2}x^2$，计算：
>
> $$L_f V = \frac{\partial V}{\partial x} f(x) = x \cdot (-x) = -x^2$$
>
> $$L_G V = \frac{\partial V}{\partial x} G(x) = x \cdot 1 = x$$
>
> CLF 条件：$\inf_u [-x^2 + x \cdot u] < 0$ 对所有 $x \neq 0$ 成立。
>
> - 若选 $u = 0$：$\dot{V} = -x^2 < 0$ ✅（漂移本身就稳定）
> - 若选 $u = -kx$：$\dot{V} = -x^2 - kx^2 = -(1+k)x^2 < 0$ ✅（控制加速收敛）
>
> 所以 $V = \frac{1}{2}x^2$ 是 CLF。即使 $f(x)$ 让系统发散，只要 $G(x) \neq 0$ 能给一个"拉回来"的方向，就存在 CLF。

### 2.2 CLF-QP

将 CLF 条件松弛为不等式约束，并加入控制代价最小的优化目标：

> **CLF-QP 问题：**
>
> $$\begin{aligned} u^* = \argmin_{u,\; \delta} \quad & \frac{1}{2} u^\top H u + p \cdot \delta^2 \\ \text{s.t.} \quad & L_f V(x) + L_G V(x) \cdot u \le -\gamma V(x) + \delta \\ & u \in \mathcal{U} \end{aligned}$$
>
> - $\gamma > 0$ — 收敛速率
> - $\delta \ge 0$ — **松弛变量**（slack），让 CLF 约束在冲突时可被"违反"
> - $p > 0$ — 惩罚系数，越大越不愿违反 CLF

> **为什么需要 $\delta$？** 当 CLF 和 CBF 冲突时（急刹车稳住速度 vs 急刹车撞墙），CLF 是**软约束**——可以暂时牺牲稳定性来保安全。

#### 案例 6：CLF-QP 的数值手算

> 设系统 $\dot{x} = 2x + u$（漂移发散！），$V = \frac{1}{2}x^2$，$\gamma = 1$，$p = 10$，$H = 1$，当前 $x = 3$。
>
> **Step 1** — 算 $L_f V$ 和 $L_G V$：
>
> $$L_f V = x \cdot 2x = 2x^2 = 18, \qquad L_G V = x \cdot 1 = 3$$
>
> **Step 2** — 写出 QP（注意这里 CLF 约束是 $L_f V + L_G V \cdot u \le -\gamma V + \delta$）：
>
> $$\min_{u, \delta} \; \frac{1}{2}u^2 + 10\delta^2 \quad \text{s.t.} \quad 18 + 3u \le -4.5 + \delta$$
>
> 整理：$3u - \delta \le -22.5$
>
> 若无 $\delta$（$\delta$ 固定为 0）：$3u \le -22.5 \Rightarrow u \le -7.5$，最优解是 $u = -7.5$。
>
> 有 $\delta$：可以选更大的 $u$（消耗更少控制能量），但牺牲 $\delta$（付出惩罚）。这就是软约束的本质：**在控制代价和稳定速度之间做权衡**。

---

<h2 id="3">3. 控制屏障函数（CBF）—— 安全性</h2>

### 3.1 安全集

> **安全集：** $h(x) \ge 0$ 定义了一个"安全区域" $\mathcal{C}$。
>
> - $\mathcal{C} = \{x \mid h(x) \ge 0\}$ — 安全集
> - $\partial\mathcal{C} = \{x \mid h(x) = 0\}$ — 安全边界
> - **前向不变**：从 $\mathcal{C}$ 内部出发，永远不离开 $\mathcal{C}$

> **定义（CBF）：** 若 $h(x)$ 的相对度为 1（$L_G h(x) \neq 0$），则 $h$ 是 CBF 当存在 $\mathcal{K}_{\infty}^e$ 类函数 $\alpha$ 使得：
>
> $$\sup_{u} \left[ L_f h(x) + L_G h(x) \cdot u \right] \ge -\alpha(h(x))$$
>
> 即存在 $u$ 能满足 $\dot{h}(x) \ge -\alpha(h(x))$。

安全控制器集合：

$$K_{\text{cbf}}(x) = \left\{ u \mid L_f h(x) + L_G h(x) \cdot u \ge -\alpha(h(x)) \right\}$$

约束含义：$\dot{h} \ge -\alpha(h)$

- 当 $h$ 很大（远离边界）→ 约束松，$\dot{h}$ 可以很负
- 当 $h \to 0$（靠近边界）→ $\alpha(h) \to 0$，必须 $\dot{h} \ge 0$（不能再靠近边界）

#### 案例 7：一维速度限制 CBF 手算

> 系统：$\dot{x} = u$（一阶积分器，$f(x) = 0$, $G(x) = 1$），安全约束 $x \le 10$。
>
> $h(x) = 10 - x$
>
> $$L_f h = \frac{\partial h}{\partial x} \cdot f = (-1) \cdot 0 = 0$$
> $$L_G h = \frac{\partial h}{\partial x} \cdot G = (-1) \cdot 1 = -1 \neq 0 \quad \Rightarrow \quad r=1$$
>
> CBF 约束（取 $\alpha(s) = \gamma s$）：
>
> $$0 + (-1) \cdot u \ge -\gamma(10 - x) \quad \Rightarrow \quad u \le \gamma(10 - x)$$
>
> **物理含义：** 当 $x=9$（离边界 1 单位）→ $u \le \gamma$；当 $x=9.9$（离边界 0.1）→ $u \le 0.1\gamma$。越靠近上限，允许的速度越小——这正是"安全减速"的数学表达。

#### 案例 8：$\alpha$ 函数效果的数值对比

> 系统同上，$x=8$（$h=2$）：
>
> | $\alpha(s)$ | CBF 约束 | $u$ 上限 | 效果 |
> |---|---|---|---|
> | $\gamma s$ ($\gamma=2$) | $u \le 2 \times 2 = 4$ | 4 | 线性减速 |
> | $\gamma s^2$ ($\gamma=2$) | $u \le 2 \times 4 = 8$ | 8 | 离边界远时约束更松 |
> | $\gamma s$ ($\gamma=5$) | $u \le 5 \times 2 = 10$ | 10 | 更大的 $\gamma$ → 约束更激进 |

> ⚠️ **核心前提：相对度必须为 1！** 如果 $L_G h = 0$（$u$ 不能直接影响 $\dot{h}$），标准 CBF 失效，需要**HOCBF**。

---

<h2 id="4">4. CLF-CBF QP 统一框架</h2>

> **CLF-CBF QP：**
>
> $$\begin{aligned} u^* = \argmin_{u,\; \delta} \quad & \frac{1}{2} (u - u_{\text{nom}})^\top H (u - u_{\text{nom}}) + p \cdot \delta^2 \\ \text{s.t.} \quad & \underbrace{L_f V + L_G V \cdot u \le -\gamma_v V + \delta}_{\text{CLF（软）}} \\ & \underbrace{L_f h + L_G h \cdot u \ge -\gamma_h h}_{\text{CBF（硬）}} \\ & u_{\min} \le u \le u_{\max} \end{aligned}$$

| 组件 | 类型 | 作用 |
|------|------|------|
| CLF 约束 | 软约束（有 $\delta$） | 保证（可松弛的）稳定性 |
| CBF 约束 | 硬约束 | 保证安全性，不可违反 |
| 输入约束 | 硬约束 | 物理极限（电压/电流上限） |
| $u_{\text{nom}}$ | 参考 | 标称控制器输出（如 PID 给出的期望值） |

> **核心哲学：** CLF-CBF QP 是一个**安全过滤器**——接收标称控制 $u_{\text{nom}}$，在满足安全和输入约束下，输出最接近 $u_{\text{nom}}$ 的指令。凸 QP 可在微秒级求解。

#### 案例 9：冲突场景 —— 为什么 CLF 必须软、CBF 必须硬

> 自动驾驶场景：
> - 标称 PID 要加速到 80 km/h（CLF 驱动）
> - 前方 10m 有障碍物（CBF 约束 $h = d - d_{\min} \ge 0$）
>
> 若 CLF 也是硬约束：必须加速 → 必须降速 → **不可行（QP 无解）→ 系统崩溃**
>
> 软 CLF：允许暂时偏离目标速度（$\delta > 0$），先满足 CBF 刹停 → 障碍物清除后恢复加速。
>
> **这就是 $\delta$ 的存在意义：安全优先，稳定让步。**

---

<h2 id="5">5. 高阶 CBF（HOCBF）—— 解决相对度 &gt; 1</h2>

### 5.1 相对度详解

> **相对度 (Relative Degree)：** 控制输入 $u$ 第一次出现在 $\dot{h}$ 的哪一阶导数中。
>
> $$L_G h = 0,\; L_G L_f h = 0,\; \dots,\; L_G L_f^{r-1} h \neq 0$$

#### 案例 10：三种相对度的直观例子

> 系统 $\ddot{x} = u$（$x$ 是位置，$u$ 是力/加速度）：
>
> | 约束 | $h(x)$ | $\dot{h}$ | $\ddot{h}$ | $r$ | 工具 |
> |---|---|---|---|---|---|
> | 输入限幅 | $u_{\max} - u$ | — | — | 0 | Box 约束 |
> | 速度限制 | $v_{\max} - \dot{x}$ | $-\ddot{x} = -u$ | — | 1 | CBF |
> | 位置限制 | $x_{\max} - x$ | $-\dot{x}$（无 $u$！） | $-u$ | 2 | HOCBF |
>
> 位置约束 $r=2$ 是最经典的 HOCBF 场景：你不能直接"推"位置，只能通过加速度间接影响。

### 5.2 HOCBF 构造（逐层回退）

> 设 $h(x)$ 相对度为 $r$，定义序列：
>
> $$\begin{aligned} \psi_0(x) &:= h(x) \\ \psi_1(x) &:= \dot{\psi}_0 + \alpha_1(\psi_0) = \dot{h} + \alpha_1(h) \\ \psi_2(x) &:= \dot{\psi}_1 + \alpha_2(\psi_1) \\ \vdots \\ \psi_r(x) &:= \dot{\psi}_{r-1} + \alpha_r(\psi_{r-1}) \end{aligned}$$
>
> 若存在 $u$ 使 $\psi_r(x) \ge 0$，则 $h(x)$ 是 **HOCBF**。

**以 $r=2$ 为例的关键展开（最常用）：**

选 $\alpha_1(s) = \gamma_1 s$, $\alpha_2(s) = \gamma_2 s$：

$$\begin{aligned} \psi_0 &= h \\ \psi_1 &= \dot{h} + \gamma_1 h \\ \psi_2 &= \ddot{h} + \gamma_1 \dot{h} + \gamma_2(\dot{h} + \gamma_1 h) = \ddot{h} + (\gamma_1 + \gamma_2)\dot{h} + \gamma_1\gamma_2 h \ge 0 \end{aligned}$$

代入 $\ddot{h} = L_f^2 h + L_G L_f h \cdot u$，得**线性约束**：

$$L_G L_f h(x) \cdot u \ge -L_f^2 h(x) - (\gamma_1 + \gamma_2)\dot{h}(x) - \gamma_1\gamma_2 h(x)$$

#### 案例 11：位置限制 HOCBF 完整手算（对应 PMSM 速度限制）

> 系统 $\ddot{x} = u$（状态 $x_1 = x$, $x_2 = \dot{x}$），约束 $x \le x_{\max}$。
>
> 写成仿射形式：$f = [x_2,\; 0]^\top$, $G = [0,\; 1]^\top$
>
> **Step 1** — $h(x) = x_{\max} - x_1$，检查相对度：
>
> $$\dot{h} = -\dot{x}_1 = -x_2$$
> $$L_G h = \nabla h \cdot G = [-1, 0] \cdot [0, 1]^\top = 0 \quad \text{→ } u \text{ 不出现！}$$
>
> **Step 2** — 继续求导：
>
> $$\ddot{h} = -\ddot{x}_1 = -u$$
> $$L_G L_f h = \frac{\partial}{\partial x}(\dot{h}) \cdot G = [0, -1] \cdot [0, 1]^\top = -1 \neq 0 \quad \text{→ } r=2$$
>
> **Step 3** — 构造 HOCBF，取 $\gamma_1 = \gamma_2 = 5$：
>
> $$\psi_0 = h = x_{\max} - x_1$$
> $$\psi_1 = \dot{h} + 5h = -x_2 + 5(x_{\max} - x_1)$$
> $$\psi_2 = \ddot{h} + 5\dot{h} + 5\psi_1 = -u + 5(-x_2) + 5[-x_2 + 5(x_{\max} - x_1)] \ge 0$$
>
> 整理：$-u - 10x_2 + 25(x_{\max} - x_1) \ge 0$
>
> **最终线性约束：**
>
> $$u \le 25(x_{\max} - x_1) - 10x_2$$
>
> **物理含义验证：**
> - $x_1$ 离 $x_{\max}$ 远（$h$ 大）→ $u$ 上限大（允许更大加速度）
> - $x_2 > 0$（正在靠近边界）→ $u$ 上限减小（需要减速）
> - $x_2 < 0$（正在远离边界）→ $u$ 上限增大（可以加速）

### 5.3 HOCBF 参数选择

> 特征方程 $s^r + c_1 s^{r-1} + \cdots + c_r = 0$ 的根应在 $-5$ 到 $-20$ 之间。
>
> 对 $r=2$：$s^2 + (\gamma_1 + \gamma_2)s + \gamma_1\gamma_2 = 0$
>
> 例：$\gamma_1 = \gamma_2 = 10$ → 二重根 $-10$，适合大多数场景。

---

<h2 id="6">6. PMSM 控制实例</h2>

### 6.1 PMSM 数学模型

表贴式 PMSM（$L_d = L_q = L_s$），状态 $x = [i_d,\; i_q,\; \omega_m]^\top$，输入 $u = [u_d,\; u_q]^\top$：

$$\begin{aligned} \frac{di_d}{dt} &= -\frac{R_s}{L_s} i_d + p\omega_m i_q + \frac{1}{L_s} u_d \\[6pt] \frac{di_q}{dt} &= -\frac{R_s}{L_s} i_q - p\omega_m i_d - \frac{\psi_f}{L_s} p\omega_m + \frac{1}{L_s} u_q \\[6pt] \frac{d\omega_m}{dt} &= \frac{1}{J}\left( K_t i_q - T_L - B\omega_m \right) \end{aligned}$$

其中 $K_t = \frac{3}{2} p \psi_f$。

> **控制仿射形式** $\dot{x} = f(x) + G(x)u$：
>
> $$f(x) = \begin{bmatrix} -\frac{R_s}{L_s} i_d + p\omega_m i_q \\[4pt] -\frac{R_s}{L_s} i_q - p\omega_m i_d - \frac{\psi_f}{L_s} p\omega_m \\[4pt] \frac{1}{J}(K_t i_q - T_L - B\omega_m) \end{bmatrix}, \quad G(x) = \begin{bmatrix} \frac{1}{L_s} & 0 \\[4pt] 0 & \frac{1}{L_s} \\[4pt] 0 & 0 \end{bmatrix}$$

| 符号 | 含义 | 符号 | 含义 |
|------|------|------|------|
| $i_d, i_q$ | d/q 轴电流 | $R_s$ | 定子电阻 |
| $u_d, u_q$ | d/q 轴电压（控制输入） | $L_s$ | 定子电感 |
| $\omega_m$ | 机械角速度 | $\psi_f$ | 永磁磁链 |
| $p$ | 极对数 | $J$ | 转动惯量 |
| $T_L$ | 负载转矩 | $B$ | 粘滞摩擦系数 |

### 6.2 速度限制 → HOCBF（$r=2$）

> **约束：** $\omega_m \le \omega_{\max}$
>
> **Step 1** — $h_\omega(x) = \omega_{\max} - \omega_m$
>
> $$\dot{h}_\omega = -\dot{\omega}_m = -\frac{1}{J}(K_t i_q - T_L - B\omega_m)$$
>
> $L_G h_\omega = [0,\; 0]$ — **$u$ 不出现 → $r=2$，需要 HOCBF！**
>
> **Step 2** — 计算 $\ddot{h}_\omega$：
>
> $$\ddot{h}_\omega = -\frac{1}{J}\left(K_t \dot{i}_q - B\dot{\omega}_m\right) = -\frac{1}{J}\left[ K_t\left(-\frac{R_s}{L_s}i_q - p\omega_m i_d - \frac{\psi_f}{L_s}p\omega_m + \frac{u_q}{L_s}\right) - B\dot{\omega}_m \right]$$
>
> $L_G L_f h_\omega = \left[0,\; -\frac{K_t}{J L_s}\right] \neq 0$ — $u_q$ 出现！验证 $r=2$。
>
> **Step 3** — HOCBF 约束：
>
> $$\psi_0 = h_\omega, \quad \psi_1 = \dot{h}_\omega + \gamma_1 h_\omega, \quad \psi_2 = \ddot{h}_\omega + \gamma_1 \dot{h}_\omega + \gamma_2 \psi_1 \ge 0$$
>
> 展开为 $u_q$ 的**线性不等式**：
>
> $$\boxed{-\frac{K_t}{J L_s} u_q \ge \frac{K_t}{J}\left(\frac{R_s}{L_s}i_q + p\omega_m i_d + \frac{\psi_f}{L_s}p\omega_m\right) + \frac{B}{J}\dot{\omega}_m - \gamma_1 \dot{h}_\omega - \gamma_2 \psi_1}$$

#### 数值案例（速度 HOCBF）

> 设 PMSM 参数：$R_s = 0.5\,\Omega$, $L_s = 5\,\text{mH}$, $p = 4$, $\psi_f = 0.1\,\text{Wb}$, $J = 0.001\,\text{kg·m}^2$, $B = 0.0001$, $K_t = 0.6$
>
> 当前状态：$i_d = 0$, $i_q = 2\,\text{A}$, $\omega_m = 2800\,\text{rpm} \approx 293\,\text{rad/s}$, $T_L = 0.1\,\text{N·m}$
>
> 速度上限 $\omega_{\max} = 3000\,\text{rpm} \approx 314\,\text{rad/s}$
>
> **Step 1** — $h_\omega = 314 - 293 = 21$
>
> $$\dot{\omega}_m = \frac{1}{0.001}(0.6 \times 2 - 0.1 - 0.0001 \times 293) = 1070.7$$
> $$\dot{h}_\omega = -1070.7 \quad \text{→ 转速正在上升，靠近上限！}$$
>
> **Step 2** — 取 $\gamma_1 = \gamma_2 = 10$：
>
> $$\psi_1 = -1070.7 + 10 \times 21 = -860.7$$
>
> HOCBF 约束最终展开为 $u_q \le a \cdot h_\omega + b \cdot \dot{h}_\omega + \cdots$
>
> 因为 $\dot{h}_\omega$ 很负（-1070），速度上升很快，HOCBF 会对 $u_q$ 施加一个很强的上界，迫使系统减速。

### 6.3 电流限制 → CBF（$r=1$）

#### 方式 A：单轴上限

> **约束：** $i_q \le i_{q,\max}$
>
> $$h_{iq} = i_{q,\max} - i_q$$
>
> $$\dot{h}_{iq} = -\dot{i}_q = \frac{R_s}{L_s}i_q + p\omega_m i_d + \frac{\psi_f}{L_s}p\omega_m - \frac{1}{L_s}u_q$$
>
> $L_G h_{iq} = [0,\; -1/L_s] \neq 0$ → $r=1$，用 CBF：
>
> $$\boxed{-\frac{1}{L_s} u_q \ge -\frac{R_s}{L_s}i_q - p\omega_m i_d - \frac{\psi_f}{L_s}p\omega_m - \gamma_{iq} h_{iq}}$$

#### 方式 B：电流幅值圆约束

> **约束：** $i_d^2 + i_q^2 \le I_{\max}^2$
>
> $$h_I(x) = I_{\max}^2 - (i_d^2 + i_q^2)$$
>
> $$\dot{h}_I = -2i_d \dot{i}_d - 2i_q \dot{i}_q, \quad L_G h_I = \begin{bmatrix} -\frac{2i_d}{L_s}, & -\frac{2i_q}{L_s} \end{bmatrix}$$
>
> 只要 $[i_d, i_q] \neq [0, 0]$，$r=1$：
>
> $$\boxed{-\frac{2i_d}{L_s} u_d - \frac{2i_q}{L_s} u_q \ge -\dot{h}_I\big|_{u=0} - \gamma_I h_I}$$

> 方式 B 同时约束 d/q 轴，限制总电流幅值（防止过流烧毁），比分别限制更符合物理实际。

#### 案例 12：电流 CBF 数值手算

> 续案例 11 参数，$i_q = 2\,\text{A}$, $i_{q,\max} = 5\,\text{A}$, $h_{iq} = 3$, $\gamma_{iq} = 10$：
>
> $$\dot{h}_{iq}|_{u=0} = \frac{0.5}{0.005} \times 2 + 4 \times 293 \times 0 + \frac{0.1}{0.005} \times 4 \times 293 = 200 + 0 + 23440 = 23640$$
>
> CBF 约束：$-200 \cdot u_q \ge -23640 - 10 \times 3 = -23670$，即 $u_q \le 118.35$
>
> 这个上界很大（远大于额定电压），说明当前 $i_q = 2$ 离 $5$ 还远，电流 CBF 约束很松——这正是我们期望的：离边界越远，约束越不活跃。

### 6.4 电压限制 → QP Box 约束

> **约束：** $|u_d| \le U_{\max},\; |u_q| \le U_{\max}$
>
> 这是**输入硬约束**，直接放在 QP 的 bounds 里：
>
> $$\boxed{-U_{\max} \le u_d \le U_{\max}, \quad -U_{\max} \le u_q \le U_{\max}}$$

> 电压圆约束 $u_d^2 + u_q^2 \le U_{\max}^2$ 不是线性的——可做八角形近似，或用 CBF 化到下一层：$h_u(x) = U_{\max}^2 - (u_d^2 + u_q^2)$。

### 6.5 完整 QP 统一公式

> **PMSM 完整 CLF-CBF-HOCBF QP：**
>
> $$\begin{aligned} \min_{u_d, u_q, \delta} \quad & \frac{1}{2}\Big[(u_d - u_d^{\text{nom}})^2 + (u_q - u_q^{\text{nom}})^2\Big] + p\cdot\delta^2 \\ \text{s.t.} \quad & \text{(CLF 稳定)} \quad L_f V + L_G V \cdot u \le -\gamma_v V + \delta \\ & \text{(CBF 电流)} \quad L_f h_I + L_G h_I \cdot u \ge -\gamma_I h_I \\ & \text{(HOCBF 速度)} \quad L_f^2 h_\omega + L_G L_f h_\omega \cdot u \ge -(\gamma_1 + \gamma_2)\dot{h}_\omega - \gamma_1\gamma_2 h_\omega \\ & \text{(电压上限)} \quad -U_{\max} \le u_d \le U_{\max},\; -U_{\max} \le u_q \le U_{\max} \\ & \text{(松弛)} \quad \delta \ge 0 \end{aligned}$$

> **QP 标准形式（可直接丢给求解器）：** 变量 $[u_d, u_q, \delta]^\top \in \mathbb{R}^3$，所有约束都是线性的，矩阵仅 $3 \times 3$。

### 6.6 约束层次总览

| 约束 | $h$ 表达式 | 工具 | 相对度 | QP 形式 |
|------|-----------|------|--------|---------|
| 速度上限 $\omega_m \le \omega_{\max}$ | $h_\omega = \omega_{\max} - \omega_m$ | **HOCBF** | $r=2$ | $u_q$ 线性不等式 |
| 电流上限 $i_q \le i_{q,\max}$ | $h_{iq} = i_{q,\max} - i_q$ | **CBF** | $r=1$ | $u_q$ 线性不等式 |
| 电流幅值 $i_d^2 + i_q^2 \le I_{\max}^2$ | $h_I = I_{\max}^2 - (i_d^2 + i_q^2)$ | **CBF** | $r=1$ | $u_d, u_q$ 线性不等式 |
| 电压上限 $|u| \le U_{\max}$ | — | **Box 约束** | $r=0$ | $lb \le u \le ub$ |
| 稳定收敛 | $V = \frac{1}{2}(\omega_m - \omega_{\text{ref}})^2$ | **CLF** | — | $u_q$ 线性不等式（软） |

---

<h2 id="7">7. 脉络总结</h2>

### 7.1 递进学习路径

1. **李导数** → 把非线性动力学"塞进"标量函数导数，全部理论的算符基础：$\dot{h} = L_f h + L_G h \cdot u$
2. **CLF** → 回答"能稳定吗"，把稳定要求转为 $u$ 的不等式（软约束）
3. **CBF** → 回答"安全吗"，把安全集不变性转为 $u$ 的不等式（硬约束），前提 $r=1$
4. **CLF-CBF QP** → 统一框架：安全 + 稳定 + 最小偏离，凸 QP 微秒级求解
5. **HOCBF** → 解决 $r>1$：逐层回退 $\psi_0 \to \psi_1 \to \cdots \to \psi_r$，最终化为 $u$ 的线性约束

### 7.2 约束工具决策流程

```
给定约束 h(x) ≥ 0
    │
    ├─ 是否涉及 u 本身？(如电压上限)
    │   → 是 → Box 约束 (直接设 u 的上下界)
    │
    └─ 否 → 计算 L_G h, L_G L_f h, ...
            │
            ├─ L_G h ≠ 0          → r = 1 → CBF
            ├─ L_G h = 0, L_G L_f h ≠ 0 → r = 2 → HOCBF
            ├─ L_G L_f h = 0, L_G L_f² h ≠ 0 → r = 3 → HOCBF
            └─ ...
```

### 7.3 PMSM 三大约束对应工具速查

| 约束 | 工具 | r | 原因 |
|------|------|---|------|
| 速度上限 | HOCBF | 2 | $u$ 不直接影响 $\dot{h}_\omega$（转速） |
| 电流上限 | CBF | 1 | $u_q$ 直接影响 $\dot{i}_q$ |
| 电压上限 | Box | 0 | 约束本身就是 $u$ |

### 7.4 核心公式速查

| 概念 | 核心公式 |
|------|---------|
| 李导数 | $L_f h = \nabla h \cdot f(x)$ |
| 仿射系统 | $\dot{x} = f(x) + G(x)u$ |
| CLF 条件 | $\inf_u [L_f V + L_G V \cdot u] < 0$ |
| CBF 约束 | $L_f h + L_G h \cdot u \ge -\alpha(h)$ |
| HOCBF ($r=2$) | $\ddot{h} + (\gamma_1 + \gamma_2)\dot{h} + \gamma_1\gamma_2 h \ge 0$ |
| QP 目标 | $\min \frac{1}{2}(u - u_{\text{nom}})^\top H (u - u_{\text{nom}}) + p\delta^2$ |

---

### 参考文献

1. Ames, A. D., et al. "Control Barrier Functions: Theory and Applications." *ECC 2019*.
2. Xu, X., et al. "Robustness of Control Barrier Functions." *HSCC 2015*.
3. Xiao, W., & Belta, C. "High Order Control Barrier Functions." *TAC 2022*.
4. Nguyen, Q., & Sreenath, K. "Exponential CBF for High-Relative-Degree Safety Constraints." *ACC 2016*.
5. Ames, A. D., et al. "Control Lyapunov-Barrier Functions." *CDC 2017*.

---

*2026-08-04 · 整理于 WorkBuddy*
