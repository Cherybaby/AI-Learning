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
7. [稳定性与安全性证明](#8-stability)
8. [脉络总结](#7)

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

<h2 id="8-stability">8. 稳定性与安全性证明</h2>

> 前面章节给出了 CLF/CBF 的**定义**与**构造方法**，本章回答更根本的问题：**为什么这样设计的控制器真的能稳定、真的能保证安全？**
>
> 证明思路统一为两条主线：
> - **CLF → 稳定性**：构造 Lyapunov 函数 $V$，证明受控闭环满足 $\dot{V} \le -\gamma V + \delta$（指数类收敛，松弛项有界）。
> - **CBF → 安全性**：证明安全集 $\mathcal{C}$ 是**前向不变**的，即 $h(x(t)) \ge 0$ 对所有 $t \ge 0$ 成立。
>
> 所有证明都建立在**控制仿射系统** $\dot{x} = f(x) + G(x)u$ 上，并沿用前文章号约定。

---

### 8.1 预备知识：类 $\mathcal{K}$、类 $\mathcal{K}_\infty$ 与比较引理

> **定义（类 $\mathcal{K}$ / $\mathcal{K}_\infty$ 函数）：**
> - $\alpha: [0, a) \to [0, \infty)$ 属于类 $\mathcal{K}$，若它连续、严格递增且 $\alpha(0)=0$。
> - 若定义域是 $[0, \infty)$ 且 $\alpha(r) \to \infty$（当 $r \to \infty$），则属于类 $\mathcal{K}_\infty$。
> - $\mathcal{K}_\infty^e$（扩展类）允许在 $r>0$ 时取正值，是 CBF 定义中 $\alpha(h)$ 常用的形式。

> **比较引理（Comparison Lemma）：** 设 $\dot{y} = g(t, y)$ 满足 $g(t, y) \le \tilde{g}(t, y)$（其中 $\tilde{g}$ 是右端更"差"的标量 ODE），且 $y(t_0) \le \tilde{y}(t_0)$，则 $y(t) \le \tilde{y}(t)$ 对所有 $t \ge t_0$ 成立。
>
> **作用**：把 $\dot{h} \ge -\alpha(h)$ 这样的**微分不等式**"放缩"成可解 ODE $\dot{\xi} = -\alpha(\xi)$，直接得到 $h(x(t)) \ge \xi(t)$ 的下界估计。这是 CBF 安全性证明的核心工具。

#### 案例 1：比较引理的显式求解（CBF 证明的模板）

> 设 $\dot{h} \ge -2h$，$h(0) = h_0 > 0$。取比较系统 $\dot{\xi} = -2\xi$，$\xi(0)=h_0$，解为 $\xi(t) = h_0 e^{-2t}$。
>
> 由比较引理：$h(t) \ge h_0 e^{-2t} > 0$ 对所有 $t \ge 0$ 成立。
>
> **结论**：只要初始在安全集内（$h_0>0$），$h(t)$ 永远为正——这正是"前向不变"的定量版本。注意指数衰减率 $2$ 正好对应 CBF 参数 $\gamma$，**$\gamma$ 越大，越远离边界的速度越快**。

---

### 8.2 CLF 控制器稳定性证明

#### 8.2.1 理想 CLF 控制器（无松弛，$\delta \equiv 0$）

> **定理 1（理想 CLF 渐近稳定）：** 考虑 $\dot{x} = f(x) + G(x)u$，设 $V(x)$ 连续可微、正定、径向无界，且满足 CLF 条件
>
> $$\inf_{u \in \mathbb{R}^m} \left[ L_f V + L_G V \cdot u \right] < 0, \quad \forall x \neq 0$$
>
> 若选取控制律 $u = k(x)$ 使得
>
> $$L_f V + L_G V \cdot k(x) \le -\gamma V(x), \quad \gamma > 0$$
>
> 则闭环系统在原点**一致渐近稳定**。

**证明：**

**Step 1（Lyapunov 函数）**——取 $V(x)$ 本身。由 CLF 假设，$V$ 正定且径向无界（满足 Lyapunov 函数的前两条）。

**Step 2（沿闭环轨线的导数）**——由链式求导与李导数定义：

$$\dot{V}(x) = \frac{\partial V}{\partial x}\dot{x} = L_f V + L_G V \cdot k(x) \le -\gamma V(x)$$

**Step 3（比较引理）**——考虑比较方程 $\dot{\xi} = -\gamma \xi$，$\xi(0)=V(x(0))$。由比较引理：

$$0 \le V(x(t)) \le V(x(0)) e^{-\gamma t}$$

**Step 4（收敛结论）**——当 $t \to \infty$ 时 $V(x(t)) \to 0$，而 $V$ 正定 $\Rightarrow$ $x(t) \to 0$。指数衰减率 $\gamma$ 直接给出**收敛速度**。∎

> **关键观察**：证明中唯一用到的控制信息是"$\dot{V} \le -\gamma V$"这个不等式——**它正是由 CLF-QP 的约束强制保证的**。也就是说，只要 QP 可行，稳定性自动成立。

#### 案例 2：理想 CLF 衰减率的数值验证

> 系统 $\dot{x} = -x + u$（同前文章节 2.1），$V = \tfrac{1}{2}x^2$，选 $u = -2x$（即 $k=2$）：
>
> $$\dot{V} = x(-x + u) = x(-x -2x) = -3x^2 = -3 \cdot (2V) = -6V$$
>
> 故 $\gamma = 6$，解为 $V(t) = V_0 e^{-6t}$，$|x(t)| = |x_0| e^{-3t}$。
>
> 若选更激进的 $u = -10x$ → $\dot{V} = -11x^2 = -22V$，$\gamma = 22$，收敛更快但控制代价更大。**$\gamma$ 是"收敛速度 vs 控制能量"的旋钮**，与 QP 中 $\frac{1}{2}u^\top H u$ 的权重 $H$ 共同决定实际收敛率。

#### 8.2.2 带松弛的 CLF-QP 控制器（实际情形）

现实 QP 引入松弛 $\delta \ge 0$ 以处理 CLF-CBF 冲突。此时约束变为：

$$L_f V + L_G V \cdot u \le -\gamma V + \delta$$

稳定性结论需相应弱化：

> **定理 2（松弛 CLF 的实际稳定性）：** 设 QP 给出的控制律 $u^*(x)$ 满足约束
>
> $$L_f V + L_G V \cdot u^*(x) \le -\gamma V(x) + \delta(x), \quad \delta(x) \ge 0$$
>
> 且松弛项满足**一致上界** $\delta(x) \le \bar{\delta}$（例如由 QP 的 box 约束 $u \in \mathcal{U}$ 有界保证）。则闭环解满足
>
> $$V(x(t)) \le V(x(0)) e^{-\gamma t} + \frac{\bar{\delta}}{\gamma}\left(1 - e^{-\gamma t}\right)$$
>
> 即 $x(t)$ **指数收敛到半径为 $\sqrt{2\bar{\delta}/\gamma}$ 的球域**（而非原点）。

**证明：**

由约束直接得 $\dot{V} \le -\gamma V + \delta$。考虑比较系统 $\dot{\xi} = -\gamma \xi + \bar{\delta}$，$\xi(0)=V(0)$，解为

$$\xi(t) = V(0) e^{-\gamma t} + \frac{\bar{\delta}}{\gamma}\left(1 - e^{-\gamma t}\right)$$

由比较引理 $V(x(t)) \le \xi(t)$。当 $t \to \infty$ 时，$V(x(t)) \le \bar{\delta}/\gamma$。因 $V = \tfrac{1}{2}x^2$（径向等价），故 $\|x\|_\infty \le \sqrt{2\bar{\delta}/\gamma}$。∎

> **工程解读**：
> - $\bar{\delta} = 0$ → 回到理想渐近稳定（定理 1）。
> - $\bar{\delta}$ 越小（QP 惩罚 $p$ 越大）→ 最终残差越小，但应对 CBF 冲突的"让步空间"也越小。
> - **这是"安全优先"的代价**：牺牲了原点精确收敛，换来可行性。

#### 案例 3：松弛残差的定量估算

> 沿用案例 6 的参数：$\gamma = 1$。假设输入受限于 $|u| \le 10$，QP 中 $u^2$ 项有限，可得松弛上界 $\bar{\delta} \approx 22.5$（来自约束 $3u - \delta \le -22.5$ 在 $u=-10$ 时的边界）。
>
> 最终残差：$V_\infty \le \bar{\delta}/\gamma = 22.5$，对应 $\|x\|_\infty \le \sqrt{2 \times 22.5} \approx 6.7$。
>
> 若把惩罚提高到 $p=100$（更不愿松弛），残差可压到约 $2.25$，代价是冲突时 QP 更容易不可行。**这解释了为什么实际系统常采用"分层恢复"策略：障碍物清除后把 $\delta$ 权重调回大值，重新精确收敛到原点。**

---

### 8.3 CBF 控制器安全性证明

#### 8.3.1 ZCBF 与零化控制障碍函数

为证明不变性，采用**零化控制障碍函数（ZCBF）** 框架（Ames et al.），它比一般 CBF 条件更便于构造控制器：

> **定义（ZCBF）：** 设 $\mathcal{C} = \{x \mid h(x) \ge 0\}$，函数 $h$ 是**零化控制障碍函数**（相对度 1），若存在 $\mathcal{K}_\infty^e$ 函数 $\alpha$ 使得
>
> $$\sup_{u \in \mathcal{U}} \left[ L_f h + L_G h \cdot u + \alpha(h) \right] \ge 0$$
>
> 即存在 $u$ 使 $\dot{h} + \alpha(h) \ge 0$。这与前文章节 3.1 的 CBF 定义等价（$\dot{h} \ge -\alpha(h)$）。

> **引理（不变性引理）：** 设 $h$ 是 ZCBF，$\mathcal{U}$ 是 closed（闭集），$f, G$ 足够光滑。定义控制律
>
> $$u_{\text{cbf}}(x) \in \argmax_{u \in \mathcal{K}_{\text{cbf}}(x)} \|u\| \quad \text{（任取可行控制器）}$$
>
> 其中 $K_{\text{cbf}}(x) = \{u \mid L_f h + L_G h \cdot u + \alpha(h) \ge 0\}$。若 $x(0) \in \mathcal{C}$，则 $x(t) \in \mathcal{C}$ 对所有 $t \ge 0$。

**证明（反证法）：**

**Step 1（假设逃逸）**——假设结论不成立，则存在最早逃逸时刻 $t^* > 0$ 使得 $h(x(t^*)) = 0$ 且 $\dot{h}(x(t^*)) < 0$（否则无法从非负穿过 0 变为负）。

**Step 2（在边界上应用 CBF 条件）**——在 $x(t^*)$ 处，$h=0$，由 $\alpha \in \mathcal{K}_\infty^e$ 得 $\alpha(0) = 0$。CBF 定义要求存在 $u$ 使

$$\dot{h}(x(t^*)) = L_f h + L_G h \cdot u \ge -\alpha(0) = 0$$

**Step 3（矛盾）**——取可行控制 $u_{\text{cbf}}(t^*)$，必有 $\dot{h}(x(t^*)) \ge 0$，与 Step 1 的 $\dot{h} < 0$ 矛盾。∎

> **要点**：证明的关键是 **"边界处 $\dot{h} \ge 0$"**——一旦靠近边界，CBF 控制器就强制状态"不再继续靠近"，这正是前文章节 3.1 所述"$h \to 0$ 时 $\dot{h} \ge 0$"的几何含义。

#### 案例 4：ZCBF 控制器在边界的行为验证

> 系统 $\dot{x} = u$，$h(x) = 10 - x$（同前文章节 3.1 案例 7），$\alpha(s) = \gamma s$。CBF 约束 $u \le \gamma(10-x)$。
>
> **在边界上** $x = 10$：$h=0$，约束给出 $u \le 0$。故 $\dot{h} = -\dot{x} = -u \ge 0$。
>
> 取 CBF 控制器 $u_{\text{cbf}} = \min\{u_{\text{nom}},\, \gamma(10-x)\}$：
> - 若 $u_{\text{nom}} = 3$，$x=9.9$（接近边界，$\gamma=5$）：允许上限 $\gamma h = 0.5$，故 $u_{\text{cbf}} = 0.5$，$\dot{h} = -0.5 < 0$？
>
> ⚠️ **检查**：$\dot{h} = -u_{\text{cbf}} = -0.5$ 看似违反 $\dot{h}\ge 0$！问题出在 **$\alpha(h)=\gamma h = 0.5$ 而非 0**。正确计算：
>
> $$\dot{h} + \alpha(h) = -0.5 + 0.5 = 0 \ge 0 \quad ✅$$
>
> 即 CBF 保证的是 $\dot{h} \ge -\alpha(h) = -0.5$，允许 $h$ 轻微下降，但下降速度被 $\alpha(h)$ 压制。**当 $h \to 0$ 时 $\alpha(h) \to 0$，压制趋于无穷强，从而 $h$ 无法真正穿越 0。** 这正是证明中"$h$ 最多渐近趋近 0 而不穿越"的定量体现。

#### 8.3.2 指数类安全：显式下界

> **定理 3（指数类安全界）：** 若 CBF 条件 $\dot{h} \ge -\gamma h$（即 $\alpha(h)=\gamma h$）成立，且 $h(x(0)) \ge 0$，则
>
> $$h(x(t)) \ge h(x(0)) e^{-\gamma t} \ge 0, \quad \forall t \ge 0$$
>
> **特别地**，$h(x(0)) > 0 \Rightarrow h(x(t)) > 0$（严格保持在安全集内部）。

**证明：** 微分不等式 $\dot{h} \ge -\gamma h$ 与比较引理（案例 1）直接给出 $h(t) \ge h(0)e^{-\gamma t} \ge 0$。∎

> **物理含义**：参数 $\gamma$ 是"排斥边界的强度"。$\gamma$ 越大，一旦 $h$ 变小，控制器越剧烈地把状态推回安全区——这与前文章节 3.1 案例 8 中"$\gamma=5$ 比 $\gamma=2$ 约束更激进"完全一致。

#### 案例 5：双 CBF 的安全性叠加

> 设同时存在速度约束 $h_1 = 10 - x$（上限）与位置约束 $h_2 = x - 2$（下限），均取 $\gamma=3$。QP 同时施加
>
> $$u \le 3(10 - x), \qquad u \ge -3(x - 2)$$
>
> 由定理 3，各自满足 $h_1(t) \ge h_1(0)e^{-3t}$、$h_2(t) \ge h_2(0)e^{-3t}$。只要初始 $x(0) \in [2, 10]$，两者同时成立 → 状态始终被"夹"在带域内。
>
> **推广**：有限个 CBF 约束 $\{h_i\}_{i=1}^{N}$ 在 QP 中做交集 $\bigcap_i \{x \mid h_i(x) \ge 0\}$，**每个 $h_i$ 的安全性都由各自的不变性引理独立保证**——这是 CBF 框架模块化、可叠加的根本原因。

---

### 8.4 CLF-CBF 统一 QP 的可行性论证

> 仅有 CLF 稳定性 + CBF 安全性各自成立还不够——**QP 必须同时可行**，否则控制器无输出。本节论证统一框架的可行性条件。

> **定理 4（统一 QP 可行域非空）：** 考虑 CLF-CBF QP（前文章节 4）：
>
> $$\min_{u,\delta}\; \tfrac{1}{2}\|u-u_{\text{nom}}\|^2 + p\delta^2 \quad \text{s.t.}\quad \dot{V} + \gamma_v V - \delta \le 0,\; \dot{h} + \gamma_h h \ge 0,\; u \in \mathcal{U}$$
>
> 设 $\mathcal{U}$ 为闭凸集（如 box 约束），且 **CLF 与 CBF 约束各自在 $\mathcal{U}$ 内可行**，即存在 $u_{\text{clf}} \in \mathcal{U}$ 满足 CLF、存在 $u_{\text{cbf}} \in \mathcal{U}$ 满足 CBF。则：
>
> 1. **当 CLF、CBF 约束相容**（交集非空）时，QP 有唯一解（目标函数强凸），闭环同时稳定且安全。
> 2. **当二者冲突**（交集为空）时，松弛变量 $\delta$ 保证 QP **恒可行**：优先满足 CBF 硬约束，CLF 被适度违反。

**论证：**

**Part 1（相容情形）**——可行域是闭凸集（线性不等式 + box），目标函数 $\tfrac{1}{2}\|u-u_{\text{nom}}\|^2 + p\delta^2$ 关于 $(u,\delta)$ 强凸（$p>0$），故若可行域非空，**存在唯一全局最优解**。闭环由定理 1 + 定理 3 同时保证稳定与安全。

**Part 2（冲突情形）**——这是引入 $\delta$ 的关键。重写 CLF 约束为

$$L_G V \cdot u \le -L_f V - \gamma_v V + \delta$$

- 若左侧在 $u \in \mathcal{U}$ 上的最大值仍大于右端 → 取足够大的 $\delta$ 即可满足。
- 因 $\delta$ 无上界（仅受目标函数惩罚），**总能找到 $(u,\delta)$ 满足 CLF 约束**，同时保持 CBF 与 $u\in\mathcal{U}$ 不变（后二者不含 $\delta$）。

故可行域恒非空，QP 恒有解。此时由定理 2，稳定性退化为"实际稳定"（残差界 $\bar{\delta}/\gamma_v$）。∎

#### 案例 6：冲突场景的可行性验证（呼应章节 4 案例 9）

> 自动驾驶：$u_{\text{nom}} = 5$（加速），$u \in [-10, 10]$。
> - CBF 约束（距障碍 2m，$\gamma_h=3$）：$u \le 3 \times 2 = 6$ → $u \le 6$。
> - CLF 约束（目标速度 80km/h，当前 60，$\gamma_v=1$，$V=200$，$L_G V=3$，$L_f V=0$）：$3u \le -200 + \delta$。
>
> **相容检查**：CBF 要求 $u \le 6$，CLF 要求 $u \le (3u+\delta)/...$ 实际为 $u \le -66.7 + \delta/3$。
>
> - 无松弛（$\delta=0$）：CLF 要 $u \le -66.7$，但 $u \ge -10$（box）→ **无解，QP 不可行**。
> - 有松弛：取 $u = -10$（box 下界），则 $\delta \ge 3(-10) + 200 = 170$。QP 选 $\delta = 170$（付出代价 $p\delta^2$），输出 $u=-10$（全力刹车）。
>
> **解读**：$\delta=170$ 很大 → 稳定性被严重牺牲（按定理 2，残差 $\bar{\delta}/\gamma_v = 170$），但 CBF 硬约束保住安全。**这正是章节 4 所述"安全优先，稳定让步"的严格数学表述**：$\delta$ 不仅让 QP 可行，更精确刻画了"让步多少"。

---

### 8.5 统一闭环的性能画像

> 综合定理 1~4，采用 CLF-CBF QP 的统一控制器，闭环具有如下性能：

| 性质 | 条件 | 结论 | 对应定理 |
|------|------|------|---------|
| **安全性** | CBF 硬约束可行 | $h(x(t)) \ge h_0 e^{-\gamma_h t} \ge 0$ | 定理 3 |
| **渐近稳定** | $\delta \equiv 0$（无冲突） | $V(t) \le V_0 e^{-\gamma_v t}$ | 定理 1 |
| **实际稳定** | $\delta > 0$（有冲突） | $V(t) \le V_0 e^{-\gamma_v t} + \bar{\delta}/\gamma_v$ | 定理 2 |
| **可行性** | $\mathcal{U}$ 闭凸 + $\delta$ 松弛 | QP 恒有唯一解 | 定理 4 |

> **核心结论**：CLF 与 CBF 不是"两个独立目标"，而是通过 QP 形成**偏序关系**——CBF（安全）是**硬约束**，CLF（稳定）是**软约束**。数学上体现为：
> - CBF 决定可行域是否为空（生存问题）；
> - CLF 决定可行域内的优化方向（性能问题）。
>
> 这正是"**安全过滤器**"这一称谓的本质：先保证活下来（安全不变集），再谈收敛快慢（Lyapunov 衰减）。

---

### 8.6 与 HOCBF 的衔接

> 第 5 章的 HOCBF 处理 $r>1$ 情形。其稳定性/安全性证明可通过**动态扩展**归约到本章框架：

> **归约思路**：对相对度 $r$ 的 $h$，构造序列 $\psi_0 = h,\ \psi_1 = \dot{h}+\alpha_1(h),\ \dots,\ \psi_r$ 如第 5 章。定义**扩张状态** $z = [\psi_0, \psi_1, \dots, \psi_{r-1}]^\top$，则 $\dot{z}_i = \psi_{i+1}$（对 $i<r-1$），而 $\dot{z}_{r-1} = \psi_r$ **显式含控制 $u$**（因 $L_G L_f^{r-1}h \neq 0$）。

> **结论**：在扩张坐标下，$\psi_r \ge 0$ 成为**相对度 1 的 ZCBF 约束**（关于扩张系统），可直接套用定理 3 得 $\psi_r(t) \ge \psi_r(0)e^{-\gamma_r t} \ge 0$，再递归推出 $h(x(t)) \ge 0$。

#### 案例 7：速度限制 HOCBF（$r=2$）的安全性验证

> 沿用第 6 章：$h_\omega = \omega_{\max} - \omega_m$，$r=2$，$\gamma_1=\gamma_2=10$。
>
> $$\psi_0 = h_\omega, \quad \psi_1 = \dot{h}_\omega + 10h_\omega, \quad \psi_2 = \ddot{h}_\omega + 10\dot{h}_\omega + 10\psi_1 \ge 0$$
>
> 由定理 3（对 $\psi_1$ 的相对度 1 约束 $\psi_2 \ge 0$，取 $\alpha=\gamma_2 s$）：

$$\psi_1(t) \ge \psi_1(0) e^{-10t}$$

> 又 $\psi_0 = h_\omega$ 满足 $\dot{\psi}_0 = \psi_1 - 10\psi_0 \ge \psi_1(0)e^{-10t} - 10\psi_0$……
>
> 递归求解可得 $h_\omega(t) \ge 0$。**数值上**（第 6 章案例：$h_\omega=21,\ \dot{h}_\omega=-1070.7$）：$\psi_1(0) = -1070.7 + 210 = -860.7 < 0$！
>
> ⚠️ **这揭示了一个关键点**：$\psi_1(0) < 0$ 意味着**初始不满足 HOCBF 条件**——系统已经"太快了"。此时需要：
> 1. 在 $u_q$ 上限约束下尽快把 $\psi_1$ 拉回非负（定理保证指数收敛，但需要控制权限足够）；
> 2. 或降低参考速度，使初始状态落在可行域内。
>
> **工程启示**：HOCBF 的安全性证明**依赖于初始可行性**（$h, \psi_1, \dots, \psi_{r-1}$ 在 $t=0$ 满足约束）。这对应第 5 章参数选择中"特征根 $-5$ 到 $-20$"的要求——**参数决定收敛快慢，而收敛快慢决定能否在越界前把状态拉回**。

---

### 8.7 证明脉络小结

> **统一证明范式**：
>
> 1. **写全导数** $\dot{V}$ 或 $\dot{h}$ → 用李导数展开 $L_f + L_G u$。
> 2. **施加控制器/QP 约束** → 得到微分不等式 $\dot{V} \le -\gamma V + \delta$ 或 $\dot{h} \ge -\alpha(h)$。
> 3. **调用比较引理** → 解出 $V(t), h(t)$ 的显式下界。
> 4. **下界非负** → 不变性（安全）或收敛（稳定）。

> 这一范式的力量在于：**它把"控制器设计"完全归约为"构造满足不等式的 $u$"**——而 QP 正是求解这个不等式的最优化机器。理解这层联系，也就理解了为什么 CLF-CBF-QP 能成为安全关键控制（机器人、自动驾驶、PMSM 弱磁限速）的通用语言。

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
