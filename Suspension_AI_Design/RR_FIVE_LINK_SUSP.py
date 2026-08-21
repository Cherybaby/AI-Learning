#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五连杆独立后悬架运动学求解器（RR_FIVE_LINK_SUSP.m 的 Python 等价实现）。

物理背景：
  五连杆悬架用 5 根连杆（对应本文件里的 A1-B1 ~ A5-B5 五段"杆"）把车轮定位在车身上，
  每根连杆的两端是固定长度的球销连接（内点 Ai 固定在车身，外点 Bi 固定在轮边/转向节上）。
  当车轮沿 Z 向上下跳动（轮跳行程 wc_stroke）时，轮边转向节的空间姿态（3个转角+3个平移，
  但本模型把转向节视为随轮心刚体运动，姿态由5根连杆的固定长度约束共同决定）会随之变化，
  进而带出前束角(toe)、外倾角(camber)、后倾角(caster)等一系列随行程变化的运动学曲线。

求解思路（每个行程点独立求解，逐点扫描 wc_stroke）：
  1. 已知车身端硬点 A1..A5（固定不动）与轮边端硬点 B1..B5（固定在转向节局部坐标系中）；
  2. 给定当前轮心目标高度 Ctz，用 fsolve 数值求解转向节的刚体位姿参数 x=[三个欧拉角, 平移x, 平移y]，
     使得变换后的 5 个 Bi 点到对应 Ai 点的距离，与设计时的原长完全一致（5 个约束方程，5 个未知数）；
  3. 位姿求出后，用旋转矩阵 E + 平移 Y 把转向节上其余关键点（C1、主销上下球销 LUP/LLWR 等）
     变换到当前姿态下的世界坐标，再由几何关系推导出 toe/camber/caster/接地点等输出量。

与 XP_algorithm.m / RR_FIVE_LINK_SUSP.m 的对应关系：
  - _rotation_matrix_for_constraint() 对应 XP_algorithm.m 里 fsolve 迭代过程中使用的旋转矩阵；
  - _rotation_matrix_for_main() 对应主脚本收敛后，用最终解构造旋转矩阵时的写法
    （注意 d23 符号与前者相反，这是 MATLAB 原始代码中两处旋转矩阵定义即存在的细节差异，
    这里原样保留，不做"统一化"改动，以保证与 MATLAB 版数值完全对齐）；
  - _xp_algorithm() 是 fsolve 每次迭代调用的残差函数，对应 XP_algorithm.m 的核心逻辑；
  - run_simulation() 对应 RR_FIVE_LINK_SUSP.m 主脚本的行程扫描循环 + 侧倾中心高度(RCH)计算。

本文件被 数据处理/generate_dataset.py 复用：generate_dataset.py 直接调用本文件的
HardPoints/_xp_algorithm/_rotation_matrix_for_main/_transform_point/_solve_contact_point
等底层函数，对硬点做随机扰动后重新求解，批量生成训练样本（DOE 数据增广）。
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve


@dataclass
class HardPoints:
    """五连杆悬架的全部关键硬点坐标（单位 mm，均在车身固定坐标系下的初始/标称位置）。

    五根连杆的内外点（内点 A 固定在车身，外点 B 固定在轮边转向节上，两点间距离即杆长，
    是刚体运动过程中保持不变的约束）：
        A1-B1, A2-B2, A3-B3, A4-B4, A5-B5   —— 分别对应上控制臂1、上控制臂2、
                                                下控制臂1、下控制臂2、转向拉杆 5 根连杆
    其余关键点：
        C    —— 轮心（Wheel Center），用于表达当前轮跳目标高度 Ctz，也是刚体姿态的平移基准点
        C1   —— 轮心在转向节局部坐标系下对应的另一参考点（用于计算 toe/camber 的方向向量）
        LUP  —— 主销上球销（转向节上的固定点，随刚体一起变换，构成主销轴的上端点）
        LLWR —— 主销下球销（构成主销轴的下端点；LUP-LLWR 连线即"主销轴"，
                 caster/caster_arm/scrub_radius 均由该轴与地面的几何关系推导而来）
        SCLP —— 转向拉杆外点参考（本文件未直接使用于运动学量计算，保留供扩展/校验用）
    """
    A1: np.ndarray
    B1: np.ndarray
    A2: np.ndarray
    B2: np.ndarray
    A3: np.ndarray
    B3: np.ndarray
    A4: np.ndarray
    B4: np.ndarray
    A5: np.ndarray
    B5: np.ndarray
    C: np.ndarray
    C1: np.ndarray
    LUP: np.ndarray
    LLWR: np.ndarray
    SCLP: np.ndarray


def _rotation_matrix_for_constraint(a: np.ndarray) -> np.ndarray:
    """构造 fsolve 迭代过程中使用的 3x3 旋转矩阵（对应 XP_algorithm.m 内的旋转矩阵定义）。

    参数:
        a: 长度>=3 的数组，a[0]/a[1]/a[2] 为绕三个轴的欧拉角（弧度），是 fsolve 正在
           试探的未知量（残差函数 _xp_algorithm 会把完整的 5 维未知向量传进来，
           这里只用到前 3 个转角分量）。

    返回:
        (3,3) 旋转矩阵 E1，满足 世界坐标 = E1 @ 局部坐标 + 平移。

    注意：这里的 d23 符号（-math.cos(a[1])*math.sin(a[2])）与
    _rotation_matrix_for_main() 中 d23 符号相反 —— 这是原始 MATLAB 代码
    (XP_algorithm.m vs RR_FIVE_LINK_SUSP.m) 里两处独立实现本身就存在的差异，
    照抄保留是为了保证 Python 版与 MATLAB 版数值完全对齐，不做"看起来该统一"的修改。
    """
    d11 = math.cos(a[0]) * math.cos(a[1])
    d12 = -math.sin(a[0]) * math.cos(a[1])
    d13 = math.sin(a[1])
    d21 = math.sin(a[0]) * math.cos(a[2]) + math.cos(a[0]) * math.sin(a[1]) * math.sin(a[2])
    d22 = math.cos(a[0]) * math.cos(a[2]) - math.sin(a[0]) * math.sin(a[1]) * math.sin(a[2])
    d23 = -math.cos(a[1]) * math.sin(a[2])
    d31 = math.sin(a[0]) * math.sin(a[2]) - math.cos(a[0]) * math.sin(a[1]) * math.cos(a[2])
    d32 = math.cos(a[0]) * math.sin(a[2]) + math.sin(a[0]) * math.sin(a[1]) * math.cos(a[2])
    d33 = math.cos(a[1]) * math.cos(a[2])
    return np.array(
        [[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]], dtype=float
    )


def _rotation_matrix_for_main(x: np.ndarray) -> np.ndarray:
    """构造 fsolve 收敛后，用最终解 x 构造旋转矩阵时使用的版本
    （对应 RR_FIVE_LINK_SUSP.m 主循环里的旋转矩阵定义）。

    参数:
        x: fsolve 求解出的 5 维姿态向量 [欧拉角a,欧拉角b,欧拉角c, 平移x, 平移y]。

    返回:
        (3,3) 旋转矩阵 E。

    与 _rotation_matrix_for_constraint() 的唯一区别是 d23 的符号相反
    （此处为 +math.cos(x[1])*math.sin(x[2])），原因见该函数的注释。
    """
    d11 = math.cos(x[0]) * math.cos(x[1])
    d12 = -math.sin(x[0]) * math.cos(x[1])
    d13 = math.sin(x[1])
    d21 = math.sin(x[0]) * math.cos(x[2]) + math.cos(x[0]) * math.sin(x[1]) * math.sin(x[2])
    d22 = math.cos(x[0]) * math.cos(x[2]) - math.sin(x[0]) * math.sin(x[1]) * math.sin(x[2])
    d23 = math.cos(x[1]) * math.sin(x[2])
    d31 = math.sin(x[0]) * math.sin(x[2]) - math.cos(x[0]) * math.sin(x[1]) * math.cos(x[2])
    d32 = math.cos(x[0]) * math.sin(x[2]) + math.sin(x[0]) * math.sin(x[1]) * math.cos(x[2])
    d33 = math.cos(x[1]) * math.cos(x[2])
    return np.array(
        [[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]], dtype=float
    )


def _transform_point(E: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """把转向节局部坐标系下的点 p，按当前刚体姿态(旋转 E + 平移 Y)变换到世界坐标系。

    刚体运动的标准表达：世界坐标 = E @ 局部坐标 + Y。
    转向节上所有随刚体一起运动的点（B1..B5、C1、主销球销 LUP/LLWR 等）都通过这个
    公式，从"设计时的标称局部坐标"变换到"当前轮跳行程下的世界坐标"。

    参数:
        E: (3,3) 旋转矩阵（见 _rotation_matrix_for_main/_rotation_matrix_for_constraint）
        Y: (3,) 平移向量
        p: (3,) 待变换点在局部坐标系下的坐标

    返回:
        (3,) 变换后的世界坐标
    """
    return E @ p + Y


def _xp_algorithm(a: np.ndarray, Ctz: float, hp: HardPoints) -> np.ndarray:
    """fsolve 残差函数：给定一组姿态参数 a，返回 5 根连杆"当前长度 - 标称长度"的残差向量。

    这是整个求解器的数学核心 —— 五连杆悬架的运动学约束本质上是：
      转向节（轮边刚体）在空间中只有一个自由度（沿轮心 Z 向的行程 Ctz 是外部给定的驱动量），
      因为 5 根连杆的固定长度约束，联合车身端 5 个固定点 A1..A5，恰好能解出转向节的完整姿态
      （3 个转角 + 2 个平动，共 5 个未知数，对应 5 个杆长约束方程，构成一个恰定非线性方程组）。

    求解流程：
      1. 用当前试探的姿态参数 a 构造旋转矩阵 E1；
      2. 由"轮心 C 在旋转后必须平移到 (a[3], a[4], Ctz)"这一条件反解出平移向量 Y1
         （即先固定轮心的世界坐标目标，再反推整体平移量，见 Y1 三个分量的计算）；
      3. 用 (E1, Y1) 把转向节局部坐标系下的 5 个外点 B1..B5 变换到世界坐标，得到 B11..B55；
      4. 计算每根连杆变换后的实际长度 |B_it - A_i|，与设计标称长度 |B_i - A_i| 的差值，
         这个差值理论上应该在收敛时趋于 0（5 根连杆长度都不能被拉伸/压缩）。

    参数:
        a: 长度为 5 的姿态试探向量 [欧拉角a, 欧拉角b, 欧拉角c, 轮心世界坐标x, 轮心世界坐标y]
           （fsolve 会不断调整这个向量试图让残差趋于 0）
        Ctz: 当前扫描到的轮心目标高度（世界坐标 Z），由外部行程扫描循环给定，不参与求解
        hp: 悬架硬点数据

    返回:
        (5,) 残差向量 [ss1..ss5]，每个元素是对应连杆的"当前长度-标称长度"误差(mm)，
        fsolve 的目标就是找到使这个向量整体趋于零向量的姿态参数 a。
    """
    E1 = _rotation_matrix_for_constraint(a)
    # 反解平移向量 Y1：约束"轮心 C 变换后必须落在 (a[3], a[4], Ctz) 这个世界坐标点"，
    # 即 E1@C + Y1 = [a[3], a[4], Ctz]，整理得 Y1 = [a[3],a[4],Ctz] - E1@C。
    Y1 = np.array(
        [
            a[3] - np.dot(E1[0, :], hp.C),
            a[4] - np.dot(E1[1, :], hp.C),
            Ctz - np.dot(E1[2, :], hp.C),
        ],
        dtype=float,
    )

    # 把 5 个连杆外点分别变换到当前试探姿态下的世界坐标
    B11 = _transform_point(E1, Y1, hp.B1)
    B22 = _transform_point(E1, Y1, hp.B2)
    B33 = _transform_point(E1, Y1, hp.B3)
    B44 = _transform_point(E1, Y1, hp.B4)
    B55 = _transform_point(E1, Y1, hp.B5)

    # 每根连杆的残差 = 变换后实际长度(Bi_t 到 Ai 的距离) - 设计标称长度(初始 Bi 到 Ai 的距离)
    ss1 = np.linalg.norm(B11 - hp.A1) - np.linalg.norm(hp.B1 - hp.A1)
    ss2 = np.linalg.norm(B22 - hp.A2) - np.linalg.norm(hp.B2 - hp.A2)
    ss3 = np.linalg.norm(B33 - hp.A3) - np.linalg.norm(hp.B3 - hp.A3)
    ss4 = np.linalg.norm(B44 - hp.A4) - np.linalg.norm(hp.B4 - hp.A4)
    ss5 = np.linalg.norm(B55 - hp.A5) - np.linalg.norm(hp.B5 - hp.A5)
    return np.array([ss1, ss2, ss3, ss4, ss5], dtype=float)


def _solve_y_with_fixed_length(Bt: np.ndarray, A: np.ndarray, L: float) -> float:
    """已知点 Bt 的 x/z 坐标和到定点 A 的固定距离 L，反解 Bt 的 y 坐标（取两个解中较小的一个）。

    背景：fsolve 求解出的姿态参数只保证了"3D 距离"约束满足（见 _xp_algorithm），
    但由于该约束方程在数值上对 y 分量不敏感（欠定/病态），主脚本额外用这个函数
    按"固定 x/z、只解 y"的思路重新精确计算一次 y 坐标，消除数值误差累积。

    几何原理：把约束 |Bt - A|² = L² 展开成关于 Bt[1](y) 的一元二次方程：
        (Bt[0]-A[0])² + (Bt[1]-A[1])² + (Bt[2]-A[2])² = L²
        => (Bt[1]-A[1])² = L² - (Bt[0]-A[0])² - (Bt[2]-A[2])²   （即 delta，判别式）
        => Bt[1] = A[1] ± sqrt(delta)
    两个根对应两个几何上都合法的解（连杆外点在 y 方向的两侧），根据 MATLAB 原逻辑
    统一取较小的一个（min(y1, y2)），以保持与其分支选择行为一致。

    参数:
        Bt: (3,) 当前试探/变换后的连杆外点坐标（x/z 分量已知有效，y 分量待重新求解）
        A:  (3,) 该连杆对应的车身端固定内点坐标
        L:  该连杆的设计标称长度（|B-A|）

    返回:
        float: 修正后的 y 坐标（两个根中较小的一个）
    """
    delta = L * L - (Bt[0] - A[0]) ** 2 - (Bt[2] - A[2]) ** 2
    if delta < 0 and abs(delta) < 1e-8:
        delta = 0.0            # 浮点误差导致的微小负数，钳位为0（几何上应视为刚好切线的临界解）
    if delta < 0:
        raise ValueError(f"Invalid geometry, negative square term: {delta}")
    root = math.sqrt(delta)
    y1 = A[1] - root
    y2 = A[1] + root
    return min(y1, y2)


def _solve_contact_point(Ct: np.ndarray, C1t: np.ndarray, tire_radius: float, Ctz: float) -> np.ndarray:
    """求解轮胎与地面的接地点坐标（对应 MATLAB 脚本"接地点坐标"章节的等价实现）。

    几何原理：
      轮胎近似为一个圆环（简化为圆），圆心在轮心 Ct，圆面法线方向由 wheel = C1t - Ct
      （轮心到 C1 参考点的方向向量，代表车轮旋转平面的朝向）决定。
      地面是水平面 z = Jz（Jz 取 Ctz - 10.0，即比当前轮心高度低 10mm 的一个近似地平面，
      10mm 是模型里对"轮胎变形/接地间隙"的一个经验修正量，不是轮胎半径本身）。
      接地点 = 轮胎圆周上，z 坐标恰好落在地平面 Jz 上的那个点（同一圆周上通常有两个
      这样的候选点，取两者中点方向再按半径重新归一化，得到最终唯一的接地点）。

    求解步骤：
      1. 把"轮胎圆周上一点 z=Jz"这一约束，结合"该点到轮心距离=tire_radius"的圆方程，
         转化成 (dx, dy) 平面内的"直线(轮胎平面在该高度的截线) 与圆(轮胎半径为半径的圆)
         求交点"问题；
      2. 用点到直线最近点 + 垂直方向偏移，解出两个交点 J1、J2；
      3. 取 J1、J2 的中点方向 TT_CTR，再沿这个方向重新按真实轮胎半径 tire_radius
         归一化，得到最终接地点 CONT（这一步保证返回点确实在半径为 tire_radius 的
         圆周上，而不只是"z=Jz 平面与轮胎平面的交线上"）。

    参数:
        Ct: (3,) 当前姿态下的轮心世界坐标
        C1t: (3,) 当前姿态下 C1 参考点的世界坐标（决定轮胎旋转平面朝向）
        tire_radius: 轮胎静力半径(mm)
        Ctz: 当前轮心目标高度（用于确定近似地平面 Jz = Ctz - 10）

    返回:
        (3,) 接地点世界坐标 [x, y, z]
    """
    Jz = Ctz - 10.0                    # 近似地平面高度：比当前轮心低 10mm（模型经验修正）
    wheel = C1t - Ct                   # 轮胎旋转平面的法线方向向量

    dz = Jz - Ct[2]                    # 从轮心到地平面的高度差
    rhs = -wheel[2] * dz               # 平面方程在 (dx,dy) 空间里对应的直线方程右端项

    # 求解"轮胎平面与地平面 z=Jz 的交线" 与 "半径为 tire_radius 的圆" 的交点，
    # 在 (dx, dy) 局部平面坐标系下：
    #   直线方程: wheel_x*dx + wheel_y*dy = rhs
    #   圆方程:   dx^2 + dy^2 = tire_radius^2 - dz^2   (r2_xy)
    r2_xy = tire_radius * tire_radius - dz * dz
    if r2_xy < 0 and abs(r2_xy) < 1e-8:
        r2_xy = 0.0
    if r2_xy < 0:
        raise ValueError(f"No real contact solution, negative planar radius squared: {r2_xy}")

    wx, wy = wheel[0], wheel[1]
    norm_w = math.hypot(wx, wy)
    if norm_w < 1e-12:
        raise ValueError("Degenerate wheel axis projection in XY plane")

    # 直线上距离原点最近的点（点到直线的垂足），作为后续沿垂直方向偏移的基准点
    d0x = (rhs / (norm_w * norm_w)) * wx
    d0y = (rhs / (norm_w * norm_w)) * wy

    # 直线的垂直方向单位向量（用于从基准点沿直线正负两侧移动，找到与圆的两个交点）
    px = -wy / norm_w
    py = wx / norm_w

    off2 = r2_xy - (d0x * d0x + d0y * d0y)
    if off2 < 0 and abs(off2) < 1e-8:
        off2 = 0.0
    if off2 < 0:
        raise ValueError(f"No real line-circle intersection, negative offset squared: {off2}")
    off = math.sqrt(off2)

    # 直线与圆的两个交点（对应轮胎圆周上，落在地平面高度的两个候选接地位置）
    dx1, dy1 = d0x + off * px, d0y + off * py
    dx2, dy2 = d0x - off * px, d0y - off * py

    J1 = np.array([Ct[0] + dx1, Ct[1] + dy1, Jz], dtype=float)
    J2 = np.array([Ct[0] + dx2, Ct[1] + dy2, Jz], dtype=float)

    # 取两个候选点的中点方向，作为最终接地点相对轮心的"参考方向"
    # （MATLAB 原逻辑对两个符号解取平均，这里保持一致）
    TT_CTR = 0.5 * (J1 + J2)

    # 沿轮心->TT_CTR方向，重新按真实轮胎半径 tire_radius 归一化，
    # 确保最终返回的接地点严格落在"以轮心为球心、半径为tire_radius的球面"上
    # （即 |CONT - Ct| = tire_radius）。
    v = TT_CTR - Ct
    vv = float(np.dot(v, v))
    if vv < 1e-20:
        raise ValueError("Degenerate TT_CTR direction vector")

    t_abs = tire_radius / math.sqrt(vv)
    t = t_abs                          # 取正值分支（与 MATLAB 标量解行为一致）

    CONT = Ct + t * v
    return CONT


def _solve_rch_point(yc: float, zc: float, ya: float, za: float, yb: float, zb: float) -> tuple[float, float]:
    """求侧倾中心(Roll Center)相关计算中用到的辅助点坐标 (y0, z0)。

    几何原理：给定 y-z 平面（侧视投影）上三个点 C(yc,zc)、A(ya,za)、B(yb,zb)，
    求一个点 (y0, z0)，使其到 C 和 A 的距离相等、且到 A 和 B 的距离也相等
    （即该点是三点两两组合的等距点，对应"接地点轨迹上局部圆弧的圆心"这一思路，
    用来估计轮跳过程中侧倾中心高度 RCH 的变化）。

    数学推导：
      "到C与到A距离相等" 展开为关于 (y0,z0) 的线性方程：
        |P-C|² = |P-A|²  =>  2(yc-ya)*y0 + 2(zc-za)*z0 = yc²+zc²-ya²-za²
      同理"到A与到B距离相等"给出第二个线性方程，两式联立解出唯一交点 (y0, z0)。
      这本质上是"三点确定一个圆，求圆心"问题的线性化解法（用两条弦的垂直平分线
      的思想，但这里写成了距离相等的直接代数形式，效果等价）。

    参数:
        yc, zc: 点 C 的 y/z 坐标（相邻窗口起始点，通常是较早的接地点轨迹采样）
        ya, za: 点 A 的 y/z 坐标（窗口中间点）
        yb, zb: 点 B 的 y/z 坐标（窗口末尾点）

    返回:
        (y0, z0): 满足上述等距条件的辅助点坐标
    """
    a11 = 2.0 * (yc - ya)
    a12 = 2.0 * (zc - za)
    b1 = yc * yc + zc * zc - ya * ya - za * za

    a21 = 2.0 * (ya - yb)
    a22 = 2.0 * (za - zb)
    b2 = ya * ya + za * za - yb * yb - zb * zb

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise ValueError("Degenerate points for RCH solving (near-collinear in y-z projection)")

    y0 = (b1 * a22 - a12 * b2) / det
    z0 = (a11 * b2 - b1 * a21) / det
    return y0, z0


def run_simulation() -> dict[str, np.ndarray]:
    """主仿真流程：扫描轮跳行程，逐点求解姿态并计算全部运动学输出量。

    对应 RR_FIVE_LINK_SUSP.m 主脚本的整体逻辑：
      1. 定义标称硬点坐标（五连杆悬架的设计输入，本函数内硬编码了一组标称几何）；
      2. 按固定步长扫描轮心高度 Ctz，从回弹极限(REBstroke)到压缩极限(BOUstroke)；
      3. 每个 Ctz 用 fsolve 求解转向节姿态（调用 _xp_algorithm 作为残差函数）；
      4. 用求解出的姿态计算 toe/camber/接地点等瞬时量；
      5. 扫描完成后，用接地点轨迹的滑动窗口（每次跨 aa=int(40/step) 个点）计算
         侧倾中心高度(RCH)随行程变化的曲线。

    返回:
        dict，包含以下键（均为 numpy 数组，除 last_opt 是最后一步的求解诊断信息 dict）：
          Ctz_plot     : 相对轮心标称高度的行程(mm)，即 wc_stroke
          Ct_write     : 各行程点的轮心世界坐标 (N,3)
          KNU_write    : 各行程点 C1 参考点的世界坐标 (N,3)
          Blt_write..B5t_write : 各行程点 5 个连杆外点的世界坐标 (N,3) 各一个数组
          camber       : 外倾角(°) 曲线
          toe          : 前束角(°) 曲线
          CONT_PONT    : 各行程点的接地点世界坐标 (N,3)
          ab           : RCH 滑动窗口计算时对应的采样序号(1-based)
          RCH          : 侧倾中心高度(mm) 曲线
          WC_RCH       : RCH 曲线对应的行程点(取窗口中点)
          ycc/zcc/yaa/zaa/ybb/zbb : RCH 计算用到的三点 y/z 坐标，保留供调试/复核
          last_opt     : 最后一个行程点的 fsolve 求解诊断信息（收敛状态/残差等），
                         用于 main() 打印结果摘要
    """
    # MATLAB 脚本里的常量：轮跳扫描范围与步长
    BOUstroke = 100.0     # 压缩(bump)方向的最大行程(mm)
    REBstroke = -100.0    # 回弹(rebound)方向的最大行程(mm，负值)
    tire_radius = 352.0   # 轮胎静力半径(mm)
    step = 5.0            # 扫描步长(mm)

    # 标称硬点坐标（设计基准几何，单位 mm，车身固定坐标系）
    hp = HardPoints(
        A1=np.array([4313.0, -428.0, 1054.5], dtype=float),
        B1=np.array([4365.008, -715.386, 1065.326], dtype=float),
        A2=np.array([4588.0, -357.0, 1090.5], dtype=float),
        B2=np.array([4444.123, -697.692, 1111.512], dtype=float),
        A3=np.array([4242.0, -413.0, 863.5], dtype=float),
        B3=np.array([4391.608, -730.852, 801.842], dtype=float),
        A4=np.array([4709.907, -292.376, 865.386], dtype=float),
        B4=np.array([4535.801, -714.774, 889.46], dtype=float),
        A5=np.array([4316.0, -446.0, 936.5], dtype=float),
        B5=np.array([4292.353, -722.178, 943.276], dtype=float),
        C=np.array([4421.553, -825.1, 972.284], dtype=float),
        C1=np.array([4421.811, -745.143, 970.287], dtype=float),
        LUP=np.array([4630.495, -640.275, 1044.481], dtype=float),
        LLWR=np.array([4617.547, -667.336, 882.149], dtype=float),
        SCLP=np.array([4881.0, -480.0, 1002.5], dtype=float),
    )

    # 轮心目标高度的扫描区间：标称高度 + 行程上下限
    UPRBOU = BOUstroke + hp.C[2]
    LWRREB = REBstroke + hp.C[2]

    # 按 MATLAB 风格的"含端点"等步长扫描（+step*0.5 是为了在浮点误差下仍能取到终点）
    ctz_values = np.arange(LWRREB, UPRBOU + step * 0.5, step, dtype=float)

    # 各输出量的逐行程点累积列表，扫描结束后统一转成 numpy 数组
    Ctz_plot = []
    Ct_write = []
    KNU_write = []
    Blt_write = []
    B2t_write = []
    B3t_write = []
    B4t_write = []
    B5t_write = []
    camber = []
    toe = []
    CONT_PONT = []

    # fsolve 初始猜测值：[3个欧拉角初值=0, 轮心x初值, 轮心y初值]（取标称轮心的x/y）
    x_init = np.array([0.0, 0.0, 0.0, 4421.553, -825.1], dtype=float)
    last_opt: dict[str, float | int | np.ndarray] | None = None

    for Ctz in ctz_values:
        # 与 MATLAB 行为保持一致：每个 Ctz 步都使用同一个固定初始猜测（不用上一步的解
        # 作为热启动），这样每步求解相互独立，避免累积误差沿扫描方向传播。
        x, info, ier, msg = fsolve(
            func=lambda a: _xp_algorithm(a, Ctz, hp),
            x0=x_init,
            xtol=1e-12,
            maxfev=10000,
            full_output=True,
        )
        # SciPy 在 "xtol 设置过小" 时可能返回 ier=3，但此时残差往往已经足够小，
        # 这里额外用残差范数兜底判断，只要足够小就视为收敛成功（不严格依赖 ier==1）。
        residual_norm = float(np.linalg.norm(info["fvec"]))
        if ier not in (1, 3) or residual_norm > 1e-8:
            raise RuntimeError(
                f"fsolve failed at Ctz={Ctz:.6f}: ier={ier}, residual={residual_norm:.3e}, msg={msg}"
            )

        # 记录本次求解的诊断信息（循环结束后 last_opt 即为最后一个行程点的求解详情）
        last_opt = {
            "Ctz": float(Ctz),
            "x": x.copy(),
            "ier": int(ier),
            "residual_norm": residual_norm,
            "fvec": info["fvec"].copy(),
        }

        # 用收敛后的姿态参数 x，构造正式的旋转矩阵 E 与平移向量 Y
        # （此处用的是 _rotation_matrix_for_main，而不是求解迭代时用的
        #   _rotation_matrix_for_constraint，两者 d23 符号相反，见各自函数注释）
        E = _rotation_matrix_for_main(x)
        Y = np.array(
            [
                x[3] - np.dot(E[0, :], hp.C),
                x[4] - np.dot(E[1, :], hp.C),
                Ctz - np.dot(E[2, :], hp.C),
            ],
            dtype=float,
        )

        # 把转向节上的关键点变换到当前姿态下的世界坐标
        B11t = _transform_point(E, Y, hp.B1)
        B22t = _transform_point(E, Y, hp.B2)
        B33t = _transform_point(E, Y, hp.B3)
        B44t = _transform_point(E, Y, hp.B4)
        B55t = _transform_point(E, Y, hp.B5)
        C11t = _transform_point(E, Y, hp.C1)

        # 与 MATLAB 分支选择规则一致：对每个连杆外点，用固定长度约束重新精确求解 y 坐标
        # （取两个数学解中较小的一个），修正 fsolve 数值解在 y 方向上可能存在的病态误差。
        B11t[1] = _solve_y_with_fixed_length(B11t, hp.A1, np.linalg.norm(hp.B1 - hp.A1))
        B22t[1] = _solve_y_with_fixed_length(B22t, hp.A2, np.linalg.norm(hp.B2 - hp.A2))
        B33t[1] = _solve_y_with_fixed_length(B33t, hp.A3, np.linalg.norm(hp.B3 - hp.A3))
        B44t[1] = _solve_y_with_fixed_length(B44t, hp.A4, np.linalg.norm(hp.B4 - hp.A4))
        B55t[1] = _solve_y_with_fixed_length(B55t, hp.A5, np.linalg.norm(hp.B5 - hp.A5))

        Ct = np.array([x[3], x[4], Ctz], dtype=float)   # 当前姿态下的轮心世界坐标
        C1t = C11t.copy()

        Ctz_plot.append(Ctz - hp.C[2])     # 转成"相对标称高度的行程"，即 wc_stroke
        Ct_write.append(Ct)
        KNU_write.append(C1t)
        Blt_write.append(B11t)
        B2t_write.append(B22t)
        B3t_write.append(B33t)
        B4t_write.append(B44t)
        B5t_write.append(B55t)

        # 由轮心 Ct 与 C1 参考点的相对位置关系，推导 camber(外倾角) 和 toe(前束角)：
        #   camber: 侧视(y-z平面)方向的倾斜角，delta_z(垫直方向差)/delta_y(横向差) 的反正切
        #   toe:    俯视(x-y平面)方向的偏转角，delta_x(纵向差)/delta_y(横向差) 的反正切
        # 这里的 delta_y 是分母，代表车轮"厚度方向"的基准距离。
        delta_z = Ctz - C1t[2]
        delta_y = Ct[1] - C1t[1]
        delta_x = Ct[0] - C1t[0]
        camber.append(math.degrees(math.atan(delta_z / delta_y)))
        toe.append(math.degrees(math.atan(delta_x / delta_y)))

        # 求解轮胎接地点（见 _solve_contact_point 详细说明）
        cont = _solve_contact_point(Ct, C1t, tire_radius, Ctz)
        CONT_PONT.append(cont)

    CONT_PONT_arr = np.array(CONT_PONT, dtype=float)

    # ------------------------------------------------------------------- #
    # 侧视投影下的侧倾中心高度(Roll Center Height, RCH)计算
    # 用接地点轨迹上间隔 aa 个采样点的"三点窗口"，估计局部圆弧圆心的高度，
    # 近似代表车辆侧倾时车身绕之转动的瞬时中心高度随行程变化的规律。
    # ------------------------------------------------------------------- #
    num = int((UPRBOU - LWRREB) / step + 1)    # 总扫描点数
    aa = int(40 / step)                         # 三点窗口的跨度（对应约40mm行程的采样间隔）

    ab = []
    RCH = []
    ycc, zcc, yaa, zaa, ybb, zbb = [], [], [], [], [], []
    WC_RCH = []

    for i in range(0, num - 2 * aa):
        ab.append(i + 1)  # 与 MATLAB 保持一致，序号从 1 开始计数（1-based）

        # 取三个间隔 aa 的接地点采样(y,z坐标)，构成一个滑动窗口
        yc = CONT_PONT_arr[i, 1]
        zc = CONT_PONT_arr[i, 2]
        ya = CONT_PONT_arr[i + aa, 1]
        za = CONT_PONT_arr[i + aa, 2]
        yb = CONT_PONT_arr[i + 2 * aa, 1]
        zb = CONT_PONT_arr[i + 2 * aa, 2]

        ycc.append(yc)
        zcc.append(zc)
        yaa.append(ya)
        zaa.append(za)
        ybb.append(yb)
        zbb.append(zb)

        # 求三点等距辅助点 (y0,z0)，再用相似三角形关系反推出侧倾中心高度 h
        y00, z00 = _solve_rch_point(yc, zc, ya, za, yb, zb)
        h = ya * (z00 - za) / (ya - y00)
        RCH.append(h)
        WC_RCH.append(Ctz_plot[i + aa])     # RCH 曲线对应窗口中点的行程值

    return {
        "Ctz_plot": np.array(Ctz_plot, dtype=float),
        "Ct_write": np.array(Ct_write, dtype=float),
        "KNU_write": np.array(KNU_write, dtype=float),
        "Blt_write": np.array(Blt_write, dtype=float),
        "B2t_write": np.array(B2t_write, dtype=float),
        "B3t_write": np.array(B3t_write, dtype=float),
        "B4t_write": np.array(B4t_write, dtype=float),
        "B5t_write": np.array(B5t_write, dtype=float),
        "camber": np.array(camber, dtype=float),
        "toe": np.array(toe, dtype=float),
        "CONT_PONT": CONT_PONT_arr,
        "ab": np.array(ab, dtype=int),
        "RCH": np.array(RCH, dtype=float),
        "WC_RCH": np.array(WC_RCH, dtype=float),
        "ycc": np.array(ycc, dtype=float),
        "zcc": np.array(zcc, dtype=float),
        "yaa": np.array(yaa, dtype=float),
        "zaa": np.array(zaa, dtype=float),
        "ybb": np.array(ybb, dtype=float),
        "zbb": np.array(zbb, dtype=float),
        "last_opt": last_opt,
    }


def main() -> None:
    """命令行入口：运行一次完整的行程扫描仿真，并打印关键结果摘要（用于独立验证求解器本身，
    不涉及神经网络训练/推理；日常训练流程会绕过 main()，直接由 generate_dataset.py 之类的
    脚本导入并调用本文件里的底层函数）。"""
    result = run_simulation()

    print("Simulation complete.")
    print(f"Points count: {len(result['Ctz_plot'])}")
    print(f"Camber range (deg): {result['camber'].min():.6f} ~ {result['camber'].max():.6f}")
    print(f"Toe range (deg): {result['toe'].min():.6f} ~ {result['toe'].max():.6f}")
    print(f"RCH points: {len(result['RCH'])}")

    # 打印最后一个行程点的 fsolve 求解诊断信息，用于快速核对收敛状态是否正常
    last_opt = result["last_opt"]
    if last_opt is not None:
        x = last_opt["x"]
        fvec = last_opt["fvec"]
        print("Final optimization result (Python):")
        print(f"  Ctz = {last_opt['Ctz']:.6f}")
        print(f"  ier = {last_opt['ier']}")
        print(f"  residual_norm = {last_opt['residual_norm']:.12e}")
        print(
            "  x = "
            f"[{x[0]:.12f}, {x[1]:.12f}, {x[2]:.12f}, {x[3]:.12f}, {x[4]:.12f}]"
        )
        print(
            "  residual fvec = "
            f"[{fvec[0]:.12e}, {fvec[1]:.12e}, {fvec[2]:.12e}, {fvec[3]:.12e}, {fvec[4]:.12e}]"
        )


if __name__ == "__main__":
    main()
