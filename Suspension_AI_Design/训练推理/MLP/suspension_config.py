#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
悬架 AI 代理模型 · 共享配置模块。

集中存放整套「硬点 → 运动学曲线」神经网络方案里，各脚本都要用到的常量与列定义，
避免在 parse_excel_to_dataset.py（数据解析）、train_nn.py / train_nn_attention.py（训练）、
infer.py / infer_attention.py（推理）、visualize_results*.py（可视化）等多处重复硬编码，
一旦这些物理/工程常量需要调整，只需改这一处。

被以下脚本 import：
  - parse_excel_to_dataset.py：读取 STROKE_VAR/N_HARDPOINT/TARGET_NAMES 来解析 Excel 列
  - generate_dataset.py / merge_datasets.py：复用 TARGET_NAMES 保持列名一致
  - train_nn.py / train_nn_attention.py：读取 DESIGN_POS_TOL_MM 做设计位校验，
    presets_dict() 的返回值会存进 model.pt 的 meta，供推理时做「设计位刚体平移校正」
  - infer.py / infer_attention.py：读取 presets_dict() 里的预设值用于校正模型输出
"""

# --------------------------------------------------------------------------- #
# 模型预设常量（来自主机厂/设计输入需求，非拟合得出，用于校验与校正模型输出）
# --------------------------------------------------------------------------- #

# ① 模型轮胎静力半径（mm）：轮胎在静止承载状态下的滚动半径，用于换算接地点几何。
TIRE_STATIC_RADIUS_MM = 352.0

# ② 模型初始前束角（°）：在设计位（wc_stroke ≈ 0，即车辆静止/满载设计姿态）处，
#    前轮/后轮的前束应达到的目标值。正值表示前束（内束），负值表示外倾（开度）。
#    训练完成后会在“设计位校验”环节，对比模型在 wc_stroke≈0 处的预测值与该常量，
#    确认模型学到的整体趋势与设计输入一致（见 train_nn.py 的 design_position_validation）。
INIT_TOE_DEG = 0.2

#    模型初始外倾角（°）：设计位处车轮相对地面法线的倾斜角度目标值，负值表示车轮顶部向内倾。
INIT_CAMBER_DEG = -1.4

# --------------------------------------------------------------------------- #
# 数据列定义（与 Excel 原始表 / 求解器输出的列名/维度保持一致，供各脚本解析数据用）
# --------------------------------------------------------------------------- #

# 自变量列名：轮心沿 Z 向的行程（wheel-center stroke，简称 wc_stroke），单位 mm。
# 悬架运动学曲线本质上就是各输出量随这个行程变量的函数关系。
STROKE_VAR = "wc_stroke"

# 输入硬点坐标个数：33 = 11 个硬点 × 3 个坐标轴(x/y/z)。
# 11 个硬点对应五连杆悬架的上/下控制臂内外点、转向拉杆内外点、轮心等关键几何点
# （在 Excel 原表中对应 B..AH 列，在求解器 RR_FIVE_LINK_SUSP.py 中对应 A1..B5,C 命名）。
N_HARDPOINT = 33

# 8 个随轮跳行程(wc_stroke)变化的输出量（神经网络需要学习/拟合的目标）：
#   toe               前束角 (°)          —— 车轮绕Z轴（俯视）相对车身纵轴的偏转角
#   camber            外倾角 (°)          —— 车轮绕X轴（正视）相对地面法线的倾斜角
#   caster            后倾角 (°)          —— 主销轴在侧视投影下相对垂线的倾角
#   caster_arm        后倾拖距 (mm)       —— 主销轴与地面交点到接地点的纵向(X)距离
#   scrub_radius      主销偏移距 (mm)     —— 主销轴与地面交点到接地点的横向(Y)距离
#   tire_con_point_x  轮胎接地点 X 坐标 (mm)
#   tire_con_point_y  轮胎接地点 Y 坐标 (mm)
#   tire_con_point_z  轮胎接地点 Z 坐标 (mm)
# 顺序在全项目内保持固定，训练/推理/可视化脚本均按此顺序索引，不应随意调整。
TARGET_NAMES = [
    "toe", "camber", "caster", "caster_arm", "scrub_radius",
    "tire_con_point_x", "tire_con_point_y", "tire_con_point_z",
]

# 设计位判定阈值（mm）：|wc_stroke| 小于该值时，认为该采样点处于「接近设计位」，
# 用于从测试集里筛出 wc_stroke≈0 附近的行，与 INIT_TOE_DEG / INIT_CAMBER_DEG 预设对照，
# 检验模型在这一关键工况点的预测是否符合设计输入（而不是只看全局平均误差）。
DESIGN_POS_TOL_MM = 2.5


def presets_dict():
    """把上面的预设常量打包成 dict，随 meta.json / model.pt 一起保存。

    下游用途：
      - train_nn.py 训练完成后写入 metrics.json 的 "presets" 字段，方便复现实验时核对当时用的常量；
      - infer.py / infer_attention.py 推理时读取该 dict，对模型在设计位(stroke=0)处的
        toe/camber 预测值做「刚体平移校正」（只平移曲线整体水平，不改变形状/梯度），
        使推理结果在设计位处精确等于此处定义的预设值，消除训练带来的系统性偏差。

    返回:
        dict: {"tire_static_radius_mm": ..., "init_toe_deg": ..., "init_camber_deg": ...}
    """
    return {
        "tire_static_radius_mm": TIRE_STATIC_RADIUS_MM,
        "init_toe_deg": INIT_TOE_DEG,
        "init_camber_deg": INIT_CAMBER_DEG,
    }
