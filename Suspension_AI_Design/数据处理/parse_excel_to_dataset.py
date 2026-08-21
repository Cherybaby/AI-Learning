#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 步：把 AI_HP.xlsx 解析成神经网络可直接使用的数据集。

输出（默认到 nn_data/）：
  - dataset.npz : X(N,34), Y(N,8), groups(N,)  —— 长格式
        X = [33 个硬点坐标] + [wc_stroke]（自变量），Y = 8 个运动学量
        groups = 每行所属的“样本(几何/整条曲线)”编号，供训练时按曲线划分数据集
  - meta.json   : 特征名、目标名、stroke 列表、样本数、wc_stroke 范围、
                  预设常量（轮胎静力半径/初始前束/初始外倾）、设计位实测统计等

用法：
  python3 parse_excel_to_dataset.py                       # 默认读 ./AI_HP.xlsx
  python3 parse_excel_to_dataset.py --xlsx AI_HP.xlsx --out nn_data
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

_TRAIN_DIR = os.path.join(os.path.dirname(_HERE), "训练推理")
if os.path.isdir(_TRAIN_DIR) and _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

import numpy as np                       # noqa: E402
from openpyxl import load_workbook       # noqa: E402
import suspension_config as cfg          # 位于 ../训练推理/  # noqa: E402


def parse_hardpoints_only(xlsx_path_or_stream):
    """仅提取 B..AH 列（33 个硬点坐标），不要求曲线真值列存在。

    用于批量推理场景：用户只有硬点坐标表（如新方案的 AI_HP.xlsx 同格式表），
    还没有（也不需要）实测的运动学曲线列。
    每行对应一组几何方案；跳过硬点不全的行。

    返回: (names: [33 个列名], rows: [[33 floats], ...], skipped: 跳过的行号列表(1-based，含表头))
    """
    wb = load_workbook(xlsx_path_or_stream, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    it = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(it)]
    input_cols = list(range(1, 1 + cfg.N_HARDPOINT))      # B..AH
    names = [header[c] for c in input_cols]

    rows, skipped = [], []
    for ri, row in enumerate(it, start=2):    # ri 从 2 开始计数，对应 Excel 里的真实行号（第1行是表头）
        if row is None:
            continue
        if len(row) <= max(input_cols):        # 行的实际列数不够（比 AH 列还短），说明这行是空行/损坏行
            skipped.append(ri)
            continue
        hp = [row[c] for c in input_cols]
        # 33 个坐标中只要有一个是 None（空单元格）或非数字类型（如误填了文字），整行跳过不用
        if any(v is None or not isinstance(v, (int, float)) for v in hp):
            skipped.append(ri)
            continue
        rows.append([float(v) for v in hp])
    wb.close()
    return names, rows, skipped


def parse(xlsx: str):
    """核心解析函数：把宽格式(wide format) Excel 表转换成神经网络训练所需的长格式(long format)数组。

    Excel 原始表结构（宽格式）：
      每一行 = 一组几何方案（一套硬点坐标），横向按 "<量名>[编号]" 命名的列存放该方案
      在不同轮跳行程点(编号对应 stroke 序号)下的 8 个运动学量取值，例如：
        toe[50], toe[51], ..., toe[150]        <- 前束角在 101 个行程点的值
        camber[50], camber[51], ..., camber[150]
      这种"一行一方案、多列按行程展开"的宽表适合人工在 Excel 里查看，但不适合直接喂给
      神经网络（网络的每次前向传播只处理"一组硬点 + 一个 wc_stroke -> 一组输出"）。

    长格式(long format)转换规则：
      把宽表"炸开"成多行：原来 1 行(1组硬点) × 101 个行程点 => 展开成 101 行，
      每行 = [33 个硬点坐标, 该行程点的 wc_stroke] -> [该行程点的 8 个输出量]。
      用 groups 数组记录每一行"炸开前属于哪一组几何"，方便训练时按整条曲线(同一 group)
      切分 train/val/test，避免同一条曲线的行程点分别落入训练集和测试集导致的数据泄漏。

    返回:
        X: (n_rows, 34) 硬点坐标(33) + wc_stroke(1)
        Y: (n_rows, 8)  8 个运动学量
        groups: (n_rows,) 每行所属的几何编号（原 Excel 行号/序号）
        feat_names: 长度 34 的列名列表（33 个硬点列名 + "wc_stroke"）
        strokes: 该 Excel 表实际包含的行程点编号列表（升序），如 [50, 51, ..., 150]
    """
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    it = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(it)]     # 第一行是表头，先取出列名

    # 硬点坐标固定占用 B..AH 列（Excel 里第 2~34 列，0-based 索引 1~33），共 33 列。
    input_cols = list(range(1, 1 + cfg.N_HARDPOINT))      # B..AH
    input_names = [header[c] for c in input_cols]

    # 扫描表头，把形如 "toe[50]" 的列名解析成 {量名: {行程编号: 列索引}} 的映射表，
    # 这样后面按 (量名, 行程编号) 就能直接查到对应的列号，不用每次都重新用正则匹配。
    qcol = defaultdict(dict)                              # <quantity>[idx] -> 列号
    for ci, name in enumerate(header):
        m = re.match(r"^(.*)\[(\d+)\]$", name)
        if m:
            qcol[m.group(1)][int(m.group(2))] = ci
    strokes = sorted(qcol[cfg.STROKE_VAR].keys())         # 该表实际有哪些行程编号，升序排列

    # 提前检查 8 个目标量的列是否齐全，缺列直接报错，避免后面处理到一半才发现数据不完整。
    missing = [t for t in cfg.TARGET_NAMES if t not in qcol]
    if missing:
        raise ValueError(f"Excel 缺少目标列: {missing}")

    X, Y, groups = [], [], []
    for si, row in enumerate(it):                          # si = 该几何方案在表中的行序号，同时作为 group id
        if row is None:
            continue
        hp = [row[c] for c in input_cols]                   # 取出这一行的 33 个硬点坐标
        if any(v is None for v in hp):                       # 硬点坐标不全（有空值）的行整行跳过
            continue
        for s in strokes:                                    # 对该几何的每一个行程点，展开成一行长格式记录
            wc = row[qcol[cfg.STROKE_VAR][s]]                # 该行程点对应的 wc_stroke 真实数值
            yv = [row[qcol[t][s]] for t in cfg.TARGET_NAMES] # 该行程点对应的 8 个输出量取值
            if wc is None or any(v is None for v in yv):      # 该行程点数据不全则只跳过这一个点，不影响同几何其它点
                continue
            X.append([float(v) for v in hp] + [float(wc)])
            Y.append([float(v) for v in yv])
            groups.append(si)
    wb.close()

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    feat_names = input_names + [cfg.STROKE_VAR]
    return X, Y, groups, feat_names, strokes


def design_position_stats(X, Y):
    """统计"设计位"(|wc_stroke| < DESIGN_POS_TOL_MM，即接近轮心行程为0的静态工况)处
    toe/camber 的实测均值，用于和 suspension_config.py 里的设计预设常量做对照，
    快速判断解析出来的数据在这个关键工况点是否符合工程输入（而非等到训练完才发现问题）。
    """
    wc = X[:, -1]
    mask = np.abs(wc) < cfg.DESIGN_POS_TOL_MM
    out = {"n_points": int(mask.sum()), "tol_mm": cfg.DESIGN_POS_TOL_MM}
    if mask.any():
        for name in ("toe", "camber"):
            j = cfg.TARGET_NAMES.index(name)
            out[f"{name}_mean"] = float(Y[mask, j].mean())
    return out


def detect_anomalous_groups(X, Y, groups, zscore=3.0, jump_k=6.0):
    """对每个几何(group)做异常检测，返回应剔除的 group id 集合 + 诊断信息。

    检测维度（与 analyze_data_anomalies.py 一致的方法论，供 parse 阶段直接复用）：
      1. 曲线跳变：同一几何内某曲线的二阶差分远超全体样本的正常跳动基准 -> 数据噪声/野点。
      2. 曲线整体幅值离群：该几何 8 条曲线的均值向量相对全体样本的 z-score 超阈值。
      3. 设计位偏离：|wc_stroke|<DESIGN_POS_TOL_MM 处 toe/camber 与预设常量偏差的 z-score 超阈值。

    返回: (bad_group_ids: set[int], reasons: dict[int, list[str]])
    """
    target_names = cfg.TARGET_NAMES
    uniq = np.unique(groups)
    j_toe, j_camber = target_names.index("toe"), target_names.index("camber")
    preset = cfg.presets_dict()

    # 按几何切片，预先缓存每个几何的 (stroke 排序后的) Y 曲线与设计位均值，
    # 避免下面三个检测维度各自重复做同样的切片/排序操作。
    per_group_Y, per_group_stroke, per_group_mean = {}, {}, {}
    for g in uniq:
        idx = np.where(groups == g)[0]
        order = np.argsort(X[idx, -1])          # 按 wc_stroke 升序排列，保证曲线是按行程顺序连续的
        idx = idx[order]
        per_group_Y[g] = Y[idx]
        per_group_stroke[g] = X[idx, -1]
        per_group_mean[g] = Y[idx].mean(0)

    # ---- 检测维度①：幅值离群 —— 该几何 8 条曲线的均值向量，相对全体几何的 z-score ----
    Ymean_all = np.stack([per_group_mean[g] for g in uniq])         # (n_geo, 8)
    y_mean, y_std = Ymean_all.mean(0), Ymean_all.std(0)
    y_std_safe = np.where(y_std < 1e-9, 1.0, y_std)                  # 防止某量标准差趋近0导致除零放大
    y_z = np.abs((Ymean_all - y_mean) / y_std_safe)               # (n_geo, 8)
    y_outlier_maxz = y_z.max(1)                     # 每个几何取8个量里最严重的那个z-score
    y_worst_target = np.array(target_names)[y_z.argmax(1)]          # 对应是哪个量最异常

    # ---- 检测维度②：曲线跳变 —— 每个目标量先算全体几何的二阶差分标准差作为"正常波动"基准，
    #      再看某个几何在该量上的最大二阶差分是否远超这个基准（说明局部有突然的数值抽风）----
    jump_ratio = {g: 0.0 for g in uniq}
    jump_target = {g: "" for g in uniq}
    for j, tname in enumerate(target_names):
        # 汇总全体几何在该量上的二阶差分，取标准差作为"正常波动幅度"的基准值
        d2_all = [np.diff(per_group_Y[g][:, j], n=2) for g in uniq if len(per_group_Y[g]) >= 3]
        d2_all = np.concatenate(d2_all) if d2_all else np.array([0.0])
        base_std = d2_all.std()
        base_std = base_std if base_std > 1e-9 else 1.0
        for g in uniq:
            y = per_group_Y[g][:, j]
            if len(y) < 3:                          # 二阶差分至少需要3个点才能算
                continue
            ratio = float(np.abs(np.diff(y, n=2)).max() / base_std)
            if ratio > jump_ratio[g]:                # 记录该几何在所有8个量里"跳变最严重"的那一次
                jump_ratio[g] = ratio
                jump_target[g] = tname

    # ---- 检测维度③：设计位偏离 —— |wc_stroke|<容差 处的 toe/camber 实测值，
    #      与预设常量(cfg.INIT_TOE_DEG/INIT_CAMBER_DEG)的偏差，相对全体几何做z-score ----
    design_dev = {}
    for g in uniq:
        mask = np.abs(per_group_stroke[g]) < cfg.DESIGN_POS_TOL_MM
        if mask.any():
            toe_dev = abs(float(per_group_Y[g][mask, j_toe].mean()) - preset["init_toe_deg"])
            camber_dev = abs(float(per_group_Y[g][mask, j_camber].mean()) - preset["init_camber_deg"])
            design_dev[g] = max(toe_dev, camber_dev)     # 取 toe/camber 两者里偏离更严重的一个
    if design_dev:
        dd_vals = np.array(list(design_dev.values()))
        dd_mean, dd_std = dd_vals.mean(), dd_vals.std()
        dd_std_safe = dd_std if dd_std > 1e-9 else 1.0
    else:
        dd_mean, dd_std_safe = 0.0, 1.0

    # ---- 汇总三个维度的判定结果：任一维度超过阈值即判定该几何为异常，记录全部触发原因 ----
    bad, reasons = set(), {}
    for i, g in enumerate(uniq):
        rs = []
        if jump_ratio[g] > jump_k:
            rs.append(f"曲线跳变(量={jump_target[g]}, ratio={jump_ratio[g]:.1f}>{jump_k})")
        if y_outlier_maxz[i] > zscore:
            rs.append(f"曲线幅值离群(量={y_worst_target[i]}, z={y_outlier_maxz[i]:.2f}>{zscore})")
        if g in design_dev:
            dz = abs((design_dev[g] - dd_mean) / dd_std_safe)
            if dz > zscore:
                rs.append(f"设计位偏离预设(偏差={design_dev[g]:.2f}°, z={dz:.2f}>{zscore})")
        if rs:
            bad.add(int(g))
            reasons[int(g)] = rs
    return bad, reasons


def main():
    ap = argparse.ArgumentParser(description="解析 AI_HP.xlsx 为 NN 数据集")
    ap.add_argument("--xlsx", default=os.path.join(_HERE, "AI_HP.xlsx"))
    ap.add_argument("--out", default=os.path.join(_HERE, "nn_data"))
    ap.add_argument("--no-filter", action="store_true",
                     help="跳过异常数据剔除，保留全部解析出的行(默认会剔除)")
    ap.add_argument("--zscore", type=float, default=3.0,
                     help="异常剔除的 z-score 阈值(默认3.0，越小越严格)")
    ap.add_argument("--jump-k", type=float, default=6.0,
                     help="曲线跳变剔除的比例阈值(默认6.0，越小越严格)")
    args = ap.parse_args()

    print(f"解析 {args.xlsx} …", flush=True)
    # 第一步：把宽格式 Excel 表解析成长格式 (X, Y, groups)，见 parse() 详细说明
    X, Y, groups, feat_names, strokes = parse(args.xlsx)
    n_parsed = len(np.unique(groups))
    print(f"样本(曲线)数={n_parsed}  展开行={X.shape[0]}  "
          f"输入维={X.shape[1]}  输出维={Y.shape[1]}  stroke点数={len(strokes)}")

    # 第二步（默认开启）：对解析出的每条曲线做异常检测，剔除数据质量有问题的几何方案，
    # 避免训练时被少数"跳变/离群/严重偏离设计位"的坏样本拖累模型学习效果。
    removed_log = []
    if not args.no_filter:
        bad_groups, reasons = detect_anomalous_groups(
            X, Y, groups, zscore=args.zscore, jump_k=args.jump_k)
        if bad_groups:
            keep_mask = ~np.isin(groups, list(bad_groups))     # 保留"不在坏几何列表里"的所有行
            for g in sorted(bad_groups):
                removed_log.append({"group_id": g, "reasons": reasons[g]})
            X, Y, groups = X[keep_mask], Y[keep_mask], groups[keep_mask]
            print(f"\n⚠️ 异常检测剔除 {len(bad_groups)}/{n_parsed} 组几何 "
                  f"(z-score阈值={args.zscore}, 跳变阈值={args.jump_k}):", flush=True)
            for item in removed_log:
                print(f"  几何#{item['group_id']}: " + "; ".join(item["reasons"]), flush=True)
        else:
            print("\n异常检测：未发现需剔除的异常几何。", flush=True)
    else:
        print("\n--no-filter 已指定，跳过异常剔除。", flush=True)

    # 第三步：保存最终数据集 dataset.npz（压缩的 numpy 数组包）+ meta.json（元信息，
    # 供 train_nn.py / infer.py 等下游脚本读取列名/目标名/预设常量等）。
    n_samples = len(np.unique(groups))
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "dataset.npz"),
                        X=X, Y=Y, groups=groups)

    dps = design_position_stats(X, Y)
    meta = {
        "feat_names": feat_names,
        "target_names": cfg.TARGET_NAMES,
        "stroke_var": cfg.STROKE_VAR,
        "strokes": strokes,
        "n_samples": n_samples,
        "n_rows": int(X.shape[0]),
        "n_inputs": int(X.shape[1]),
        "n_outputs": int(Y.shape[1]),
        "wc_stroke_range": [float(X[:, -1].min()), float(X[:, -1].max())],
        "presets": cfg.presets_dict(),
        "design_position_measured": dps,
        "anomaly_filter": {
            "enabled": not args.no_filter,
            "zscore_threshold": args.zscore,
            "jump_k_threshold": args.jump_k,
            "n_parsed_geo": n_parsed,
            "n_removed_geo": len(removed_log),
            "n_kept_geo": n_samples,
            "removed": removed_log,
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 剔除日志单独存一份，方便人工复核（不用每次都翻 meta.json）
    if removed_log:
        log_path = os.path.join(args.out, "removed_rows_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(removed_log, f, ensure_ascii=False, indent=2)
        print(f"已保存剔除日志: {log_path}")

    print(f"已保存: {os.path.join(args.out, 'dataset.npz')}")
    print(f"已保存: {os.path.join(args.out, 'meta.json')}")
    # 预设 vs 实测 设计位对照（信息性，帮助人工确认解析结果与设计输入一致）
    p = cfg.presets_dict()
    print("\n预设常量：轮胎静力半径={}mm  初始前束={}°  初始外倾={}°".format(
        p["tire_static_radius_mm"], p["init_toe_deg"], p["init_camber_deg"]))
    if "toe_mean" in dps:
        print("设计位(|wc_stroke|<{}mm)实测：toe≈{:.3f}°(预设{}°)  camber≈{:.3f}°(预设{}°)"
              .format(dps["tol_mm"], dps["toe_mean"], p["init_toe_deg"],
                      dps["camber_mean"], p["init_camber_deg"]))


if __name__ == "__main__":
    main()
