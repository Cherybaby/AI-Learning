#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 / 说明 dataset.npz 的数据格式。

- 终端打印：数组名、形状、dtype、每列含义（来自 meta.json）、样例行。
- 生成图 nn_out/dataset_format.png：①样例数据表 ②X各列取值范围 ③Y各列取值范围
  ④某一条几何(整条曲线)的 8 个量随 wc_stroke 的曲线（展示“长格式行→曲线”的关系）。
- 另导出样例 CSV：nn_out/dataset_sample_head.csv

用法：
  python3 inspect_dataset.py                       # 默认读 nn_data/
  python3 inspect_dataset.py --data nn_data --out nn_out
"""
from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

import numpy as np                    # noqa: E402
import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402


def main():
    """检视并可视化说明 dataset.npz 的数据格式，帮助人理解「长格式」数据集结构。

    流程：
      1. 解析命令行参数（--data 数据集目录、--out 图/CSV 输出目录），并确保输出目录存在。
      2. 加载 dataset.npz（含 X / Y / groups 三个 numpy 数组）与 meta.json（列名等元信息）。
      3. 在终端打印：各数组的形状/dtype、长格式含义说明、某条几何前 5 个行程点的样例行。
      4. 导出样例 CSV（前 20 行完整列）到 <out>/dataset_sample_head.csv。
      5. 生成 4 子图总览图 <out>/dataset_format.png：
           面板1 样例数据表、面板2 X 各列取值范围、面板3 Y 各列取值范围、
           面板4 一条几何的 8 条曲线（演示「长格式行 -> 曲线」的关系）。

    参数：
        无（通过 argparse 从命令行读取 --data / --out）。

    返回值：
        None。副作用为终端打印，并产出以下文件：
          - <out>/dataset_sample_head.csv  样例数据（前 20 行）
          - <out>/dataset_format.png       4 子图数据格式总览图
    """
    # ---------- 命令行参数 ----------
    ap = argparse.ArgumentParser()
    # --data：数据集目录，默认脚本同级的 nn_data/
    ap.add_argument("--data", default=os.path.join(_HERE, "nn_data"))
    # --out：图与 CSV 的输出目录，默认指向同级的 训练推理/nn_out
    ap.add_argument("--out", default=os.path.join(_HERE, "nn_data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)   # 输出目录不存在则创建

    # ---------- 加载数据与元信息 ----------
    # dataset.npz：一个压缩包，内含 X（输入）、Y（输出）、groups（几何编号）三个数组
    npz = np.load(os.path.join(args.data, "dataset.npz"))
    # meta.json：记录列名、几何数、行程点、取值范围等人类可读的元信息
    meta = json.load(open(os.path.join(args.data, "meta.json"), encoding="utf-8"))
    X, Y, groups = npz["X"], npz["Y"], npz["groups"]
    feat = meta["feat_names"]          # 34: 33 硬点 + wc_stroke（输入 X 的列名）
    tgt = meta["target_names"]         # 8（输出 Y 的列名，即 8 个运动学量）
    strokes = meta["strokes"]          # 预设的轮跳行程点列表（每条几何采样的 wc_stroke 值）

    # ---------- 终端说明 ----------
    print("=" * 70)
    print("dataset.npz 内含 3 个数组（NPZ = 多个 numpy 数组的压缩包）：")
    for k in npz.files:
        a = npz[k]
        print(f"  {k:8s} shape={str(a.shape):16s} dtype={a.dtype}")
    print("-" * 70)
    print(f"格式：长格式(long format)。每一行 = 某个几何(样本) 在某个 wc_stroke 点的一条记录。")
    print(f"总几何数={meta['n_samples']}  行数={X.shape[0]} (= 几何数 × {len(strokes)} 个行程点)")
    print(f"X[{X.shape[1]}] = 33 个硬点坐标 + wc_stroke(自变量)")
    print(f"   前 6 列: {feat[:6]} ...  末列: {feat[-1]}")
    print(f"Y[{Y.shape[1]}] = 8 个运动学量: {tgt}")
    print(f"groups = 每行所属几何编号(用于按曲线划分train/val/test)")
    print(f"wc_stroke 范围: {meta['wc_stroke_range']}  预设: {meta['presets']}")
    print("-" * 70)
    # 样例：第一条几何的前 5 行
    g0 = groups[0]                       # 取数据集中第一条几何的编号作为展示对象
    idx = np.where(groups == g0)[0]      # 找出属于该几何的所有行下标
    idx = idx[np.argsort(X[idx, -1])][:5]  # 按 wc_stroke（X 末列）升序排序后取前 5 行
    print(f"样例（几何 Run#{g0} 的前 5 个行程点）：")
    print("  wc_stroke |", "  ".join(f"{t}" for t in tgt))
    for i in idx:
        # 逐行打印：该行程点的 wc_stroke 值 + 对应的 8 个目标量真值
        print(f"  {X[i,-1]:9.2f} |", "  ".join(f"{v:.3f}" for v in Y[i]))
    print("=" * 70)

    # 导出样例 CSV（前 20 行完整列）
    import csv
    with open(os.path.join(args.out, "dataset_sample_head.csv"), "w", newline="", encoding="utf-8") as f:
        # 表头：group 编号 + 全部输入列名 + 全部输出列名
        w = csv.writer(f); w.writerow(["group"] + feat + tgt)
        for i in range(min(20, X.shape[0])):
            # 每行：几何编号 + 34 个输入值 + 8 个输出值（数值统一格式化为 5 位有效数字）
            w.writerow([int(groups[i])] + [f"{v:.5g}" for v in X[i]] + [f"{v:.5g}" for v in Y[i]])

    # ---------- 图形 ----------
    fig = plt.figure(figsize=(20, 12))

    # 面板1：样例数据表（几何g0前8行，选 wc_stroke + 8 目标）
    ax1 = fig.add_subplot(2, 2, 1); ax1.axis("off")   # 关闭坐标轴，纯做表格容器
    ax1.set_title("(1) Sample rows (one geometry, long format)", fontsize=12, loc="left")
    # 取几何 g0 的行，按 wc_stroke 升序后取前 8 个行程点
    idx8 = np.where(groups == g0)[0]; idx8 = idx8[np.argsort(X[idx8, -1])][:8]
    col = ["wc_stroke"] + tgt                          # 表格列：自变量 + 8 个目标量
    # 每行单元格：wc_stroke 值 + 8 个目标真值（格式化为字符串）
    cell = [[f"{X[i,-1]:.1f}"] + [f"{v:.3f}" for v in Y[i]] for i in idx8]
    tb = ax1.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.4)  # 固定字号并放大行高

    # 面板2：X 各列取值范围（min–max，log 不用；标注 wc_stroke）
    ax2 = fig.add_subplot(2, 2, 2)
    xmin, xmax = X.min(0), X.max(0)         # 每一列（跨所有行）的最小/最大值
    ypos = np.arange(len(feat))             # 每个输入特征对应一条水平线的 y 位置
    ax2.hlines(ypos, xmin, xmax, color="steelblue", lw=3)  # 用横线画出各列 min–max 区间
    ax2.plot(X.mean(0), ypos, "o", color="navy", ms=3)     # 用圆点标注各列均值
    ax2.set_yticks(ypos); ax2.set_yticklabels(feat, fontsize=6)
    ax2.invert_yaxis(); ax2.grid(True, axis="x", alpha=0.3)  # y 轴翻转使第一列在顶部
    ax2.set_title("(2) Input X: per-column min–mean–max (34 dims)", fontsize=12, loc="left")

    # 面板3：Y 各列取值范围（各自量纲差异大，用归一化条 + 标注真实范围）
    ax3 = fig.add_subplot(2, 2, 3)
    yy = np.arange(len(tgt))                # 每个输出量对应一根水平条
    ymin, ymax = Y.min(0), Y.max(0)         # 每个输出量的最小/最大值
    ax3.barh(yy, ymax - ymin, left=ymin, color="salmon", alpha=0.7)  # 条长=取值跨度，起点=最小值
    for i, t in enumerate(tgt):
        # 在每根条起点处标注真实 [min, max] 区间（因各量纲不同，文字标注更直观）
        ax3.text(ymin[i], i, f" [{ymin[i]:.2f}, {ymax[i]:.2f}]", va="center", fontsize=7)
    ax3.set_yticks(yy); ax3.set_yticklabels(tgt, fontsize=8); ax3.invert_yaxis()
    ax3.grid(True, axis="x", alpha=0.3)
    ax3.set_title("(3) Output Y: value range per quantity (8 dims)", fontsize=12, loc="left")

    # 面板4：一条几何的 8 条曲线（长格式行 -> 曲线）
    ax4 = fig.add_subplot(2, 2, 4)
    im = np.where(groups == g0)[0]; im = im[np.argsort(X[im, -1])]  # 该几何全部行，按行程升序
    st = X[im, -1]                          # 横坐标：wc_stroke 序列
    for j, t in enumerate(tgt):
        yv = Y[im, j]                       # 第 j 个目标量随行程变化的原始值
        yn = (yv - yv.min()) / (np.ptp(yv) + 1e-9)   # 归一化到[0,1]同图显示（+1e-9 防止除零）
        ax4.plot(st, yn, label=t, lw=1.5)
    ax4.set_xlabel("wc_stroke"); ax4.set_ylabel("normalized value")
    ax4.set_title(f"(4) One geometry Run#{g0}: 8 curves vs wc_stroke (normalized)", fontsize=12, loc="left")
    ax4.legend(fontsize=7, ncol=2); ax4.grid(True, alpha=0.3)

    # 总标题汇总各数组形状与「几何数 × 行程点数」的规模信息
    fig.suptitle(f"dataset.npz overview | X{X.shape} Y{Y.shape} groups{groups.shape} | "
                 f"{meta['n_samples']} geometries × {len(strokes)} strokes", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(args.out, "dataset_format.png")
    fig.savefig(p, dpi=110); plt.close(fig)   # 保存图并释放画布资源
    print(f"已保存图: {p}")
    print(f"已保存样例CSV: {os.path.join(args.out, 'dataset_sample_head.csv')}")


if __name__ == "__main__":
    main()
