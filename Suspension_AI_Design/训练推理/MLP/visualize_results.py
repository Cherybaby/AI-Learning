#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化训练结果：从 nn_out/model.pt + metrics.json 生成多张图。
  - results_metrics.png : 各量 R²(train/val/test) 分组柱状 + RMSE/MAE 表
  - results_parity.png  : 测试集 预测 vs 真实 散点(parity)，每量一格，越贴对角线越好
  - results_curves.png  : 若干条测试几何的 预测 vs 真实 曲线对照（每量一格叠多条）
  - results_residual.png: 测试集各量残差(预测-真实)分布直方图
  - results_curve_table.png : 逐量详细指标表(MAE/Median/P90/MaxError/D2)，与 metrics.json 的
    test_curve_metrics 对应，直观展示"是否有部分量学得特别好，把整体均值拉低了对比效果"
    同时给出"全数据集(train+val+test 全部曲线)"版本的对照，避免只看测试集94条样本的统计噪声
  - results_worst_samples.png / .json : 各量按"逐条曲线自己的 MAE"排序的最差 Top-N 样本
    （metrics.json 里的 MAE 是全部测试曲线揉在一起算的汇总值，看不出是否由少数离群样本主导；
    这里逐条曲线单独算 MAE 再排序，可回答"是否存在个别样本把整体指标拖累"这类问题）
    默认统计全数据集(train+val+test)的全部曲线，可用 --scope test 只看测试集

用法：
  python3 visualize_results.py                # 默认统计全部数据(train+val+test)
  python3 visualize_results.py --scope test    # 只统计测试集(与之前版本一致)
  python3 visualize_results.py --topk 5        # 每个量列出最差 5 条曲线(默认5)
"""
from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

import numpy as np                    # noqa: E402
import torch                          # noqa: E402
import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import train_nn as tn                 # 复用模型加载/数据/推理  # noqa: E402


def _setup_chinese_font():
    """自动探测系统里可用的中文字体，避免新增的中文图表(表格/最差样本图)标题乱码。
    找不到任何中文字体时静默跳过，不影响图表生成（原有4张图标题本来就是英文，不受影响）。"""
    import matplotlib.font_manager as fm
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                  "Noto Sans CJK JP", "Source Han Sans SC", "WenQuanYi Zen Hei",
                  "Heiti SC", "STHeiti", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


_CJK_FONT = _setup_chinese_font()


def per_curve_mae(model, xs, ys, X, Y, groups, curve_ids, names):
    """逐条曲线(逐几何)各自算一次 MAE，而非把所有曲线的行混在一起算。
    curve_ids: 要统计的几何编号集合（可以是仅测试集，也可以是 train+val+test 全部曲线）。
    返回 dict: {name: [(sample_id, mae), ...]}（未排序，按 curve_ids 顺序）。
    用于回答"整体 MAE/R² 是否被少数离群样本主导，还是普遍中等偏差"。"""
    per_curve = {n: [] for n in names}
    for sid in curve_ids:
        idx = np.where(groups == sid)[0]
        idx = idx[np.argsort(X[idx, -1])]
        yp = tn.pred_orig(model, xs, ys, X[idx])
        yt = Y[idx]
        ae = np.abs(yp - yt)                      # (n_stroke, n_out)
        for j, n in enumerate(names):
            per_curve[n].append((int(sid), float(ae[:, j].mean())))
    return per_curve


def curve_metrics_from_per_curve(per_curve, names):
    """把 per_curve_mae() 的逐曲线 MAE，汇总成与 train_nn.curve_error_metrics 同口径的
    {name: {MAE, Median, P90, MaxError}} 统计（按"逐曲线MAE"汇总，而非按行汇总；
    两者数值上接近但不完全相同——本函数是"先按曲线聚合再统计"，metrics.json 里的是
    "所有行直接混合统计"，都是合理的口径，这里保持与本文件的 per_curve 分析口径一致）。
    不含 D2(平滑度)，D2 定义依赖曲线内部相邻点，无法从单个 MAE 值反推。"""
    out = {}
    for n in names:
        maes = np.array([m for _, m in per_curve[n]])
        out[n] = {"MAE": float(maes.mean()), "Median": float(np.median(maes)),
                  "P90": float(np.percentile(maes, 90)), "MaxError": float(maes.max())}
    return out


def main():
    """主流程：加载 MLP 模型(model.pt)与 metrics.json，对测试集做推理，依次生成 6 张分析图。

    命令行参数：
      --out   : 输出目录，默认 train_nn.OUT_DIR（即 nn_out），图表与 json 均写入此处。
      --data  : 数据集目录，默认沿用训练时的 nn_data；指定后会覆盖 train_nn.DATA_DIR。
      --topk  : 图6 中每个量列出的"最差曲线"条数，默认 5。
      --scope : 图5/图6 的统计范围。all=对 train+val+test 全部曲线逐条统计（默认，
                样本更多、统计更稳，并可通过对比训练/测试几何的差异观察过拟合）；
                test=仅统计测试集（复用 metrics.json 中预先算好的 test_curve_metrics）。

    产出文件（均写入 --out 目录）：
      results_metrics.png / results_parity.png / results_curves.png /
      results_residual.png / results_curve_table.png /
      results_worst_samples.png / results_worst_samples.json

    无返回值，结果以图片和 json 文件形式落盘，并在 stdout 打印保存路径与逐量统计。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=tn.OUT_DIR)
    ap.add_argument("--data", default=None, help="数据集目录(默认与训练一致 nn_data)")
    ap.add_argument("--topk", type=int, default=5, help="每个量列出最差 N 条曲线(默认5)")
    ap.add_argument("--scope", choices=["all", "test"], default="all",
                    help="图5/图6 的统计范围：all=train+val+test 全部曲线(默认)，test=仅测试集")
    args = ap.parse_args()
    if args.data:
        tn.DATA_DIR = __import__('os').path.abspath(args.data)
    os.makedirs(args.out, exist_ok=True)

    # 加载训练产物：model=已训练网络，xs/ys=输入/输出的标准化器，meta=元信息(含各量名称)
    model, xs, ys, meta = tn.load_bundle()
    names = meta["target_names"]              # 8 个运动学量的名称列表，决定后续所有子图的排布顺序
    # 读取训练阶段已算好的整体指标(train/val/test 的 R²/RMSE/MAE 等)，图1/图2/图4 直接引用其数值
    metrics = json.load(open(os.path.join(args.out, "metrics.json"), encoding="utf-8"))

    # 载入全量长格式数据并按几何分组划分 train/val/test（与训练时同一套 split，保证测试集口径一致）
    X, Y, groups, _, _ = tn.load_data()
    tr, va, te, _ = tn.split_by_group(groups)
    # 对测试集样本前向推理，并用 ys.inv 把标准化输出还原回物理量纲，得到测试集预测 yte 与真实 Yte
    with torch.no_grad():
        yte = ys.inv(model(torch.tensor(xs.tf(X[te]), dtype=torch.float32)).numpy())
    Yte = Y[te]

    # ---------- 图1：指标 ----------
    # 目的：一眼看清每个量在 train/val/test 三个集合上的 R²，判断整体拟合水平与泛化差距。
    # 解读：柱越高越好；红色虚线是 0.98 目标线；同一量的 train 柱明显高于 test 柱 = 过拟合信号。
    # 右侧表格列出测试集每个量的 R²/RMSE/MAE，便于精确查数而非只看柱状高度。
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7))
    r2 = {s: [metrics[s][n]["R2"] for n in names] for s in ("train", "val", "test")}
    xpos = np.arange(len(names)); w = 0.26
    for k, s in enumerate(("train", "val", "test")):
        a1.bar(xpos + (k - 1) * w, r2[s], w, label=f"{s} (mean={metrics[s]['_mean_R2']:.3f})")
    a1.axhline(metrics.get("target_mean_R2", 0.98), color="r", ls="--", lw=1, label="target 0.98")
    a1.set_xticks(xpos); a1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a1.set_ylabel("R²"); a1.set_ylim(min(0, min(r2["test"]) - 0.1), 1.05)
    a1.set_title("Per-quantity R² (train/val/test)"); a1.legend(fontsize=8); a1.grid(axis="y", alpha=0.3)

    a2.axis("off")
    rows = [[n, f"{metrics['test'][n]['R2']:.3f}", f"{metrics['test'][n]['RMSE']:.4g}",
             f"{metrics['test'][n]['MAE']:.4g}"] for n in names]
    tb = a2.table(cellText=rows, colLabels=["quantity", "test R²", "test RMSE", "test MAE"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
    a2.set_title("Test-set metrics per quantity", loc="left")
    fig.suptitle(f"Training results | test mean R²={metrics['test']['_mean_R2']:.3f} "
                 f"| {'PASS' if metrics.get('passed') else 'not reaching 0.98 (data-limited)'}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(args.out, "results_metrics.png"), dpi=110); plt.close(fig)

    # ---------- 图2：parity 散点 ----------
    # 目的：把测试集"预测值 vs 真实值"画成散点，直观检查预测偏差的形态与是否存在系统性偏移。
    # 解读：点越贴近红色对角线(y=x)越准；整体偏离对角线=有系统性高估/低估；点云越发散=随机误差越大。
    # 为控制绘图规模，用固定随机种子(0)最多抽样 3000 个点，保证每次运行抽样一致、图可复现。
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Yte), size=min(3000, len(Yte)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        yt, yp = Yte[sel, i], yte[sel, i]
        ax.scatter(yt, yp, s=5, alpha=0.3, color="steelblue")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{n} (R²={metrics['test'][n]['R2']:.3f})")
        ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.grid(alpha=0.3)
    fig.suptitle("Test-set parity: predicted vs actual (closer to red line = better)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_parity.png"), dpi=110); plt.close(fig)

    # ---------- 图3：多条测试曲线对照 ----------
    # 目的：抽取 4 条测试几何，画出 8 个量随轮跳行程(wc_stroke)变化的完整曲线，检验模型是否
    #       还原了曲线的形状/趋势/拐点，而不仅是逐点误差小。
    # 解读：实线=真实曲线，虚线=预测曲线，同色为同一条几何；虚线越贴实线越好，
    #       形状/斜率对不上说明模型没学到该量的物理规律。
    te_ids = np.unique(groups[te])[:4]        # 取测试集里前 4 条几何编号做对照
    colors = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        for c, sid in zip(colors, te_ids):
            # 取出该几何的所有行并按轮跳行程升序排列，保证曲线沿 x 轴平滑连续
            idx = np.where(groups == sid)[0]; idx = idx[np.argsort(X[idx, -1])]
            st = X[idx, -1]                    # 该几何的轮跳行程序列(曲线横坐标)
            # predict_curves：固定硬点坐标(X[...,:-1])，沿行程 st 批量推理整条曲线
            yp = tn.predict_curves(model, xs, ys, X[idx[0], :-1], st)
            ax.plot(st, Y[idx, i], color=c, lw=1.6)      # 真实曲线：实线
            ax.plot(st, yp[:, i], color=c, ls="--", lw=1.4)  # 预测曲线：同色虚线
        ax.set_title(n); ax.set_xlabel("wc_stroke"); ax.grid(alpha=0.3)
    axes[0][0].plot([], [], "k-", label="actual"); axes[0][0].plot([], [], "k--", label="predicted")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("4 test geometries: actual (solid) vs predicted (dashed)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_curves.png"), dpi=110); plt.close(fig)

    # ---------- 图4：残差分布 ----------
    # 目的：统计测试集每个量的残差(预测-真实)分布，检查误差是否零均值、对称、无长尾。
    # 解读：理想残差应集中在红色 0 线附近且左右对称；整体偏离 0=系统性偏差；
    #       出现长尾/双峰=部分样本误差异常大，需结合图6 定位离群曲线。
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        res = yte[:, i] - Yte[:, i]
        ax.hist(res, bins=60, color="mediumpurple", alpha=0.8)
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.set_title(f"{n}  MAE={metrics['test'][n]['MAE']:.3g}")
        ax.set_xlabel("residual (pred - actual)"); ax.grid(alpha=0.3)
    fig.suptitle("Test-set residual distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_residual.png"), dpi=110); plt.close(fig)

    # ---------- 图5：逐量详细指标表 (MAE/Median/P90/MaxError[/D2]) ----------
    # 默认统计范围(--scope all)：train+val+test 全部曲线，而不仅是 metrics.json 里
    # test_curve_metrics 覆盖的测试集94条曲线；用全量数据能避免测试集样本太少导致的统计噪声，
    # 也能看出模型在"见过的训练几何"上是否明显好于"未见过的测试几何"(过拟合信号)。
    # --scope test 时行为与之前版本一致，直接复用 metrics.json 里预先算好的 test_curve_metrics。
    if args.scope == "test":
        scope_ids = np.unique(groups[te])
        cm = metrics.get("test_curve_metrics", {})
        scope_label = f"测试集 {len(scope_ids)} 条曲线（与 metrics.json 一致）"
    else:
        scope_ids = np.unique(groups)
        print(f"逐条曲线计算 MAE 中（全数据集共 {len(scope_ids)} 条曲线，含 train+val+test）...", flush=True)
        _per_curve_all = per_curve_mae(model, xs, ys, X, Y, groups, scope_ids, names)
        cm = curve_metrics_from_per_curve(_per_curve_all, names)
        scope_label = f"全数据集 {len(scope_ids)} 条曲线（train+val+test 全部，非仅测试集）"

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(names) + 1.5))
    ax.axis("off")
    rows = []
    for n in names:
        d = cm.get(n, {})
        rows.append([tn.Q_LABEL.get(n, n),
                     f"{d.get('MAE', float('nan')):.5g}",
                     f"{d.get('Median', float('nan')):.5g}",
                     f"{d.get('P90', float('nan')):.5g}",
                     f"{d.get('MaxError', float('nan')):.5g}",
                     f"{d.get('D2', float('nan')):.5g}" if "D2" in d else "-"])
    tb = ax.table(cellText=rows, colLabels=["曲线 / Quantity", "MAE", "Median", "P90", "MaxError", "平滑度D2"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.8)
    ax.set_title(f"逐量详细指标 · 统计范围：{scope_label}", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "results_curve_table.png"), dpi=110); plt.close(fig)

    # ---------- 图6 + JSON：逐条曲线单独算 MAE，找出每个量的最差 Top-K 样本 ----------
    # 目的：区分"整体普遍偏差中等"还是"少数离群样本把汇总 MAE/R² 拖低"。
    # 默认(--scope all)覆盖 train+val+test 全部曲线；--scope test 时只看测试集(94条)。
    if args.scope == "test":
        per_curve = per_curve_mae(model, xs, ys, X, Y, groups, scope_ids, names)
    else:
        per_curve = _per_curve_all   # 复用图5已经算好的全量结果，避免重复推理

    scope_desc = "全数据集" if args.scope == "all" else "测试集"
    worst = {}
    for n in names:
        lst = sorted(per_curve[n], key=lambda t: -t[1])   # 按 MAE 降序
        worst[n] = lst[:args.topk]
        all_mae = np.array([m for _, m in per_curve[n]])
        # 剔除最差 10% 样本后的 MAE，量化离群样本对整体指标的拉高/拉低作用
        k10 = max(1, int(0.1 * len(all_mae)))
        trimmed = np.sort(all_mae)[:-k10] if len(all_mae) > k10 else all_mae
        print(f"  {tn.Q_LABEL.get(n, n):<20} {scope_desc}MAE={all_mae.mean():.4g}  "
              f"剔除最差10%({k10}条)后MAE={trimmed.mean():.4g}  "
              f"最差样本 Run#{worst[n][0][0]}(MAE={worst[n][0][1]:.4g})", flush=True)

    with open(os.path.join(args.out, "results_worst_samples.json"), "w", encoding="utf-8") as f:
        json.dump({n: [{"sample_id": sid, "mae": mae} for sid, mae in worst[n]] for n in names},
                  f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        all_mae = np.array([m for _, m in per_curve[n]])
        ax.hist(all_mae, bins=40, color="darkorange", alpha=0.75)
        for sid, mae in worst[n]:
            ax.axvline(mae, color="r", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(f"{n}\n均值MAE={all_mae.mean():.3g} | 最差5条均值={np.mean([m for _, m in worst[n]]):.3g}",
                     fontsize=9)
        ax.set_xlabel(f"单条曲线的 MAE（{scope_desc}）"); ax.set_ylabel("曲线条数"); ax.grid(alpha=0.3)
    fig.suptitle(f"每条曲线各自的 MAE 分布 · 统计范围：{scope_label}\n"
                 f"（红虚线=最差{args.topk}条位置，右侧长尾=离群样本）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out, "results_worst_samples.png"), dpi=110); plt.close(fig)

    for f in ("results_metrics.png", "results_parity.png", "results_curves.png", "results_residual.png",
              "results_curve_table.png", "results_worst_samples.png", "results_worst_samples.json"):
        print("已保存:", os.path.join(args.out, f))


if __name__ == "__main__":
    main()
