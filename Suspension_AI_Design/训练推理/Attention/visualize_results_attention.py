#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化训练结果（Attention/Seq2Seq 模型版）：从 nn_out_attention/model.pt + metrics.json 生成多张图。
与 visualize_results.py 的区别：本脚本针对 train_nn_attention.py 训练的 Seq2SeqAttnKC 模型
（规整序列数据，按几何分组，接口是 load_data_seq/pred_seq/split_geo/load_bundle(5返回值)）。
图表种类、文件名与 visualize_results.py 完全一致，只是数据来源换成 attention 模型。

  - results_metrics.png : 各量 R²(train/val/test) 分组柱状 + RMSE/MAE 表
  - results_parity.png  : 测试集 预测 vs 真实 散点(parity)，每量一格，越贴对角线越好
  - results_curves.png  : 若干条测试几何的 预测 vs 真实 曲线对照（每量一格叠多条）
  - results_residual.png: 测试集各量残差(预测-真实)分布直方图
  - results_curve_table.png : 逐量详细指标表(MAE/Median/P90/MaxError[/D2])
    默认统计"全数据集(train+val+test 全部曲线)"，避免只看测试集样本数太少的统计噪声
  - results_worst_samples.png / .json : 各量按"逐条曲线自己的 MAE"排序的最差 Top-N 样本

用法：
  cd /workspace/Suspension_AI_Design/训练推理
  python3 visualize_results_attention.py                # 默认统计全部数据(train+val+test)
  python3 visualize_results_attention.py --scope test    # 只统计测试集
  python3 visualize_results_attention.py --topk 5        # 每个量列出最差 5 条曲线(默认5)
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
import train_nn_attention as tn       # 复用模型加载/数据/推理  # noqa: E402


def _setup_chinese_font():
    """自动探测系统里可用的中文字体，避免中文图表(表格/最差样本图)标题乱码。
    找不到任何中文字体时静默跳过，不影响图表生成。"""
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


def per_curve_mae_seq(model, xs, ys, ss, hp, stroke, Yseq, curve_ids, gids, names):
    """逐条曲线(逐几何)各自算一次 MAE —— Attention/Seq2Seq 序列模型版。

    与 MLP 版 per_curve_mae 的差异：数据是规整序列(每条几何一整段 T 步)，通过 pred_seq 一次
    推理出整条曲线，而不是对长格式逐行推理。

    参数：
      model            : 已训练的 Seq2SeqAttnKC 模型。
      xs, ys, ss       : 硬点/输出/行程序列的标准化器(load_bundle 返回)。
      hp               : 各几何的硬点特征，形状 (n_geo, n_in)。
      stroke           : 各几何的轮跳行程序列，形状 (n_geo, T)。
      Yseq             : 各几何的真实输出序列，形状 (n_geo, T, n_out)。
      curve_ids        : 要统计的几何在数组里的下标集合(注意是下标，不是几何编号本身)。
      gids             : 下标到真实几何编号(sample_id)的映射数组。
      names            : 8 个量的名称列表。

    返回：
      dict {name: [(sample_id, mae), ...]}，每个量对应一串"该量在各条曲线上的平均绝对误差"，
      未排序。用于分析整体指标是否被少数离群曲线主导。
    """
    per_curve = {n: [] for n in names}
    for i in curve_ids:
        yp = tn.pred_seq(model, xs, ys, ss, hp[[i]], stroke[[i]])[0]
        yt = Yseq[i]
        ae = np.abs(yp - yt)                      # (T, n_out)：该几何每一步、每个量的绝对误差
        sid = int(gids[i])                        # 下标 i 还原成真实几何编号，便于报告定位样本
        for j, n in enumerate(names):
            per_curve[n].append((sid, float(ae[:, j].mean())))   # 对时间步取均值 = 该曲线该量的 MAE
    return per_curve


def curve_metrics_from_per_curve(per_curve, names):
    """把逐曲线 MAE 汇总成 {name: {MAE, Median, P90, MaxError}}（口径同 visualize_results.py）。
    不含 D2(平滑度)，D2 定义依赖曲线内部相邻点，无法从单个 MAE 值反推。"""
    out = {}
    for n in names:
        maes = np.array([m for _, m in per_curve[n]])
        out[n] = {"MAE": float(maes.mean()), "Median": float(np.median(maes)),
                  "P90": float(np.percentile(maes, 90)), "MaxError": float(maes.max())}
    return out


def main():
    """主流程（Attention 版）：加载 Seq2Seq 模型与 metrics.json，对测试几何做序列推理，生成 6 张分析图。

    与 MLP 版 visualize_results.main 图表种类/文件名一致，差异在于数据组织成规整序列
    (n_geo, T, n_out)，推理走 pred_seq，几何划分走 split_geo。

    命令行参数：
      --out   : 输出目录，默认 train_nn_attention.OUT_DIR（即 nn_out_attention）。
      --topk  : 图6 中每个量列出的"最差曲线"条数，默认 5。
      --scope : 图5/图6 的统计范围。all=对全部几何(train+val+test)逐条统计(默认)；
                test=仅测试几何(复用 metrics.json 里的 test_curve_metrics)。

    产出文件（均写入 --out 目录）：
      results_metrics.png / results_parity.png / results_curves.png /
      results_residual.png / results_curve_table.png /
      results_worst_samples.png / results_worst_samples.json

    无返回值，结果以图片和 json 落盘，并在 stdout 打印保存路径与逐量统计。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=tn.OUT_DIR)
    ap.add_argument("--topk", type=int, default=5, help="每个量列出最差 N 条曲线(默认5)")
    ap.add_argument("--scope", choices=["all", "test"], default="all",
                    help="图5/图6 的统计范围：all=train+val+test 全部曲线(默认)，test=仅测试集")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 加载训练产物：model=Seq2Seq模型，xs/ys/ss=硬点/输出/行程的标准化器，meta=元信息
    model, xs, ys, ss, meta = tn.load_bundle()
    names = meta["target_names"]              # 8 个运动学量名称，决定子图排布顺序
    # 训练阶段已算好的整体指标(train/val/test)，图1/图2/图4 直接引用其数值
    metrics = json.load(open(os.path.join(args.out, "metrics.json"), encoding="utf-8"))

    # 载入规整序列数据：hp=硬点特征(n_geo,n_in)，stroke=行程(n_geo,T)，
    # Yseq=真实输出序列(n_geo,T,n_out)，gids=各几何真实编号
    hp, stroke, Yseq, gids, _ = tn.load_data_seq()
    n_geo = len(gids)
    tr, va, te = tn.split_geo(n_geo)          # 按几何(整条曲线)划分 train/val/test 下标

    # 对每条测试几何整段推理，堆叠成测试集预测序列 yte(n_te,T,n_out)
    with torch.no_grad():
        yte = np.stack([tn.pred_seq(model, xs, ys, ss, hp[[i]], stroke[[i]])[0] for i in te])
    Yte = Yseq[te]                       # (n_te, T, n_out)
    # 图2/图4 需要逐点误差，把序列展平成 (n_te*T, n_out) 的长格式
    yte_flat = yte.reshape(-1, yte.shape[-1])
    Yte_flat = Yte.reshape(-1, Yte.shape[-1])

    # ---------- 图1：指标 ----------
    # 目的：一眼看清每个量在 train/val/test 上的 R²，判断整体拟合水平与泛化差距。
    # 解读：柱越高越好；红色虚线为 0.98 目标线；train 明显高于 test = 过拟合信号。右侧表格给测试集精确数值。
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
    fig.suptitle(f"Training results (Attention) | test mean R²={metrics['test']['_mean_R2']:.3f} "
                 f"| {'PASS' if metrics.get('passed') else 'not reaching 0.98 (data-limited)'}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(args.out, "results_metrics.png"), dpi=110); plt.close(fig)

    # ---------- 图2：parity 散点 ----------
    # 目的：把测试集展平后的"预测 vs 真实"画成散点，检查预测偏差形态与系统性偏移。
    # 解读：点越贴红色对角线(y=x)越准；整体偏离=系统性高/低估；点云越散=随机误差越大。
    # 用固定随机种子(0)最多抽 3000 点，保证抽样可复现。
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Yte_flat), size=min(3000, len(Yte_flat)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        yt, yp = Yte_flat[sel, i], yte_flat[sel, i]
        ax.scatter(yt, yp, s=5, alpha=0.3, color="steelblue")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{n} (R²={metrics['test'][n]['R2']:.3f})")
        ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.grid(alpha=0.3)
    fig.suptitle("Test-set parity: predicted vs actual (closer to red line = better)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_parity.png"), dpi=110); plt.close(fig)

    # ---------- 图3：多条测试曲线对照 ----------
    # 目的：取前 4 条测试几何，画出 8 个量随行程变化的完整曲线，检验模型是否还原了曲线形状/趋势/拐点。
    # 解读：实线=真实，虚线=预测，同色为同一几何；虚线越贴实线越好，形状/斜率对不上=没学到物理规律。
    te4 = te[:4]                               # 测试集前 4 条几何下标
    colors = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        for c, gi in zip(colors, te4):
            st = stroke[gi]                    # 该几何的行程序列(曲线横坐标)
            # pred_seq：对整条序列一次性推理出预测曲线(序列模型无需逐点前向)
            yp = tn.pred_seq(model, xs, ys, ss, hp[[gi]], stroke[[gi]])[0]
            ax.plot(st, Yseq[gi, :, i], color=c, lw=1.6)     # 真实曲线：实线
            ax.plot(st, yp[:, i], color=c, ls="--", lw=1.4)  # 预测曲线：同色虚线
        ax.set_title(n); ax.set_xlabel("wc_stroke"); ax.grid(alpha=0.3)
    axes[0][0].plot([], [], "k-", label="actual"); axes[0][0].plot([], [], "k--", label="predicted")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("4 test geometries: actual (solid) vs predicted (dashed)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_curves.png"), dpi=110); plt.close(fig)

    # ---------- 图4：残差分布 ----------
    # 目的：统计测试集每个量的残差(预测-真实)分布，检查误差是否零均值、对称、无长尾。
    # 解读：理想残差集中在红色 0 线附近且对称；整体偏离 0=系统性偏差；长尾/双峰=部分样本误差异常大。
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, n in enumerate(names):
        ax = axes[i // 4][i % 4]
        res = yte_flat[:, i] - Yte_flat[:, i]
        ax.hist(res, bins=60, color="mediumpurple", alpha=0.8)
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.set_title(f"{n}  MAE={metrics['test'][n]['MAE']:.3g}")
        ax.set_xlabel("residual (pred - actual)"); ax.grid(alpha=0.3)
    fig.suptitle("Test-set residual distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(args.out, "results_residual.png"), dpi=110); plt.close(fig)

    # ---------- 图5：逐量详细指标表 (MAE/Median/P90/MaxError[/D2]) ----------
    # 目的：用表格给出每个量的误差分位(MAE/中位数/P90/最大误差)，看清"是否有个别量学得特别差把均值拉高"。
    # 默认(--scope all)对全部几何逐条统计，样本更多、统计更稳，还能对比训练/测试几何差异观察过拟合；
    # --scope test 时复用 metrics.json 里预先算好的 test_curve_metrics，与训练阶段口径一致。
    if args.scope == "test":
        scope_ids = te
        cm = metrics.get("test_curve_metrics", {})
        scope_label = f"测试集 {len(scope_ids)} 条曲线（与 metrics.json 一致）"
    else:
        scope_ids = np.arange(n_geo)
        print(f"逐条曲线计算 MAE 中（全数据集共 {len(scope_ids)} 条曲线，含 train+val+test）...", flush=True)
        # 全量逐曲线 MAE 只算一次，图5 汇总统计与图6 排序找最差样本共用，避免重复推理
        _per_curve_all = per_curve_mae_seq(model, xs, ys, ss, hp, stroke, Yseq, scope_ids, gids, names)
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
    ax.set_title(f"逐量详细指标(Attention) · 统计范围：{scope_label}", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "results_curve_table.png"), dpi=110); plt.close(fig)

    # ---------- 图6 + JSON：逐条曲线单独算 MAE，找出每个量的最差 Top-K 样本 ----------
    # 目的：区分"整体普遍偏差中等"还是"少数离群样本把汇总 MAE/R² 拖低"。
    # 默认(--scope all)覆盖全部几何；--scope test 时只看测试几何。
    if args.scope == "test":
        per_curve = per_curve_mae_seq(model, xs, ys, ss, hp, stroke, Yseq, scope_ids, gids, names)
    else:
        per_curve = _per_curve_all   # 复用图5已经算好的全量结果，避免重复推理

    scope_desc = "全数据集" if args.scope == "all" else "测试集"
    worst = {}
    for n in names:
        lst = sorted(per_curve[n], key=lambda t: -t[1])   # 按 MAE 降序，排最前的即最差曲线
        worst[n] = lst[:args.topk]
        all_mae = np.array([m for _, m in per_curve[n]])
        # 剔除最差 10% 曲线后再算 MAE，与原始均值对比：两者接近=误差均匀，差距大=少数离群样本主导
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
    fig.suptitle(f"每条曲线各自的 MAE 分布(Attention) · 统计范围：{scope_label}\n"
                 f"（红虚线=最差{args.topk}条位置，右侧长尾=离群样本）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out, "results_worst_samples.png"), dpi=110); plt.close(fig)

    for f in ("results_metrics.png", "results_parity.png", "results_curves.png", "results_residual.png",
              "results_curve_table.png", "results_worst_samples.png", "results_worst_samples.json"):
        print("已保存:", os.path.join(args.out, f))


if __name__ == "__main__":
    main()
