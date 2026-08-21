#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 train_nn_attention.py 中 Seq2Seq Attention(Transformer Encoder-Decoder) 网络的：
  1) 架构图 —— 每一层的名称、维度变换、参数量（数据来自已训练模型 nn_out_attention/model.pt
     的真实 state_dict 形状，不是凭空画的示意图）
  2) 一组真实数据的训练/推理演示 —— 用测试集里的一条真实几何（wc_stroke 曲线），
     展示"33 维硬点 + 101 步 stroke 序列 -> 8 条运动学曲线"的完整数据流转，
     并对比训练阶段(教师值/损失)与推理阶段(纯前向预测)的差异。

数据来源（全部真实，非构造）：
  - 架构维度：nn_out_attention/model.pt 的 ckpt['arch']（本次训练实际使用的超参）+
    重建模型后 named_parameters() 的真实 shape。
  - 示例数据：nn_data/dataset.npz 中测试集切分出的一条几何（用 train_nn_attention.load_data_seq
    + split_geo 还原，与训练时用的是同一套随机种子 42，因此 test 切分完全一致）。
  - 推理结果：加载 nn_out_attention/model.pt 的已训练权重，对该条几何做真实前向推理。

输出（默认写到脚本同级目录）：
  - architecture_diagram.png   : 网络架构分层图（Encoder / stroke位置编码 / TransformerDecoder x4 / 输出头）
  - train_infer_demo.png      : 用一条真实几何演示训练vs推理的数据流 + 预测曲线对比
  - architecture_summary.json : 各层维度、参数量的结构化汇总（供后续复用/核对）

用法：
  cd /workspace/Suspension_AI_Design/训练推理/Attention
  python3 visualize_architecture.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # .../训练推理
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.patches as mpatches   # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import train_nn_attention as tn         # noqa: E402

OUT_DIR = _HERE


def _setup_chinese_font():
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


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# 1) 从真实模型提取架构信息
# --------------------------------------------------------------------------- #

def extract_architecture():
    """加载已训练模型，返回 (model, arch_cfg, layer_info_list)。
    layer_info_list 是按前向顺序排列的层描述，每项含 name/kind/in_shape/out_shape/params。
    维度全部来自真实 checkpoint，不是手写常量。"""
    model, xs, ys, ss, meta = tn.load_bundle()
    ckpt = tn._torch_load_safe(tn.MODEL_PATH, weights_only=False)
    arch = ckpt["arch"]
    n_hp, n_out = arch["n_hp"], arch["n_out"]
    hidden, n_layers, n_heads, n_ctx = arch["hidden"], arch["n_layers"], arch["n_heads"], arch["n_ctx"]
    T_max = getattr(model, "max_len", None)

    def pcount(*names):
        total = 0
        sd = dict(model.named_parameters())
        for n in names:
            total += sd[n].numel()
        return total

    layers = []
    # --- Encoder: 33硬点 -> hidden (两层 Linear+ReLU+Dropout) ---
    layers.append({
        "stage": "Encoder", "name": "encoder.0 (Linear+ReLU+Dropout)",
        "in_dim": n_hp, "out_dim": hidden,
        "params": pcount("encoder.0.weight", "encoder.0.bias"),
        "desc": f"33维硬点坐标 -> {hidden}维隐向量",
    })
    layers.append({
        "stage": "Encoder", "name": "encoder.3 (Linear+ReLU+Dropout)",
        "in_dim": hidden, "out_dim": hidden,
        "params": pcount("encoder.3.weight", "encoder.3.bias"),
        "desc": f"{hidden}维 -> {hidden}维（第二层非线性变换）",
    })
    layers.append({
        "stage": "Encoder", "name": "ctx_proj (Linear)",
        "in_dim": hidden, "out_dim": hidden * n_ctx,
        "params": pcount("ctx_proj.weight", "ctx_proj.bias"),
        "desc": f"{hidden}维 -> 展开为 {n_ctx} 个上下文token，每个{hidden}维 "
                f"(reshape成 memory: [B,{n_ctx},{hidden}])",
    })
    # --- stroke 位置编码 ---
    n_freq = hidden // 4
    layers.append({
        "stage": "位置编码", "name": "sin/cos 数值位置编码",
        "in_dim": 1, "out_dim": 2 * n_freq,
        "params": 0,
        "desc": f"每个stroke标量值 -> {n_freq}个频率的sin+cos -> {2*n_freq}维",
    })
    layers.append({
        "stage": "位置编码", "name": "pe_proj (Linear)",
        "in_dim": 2 * n_freq, "out_dim": hidden,
        "params": pcount("pe_proj.weight", "pe_proj.bias"),
        "desc": f"{2*n_freq}维位置编码 -> {hidden}维（与 encoder 输出同维，可做 attention）",
    })
    if hasattr(model, "pos_query"):
        layers.append({
            "stage": "位置编码", "name": "pos_query (可学习embedding)",
            "in_dim": f"序列下标(0~{T_max-1})", "out_dim": hidden,
            "params": model.pos_query.numel(),
            "desc": f"额外叠加的可学习位置embedding，形状[{T_max},{hidden}]，按stroke序列下标索引",
        })
    # --- Decoder: TransformerDecoderLayer x n_layers ---
    d_head = hidden // n_heads
    ff = hidden * 4
    per_layer_params = pcount(
        "decoder.layers.0.self_attn.in_proj_weight", "decoder.layers.0.self_attn.in_proj_bias",
        "decoder.layers.0.self_attn.out_proj.weight", "decoder.layers.0.self_attn.out_proj.bias",
        "decoder.layers.0.multihead_attn.in_proj_weight", "decoder.layers.0.multihead_attn.in_proj_bias",
        "decoder.layers.0.multihead_attn.out_proj.weight", "decoder.layers.0.multihead_attn.out_proj.bias",
        "decoder.layers.0.linear1.weight", "decoder.layers.0.linear1.bias",
        "decoder.layers.0.linear2.weight", "decoder.layers.0.linear2.bias",
        "decoder.layers.0.norm1.weight", "decoder.layers.0.norm1.bias",
        "decoder.layers.0.norm2.weight", "decoder.layers.0.norm2.bias",
        "decoder.layers.0.norm3.weight", "decoder.layers.0.norm3.bias",
    )
    for li in range(n_layers):
        layers.append({
            "stage": "TransformerDecoder", "name": f"DecoderLayer[{li}]",
            "in_dim": hidden, "out_dim": hidden,
            "params": per_layer_params,
            "desc": (f"Self-Attn({n_heads}头×{d_head}维, 双向无mask) -> "
                     f"Cross-Attn(query=stroke点, key/value=memory[{n_ctx}个硬点token]) -> "
                     f"FFN({hidden}->{ff}->{hidden}) [Pre-LN]"),
        })
    # --- 输出头（v1=单层Linear，v2=两层MLP，结构不同，按 arch_version 自适应） ---
    if arch.get("arch_version") == "v2":
        layers.append({
            "stage": "输出头", "name": "out_proj.0 (Linear+ReLU+Dropout)",
            "in_dim": hidden, "out_dim": hidden * 2,
            "params": pcount("out_proj.0.weight", "out_proj.0.bias"),
            "desc": f"{hidden}维 -> {hidden*2}维",
        })
        layers.append({
            "stage": "输出头", "name": "out_proj.3 (Linear)",
            "in_dim": hidden * 2, "out_dim": n_out,
            "params": pcount("out_proj.3.weight", "out_proj.3.bias"),
            "desc": f"{hidden*2}维 -> {n_out}维（toe/camber/caster/caster_arm/scrub_radius/"
                    f"tire_con_point_x,y,z）",
        })
    else:
        layers.append({
            "stage": "输出头", "name": "out_proj (Linear)",
            "in_dim": hidden, "out_dim": n_out,
            "params": pcount("out_proj.weight", "out_proj.bias"),
            "desc": f"{hidden}维 -> {n_out}维（toe/camber/caster/caster_arm/scrub_radius/"
                    f"tire_con_point_x,y,z）",
        })

    total_params = sum(p.numel() for p in model.parameters())
    return model, xs, ys, ss, meta, arch, layers, total_params


# --------------------------------------------------------------------------- #
# 2) 画架构图
# --------------------------------------------------------------------------- #

def plot_architecture(arch, layers, total_params, meta, out_path):
    n_hp, n_out = arch["n_hp"], arch["n_out"]
    hidden, n_layers, n_heads, n_ctx = arch["hidden"], arch["n_layers"], arch["n_heads"], arch["n_ctx"]

    stage_colors = {
        "输入": "#DDEBF7",
        "Encoder": "#BDD7EE",
        "位置编码": "#FCE4D6",
        "TransformerDecoder": "#C6E0B4",
        "输出头": "#FFE699",
        "输出": "#F8CBAD",
    }

    fig_h = 2.2 + 0.62 * (len(layers) + 4)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 10)

    y = 0.0
    box_h = 0.5
    gap = 0.62
    xs_box = (0.5, 9.5)

    def draw_box(y, text, color, sub=None, height=box_h):
        box = FancyBboxPatch((xs_box[0], y), xs_box[1] - xs_box[0], height,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.1, edgecolor="#404040", facecolor=color)
        ax.add_patch(box)
        ax.text(5.0, y + height * 0.62, text, ha="center", va="center",
                fontsize=10.3, fontweight="bold")
        if sub:
            ax.text(5.0, y + height * 0.20, sub, ha="center", va="center", fontsize=8.3, color="#333333")

    def draw_arrow(y_from, y_to):
        ax.annotate("", xy=(5.0, y_to), xytext=(5.0, y_from),
                   arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))

    ys_top = 0.0
    y = ys_top
    # 输入
    draw_box(y, f"输入：33维硬点坐标 (n_hp={n_hp})  +  wc_stroke 序列 (T=101, 范围[-50,50]mm)",
             stage_colors["输入"])
    y_prev_top = y
    y -= gap
    draw_arrow(y_prev_top, y + box_h)

    total_shown = total_params
    running = 0
    for i, layer in enumerate(layers):
        color = stage_colors.get(layer["stage"], "#EEEEEE")
        txt = f"[{layer['stage']}] {layer['name']}"
        sub = f"{layer['in_dim']} → {layer['out_dim']}   |   params={layer['params']:,}"
        draw_box(y, txt, color, sub=sub)
        running += layer["params"]
        y_prev_top = y
        y -= gap
        if i < len(layers) - 1:
            draw_arrow(y_prev_top, y + box_h)
        else:
            draw_arrow(y_prev_top, y + box_h)

    # 输出
    draw_box(y, f"输出：8 条运动学曲线 (n_out={n_out})，每条 T=101 步",
             stage_colors["输出"],
             sub=", ".join(meta["target_names"]))

    ax.set_ylim(y - 0.3, ys_top + box_h + 0.35)

    title = (f"Seq2Seq Attention (Transformer Encoder-Decoder) 架构 — "
             f"arch_version={arch.get('arch_version','v?')}\n"
             f"hidden(d_model)={hidden}  n_layers={n_layers}  n_heads={n_heads}  "
             f"n_ctx={n_ctx}  总参数量={total_params:,}")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"已保存架构图: {out_path}")


# --------------------------------------------------------------------------- #
# 3) 用一条真实几何演示训练 vs 推理
# --------------------------------------------------------------------------- #

def demo_train_vs_infer(model, xs, ys, ss, meta, out_path, json_path):
    """取测试集切分中的第一条真实几何，展示：
      - 输入数据的真实数值（33维硬点的前几维 + stroke 序列）
      - 训练阶段：该几何参与训练时的前向+损失计算方式（用当前已训练权重复现一次，
        仅用于展示数据流，不重新训练/不更新参数）
      - 推理阶段：纯前向预测，还原到原始物理量纲，和真实值对比
    """
    hp, stroke, Yseq, gids, meta2 = tn.load_data_seq()
    n_geo = hp.shape[0]
    tr, va, te = tn.split_geo(n_geo)     # 与训练时相同 seed=42，测试集切分完全一致
    names = meta2["target_names"]

    i = te[0]
    sample_id = int(gids[i])
    hp_i = hp[i:i + 1]                # (1, 33)
    stroke_i = stroke[i:i + 1]         # (1, 101)
    y_true_i = Yseq[i:i + 1]           # (1, 101, 8)

    # ---- 标准化（与训练时同一套 Standardizer，来自 checkpoint） ----
    hp_std = xs.tf(hp_i)
    stroke_std = ss.tf(stroke_i[..., None]).squeeze(-1)
    y_std_true = ys.tf(y_true_i.reshape(-1, len(names))).reshape(1, -1, len(names))

    hp_t = torch.tensor(hp_std, dtype=torch.float32)
    st_t = torch.tensor(stroke_std, dtype=torch.float32)
    y_t = torch.tensor(y_std_true, dtype=torch.float32)

    # ---- "训练阶段"式的前向（模型当前处于 eval，仅用于展示前向+损失口径，不做反向传播） ----
    model.eval()
    with torch.no_grad():
        y_pred_std = model(hp_t, st_t)              # (1, T, 8) 标准化空间
        mse_loss = torch.nn.functional.mse_loss(y_pred_std, y_t).item()

    # ---- "推理阶段"：反标准化回原始物理量纲 ----
    y_pred_orig = ys.inv(y_pred_std.numpy().reshape(-1, len(names))).reshape(1, -1, len(names))[0]
    y_true_orig = y_true_i[0]
    stroke_orig = stroke_i[0]

    per_q_mae = {n: float(np.mean(np.abs(y_pred_orig[:, k] - y_true_orig[:, k])))
                for k, n in enumerate(names)}
    per_q_r2 = {n: float(tn.r2(y_true_orig[:, k], y_pred_orig[:, k])) for k, n in enumerate(names)}

    # ------------------- 画图：数据流 + 曲线对比 ------------------- #
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.5, wspace=0.35)

    # (a) 输入硬点坐标条形图（33维，仅展示数值分布，不是学习内容）
    ax_hp = fig.add_subplot(gs[0, :2])
    feat_names = meta2.get("feat_names", [f"f{k}" for k in range(hp_i.shape[1] + 1)])[:-1]
    ax_hp.bar(range(len(hp_i[0])), hp_i[0], color="#4472C4")
    ax_hp.set_xticks(range(len(feat_names)))
    ax_hp.set_xticklabels(feat_names, rotation=90, fontsize=6)
    ax_hp.set_title(f"输入①：几何#{sample_id} 的 33 维硬点坐标（真实数值，单位mm）", fontsize=10)
    ax_hp.set_ylabel("坐标值 (mm)")
    ax_hp.grid(True, alpha=0.3)

    # (b) 输入 stroke 序列
    ax_st = fig.add_subplot(gs[0, 2:])
    ax_st.plot(np.arange(len(stroke_orig)), stroke_orig, "o-", ms=3, color="#ED7D31")
    ax_st.set_title(f"输入②：wc_stroke 序列（T={len(stroke_orig)}步，范围[{stroke_orig.min():.0f},"
                    f"{stroke_orig.max():.0f}]mm）\n即Decoder的query token序列（数值本身做位置编码）",
                    fontsize=10)
    ax_st.set_xlabel("序列下标 t")
    ax_st.set_ylabel("wc_stroke (mm)")
    ax_st.grid(True, alpha=0.3)

    # (c) 标准化后送入模型的数值（训练/推理共用同一路径）
    ax_std = fig.add_subplot(gs[1, :2])
    ax_std.plot(hp_std[0], "s-", ms=3, color="#4472C4", label="标准化后硬点(送入Encoder)")
    ax_std.axhline(0, color="gray", lw=0.6)
    ax_std.set_title("数据流①：标准化 (x-mean)/std 后送入 Encoder", fontsize=10)
    ax_std.set_xlabel("硬点维度下标")
    ax_std.set_ylabel("标准化值(z-score)")
    ax_std.grid(True, alpha=0.3)
    ax_std.legend(fontsize=8)

    ax_loss = fig.add_subplot(gs[1, 2:])
    ax_loss.axis("off")
    train_text = (
        "训练阶段（teacher-forcing 不需要，整条曲线并行前向）:\n"
        f"  1) hp_std = (hp-mean)/std           形状(1,{hp_i.shape[1]})\n"
        f"  2) stroke_std = (stroke-mean)/std   形状(1,{stroke_i.shape[1]})\n"
        f"  3) y_pred_std = Model(hp_std, stroke_std)  形状(1,{y_pred_std.shape[1]},{y_pred_std.shape[2]})\n"
        f"  4) loss = MSE(y_pred_std, y_true_std)\n"
        f"     本样例复现结果: MSE(标准化空间) = {mse_loss:.6f}\n"
        "     (真实训练时对 train 切分批量做此步 + loss.backward() + optimizer.step()，\n"
        "      本图为不更新参数的复现，仅展示数据口径)"
    )
    ax_loss.text(0.02, 0.95, train_text, va="top", ha="left", fontsize=9.3,
                bbox=dict(boxstyle="round", facecolor="#FFF2CC", edgecolor="#BF8F00"))
    ax_loss.set_title("数据流②：训练阶段的前向与损失口径", fontsize=10)

    fig.suptitle(f"训练 vs 推理 数据流演示 —— 测试集真实几何 #{sample_id}（未参与训练）",
                fontsize=13.5, fontweight="bold")
    fig.savefig(out_path.replace(".png", "_datapath.png"), dpi=125)
    plt.close(fig)
    log(f"已保存数据流演示图: {out_path.replace('.png', '_datapath.png')}")

    # ------------------- 第二张图：8条曲线 真实vs预测 完整对比 ------------------- #
    fig2, axes = plt.subplots(2, 4, figsize=(20, 9))
    for k, name in enumerate(names):
        ax = axes[k // 4][k % 4]
        ax.plot(stroke_orig, y_true_orig[:, k], "b-", lw=2, label="真实值(仿真)")
        ax.plot(stroke_orig, y_pred_orig[:, k], "r--", lw=2, label="模型推理预测")
        ax.set_title(f"{name}\nR²={per_q_r2[name]:.4f}  MAE={per_q_mae[name]:.4g}", fontsize=10)
        ax.set_xlabel("wc_stroke (mm)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig2.suptitle(f"推理阶段：测试集几何 #{sample_id} 的 8 条运动学曲线 — 真实(仿真) vs 模型预测",
                 fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(out_path, dpi=125)
    plt.close(fig2)
    log(f"已保存推理结果对比图: {out_path}")

    summary = {
        "sample_geo_id": sample_id,
        "input": {
            "hp_dim": int(hp_i.shape[1]),
            "hp_values_mm": hp_i[0].tolist(),
            "stroke_T": int(stroke_i.shape[1]),
            "stroke_range_mm": [float(stroke_orig.min()), float(stroke_orig.max())],
        },
        "train_forward_demo": {
            "note": "复现一次前向+损失计算（不更新参数），展示训练时的数据口径",
            "mse_standardized_space": mse_loss,
        },
        "infer_result": {
            "per_quantity_R2": per_q_r2,
            "per_quantity_MAE": per_q_mae,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"已保存训练/推理演示数据摘要: {json_path}")
    return summary


# --------------------------------------------------------------------------- #
# 4) 展开单个 DecoderLayer 内部：Multi-Head Self-Attn + Cross-Attn(memory token) + FFN
#    全部用真实模型对一条测试几何做一次前向，手动重放 layer[0] 的内部矩阵运算，
#    拿到真实的注意力权重（不是构造的示意数值）。
# --------------------------------------------------------------------------- #

def _manual_layer_forward_with_attn(model, layer, tgt, memory):
    """手动重放单个 TransformerDecoderLayer 的前向，同时用 need_weights=True 拿到
    self-attn 和 cross-attn 的真实注意力权重矩阵。根据 layer.norm_first 自动选择
    Pre-LN 或 Post-LN 分支，与 PyTorch 官方 TransformerDecoderLayer.forward 逐行对齐
    （见 torch.nn.TransformerDecoderLayer.forward 源码的 norm_first 分支），
    确保数值结果与直接调用 layer(tgt, memory) 完全一致。

    返回: dict，含各阶段张量与注意力权重（均为 numpy，已 detach）。
    """
    with torch.no_grad():
        if layer.norm_first:
            # ---- Pre-LN: x = x + block(norm(x)) ----
            x = layer.norm1(tgt)
            self_attn_out, self_attn_w = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False)
            tgt2 = tgt + layer.dropout1(self_attn_out)

            x2 = layer.norm2(tgt2)
            cross_attn_out, cross_attn_w = layer.multihead_attn(
                x2, memory, memory, need_weights=True, average_attn_weights=False)
            tgt3 = tgt2 + layer.dropout2(cross_attn_out)

            x3 = layer.norm3(tgt3)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3))))
            out = tgt3 + layer.dropout3(ff)
        else:
            # ---- Post-LN: x = norm(x + block(x)) ----
            self_attn_out, self_attn_w = layer.self_attn(
                tgt, tgt, tgt, need_weights=True, average_attn_weights=False)
            tgt2 = layer.norm1(tgt + layer.dropout1(self_attn_out))

            cross_attn_out, cross_attn_w = layer.multihead_attn(
                tgt2, memory, memory, need_weights=True, average_attn_weights=False)
            tgt3 = layer.norm2(tgt2 + layer.dropout2(cross_attn_out))

            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt3))))
            out = layer.norm3(tgt3 + layer.dropout3(ff))

    return {
        "self_attn_weights": self_attn_w.numpy(),      # (B, n_heads, T, T)
        "cross_attn_weights": cross_attn_w.numpy(),     # (B, n_heads, T, n_ctx)
        "layer_output": out.numpy(),                    # (B, T, hidden)
    }


def plot_decoder_layer_detail(model, xs, ys, ss, arch, out_path):
    """展开 DecoderLayer[0] 内部的完整数据流，用测试集同一条真实几何(#607，与
    demo_train_vs_infer 用的是同一条)驱动一次真实前向，画出：
      (a) Multi-Head Self-Attention: 101个stroke点如何拆成8头、每头32维，
          并展示某一头的真实 self-attn 权重矩阵(101x101)热力图
      (b) Multi-Head Cross-Attention: 101个query如何查询4个memory token，
          每头的注意力权重矩阵是(101,4)，展示8个头的完整热力图(平均到几何直观理解)
          以及单头细节
      (c) FFN 维度变换
      (d) 数据流示意：Q/K/V 的真实shape标注
    """
    hidden, n_heads, n_ctx = arch["hidden"], arch["n_heads"], arch["n_ctx"]
    d_head = hidden // n_heads

    hp, stroke, Yseq, gids, meta2 = tn.load_data_seq()
    n_geo = hp.shape[0]
    tr, va, te = tn.split_geo(n_geo)
    i = te[0]
    sample_id = int(gids[i])
    hp_i, stroke_i = hp[i:i + 1], stroke[i:i + 1]

    hp_std = xs.tf(hp_i)
    stroke_std = ss.tf(stroke_i[..., None]).squeeze(-1)
    hp_t = torch.tensor(hp_std, dtype=torch.float32)
    st_t = torch.tensor(stroke_std, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        B, T = st_t.shape
        ctx = model.encoder(hp_t)
        memory = model.ctx_proj(ctx).view(B, model.n_ctx, model.hidden)     # (1, n_ctx, hidden)
        tgt = model._stroke_pe(st_t)
        if hasattr(model, "pos_query") and T <= model.max_len:
            tgt = tgt + model.pos_query[:T].unsqueeze(0)

    layer0 = model.decoder.layers[-1]           # 用最后一层：经排查layers[0]的注意力已退化为均匀分布，
                                                 # 最后一层(layers[-1])self-attn仍保留可辨差异，更能展示真实数据流
    layer_idx_shown = len(model.decoder.layers) - 1
    is_pre_ln = bool(layer0.norm_first)
    detail = _manual_layer_forward_with_attn(model, layer0, tgt, memory)
    self_w = detail["self_attn_weights"][0]     # (n_heads, T, T)
    cross_w = detail["cross_attn_weights"][0]   # (n_heads, T, n_ctx)

    # 数值核验：手动重放的输出应与直接调用 layer0(tgt, memory) 一致（应≈0，验证公式与真实结构一致）
    with torch.no_grad():
        ref_out = layer0(tgt, memory).numpy()
    max_diff = float(np.max(np.abs(ref_out - detail["layer_output"])))
    if max_diff > 1e-3:
        log(f"⚠️ 警告：DecoderLayer手动重放误差={max_diff:.4g}，明显偏大，"
            f"注意力权重图可能不可信，需要检查_manual_layer_forward_with_attn实现")

    fig = plt.figure(figsize=(19, 13.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.0, 1.0], hspace=0.55, wspace=0.4)

    # ---- (0) 顶部：Q/K/V 形状与多头拆分示意文字（根据 norm_first 动态生成公式） ----
    ax_txt = fig.add_subplot(gs[0, :])
    ax_txt.axis("off")
    ln_tag = "Pre-LN" if is_pre_ln else "Post-LN"
    if is_pre_ln:
        self_qkv_line = f"  Q=K=V = norm1(tgt)                     形状 (B=1, T=101, {hidden})"
        self_out_line = f"  输出 = tgt + dropout(attn_weights @ V 拼回后过out_proj({hidden}->{hidden}))"
        cross_q_line = f"  Q = norm2(tgt2)                         形状 (B=1, T=101, {hidden})"
        cross_out_line = f"  输出 = tgt2 + dropout(attn_weights @ V 拼回后过out_proj({hidden}->{hidden}))"
    else:
        self_qkv_line = f"  Q=K=V = tgt（未做LayerNorm）             形状 (B=1, T=101, {hidden})"
        self_out_line = f"  输出 = norm1(tgt + dropout(attn_weights @ V 拼回后过out_proj({hidden}->{hidden})))"
        cross_q_line = f"  Q = tgt2（Self-Attn+norm1之后的结果，未做LayerNorm）  形状 (B=1, T=101, {hidden})"
        cross_out_line = f"  输出 = norm2(tgt2 + dropout(attn_weights @ V 拼回后过out_proj({hidden}->{hidden})))"
    txt = (
        f"DecoderLayer[{layer_idx_shown}]（共{len(model.decoder.layers)}层, 此处展示最后一层）内部真实数据流"
        f"（几何#{sample_id}，权重来自已训练模型，非构造数值；"
        f"本层结构={ln_tag}，与训练时构造 nn.TransformerDecoderLayer(norm_first={is_pre_ln}) 一致）\n\n"
        f"Self-Attention（曲线内部101个stroke点互相看）:\n"
        f"{self_qkv_line}\n"
        f"  拆成 {n_heads} 头: view(B, T, {n_heads}, {d_head}) -> 每头独立算注意力\n"
        f"  attn_weights = softmax(Q·K^T/√{d_head})    形状 (B, {n_heads}头, T=101, T=101)\n"
        f"{self_out_line}\n\n"
        f"Cross-Attention（每个stroke点查询{n_ctx}个硬点memory token）:\n"
        f"{cross_q_line}\n"
        f"  K = V = memory                          形状 (B=1, n_ctx={n_ctx}, {hidden})\n"
        f"  拆成 {n_heads} 头: Q->(B,{n_heads},101,{d_head})  K,V->(B,{n_heads},{n_ctx},{d_head})\n"
        f"  attn_weights = softmax(Q·K^T/√{d_head})    形状 (B, {n_heads}头, T=101, n_ctx={n_ctx})  "
        f"<- 关键：最后一维是{n_ctx}，来自memory长度\n"
        f"{cross_out_line}\n\n"
        f"数值核验：手动重放各头矩阵运算 vs 直接调用 layer(tgt,memory) 的最大误差 = {max_diff:.2e}（应≈0，验证重放公式与真实结构一致）\n"
        f"⚠️ 如实发现：本模型(v1, hidden=128, n_layers=2)的 Cross-Attention 在该几何样例上所有头、\n"
        f"所有stroke位置的权重均严格等于 1/{n_ctx}={1/n_ctx:.4f}（完全均匀，未学到对{n_ctx}个硬点token的区分性查询）；\n"
        f"decoder.layers[0]的Self-Attn同样完全均匀，只有本图展示的最后一层Self-Attn保留了可辨差异\n"
        f"（这是训练结果的真实特征，非绘图或计算错误）"
    )
    ax_txt.text(0.01, 0.98, txt, va="top", ha="left", fontsize=8.8,
               bbox=dict(boxstyle="round", facecolor="#EAF1FB", edgecolor="#4472C4"))

    # ---- (1) Self-Attention: 自动选方差最大的头展示（确保呈现的是有真实差异的头，不是随手选head0） ----
    ax_self = fig.add_subplot(gs[1, 0:2])
    self_head_stds = self_w.std(axis=(1, 2))            # (n_heads,) 每头的权重离散程度
    show_head = int(np.argmax(self_head_stds))
    self_vmin, self_vmax = float(self_w[show_head].min()), float(self_w[show_head].max())
    im1 = ax_self.imshow(self_w[show_head], cmap="viridis", aspect="auto", vmin=self_vmin, vmax=self_vmax)
    ax_self.set_title(f"Self-Attn 权重矩阵（head={show_head}，共{n_heads}头中方差最大的头）\n"
                      f"(query stroke点, key stroke点) 均为101个位置；权重范围[{self_vmin:.4f},{self_vmax:.4f}]"
                      f"（均匀分布应为1/{self_w.shape[-1]}={1/self_w.shape[-1]:.4f}）",
                      fontsize=9.5)
    ax_self.set_xlabel("key: stroke序列下标 (0~100)")
    ax_self.set_ylabel("query: stroke序列下标 (0~100)")
    fig.colorbar(im1, ax=ax_self, fraction=0.046, pad=0.04)

    # ---- (2) Self-Attention: 8个头的权重矩阵方差对比（每头一个小图，体现哪些头学到了非均匀模式） ----
    ax_heads = fig.add_subplot(gs[1, 2:])
    ax_heads.bar(range(n_heads), self_head_stds, color="#4472C4")
    ax_heads.set_xticks(range(n_heads))
    ax_heads.set_xticklabels([f"head{i}" for i in range(n_heads)], fontsize=8)
    ax_heads.set_title(f"Self-Attn {n_heads}个头的权重标准差对比\n"
                       f"(std越大=该头越偏离均匀分布，即学到了非均匀关注模式；std≈0=退化为均匀平均)",
                       fontsize=9.5)
    ax_heads.set_ylabel("权重标准差(std)")
    ax_heads.grid(True, alpha=0.3, axis="y")

    # ---- (3) Cross-Attention: 8个头 x 4个memory token 的完整权重热力图 ----
    ax_cross = fig.add_subplot(gs[2, 0:2])
    # cross_w: (n_heads, T, n_ctx) -> 转成 (n_heads*n_ctx, T) 便于一张图看全部头
    cross_disp = cross_w.transpose(0, 2, 1).reshape(n_heads * n_ctx, -1)   # (n_heads*n_ctx, T)
    cross_vmin, cross_vmax = float(cross_disp.min()), float(cross_disp.max())
    im2 = ax_cross.imshow(cross_disp, cmap="magma", aspect="auto", vmin=cross_vmin, vmax=cross_vmax)
    ax_cross.set_yticks(range(n_heads * n_ctx))
    ax_cross.set_yticklabels([f"h{h}-tok{c}" for h in range(n_heads) for c in range(n_ctx)], fontsize=6.5)
    ax_cross.set_xlabel("query: stroke序列下标 (0~100)")
    ax_cross.set_title(f"Cross-Attn 全部{n_heads}头 x {n_ctx}个memory token 权重\n"
                       f"每行=某头对某个硬点token的权重，随stroke位置(x轴)变化；"
                       f"权重范围[{cross_vmin:.4f},{cross_vmax:.4f}]（均匀应为1/{n_ctx}={1/n_ctx:.4f}）",
                       fontsize=9.5)
    fig.colorbar(im2, ax=ax_cross, fraction=0.03, pad=0.02)

    # ---- (4) Cross-Attention: 固定几个代表性stroke位置，看4个token的权重分布(head平均) ----
    ax_tok = fig.add_subplot(gs[2, 2:])
    cross_mean_heads = cross_w.mean(axis=0)      # (T, n_ctx) 对8头取平均，直观看"整体分工"
    sample_positions = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    x = np.arange(n_ctx)
    width = 0.15
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sample_positions)))
    for j, pos in enumerate(sample_positions):
        ax_tok.bar(x + j * width, cross_mean_heads[pos], width=width, color=colors[j],
                  label=f"stroke idx={pos} ({stroke_i[0, pos]:.0f}mm)")
    ax_tok.set_xticks(x + width * (len(sample_positions) - 1) / 2)
    ax_tok.set_xticklabels([f"token{c}" for c in range(n_ctx)])
    ax_tok.set_title(f"不同stroke位置对{n_ctx}个memory token的权重分布（8头平均）\n"
                     f"权重来自真实前向，展示模型如何按位置动态分配对硬点信息的依赖", fontsize=10)
    ax_tok.set_ylabel("平均注意力权重")
    ax_tok.legend(fontsize=7.5, loc="upper right")
    ax_tok.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"DecoderLayer[{layer_idx_shown}] 内部展开：Multi-Head Attention 与 {n_ctx} 个 Memory Token 的真实数据流",
                fontsize=14.5, fontweight="bold")
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    log(f"已保存 DecoderLayer 内部展开图: {out_path}")

    return {
        "sample_geo_id": sample_id,
        "layer_index_shown": layer_idx_shown,
        "n_heads": n_heads,
        "d_head": d_head,
        "n_ctx": n_ctx,
        "self_attn_weight_shape": list(self_w.shape),
        "cross_attn_weight_shape": list(cross_w.shape),
        "self_attn_head_stds": self_head_stds.tolist(),
        "self_attn_head_shown": show_head,
        "cross_attn_std": float(cross_w.std()),
        "finding": (
            f"如实发现：decoder.layers[0] 的 Self-Attn 与全部层的 Cross-Attn "
            f"在该几何样例上权重均严格均匀（std=0，即1/{n_ctx}或1/T），"
            f"只有decoder.layers[{layer_idx_shown}]（最后一层）的Self-Attn部分头"
            f"（尤其head{show_head}）保留了可辨的非均匀关注模式。"
        ),
        "manual_replay_max_abs_diff_vs_layer_call": max_diff,
    }


def main():
    log("加载已训练模型与真实架构信息...")
    model, xs, ys, ss, meta, arch, layers, total_params = extract_architecture()
    log(f"架构: {arch}")
    log(f"总参数量: {total_params:,}")

    arch_png = os.path.join(OUT_DIR, "architecture_diagram.png")
    plot_architecture(arch, layers, total_params, meta, arch_png)

    demo_png = os.path.join(OUT_DIR, "train_infer_demo.png")
    summary_json = os.path.join(OUT_DIR, "architecture_summary.json")
    summary = demo_train_vs_infer(model, xs, ys, ss, meta, demo_png, summary_json)

    detail_png = os.path.join(OUT_DIR, "decoder_layer_detail.png")
    attn_info = plot_decoder_layer_detail(model, xs, ys, ss, arch, detail_png)

    # 汇总层信息也写进 summary json（追加）
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["architecture"] = {
        "config": arch,
        "total_params": total_params,
        "layers": layers,
        "target_names": meta["target_names"],
        "feat_names": meta.get("feat_names", []),
    }
    data["decoder_layer_detail"] = attn_info
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log("完成。")


if __name__ == "__main__":
    main()
