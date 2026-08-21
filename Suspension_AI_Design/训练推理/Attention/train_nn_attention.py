#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 步（备选架构）：用 Seq2Seq(Transformer Attention Encoder-Decoder) 训练「33 硬点 ->
沿 wc_stroke 的 8 条运动学曲线」的代理模型，替代 train_nn.py 的逐点回归 MLP/PointMLP。

设计动机：
  运动学曲线本质是「同一组硬点几何」在 wc_stroke 这个序列维度上的连续输出，
  相邻 stroke 点强相关（曲线平滑）。逐点 MLP 把每个 (硬点,stroke) 当独立样本回归，
  没有显式利用这种序列结构。本版本用 Attention(Transformer) 代替最初的 GRU 实现：
  Encoder 把 33 硬点编码成若干上下文 token；Decoder 把 stroke 序列的每个点当作
  query token（用 stroke 数值本身做 sin/cos 位置编码），通过 self-attention 让
  曲线上任意两点互相感知，再通过 cross-attention 融合硬点上下文，最终整条曲线
  一次性并行输出——不需要像 GRU 版本那样逐时间步循环/teacher forcing。

⚠️ 如实说明（继承自 train_nn.py 已验证的结论，见 HELP.md §9）：
  969 组几何在 33 维硬点空间的分布密度才是测试集 R² 卡在 ~0.44 的根因
  （线性回归/kNN/MLP 三种方法在同一份数据上得到同一天花板；换成 GRU-Seq2Seq
  同样验证过卡在类似水平，见上一版本记录）。换成 Attention 改变的仍然只是
  「模型对同一条曲线内部序列结构的利用方式」，不改变「训练几何覆盖了多大的
  硬点空间」这个信息量上限。本脚本按用户要求实现该架构、跑出真实指标；
  如指标仍不达标，会如实汇报，不会为了达标而虚报或裁剪数据。

输入：nn_data/dataset.npz + nn_data/meta.json（由 parse_excel_to_dataset.py 生成，长格式）
      本脚本会按 groups 把长格式重组成规整序列 (n_geo, n_stroke, ...)
输出（默认 nn_out2/，与 train_nn.py 的 nn_out/ 分开，不覆盖已训练的 MLP 结果；
      会覆盖同目录下此前 GRU 版本训练保存的 model.pt/metrics.json）：
  - model.pt        : Attention 模型权重 + 结构 + 标准化参数 + 元数据
  - metrics.json    : 各量 R²/RMSE/MAE（train/val/test）+ 设计位校验 + 逐曲线误差指标
  - curves_test_sample_*.png/.csv : 示例几何 预测 vs 真实曲线

用法：
  python3 train_nn2.py                          # 训练+评估+保存+示例推理绘图
  python3 train_nn2.py --data nn_data            # 显式指定数据目录（默认即此）
  python3 train_nn2.py --infer-only              # 用已存 model.pt 做示例推理
  python3 train_nn2.py --epochs 300 --time-budget 240 --hidden 128 --layers 2 --heads 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time as _t

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn as nn                    # noqa: E402
import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import suspension_config as cfg          # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "数据处理", "nn_data")
OUT_DIR = os.path.join(_HERE, "nn_out_attention")
MODEL_PATH = os.path.join(OUT_DIR, "model.pt")
TARGET_MEAN_R2 = 0.98
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))


def log(m):
    print(m, flush=True)


def _torch_save_safe(obj, path):
    """torch.save 的安全封装：绕开 PyTorch 在 Windows 下用 ASCII API 打开路径的已知 bug
    （路径含中文等非 ASCII 字符时会误报 "Parent directory ... does not exist"，
    见 https://github.com/pytorch/pytorch/issues/47422）。
    做法：临时把 cwd 切到目标文件所在目录，只把纯文件名交给 torch.save，
    这样传入 PyTorch 底层 C++ 的字符串不含中文路径片段。"""
    path = os.path.abspath(path)
    folder, name = os.path.dirname(path), os.path.basename(path)
    os.makedirs(folder, exist_ok=True)
    _cwd = os.getcwd()
    try:
        os.chdir(folder)
        torch.save(obj, name)
    finally:
        os.chdir(_cwd)


def _torch_load_safe(path, **kwargs):
    """torch.load 的安全封装，原理同 _torch_save_safe。"""
    path = os.path.abspath(path)
    folder, name = os.path.dirname(path), os.path.basename(path)
    _cwd = os.getcwd()
    try:
        os.chdir(folder)
        return torch.load(name, **kwargs)
    finally:
        os.chdir(_cwd)


def _install_numpy_core_shim():
    """兼容不同 numpy 大版本保存/加载 pickle。

    model.pt 中的标准化参数是 numpy 数组，pickle 里会引用其所属模块路径：
      - numpy >= 2.0 使用 `numpy._core`
      - numpy <  2.0 使用 `numpy.core`
    若保存端与加载端 numpy 大版本不同，torch.load 反序列化会报
    `ModuleNotFoundError: No module named 'numpy._core'`（或反之）。
    这里把缺失的一侧别名到已存在的一侧，两个方向都兜底，从而两种 numpy 版本都能加载。
    """
    import importlib

    for want, have in (("numpy._core", "numpy.core"), ("numpy.core", "numpy._core")):
        if want in sys.modules:
            continue
        try:
            importlib.import_module(want)      # 本环境原生就有 → 无需别名
            continue
        except Exception:                      # noqa: BLE001
            pass
        try:
            base = importlib.import_module(have)   # 目标别名源不存在则跳过
        except Exception:                      # noqa: BLE001
            continue
        sys.modules[want] = base
        # 兜底 pickle 常引用的子模块
        for sub in ("multiarray", "_multiarray_umath", "numeric", "umath",
                    "numerictypes", "_dtype", "_dtype_ctypes", "_exceptions",
                    "_methods", "fromnumeric", "_type_aliases"):
            try:
                sys.modules[f"{want}.{sub}"] = importlib.import_module(f"{have}.{sub}")
            except Exception:                  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 数据：长格式 -> 规整序列 (n_geo, n_stroke, ...)
# --------------------------------------------------------------------------- #

def load_data_seq():
    """把 dataset.npz 的长格式 (X,Y,groups) 按几何重组为规整序列张量。

    返回：
      hp (n_geo, 33)          每条几何的硬点坐标（同一几何内恒定，取首行）
      stroke (n_geo, T)       每条几何的 wc_stroke 网格（按升序排列）
      Y (n_geo, T, 8)         对应的曲线值
      gids (n_geo,)           几何编号（原始 groups 的唯一值，用于对照/复现）
      meta                    meta.json
    要求每条几何的采样点数一致（本项目固定 T=101，来自 parse_excel_to_dataset.py）；
    若不一致会报错提示，不做静默截断/填充，避免隐藏数据问题。
    """
    d = np.load(os.path.join(DATA_DIR, "dataset.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8"))
    X, Y, groups = d["X"], d["Y"], d["groups"]
    gids = np.unique(groups)
    counts = np.array([np.sum(groups == g) for g in gids])
    T = int(counts[0])
    if not np.all(counts == T):
        raise ValueError(f"各几何的 stroke 采样点数不一致(min={counts.min()},max={counts.max()})，"
                         "Seq2Seq 需规整序列，请检查数据集。")
    n_geo = len(gids)
    hp = np.zeros((n_geo, X.shape[1] - 1), dtype=np.float64)
    stroke = np.zeros((n_geo, T), dtype=np.float64)
    Yseq = np.zeros((n_geo, T, Y.shape[1]), dtype=np.float64)
    for i, g in enumerate(gids):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(X[idx, -1])]          # 按 stroke 升序排列
        hp[i] = X[idx[0], :-1]
        stroke[i] = X[idx, -1]
        Yseq[i] = Y[idx]
    return hp, stroke, Yseq, gids, meta


def split_geo(n_geo, seed=42, ratios=(0.8, 0.1, 0.1)):
    """按几何编号切分 train/val/test（Attention 版数据已经是"一行一几何"的规整序列，
    不需要像 MLP 版 split_by_group 那样从长格式行反查几何归属，直接对几何下标切分即可）。

    参数:
        n_geo: 几何(曲线)总数
        seed: 随机种子
        ratios: (train比例, val比例)

    返回:
        (train_idx, val_idx, test_idx)：三个几何下标数组（不是布尔掩码，与 MLP 版不同）
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_geo)
    n_tr, n_va = int(ratios[0] * n_geo), int(ratios[1] * n_geo)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    return tr, va, te


class Standardizer:
    """逐列 z-score 标准化器（与 train_nn.py 同名类完全一致）：
    (x-mean)/std 标准化，以及 inv() 反变换还原回原始物理量纲。"""
    def __init__(self, A):
        self.mean = A.mean(0)
        self.std = A.std(0)
        self.std[self.std < 1e-8] = 1.0
    def tf(self, A):  return (A - self.mean) / self.std
    def inv(self, A): return A * self.std + self.mean


# --------------------------------------------------------------------------- #
# 模型：Attention Encoder-Decoder（Transformer 风格，替代原 GRU 版本）
# --------------------------------------------------------------------------- #

class Seq2SeqAttnKC(nn.Module):
    """Encoder: 33 硬点(标准化后) -> 上下文 token 序列(用作 memory，供 decoder 做 cross-attention)。
       实际上把硬点编码成 n_ctx 个 token，天然对应「11 个硬点×3坐标」这种分组结构的推广。
    Decoder: 把 wc_stroke 序列当作 query token 序列（sin/cos 位置编码 stroke 值本身，
             而非序列下标——因为 stroke 数值就是这个任务里有物理意义的"位置"），
             叠 n_layers 个 TransformerDecoderLayer：
               - self-attention：让曲线上每个 stroke 点都能看到序列内其它所有点(双向，无因果mask，
                 因为整条曲线在推理时是一次性给定 stroke 网格，不是自回归生成)；
               - cross-attention：每个 stroke 点查询硬点编码出的上下文 token，
                 把"这组硬点几何"的信息带入该点的输出。
             最后线性投影到 8 维运动学量。
    相比 GRU 版本(Seq2SeqKC)的关键差异：
      - 整条曲线并行计算（无需像 GRU 那样逐时间步循环、无需 teacher forcing），训练更快更稳定；
      - self-attention 让任意两个 stroke 点直接建立联系，不受 GRU 隐状态"只能顺序传递信息"的限制；
      - 不再有 exposure bias 问题（GRU 版本纯自回归时测试更差，就是这个原因）。
    """
    def __init__(self, n_hp, n_out, hidden=128, n_layers=2, n_heads=4,
                 n_ctx=8, ff_mult=4, p_drop=0.1):
        super().__init__()
        self.hidden, self.n_out, self.n_ctx = hidden, n_out, n_ctx

        # Encoder：33 硬点 -> n_ctx 个上下文 token（每个 token 一份独立线性头，捕获不同子结构）
        enc_layers, d = [], n_hp
        for _ in range(2):
            enc_layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
            d = hidden
        self.encoder = nn.Sequential(*enc_layers)
        self.ctx_proj = nn.Linear(hidden, hidden * n_ctx)

        # stroke 位置编码：把标准化后的 stroke 标量映射成 hidden 维（sin/cos 多频率 + 线性）
        self.n_freq = hidden // 4
        assert self.n_freq >= 1, "hidden 太小，至少需要 4"
        freq = torch.exp(torch.linspace(0, 4, self.n_freq))     # 多个频率尺度
        self.register_buffer("pe_freq", freq)
        self.pe_proj = nn.Linear(2 * self.n_freq, hidden)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * ff_mult,
            dropout=p_drop, batch_first=True, activation="relu")
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden, n_out)

    def _stroke_pe(self, stroke_seq):
        """stroke_seq: (B, T) 标准化后的 stroke 值 -> (B, T, hidden) 位置编码。"""
        s = stroke_seq.unsqueeze(-1) * self.pe_freq.view(1, 1, -1)      # (B,T,n_freq)
        pe = torch.cat([torch.sin(s), torch.cos(s)], dim=-1)            # (B,T,2*n_freq)
        return self.pe_proj(pe)                                        # (B,T,hidden)

    def forward(self, hp, stroke_seq, y_seq=None, teacher_forcing=False):
        """hp: (B, n_hp) 标准化后硬点。stroke_seq: (B, T) 标准化后 stroke 序列。
        y_seq/teacher_forcing：保留参数签名以兼容旧调用方式，Attention 版本整条序列并行计算，
        不需要也不使用 teacher forcing（无论传入与否，行为一致）。
        返回: (B, T, n_out) 标准化空间的预测曲线。
        """
        B, T = stroke_seq.shape
        ctx = self.encoder(hp)                                          # (B, hidden)
        memory = self.ctx_proj(ctx).view(B, self.n_ctx, self.hidden)     # (B, n_ctx, hidden)
        tgt = self._stroke_pe(stroke_seq)                                # (B, T, hidden)
        dec = self.decoder(tgt, memory)                                  # (B, T, hidden) 双向self-attn+cross-attn
        return self.out_proj(dec)                                       # (B, T, n_out)


# 向后兼容别名：旧代码/已保存模型如引用 Seq2SeqKC 名称，指向新的 Attention 实现。
Seq2SeqKC = Seq2SeqAttnKC


class Seq2SeqAttnKC_V2(nn.Module):
    """"方案 B"：加大容量 + 更标准的深层 Transformer 训练配方，与 Seq2SeqAttnKC 的差异：
      - d_model 128->256（默认），单头维度不变(256/8=32，与原128/4=32一致)，仅头数从4->8、层数从2->4；
      - memory token 数从 8->4（编码后的硬点上下文槽位更精简）；
      - decoder 除了 stroke 数值的 sin/cos 位置编码，额外叠加一组"可学习 position embedding"
        （按 stroke 序列下标索引，序列长度固定为 T=101，来自数据集格式约定），
        让模型除了"知道这是第几毫米的行程"外，还能学到"这个位置本身的专属模式"；
      - TransformerDecoderLayer 用 norm_first=True（Pre-LN），层数变深(4层)时训练更稳定；
      - 输出头从单层 Linear 换成两层 MLP（hidden->2*hidden->n_out），回归头本身也加深加宽。
    容量更大、更接近主流工业界的 Transformer 配方，但也更容易在小数据集上过拟合、训练更慢，
    是否比 Seq2SeqAttnKC 更好需要实测（同数据/同预算对比 R²/MAE）。
    与 Seq2SeqAttnKC 相同的输入输出形状约定：forward(hp:(B,n_hp), stroke_seq:(B,T)) -> (B,T,n_out)。
    """
    def __init__(self, n_hp, n_out, hidden=256, n_layers=4, n_heads=8,
                 n_ctx=4, ff_mult=4, p_drop=0.1, max_len=101):
        super().__init__()
        self.hidden, self.n_out, self.n_ctx, self.max_len = hidden, n_out, n_ctx, max_len

        # Encoder：33 硬点 -> n_ctx 个上下文 token
        enc_layers, d = [], n_hp
        for _ in range(2):
            enc_layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
            d = hidden
        self.encoder = nn.Sequential(*enc_layers)
        self.ctx_proj = nn.Linear(hidden, hidden * n_ctx)

        # stroke 数值的 sin/cos 位置编码（与 Seq2SeqAttnKC 一致，物理数值本身就是"位置"）
        self.n_freq = hidden // 4
        assert self.n_freq >= 1, "hidden 太小，至少需要 4"
        freq = torch.exp(torch.linspace(0, 4, self.n_freq))
        self.register_buffer("pe_freq", freq)
        self.pe_proj = nn.Linear(2 * self.n_freq, hidden)

        # 额外的可学习 position query（按序列下标索引，序列长度固定为 max_len）：
        # 与上面的数值位置编码互补——数值编码知道"这是多少毫米的行程"，
        # 这组 embedding 让模型再学到"数据集里第几个采样点通常长什么样"的专属模式。
        self.pos_query = nn.Parameter(torch.randn(max_len, hidden) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * ff_mult,
            dropout=p_drop, batch_first=True, activation="relu",
            norm_first=True)                                    # Pre-LN，深层(4层)训练更稳定
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # 两层 MLP 输出头（比单层 Linear 多一次非线性变换）
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden * 2, n_out))

    def _stroke_pe(self, stroke_seq):
        """stroke_seq: (B, T) 标准化后的 stroke 值 -> (B, T, hidden) 数值位置编码。"""
        s = stroke_seq.unsqueeze(-1) * self.pe_freq.view(1, 1, -1)
        pe = torch.cat([torch.sin(s), torch.cos(s)], dim=-1)
        return self.pe_proj(pe)

    def forward(self, hp, stroke_seq, y_seq=None, teacher_forcing=False):
        """形状约定与 Seq2SeqAttnKC 完全一致；y_seq/teacher_forcing 保留兼容旧调用签名，不使用。"""
        B, T = stroke_seq.shape
        ctx = self.encoder(hp)
        memory = self.ctx_proj(ctx).view(B, self.n_ctx, self.hidden)
        tgt = self._stroke_pe(stroke_seq)
        if T <= self.max_len:
            tgt = tgt + self.pos_query[:T].unsqueeze(0)          # 叠加可学习 position query
        # T > max_len(101) 时说明调用方传入了训练时未见过的更长序列，跳过可学习位置项
        # （数值位置编码本身仍对任意 T 有效，只是这组"专属模式"embedding 覆盖不到，不报错、静默降级）
        dec = self.decoder(tgt, memory)
        return self.out_proj(dec)


def pred_seq(model, xs, ys, ss, hp33_batch, stroke_batch):
    """给定原量纲 33 硬点(B,33) + stroke 网格(B,T) -> 原量纲曲线预测 (B,T,8)。"""
    model.eval()
    with torch.no_grad():
        hp_t = torch.tensor(xs.tf(hp33_batch), dtype=torch.float32)
        st_t = torch.tensor(ss.tf(stroke_batch[..., None]).squeeze(-1), dtype=torch.float32)
        yp = model(hp_t, st_t, y_seq=None, teacher_forcing=False).numpy()
    return ys.inv(yp)


# --------------------------------------------------------------------------- #
# 指标（与 train_nn.py 保持同口径，便于直接对比两种架构）
# --------------------------------------------------------------------------- #

def r2(yt, yp):
    """决定系数 R²（与 train_nn.py 同名函数完全一致，见该文件详细注释）。"""
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_target_metrics(y_true_flat, y_pred_flat, names):
    """逐个输出量分别计算 R²/RMSE/MAE，并汇总不加权平均R²(_mean_R2)。
    与 train_nn.py 的同名函数逻辑一致，只是这里的输入需要调用方先把序列展平成
    (n_rows, n_out) 二维数组（Attention 版数据原始形状是 (n_geo, T, n_out)）。"""
    out = {}
    for i, n in enumerate(names):
        yt, yp = y_true_flat[:, i], y_pred_flat[:, i]
        out[n] = {"R2": float(r2(yt, yp)),
                  "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
                  "MAE": float(np.mean(np.abs(yt - yp)))}
    out["_mean_R2"] = float(np.mean([out[n]["R2"] for n in names]))
    return out


def print_metrics(title, mt, names):
    """把 per_target_metrics() 的结果格式化打印成表格。"""
    log(f"\n== {title} ==  (平均 R²={mt['_mean_R2']:.4f})")
    log(f"{'quantity':<20}{'R2':>10}{'RMSE':>14}{'MAE':>14}")
    for n in names:
        d = mt[n]
        log(f"{n:<20}{d['R2']:>10.4f}{d['RMSE']:>14.5g}{d['MAE']:>14.5g}")


Q_LABEL = {
    "toe": "前束角 Toe (°)", "camber": "外倾角 Camber (°)", "caster": "后倾角 Caster (°)",
    "caster_arm": "后倾拖距 Caster Arm (mm)", "scrub_radius": "主销偏移距 Scrub Radius (mm)",
    "tire_con_point_x": "接地点 X (mm)", "tire_con_point_y": "接地点 Y (mm)",
    "tire_con_point_z": "接地点 Z (mm)",
}


def curve_error_metrics(y_true_seq, y_pred_seq, names):
    """按曲线评估：MAE/中位数/P90/最大误差/平滑度D2（与 train_nn.py 的同名函数
    统计口径一致，但输入输出的数据形状不同——这里直接接收规整的三维序列数组，
    不需要像 MLP 版那样从长格式数据里按 group 重新切片排序，因为 Attention 版
    的数据本来就已经是 (n_geo, T, n_out) 的规整形状，可以直接向量化计算二阶差分。

    参数:
        y_true_seq/y_pred_seq: (n_geo, T, n_out) 原量纲真实值/预测值
        names: 输出量名称列表

    返回:
        dict，键为各量名 -> {"MAE","Median","P90","MaxError","D2"}
    """
    ae = np.abs(y_pred_seq - y_true_seq)                       # (n_geo,T,n_out)
    n_geo = y_true_seq.shape[0]
    samp = np.arange(n_geo) if n_geo <= 300 else \
        np.random.default_rng(0).choice(n_geo, 300, replace=False)   # D2最多抽样300条几何(提速)
    dd = np.abs(y_pred_seq[samp, 2:] - 2 * y_pred_seq[samp, 1:-1] + y_pred_seq[samp, :-2])
    out = {}
    for j, n in enumerate(names):
        e = ae[:, :, j].reshape(-1)
        out[n] = {"MAE": float(e.mean()), "Median": float(np.median(e)),
                  "P90": float(np.percentile(e, 90)), "MaxError": float(e.max()),
                  "D2": float(dd[:, :, j].mean())}
    return out


def print_curve_table(cm, names):
    """把 curve_error_metrics() 的结果格式化打印成表格。"""
    log(f"\n{'曲线 / Quantity':<28}{'MAE':>10}{'Median':>10}{'P90':>10}"
        f"{'MaxError':>11}{'平滑度D2':>11}")
    for n in names:
        d = cm[n]
        log(f"{Q_LABEL.get(n, n):<28}{d['MAE']:>10.5g}{d['Median']:>10.5g}"
            f"{d['P90']:>10.5g}{d['MaxError']:>11.5g}{d['D2']:>11.5g}")


def design_position_validation(y_true_seq, y_pred_seq, stroke_seq, names):
    """设计位(|stroke|<tol)处，模型/实测 toe/camber 与预设对照（跨测试集几何插值取0点）。"""
    tol = cfg.DESIGN_POS_TOL_MM
    n_geo, T, _ = y_true_seq.shape
    idx_toe, idx_cam = names.index("toe"), names.index("camber")

    def val0(seq, j):
        return np.array([np.interp(0.0, stroke_seq[i], seq[i, :, j]) for i in range(n_geo)])

    toe_pred, cam_pred = val0(y_pred_seq, idx_toe), val0(y_pred_seq, idx_cam)
    toe_true, cam_true = val0(y_true_seq, idx_toe), val0(y_true_seq, idx_cam)
    res = {"tol_mm": tol, "n_points": int(n_geo),
           "preset_init_toe_deg": cfg.INIT_TOE_DEG, "preset_init_camber_deg": cfg.INIT_CAMBER_DEG,
           "toe_pred_mean": float(toe_pred.mean()), "toe_actual_mean": float(toe_true.mean()),
           "camber_pred_mean": float(cam_pred.mean()), "camber_actual_mean": float(cam_true.mean())}
    log("\n设计位校验(测试集 {} 条几何，stroke=0 插值):".format(n_geo))
    log("  前束 toe : 预设 {:.3f}° | 实测 {:.3f}° | 模型 {:.3f}°".format(
        cfg.INIT_TOE_DEG, res["toe_actual_mean"], res["toe_pred_mean"]))
    log("  外倾 camber: 预设 {:.3f}° | 实测 {:.3f}° | 模型 {:.3f}°".format(
        cfg.INIT_CAMBER_DEG, res["camber_actual_mean"], res["camber_pred_mean"]))
    return res


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #

def train(epochs=300, time_budget=240.0, patience=25, batch=64,
          hidden=128, n_layers=2, n_heads=4, n_ctx=8, lr=1.5e-3,
          weight_decay=2e-4, p_drop=0.1, arch="v1"):
    """Attention(Transformer Decoder) 版本：整条曲线一次性并行前向，不需要 teacher forcing。
    arch: "v1"=Seq2SeqAttnKC(原实现，默认超参 hidden=128/layers=2/heads=4/n_ctx=8)；
          "v2"=Seq2SeqAttnKC_V2("方案B"：更大容量 d_model/层数/头数 + 可学习position query +
          Pre-LN + 两层MLP输出头，默认超参见 main() 的 --hidden/--layers/--heads/--ctx 默认值，
          v2 模式下这些参数的默认值也相应调整为 256/4/8/4，可通过 CLI 覆盖）。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    hp, stroke, Yseq, gids, meta = load_data_seq()
    names = meta["target_names"]
    n_geo, T, n_out = Yseq.shape
    log(f"数据: 几何={n_geo} 每条曲线点数={T} 硬点维={hp.shape[1]} 输出维={n_out}")

    tr, va, te = split_geo(n_geo)
    log(f"划分(按几何): train={len(tr)} val={len(va)} test={len(te)}")

    xs = Standardizer(hp[tr])
    ys = Standardizer(Yseq[tr].reshape(-1, n_out))
    ss = Standardizer(stroke[tr].reshape(-1, 1))

    def to_seq_tensors(ids):
        hp_t = torch.tensor(xs.tf(hp[ids]), dtype=torch.float32)
        st_t = torch.tensor(ss.tf(stroke[ids][..., None]).squeeze(-1), dtype=torch.float32)
        y_t = torch.tensor(ys.tf(Yseq[ids].reshape(-1, n_out)).reshape(len(ids), T, n_out),
                           dtype=torch.float32)
        return hp_t, st_t, y_t

    Xtr_hp, Xtr_st, Ytr = to_seq_tensors(tr)
    Xva_hp, Xva_st, _ = to_seq_tensors(va)
    Yva_orig = Yseq[va]

    model_cls = Seq2SeqAttnKC_V2 if arch == "v2" else Seq2SeqAttnKC
    model = model_cls(hp.shape[1], n_out, hidden=hidden, n_layers=n_layers,
                      n_heads=n_heads, n_ctx=n_ctx, p_drop=p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    lossf = nn.MSELoss()

    log(f"模型={model_cls.__name__}(arch={arch}) hidden={hidden} layers={n_layers} heads={n_heads} "
        f"n_ctx={n_ctx} drop={p_drop} 参数量={sum(p.numel() for p in model.parameters())}")

    n_tr = len(tr)
    rng = np.random.default_rng(42)
    best, bstate, noimp, ep, t0 = -1e9, None, 0, 0, _t.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            yp = model(Xtr_hp[idx], Xtr_st[idx])            # 并行前向，无需 teacher forcing
            loss = lossf(yp, Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            yv = model(Xva_hp, Xva_st).numpy()
        yv_orig = ys.inv(yv.reshape(-1, n_out)).reshape(len(va), T, n_out)
        vr2 = float(np.mean([r2(Yva_orig[:, :, k].reshape(-1), yv_orig[:, :, k].reshape(-1))
                             for k in range(n_out)]))
        sched.step(vr2)
        if ep % 10 == 0 or ep == 1:
            log(f"  epoch {ep:4d}  val_meanR²={vr2:.4f}")
        if vr2 > best + 1e-4:
            best, noimp = vr2, 0
            bstate = copy.deepcopy(model.state_dict())
        else:
            noimp += 1
        if noimp >= patience or (_t.time() - t0) > time_budget:
            log(f"  早停于 epoch {ep}（{'耐心耗尽' if noimp >= patience else '超时'}）")
            break
    if bstate is not None:
        model.load_state_dict(bstate)
    log(f"训练完成：最优验证R²={best:.4f}")

    def eval_split(ids):
        hp_o, st_o, y_o = hp[ids], stroke[ids], Yseq[ids]
        yp = pred_seq(model, xs, ys, ss, hp_o, st_o)
        return y_o, yp

    y_tr_t, y_tr_p = eval_split(tr)
    y_va_t, y_va_p = eval_split(va)
    y_te_t, y_te_p = eval_split(te)

    m_tr = per_target_metrics(y_tr_t.reshape(-1, n_out), y_tr_p.reshape(-1, n_out), names)
    m_va = per_target_metrics(y_va_t.reshape(-1, n_out), y_va_p.reshape(-1, n_out), names)
    m_te = per_target_metrics(y_te_t.reshape(-1, n_out), y_te_p.reshape(-1, n_out), names)
    print_metrics("训练集", m_tr, names)
    print_metrics("验证集", m_va, names)
    print_metrics("测试集", m_te, names)
    passed = m_te["_mean_R2"] >= TARGET_MEAN_R2
    log(f"\n指标判据：测试集平均 R²={m_te['_mean_R2']:.4f} 目标>= {TARGET_MEAN_R2} "
        f"=> {'✅ 达标' if passed else '⚠️ 未达标'}")

    cm_te = curve_error_metrics(y_te_t, y_te_p, names)
    log("\n== 测试集 逐曲线误差指标 ==")
    print_curve_table(cm_te, names)

    dpv = design_position_validation(y_te_t, y_te_p, stroke[te], names)

    _torch_save_safe({
        "state_dict": model.state_dict(),
        "arch": {"n_hp": hp.shape[1], "n_out": n_out, "hidden": hidden,
                 "n_layers": n_layers, "n_heads": n_heads, "n_ctx": n_ctx, "p_drop": p_drop,
                 "arch_version": arch},
        "x_mean": xs.mean, "x_std": xs.std,
        "y_mean": ys.mean, "y_std": ys.std,
        "s_mean": ss.mean, "s_std": ss.std,
        "meta": meta,
    }, MODEL_PATH)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "val": m_va, "test": m_te, "passed": passed,
                   "target_mean_R2": TARGET_MEAN_R2, "model_kind": "seq2seq_attention",
                   "arch_version": arch,
                   "test_curve_metrics": cm_te,
                   "design_position_validation": dpv,
                   "presets": meta["presets"]}, f, ensure_ascii=False, indent=2)
    log(f"已保存模型: {MODEL_PATH}")
    log(f"已保存指标: {os.path.join(OUT_DIR, 'metrics.json')}")

    plot_example(y_te_t, y_te_p, stroke[te], te, gids, names)
    return m_te


def plot_example(y_true_seq, y_pred_seq, stroke_seq, te_ids, gids, names):
    """挑测试集里的第一条几何曲线，画出预测vs真实的对照图(与 train_nn.py 的同名函数
    功能一致)，并导出该曲线的完整数据为 CSV。注意本函数接收的是已经按几何下标切好
    的序列数据(y_true_seq/y_pred_seq形状为(n_te,T,n_out))，而不是 train_nn.py 那种
    长格式+groups需要现场按几何筛选排序的数据，因为 Attention 版数据本来就是规整序列。"""
    i = 0
    sid = int(gids[te_ids[i]])
    stroke, y_true, y_pred = stroke_seq[i], y_true_seq[i], y_pred_seq[i]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for k, name in enumerate(names):
        ax = axes[k // 4][k % 4]
        ax.plot(stroke, y_true[:, k], "b-", lw=2, label="actual")
        ax.plot(stroke, y_pred[:, k], "r--", lw=2, label="predicted")
        if name == "toe":
            ax.scatter([0], [cfg.INIT_TOE_DEG], c="g", marker="*", s=140, zorder=5,
                       label=f"preset init {cfg.INIT_TOE_DEG}")
        if name == "camber":
            ax.scatter([0], [cfg.INIT_CAMBER_DEG], c="g", marker="*", s=140, zorder=5,
                       label=f"preset init {cfg.INIT_CAMBER_DEG}")
        ax.set_title(f"{name}  (R2={r2(y_true[:, k], y_pred[:, k]):.4f})")
        ax.set_xlabel("wc_stroke"); ax.grid(True, alpha=0.3)
        if name in ("toe", "camber") or k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Test sample Run#{sid}: predicted vs actual (Seq2Seq Attention/Transformer) | "
                 f"tire R={cfg.TIRE_STATIC_RADIUS_MM}mm", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT_DIR, f"curves_test_sample_{sid}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    log(f"已保存示例曲线图: {p}")

    import csv
    cp = os.path.join(OUT_DIR, f"curves_test_sample_{sid}.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wc_stroke"] + [f"{n}_true" for n in names] + [f"{n}_pred" for n in names])
        for t in range(len(stroke)):
            w.writerow([stroke[t]] + list(y_true[t]) + list(y_pred[t]))
    log(f"已保存示例曲线数据: {cp}")


def load_bundle():
    """从 MODEL_PATH 加载 Attention 模型的完整训练产物：模型权重+3个标准化器(硬点/输出/行程)+meta。
    与 train_nn.load_bundle() 的主要区别：
      1) 无集成(Ensemble)，Attention 版本目前每次只训练单个模型；
      2) 需要额外的 ss(stroke标准化器)，因为 Attention 模型的 forward() 需要标准化后的
         stroke 序列作为 decoder 的 query 输入；
      3) 根据 checkpoint 里的 arch_version 字段自动选择 Seq2SeqAttnKC(v1,默认)还是
         Seq2SeqAttnKC_V2(v2,"方案B")重建模型结构，保证不同架构训练出的模型都能被
         正确加载(state_dict 的键名结构不同，选错类会报 key mismatch)。

    返回:
        (model, xs, ys, ss, meta)：model 已 eval()
    """
    _install_numpy_core_shim()                 # 先装兼容 shim，再反序列化
    ckpt = _torch_load_safe(MODEL_PATH, weights_only=False)
    a = ckpt["arch"]
    model_cls = Seq2SeqAttnKC_V2 if a.get("arch_version") == "v2" else Seq2SeqAttnKC
    model = model_cls(a["n_hp"], a["n_out"], hidden=a["hidden"],
                      n_layers=a["n_layers"], n_heads=a.get("n_heads", 4),
                      n_ctx=a.get("n_ctx", 8), p_drop=a["p_drop"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    xs, ys, ss = (Standardizer.__new__(Standardizer) for _ in range(3))
    xs.mean, xs.std = ckpt["x_mean"], ckpt["x_std"]
    ys.mean, ys.std = ckpt["y_mean"], ckpt["y_std"]
    ss.mean, ss.std = ckpt["s_mean"], ckpt["s_std"]
    return model, xs, ys, ss, ckpt["meta"]


def infer_only():
    """--infer-only 模式：跳过训练，直接加载已有 model.pt，对测试集第一条几何画一次
    示例推理图（与 train_nn.py 的同名函数功能一致，适配序列数据接口）。"""
    if not os.path.isfile(MODEL_PATH):
        log("未找到 model.pt，请先训练。"); return
    model, xs, ys, ss, meta = load_bundle()
    hp, stroke, Yseq, gids, _ = load_data_seq()
    _, _, te = split_geo(len(gids))
    y_pred = np.stack([pred_seq(model, xs, ys, ss, hp[[i]], stroke[[i]])[0] for i in te[:1]])
    plot_example(Yseq[te[:1]], y_pred, stroke[te[:1]], te[:1], gids, meta["target_names"])


def main():
    """命令行入口：解析参数，按 --arch 选择 v1/v2 架构默认超参后调用 train() 或 infer_only()。
    v1(默认)=Seq2SeqAttnKC 原实现；v2="方案B"=Seq2SeqAttnKC_V2(更大容量+可学习position
    query+Pre-LN+两层MLP输出头)。若用户显式传入 --hidden/--layers/--heads/--ctx 中的任意
    一个，则该参数以用户传入值为准，其余未传入的参数仍按 --arch 对应的默认值补全。"""
    ap = argparse.ArgumentParser(description="Seq2Seq(Transformer Attention Encoder-Decoder) "
                                              "训练/推理 悬架运动学代理模型")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--data", default=None, help="数据集目录(默认 nn_data)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--time-budget", type=float, default=240.0)
    ap.add_argument("--batch", type=int, default=64, help="按几何计的批大小(整条曲线为一个样本)")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1",
                    help="v1=Seq2SeqAttnKC(原实现,默认); "
                         "v2=\"方案B\"(更大容量+可学习position query+Pre-LN+两层MLP输出头,"
                         "见 Seq2SeqAttnKC_V2 类注释)。--hidden/--layers/--heads/--ctx 未显式传入时，"
                         "v2 会使用更大的默认值(256/4/8/4)而不是 v1 的默认值(128/2/4/8)。")
    ap.add_argument("--hidden", type=int, default=None, help="Transformer 隐藏维度(d_model)"
                    "，未传时按--arch取默认值(v1=128,v2=256)")
    ap.add_argument("--layers", type=int, default=None, help="TransformerDecoder 层数"
                    "，未传时按--arch取默认值(v1=2,v2=4)")
    ap.add_argument("--heads", type=int, default=None, help="多头注意力头数"
                    "，未传时按--arch取默认值(v1=4,v2=8)")
    ap.add_argument("--ctx", type=int, default=None, help="硬点编码成的上下文 token 数(cross-attention用)"
                    "，未传时按--arch取默认值(v1=8,v2=4)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()
    if args.data:
        globals()["DATA_DIR"] = os.path.abspath(args.data)
        log(f"使用数据集目录: {DATA_DIR}")

    v1_defaults = {"hidden": 128, "layers": 2, "heads": 4, "ctx": 8}
    v2_defaults = {"hidden": 256, "layers": 4, "heads": 8, "ctx": 4}
    d = v2_defaults if args.arch == "v2" else v1_defaults
    hidden = args.hidden if args.hidden is not None else d["hidden"]
    n_layers = args.layers if args.layers is not None else d["layers"]
    n_heads = args.heads if args.heads is not None else d["heads"]
    n_ctx = args.ctx if args.ctx is not None else d["ctx"]

    if args.infer_only:
        infer_only()
    else:
        train(epochs=args.epochs, time_budget=args.time_budget, batch=args.batch,
              hidden=hidden, n_layers=n_layers, n_heads=n_heads, n_ctx=n_ctx,
              p_drop=args.dropout, weight_decay=args.weight_decay, lr=args.lr, arch=args.arch)


if __name__ == "__main__":
    main()
