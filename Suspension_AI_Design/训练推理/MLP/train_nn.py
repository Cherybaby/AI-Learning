#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 步：用 PyTorch 训练/验证「硬点+wc_stroke -> 8 个运动学量」的 MLP，
并对给定硬点扫掠 wc_stroke 推理出曲线。

输入：nn_data/dataset.npz + nn_data/meta.json（由 parse_excel_to_dataset.py 生成）
输出（默认 nn_out/）：
  - model.pt        : 模型权重 + 结构 + 标准化参数 + 元数据 + 预设常量
  - metrics.json    : 各量 R²/RMSE/MAE（train/val/test）+ 设计位校验
  - curves_test_sample_*.png / .csv : 某测试几何的预测 vs 真实曲线（含预设参考点）

用法：
  python3 train_nn.py                     # 训练+评估+保存+示例推理绘图
  python3 train_nn.py --infer-only        # 用已存 model.pt 做示例推理
  python3 train_nn.py --epochs 400 --time-budget 240
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

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
OUT_DIR = os.path.join(_HERE, "nn_out")
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
# 数据
# --------------------------------------------------------------------------- #

def load_data():
    """从 DATA_DIR 读取长格式数据集（由 parse_excel_to_dataset.py / generate_dataset.py /
    merge_datasets.py 之一生成）。

    返回:
        X: (n_rows, 34) 33个硬点坐标 + wc_stroke
        Y: (n_rows, 8) 8个运动学量
        groups: (n_rows,) 每行所属的几何编号
        meta: dict，来自 meta.json，含 target_names/feat_names/presets 等元信息
        source: (n_rows,) 或 None —— 仅合并数据集(nn_data_merged)含有此字段，
                0=Excel真实数据，1=求解器生成数据；单一来源数据集(nn_data/nn_data_gen)
                没有这个字段，返回 None。
    """
    d = np.load(os.path.join(DATA_DIR, "dataset.npz"))
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8"))
    source = d["source"] if "source" in d.files else None   # 合并数据集专用：0=Excel真实,1=求解器生成
    return d["X"], d["Y"], d["groups"], meta, source


def split_by_group(groups, seed=42, ratios=(0.8, 0.1, 0.1)):
    """按"几何编号(group)"整体切分 train/val/test，而不是按行随机切分。

    为什么要按几何切分：数据集是长格式，同一条曲线(同一几何)在 101 个行程点上
    展开成 101 行。如果直接按行随机切分，同一条曲线的行程点会同时出现在训练集
    和测试集里，模型只需要"记住这条曲线的形状"就能在测试集上刷出虚高的分数，
    这是典型的数据泄漏。按整条曲线(group)切分能保证测试集里的几何是模型完全
    没见过的，指标才能真实反映泛化能力。

    参数:
        groups: (n_rows,) 每行所属的几何编号
        seed: 随机种子，保证多次运行的切分结果可复现
        ratios: (train比例, val比例)，测试集占比为 1-train-val

    返回:
        (train_mask, val_mask, test_mask, (n_train_geo, n_val_geo, n_test_geo))
        前三者是长度为 n_rows 的布尔掩码，最后是三个集合各自的几何(曲线)数量。
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)                       # 先打乱全部几何编号，再顺序切片
    n = len(uniq)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    tr, va, te = uniq[:n_tr], uniq[n_tr:n_tr + n_va], uniq[n_tr + n_va:]
    m = lambda ids: np.isin(groups, ids)    # 把"属于某个几何集合"转成行级布尔掩码
    return m(tr), m(va), m(te), (len(tr), len(va), len(te))


class Standardizer:
    """逐列 z-score 标准化器：(x - mean) / std，以及反变换 x*std + mean。

    神经网络对输入/输出的数值量级很敏感（硬点坐标量级是千位mm，运动学角度量级
    是个位数度数，量级差异过大会导致梯度更新不均衡），训练前先对 X 和 Y 分别做
    标准化（只用训练集统计量，避免验证/测试集信息泄漏到标准化参数里），
    推理时用 inv() 把网络输出的标准化空间数值还原回真实物理量纲。
    """
    def __init__(self, A):
        self.mean = A.mean(0)
        self.std = A.std(0)
        self.std[self.std < 1e-8] = 1.0        # 防止近似常数列除零放大
    def tf(self, A):  return (A - self.mean) / self.std
    def inv(self, A): return A * self.std + self.mean


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    """残差块：x -> Linear->ReLU->Dropout->Linear -> (+x) -> ReLU。
    输入输出维度相同时直接相加；维度不同的第一层用于变维，其后的块维度都相同可以恒等相加。
    这样加深网络时能保留浅层特征的直连梯度通路，缓解深网络训练变难/梯度消失的问题。"""
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(p_drop)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(x + h)


class ResTrunk(nn.Module):
    """残差版主干：先投影到统一宽度 d，再堆叠 n_blocks 个 ResBlock，每块输出维度不变。
    用于替代 MLP/PointMLP 里原来的"逐层变宽变窄"的 Sequential 主干；
    residual=False 时退化为普通逐层 Sequential（与原实现等价），便于对比。"""
    def __init__(self, d_in, hidden, p_drop=0.1, residual=True):
        super().__init__()
        self.residual = residual
        if residual:
            d = hidden[0] if hidden else d_in
            self.in_proj = nn.Linear(d_in, d) if d_in != d else nn.Identity()
            self.blocks = nn.ModuleList([ResBlock(d, p_drop=p_drop) for _ in hidden])
            self.out_dim = d
        else:
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
                d = h
            self.plain = nn.Sequential(*layers)
            self.out_dim = d

    def forward(self, x):
        if self.residual:
            x = self.in_proj(x)
            for b in self.blocks:
                x = b(x)
            return x
        return self.plain(x)


class MultiHead(nn.Module):
    """输出头：共享主干特征 -> 大多数量走一个共享的 nn.Linear（和原实现等价）；
    hard_idx 指定的"难学"量（如 caster_arm/scrub_radius）各自额外配一个独立的
    2 层小 MLP 头（trunk_dim -> head_hidden -> 1），再与共享输出的对应位置相加。

    动机：单纯加大这些量在损失函数里的权重，训练早期确实能让梯度更偏向它们，
    但共享的最后一层 nn.Linear 的权重矩阵仍然是"一次性"给全部 n_out 个量、
    容量没有变化——加权重加到一定程度后，继续加大反而挤占其它量的优化预算，
    整体变差（实测：权重从2.5提到4.0后caster_arm/scrub_radius的MAE不降反升）。
    独立小头给这些量单独的非线性变换能力，不占用共享层的容量，
    是比"继续加大权重"更根本的解法；跟按量加权配合使用效果更好（本身仍支持
    target_weights，只是现在有更大的容量去响应权重带来的梯度）。

    hard_idx=None（默认）时退化为单一 nn.Linear(trunk_dim, n_out)，与原实现完全等价。"""
    def __init__(self, trunk_dim, n_out, hard_idx=None, head_hidden=64, p_drop=0.1):
        super().__init__()
        self.hard_idx = sorted(set(hard_idx)) if hard_idx else []
        self.shared = nn.Linear(trunk_dim, n_out)
        self.extra_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(trunk_dim, head_hidden), nn.ReLU(),
                          nn.Dropout(p_drop), nn.Linear(head_hidden, 1))
            for _ in self.hard_idx
        ])

    def forward(self, feat):
        out = self.shared(feat)
        if not self.hard_idx:
            return out
        out = out.clone()
        for j, head in zip(self.hard_idx, self.extra_heads):
            out[:, j] = out[:, j] + head(feat).squeeze(-1)
        return out


class MLP(nn.Module):
    """打平版 MLP：把 33 维硬点 + wc_stroke 直接拼成 34 维向量整体送入主干网络
    （不做逐点分组编码，对应 CLI 的 --model flat 选项，用于和 PointMLP 对比）。"""
    def __init__(self, n_in, n_out, hidden=(256, 256, 128), p_drop=0.1, residual=False,
                 hard_idx=None, head_hidden=64):
        super().__init__()
        self.trunk = ResTrunk(n_in, hidden, p_drop=p_drop, residual=residual)
        self.head = MultiHead(self.trunk.out_dim, n_out, hard_idx=hard_idx,
                              head_hidden=head_hidden, p_drop=p_drop)

    def forward(self, x):
        return self.head(self.trunk(x))


class PointMLP(nn.Module):
    """逐点三维编码器：把 33 维硬点拆成 n_points 个 (x,y,z) 单元，
    每个点用**独立**权重的两层小 MLP 编码成 emb 维（保留点的身份），
    再拼接所有点嵌入 + wc_stroke 送入主干网络。
    这样注入“同一硬点的三坐标是一个整体”的归纳偏置。
    residual=True 时主干换成 ResTrunk(残差块堆叠)，支持更深的网络而不易梯度消失。
    hard_idx: 见 MultiHead 说明，给指定输出维度额外配独立小头。"""
    def __init__(self, n_in, n_out, n_points=11, coord=3, emb=16,
                 trunk=(256, 128), p_drop=0.1, residual=False,
                 hard_idx=None, head_hidden=64):
        super().__init__()
        assert n_points * coord + 1 == n_in, "输入维应为 n_points*3 + 1(wc_stroke)"
        self.n_points, self.coord, self.emb = n_points, coord, emb
        # 每个点独立的 3->emb->emb（用 einsum 做“分组”线性）
        self.W1 = nn.Parameter(torch.randn(n_points, coord, emb) * 0.2)
        self.b1 = nn.Parameter(torch.zeros(n_points, emb))
        self.W2 = nn.Parameter(torch.randn(n_points, emb, emb) * 0.2)
        self.b2 = nn.Parameter(torch.zeros(n_points, emb))
        self.drop = nn.Dropout(p_drop)
        self.trunk_net = ResTrunk(n_points * emb + 1, trunk, p_drop=p_drop, residual=residual)
        self.head = MultiHead(self.trunk_net.out_dim, n_out, hard_idx=hard_idx,
                              head_hidden=head_hidden, p_drop=p_drop)

    def forward(self, x):
        m = self.n_points * self.coord
        hp = x[:, :m].reshape(-1, self.n_points, self.coord)     # (B,P,3)
        wc = x[:, m:]                                            # (B,1)
        e = torch.relu(torch.einsum("bpi,pio->bpo", hp, self.W1) + self.b1)
        e = torch.relu(torch.einsum("bpo,poq->bpq", e, self.W2) + self.b2)
        e = self.drop(e).reshape(e.shape[0], -1)                # (B,P*emb)
        feat = self.trunk_net(torch.cat([e, wc], dim=1))
        return self.head(feat)


def build_model(kind, n_in, n_out, hidden=(256, 256, 128), emb=16,
                trunk=(256, 128), p_drop=0.1, residual=False,
                hard_idx=None, head_hidden=64):
    """按 kind 选择并构造模型实例（PointMLP 或打平版 MLP），统一透传网络结构参数。

    参数:
        kind: "point"=PointMLP(默认，逐点编码器)；其它值=打平版 MLP
        n_in/n_out: 输入/输出维度
        hidden: 打平版 MLP 主干各层宽度
        emb: PointMLP 逐点编码维度
        trunk: PointMLP 主干各层宽度
        p_drop: Dropout 比例
        residual: 主干是否用残差块堆叠
        hard_idx/head_hidden: 见 MultiHead 说明

    返回:
        nn.Module 实例（PointMLP 或 MLP）
    """
    if kind == "point":
        return PointMLP(n_in, n_out, emb=emb, trunk=trunk, p_drop=p_drop, residual=residual,
                        hard_idx=hard_idx, head_hidden=head_hidden)
    return MLP(n_in, n_out, hidden=hidden, p_drop=p_drop, residual=residual,
              hard_idx=hard_idx, head_hidden=head_hidden)


class Ensemble(nn.Module):
    """多个模型预测求平均，降低方差与外推异常。"""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], 0).mean(0)


def pred_orig(model, xs, ys, Xnp):
    """标准化→前向→反标准化→（可选）裁剪到训练值域。返回原量纲预测。"""
    model.eval()
    with torch.no_grad():
        y = ys.inv(model(torch.tensor(xs.tf(Xnp), dtype=torch.float32)).numpy())
    lo, hi = getattr(model, "_y_lo", None), getattr(model, "_y_hi", None)
    if lo is not None and hi is not None:
        y = np.clip(y, lo, hi)
    return y


def _fit_one(Xtr, Ytr, Xva_t, Yva_orig, ys, names, *, kind, hidden, emb, trunk,
             p_drop, lr, wd, batch, epochs, time_budget, patience, seed, sample_weight=None,
             residual=False, target_weights=None, hard_idx=None, head_hidden=64):
    """训练单个模型，按验证集加权平均 R² 早停并保留最优权重。返回 (model, best_val_r2, epochs)。
    sample_weight: 可选 (n,) 数组，每个训练行的采样权重（未归一化即可）；
                   用于给某个数据源(如 Excel 真实数据)更高的采样概率，抵消数量劣势。
                   None 时退化为原来的均匀随机打乱。
    residual: 主干是否用残差块(ResBlock)堆叠，支持更深网络时缓解梯度消失。
    target_weights: 可选 (n_out,) 数组，每个输出量在损失函数和早停判据中的权重。
                   None 时退化为全 1（等权重），与原行为完全一致。用于让训练更关注
                   caster/caster_arm/scrub_radius 这类物理上对硬点误差更敏感、
                   容易被其它量的高R²掩盖的目标量。
    hard_idx/head_hidden: 见 MultiHead 说明，给指定输出维度额外配独立小头。"""
    import time as _t
    torch.manual_seed(seed)
    model = build_model(kind, Xtr.shape[1], Ytr.shape[1],
                        hidden=hidden, emb=emb, trunk=trunk, p_drop=p_drop, residual=residual,
                        hard_idx=hard_idx, head_hidden=head_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=5)
    n_out = Ytr.shape[1]
    if target_weights is None:
        w_t = None
    else:
        w_t = torch.tensor(np.asarray(target_weights, dtype=np.float32))
        assert w_t.shape[0] == n_out, "target_weights 长度必须等于输出量个数"

    def weighted_mse(pred, target):
        """按量加权的 MSE：先算每个输出维度自己的 MSE，再按 target_weights 加权平均。
        w_t=None 时等价于普通 nn.MSELoss()（逐元素误差平方后整体求平均，与手动
        实现的“各维度先平均、再等权重平均”在数学上一致，保证向后兼容）。"""
        se = (pred - target) ** 2                       # (B, n_out)
        per_dim = se.mean(dim=0)                         # (n_out,)
        if w_t is None:
            return per_dim.mean()
        return (per_dim * w_t).sum() / w_t.sum()

    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    probs = None
    if sample_weight is not None:
        probs = sample_weight / sample_weight.sum()
    best, bstate, noimp, ep = -1e9, None, 0, 0
    t0 = _t.time()
    n_batches = max(1, n // batch)
    for ep in range(1, epochs + 1):
        model.train()
        if probs is None:
            perm = rng.permutation(n)
            batches = [perm[i:i + batch] for i in range(0, n, batch)]
        else:
            # 按权重有放回采样出与原数据等量的行，再按批次训练(一个epoch内样本量不变，
            # 但高权重来源(如Excel真实数据)会被更频繁地抽到)
            batches = [rng.choice(n, size=batch, replace=True, p=probs)
                      for _ in range(n_batches)]
        for idx in batches:
            opt.zero_grad()
            weighted_mse(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            yv = ys.inv(model(Xva_t).numpy())
        r2s = np.array([r2(Yva_orig[:, k], yv[:, k]) for k in range(len(names))])
        if target_weights is None:
            vr2 = float(r2s.mean())
        else:
            tw = np.asarray(target_weights, dtype=np.float64)
            vr2 = float(np.sum(r2s * tw) / np.sum(tw))
        sched.step(vr2)
        if vr2 > best + 1e-4:
            best, noimp = vr2, 0
            bstate = copy.deepcopy(model.state_dict())
        else:
            noimp += 1
        if noimp >= patience or (_t.time() - t0) > time_budget:
            break
    if bstate is not None:
        model.load_state_dict(bstate)
    return model, best, ep


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def r2(yt, yp):
    """计算决定系数 R²（拟合优度）：1 - 残差平方和/总方差。
    R²=1 表示完美预测；R²=0 表示预测水平等同于"永远猜均值"；R²<0 表示预测比猜均值还差。
    """
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_target_metrics(y_true, y_pred, names):
    """逐个输出量分别计算 R²/RMSE/MAE，并汇总一个不加权的平均R²(_mean_R2)。

    参数:
        y_true/y_pred: (n_rows, n_out) 真实值/预测值（原始物理量纲，非标准化空间）
        names: 长度 n_out 的输出量名称列表

    返回:
        dict，键为各量名 -> {"R2":..,"RMSE":..,"MAE":..}，外加一个 "_mean_R2" 汇总键
        （8个量R²的简单算术平均，注意这不是按 target_weights 加权的版本，
        仅用于报告展示；训练时的加权早停判据在 _fit_one 内单独计算）。
    """
    out = {}
    for i, n in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[n] = {"R2": float(r2(yt, yp)),
                  "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
                  "MAE": float(np.mean(np.abs(yt - yp)))}
    out["_mean_R2"] = float(np.mean([out[n]["R2"] for n in names]))
    return out


def print_metrics(title, mt, names):
    """把 per_target_metrics() 的结果格式化打印成表格（用于训练/验证/测试集的日志展示）。"""
    log(f"\n== {title} ==  (平均 R²={mt['_mean_R2']:.4f})")
    log(f"{'quantity':<20}{'R2':>10}{'RMSE':>14}{'MAE':>14}")
    for n in names:
        d = mt[n]
        log(f"{n:<20}{d['R2']:>10.4f}{d['RMSE']:>14.5g}{d['MAE']:>14.5g}")


# 输出量的中文/单位标签（用于误差指标表）
Q_LABEL = {
    "toe": "前束角 Toe (°)", "camber": "外倾角 Camber (°)", "caster": "后倾角 Caster (°)",
    "caster_arm": "后倾拖距 Caster Arm (mm)", "scrub_radius": "主销偏移距 Scrub Radius (mm)",
    "tire_con_point_x": "接地点 X (mm)", "tire_con_point_y": "接地点 Y (mm)",
    "tire_con_point_z": "接地点 Z (mm)",
}


def curve_error_metrics(model, xs, ys, X, Y, groups, mask, names):
    """按曲线评估：MAE / 中位数 / P90 / 最大误差 / 平滑度D2。

    与 per_target_metrics() 的区别：per_target_metrics 把所有行(不区分曲线)混在一起
    算整体统计；本函数额外计算"平滑度D2"这个逐曲线才有意义的指标，且提供更细的
    分位数(P90)/最大误差，用于发现"整体均值看起来还行，但个别曲线/个别行程点误差
    很大"的情况。

    D2 = 预测曲线沿 wc_stroke 的二阶差分绝对值均值(逐几何后再平均)，越小越平滑。
         二阶差分反映曲线的"抖动程度"——如果模型预测的曲线在相邻行程点之间
         忽大忽小(不符合悬架运动学应有的连续平滑特性)，D2 会明显偏大，
         可以用来发现模型输出"物理上不合理的抖动"，而不仅仅是绝对误差大小。

    参数:
        model/xs/ys: 训练好的模型 + 标准化器
        X/Y/groups: 完整数据集
        mask: 布尔掩码，指定要评估哪个子集（通常是测试集 te）
        names: 输出量名称列表

    返回:
        dict，键为各量名 -> {"MAE","Median","P90","MaxError","D2"}
    """
    yp = pred_orig(model, xs, ys, X[mask])
    ae = np.abs(yp - Y[mask])                      # 逐行绝对误差
    d2_acc = {n: [] for n in names}
    d2_ids = np.unique(groups[mask])
    if len(d2_ids) > 300:                          # D2 最多抽样 300 条几何(提速,不影响均值代表性)
        d2_ids = np.random.default_rng(0).choice(d2_ids, 300, replace=False)
    for sid in d2_ids:
        idx = np.where(groups == sid)[0]
        idx = idx[np.argsort(X[idx, -1])]          # 该几何按行程排序
        pc = pred_orig(model, xs, ys, X[idx])      # 预测曲线 (n_stroke, n_out)
        if len(pc) >= 3:
            dd = np.abs(pc[2:] - 2 * pc[1:-1] + pc[:-2])   # 二阶差分
            for j, n in enumerate(names):
                d2_acc[n].append(float(dd[:, j].mean()))
    out = {}
    for j, n in enumerate(names):
        e = ae[:, j]
        out[n] = {"MAE": float(e.mean()), "Median": float(np.median(e)),
                  "P90": float(np.percentile(e, 90)), "MaxError": float(e.max()),
                  "D2": float(np.mean(d2_acc[n]) if d2_acc[n] else 0.0)}
    return out


def print_curve_table(cm, names):
    """把 curve_error_metrics() 的结果格式化打印成表格。"""
    log(f"\n{'曲线 / Quantity':<28}{'MAE':>10}{'Median':>10}{'P90':>10}"
        f"{'MaxError':>11}{'平滑度D2':>11}")
    for n in names:
        d = cm[n]
        log(f"{Q_LABEL.get(n, n):<28}{d['MAE']:>10.5g}{d['Median']:>10.5g}"
            f"{d['P90']:>10.5g}{d['MaxError']:>11.5g}{d['D2']:>11.5g}")


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #

def train(epochs=800, time_budget=240.0, patience=40, batch=2048,
          hidden=(512, 256, 128), lr=2e-3, weight_decay=2e-4, p_drop=0.0,
          model_kind="point", emb=32, trunk=(512, 256, 128), n_ensemble=3,
          excel_weight=1.0, residual=False, target_weights=None, hard_targets=None, head_hidden=64):
    """excel_weight: 仅当数据集含 source 字段(合并数据集)时生效——
    Excel 真实数据(source=0)相对求解器生成数据(source=1)的采样权重倍数。
    1.0=不加权(等同原行为)；>1 会让训练时更频繁抽到 Excel 真实数据的行，
    用于抵消其数量远少于求解器生成数据的劣势。
    residual: 主干换成残差块(ResBlock)堆叠，配合更深的 hidden/trunk 使用效果更明显。
    target_weights: 可选 {量名: 权重} 字典，None(默认)=全部量权重相等，与原行为一致。
    对 caster/caster_arm/scrub_radius 这类物理上对硬点误差更敏感、容易被其它量的高R²
    掩盖真实短板的量，可传入更高权重（如 2.0），同时影响损失函数和早停判据，
    让训练过程更关注这些量，而不是被"8量简单平均"掩盖。
    hard_targets: 可选量名列表(如 ["caster_arm","scrub_radius"])，给这些量额外配一个
    独立的 2 层小 MLP 输出头（不与其它量共享最后一层权重），单纯加大 target_weights
    到一定程度后收益会递减甚至让其它量变差（共享输出层容量没变），独立头是更根本的解法，
    推荐与 target_weights 搭配使用。None(默认)=不加头，与原行为完全一致。
    head_hidden: 独立头的隐藏层宽度，默认 64。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    X, Y, groups, meta, source = load_data()
    names = meta["target_names"]
    log(f"数据: 曲线={meta['n_samples']} 行={X.shape[0]} 入={X.shape[1]} 出={Y.shape[1]}")

    tw_vec = None
    if target_weights:
        tw_vec = np.array([float(target_weights.get(n, 1.0)) for n in names], dtype=np.float64)
        log("按量加权: " + ", ".join(f"{n}={w:g}" for n, w in zip(names, tw_vec)))

    hard_idx = None
    if hard_targets:
        hard_idx = [names.index(n) for n in hard_targets if n in names]
        missing = [n for n in hard_targets if n not in names]
        if missing:
            log(f"[WARN] hard_targets 中以下量名不存在于 target_names，已忽略: {missing}")
        if hard_idx:
            log(f"独立输出头(head_hidden={head_hidden}): " +
                ", ".join(names[j] for j in hard_idx))

    tr, va, te, c = split_by_group(groups)
    log(f"划分(按曲线): train={c[0]} val={c[1]} test={c[2]}")

    xs, ys = Standardizer(X[tr]), Standardizer(Y[tr])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Ytr = to_t(xs.tf(X[tr])), to_t(ys.tf(Y[tr]))
    Xva_t, Yva_orig = to_t(xs.tf(X[va])), Y[va]

    sample_weight = None
    if source is not None and excel_weight != 1.0:
        src_tr = source[tr]
        sample_weight = np.where(src_tr == 0, excel_weight, 1.0).astype(np.float64)
        n_ex, n_gn = int((src_tr == 0).sum()), int((src_tr == 1).sum())
        eff_ratio = (n_ex * excel_weight) / max(1, n_gn)
        log(f"启用加权采样: Excel真实数据×{excel_weight:g} (训练集内 Excel={n_ex}行/"
            f"求解器={n_gn}行, 加权后有效占比≈{eff_ratio:.2f}:1)")
    elif source is not None:
        log("数据集含 source 字段但 excel_weight=1.0，未启用加权(等同均匀采样)。")

    per_budget = max(20.0, time_budget / max(1, n_ensemble))
    log(f"模型={model_kind} 集成成员数={n_ensemble} 每成员预算≈{per_budget:.0f}s "
        f"(emb={emb}, trunk={trunk}, wd={weight_decay}, drop={p_drop}, residual={residual})")

    # 用 torchinfo 打印单个成员的结构概要
    probe = build_model(model_kind, X.shape[1], Y.shape[1],
                        hidden=hidden, emb=emb, trunk=trunk, p_drop=p_drop, residual=residual,
                        hard_idx=hard_idx, head_hidden=head_hidden)
    try:
        from torchinfo import summary
        log("\n===== 模型结构 (torchinfo, 单个集成成员) =====")
        log(str(summary(probe, input_size=(2, X.shape[1]), verbose=0,
                        col_names=("input_size", "output_size", "num_params", "trainable"))))
        log("=" * 46)
    except Exception as e:   # noqa: BLE001
        log(f"(torchinfo 概要不可用: {e})")

    members, vr2s = [], []
    for s in range(n_ensemble):
        m, vr2, ep = _fit_one(Xtr, Ytr, Xva_t, Yva_orig, ys, names,
                              kind=model_kind, hidden=hidden, emb=emb, trunk=trunk,
                              p_drop=p_drop, lr=lr, wd=weight_decay, batch=batch,
                              epochs=epochs, time_budget=per_budget,
                              patience=patience, seed=42 + s, sample_weight=sample_weight,
                              residual=residual, target_weights=tw_vec,
                              hard_idx=hard_idx, head_hidden=head_hidden)
        log(f"  成员 {s+1}/{n_ensemble}: val_{'加权' if tw_vec is not None else ''}meanR²={vr2:.4f}  epochs={ep}")
        members.append(m); vr2s.append(vr2)

    model = members[0] if n_ensemble == 1 else Ensemble(members)
    # 裁剪范围：训练集各目标 min/max（留 2% 余量），抑制外推异常值
    rng_y = Y[tr].max(0) - Y[tr].min(0)
    y_lo = Y[tr].min(0) - 0.02 * rng_y
    y_hi = Y[tr].max(0) + 0.02 * rng_y
    model._y_lo, model._y_hi = y_lo, y_hi
    log(f"集成完成：成员平均验证R²={np.mean(vr2s):.4f}（已启用预测裁剪到训练值域）")

    def ev(mask):
        return per_target_metrics(Y[mask], pred_orig(model, xs, ys, X[mask]), names)

    m_tr, m_va, m_te = ev(tr), ev(va), ev(te)
    print_metrics("训练集", m_tr, names)
    print_metrics("验证集", m_va, names)
    print_metrics("测试集", m_te, names)
    passed = m_te["_mean_R2"] >= TARGET_MEAN_R2
    log(f"\n指标判据：测试集平均 R²={m_te['_mean_R2']:.4f} 目标>= {TARGET_MEAN_R2} "
        f"=> {'✅ 达标' if passed else '⚠️ 未达标'}")

    # 逐曲线误差指标（MAE/Median/P90/MaxError/平滑度D2）——测试集
    cm_te = curve_error_metrics(model, xs, ys, X, Y, groups, te, names)
    log("\n== 测试集 逐曲线误差指标 ==")
    print_curve_table(cm_te, names)

    # 若为合并数据集(含 source)，额外分数据源汇报测试集表现，避免被整体平均掩盖差异
    m_te_by_source = None
    if source is not None:
        m_te_by_source = {}
        log("\n== 测试集 按数据源分开评估 ==")
        for label, src_val in (("excel_real", 0), ("solver_generated", 1)):
            mask = te & (source == src_val)
            if mask.sum() == 0:
                continue
            mt = per_target_metrics(Y[mask], pred_orig(model, xs, ys, X[mask]), names)
            m_te_by_source[label] = {"n_rows": int(mask.sum()), **mt}
            log(f"  [{label}] 行数={mask.sum()}  平均R²={mt['_mean_R2']:.4f}  "
               f"(toe MAE={mt['toe']['MAE']:.4f}, camber MAE={mt['camber']['MAE']:.4f})")

    # 保存（集成成员 + 结构 + 标准化参数 + 裁剪范围 + 元数据）
    _torch_save_safe({
        "members": [m.state_dict() for m in members],
        "arch": {"model_kind": model_kind, "n_in": X.shape[1], "n_out": Y.shape[1],
                 "hidden": list(hidden), "emb": emb, "trunk": list(trunk),
                 "p_drop": p_drop, "residual": residual,
                 "hard_idx": hard_idx, "head_hidden": head_hidden},
        "x_mean": xs.mean, "x_std": xs.std, "y_mean": ys.mean, "y_std": ys.std,
        "y_lo": y_lo, "y_hi": y_hi, "meta": meta,
        "target_weights": dict(zip(names, tw_vec.tolist())) if tw_vec is not None else None,
    }, MODEL_PATH)
    dpv = design_position_validation(model, xs, ys, X, Y, te, meta)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "val": m_va, "test": m_te, "passed": passed,
                   "target_mean_R2": TARGET_MEAN_R2, "n_ensemble": n_ensemble,
                   "excel_weight": excel_weight,
                   "target_weights": dict(zip(names, tw_vec.tolist())) if tw_vec is not None else None,
                   "hard_targets": [names[j] for j in hard_idx] if hard_idx else None,
                   "test_by_source": m_te_by_source,
                   "test_curve_metrics": cm_te,
                   "design_position_validation": dpv,
                   "presets": meta["presets"]}, f, ensure_ascii=False, indent=2)
    log(f"已保存模型: {MODEL_PATH}")
    log(f"已保存指标: {os.path.join(OUT_DIR, 'metrics.json')}")

    plot_example(model, xs, ys, X, Y, groups, te, meta)
    return m_te


# --------------------------------------------------------------------------- #
# 推理 / 校验 / 绘图
# --------------------------------------------------------------------------- #

def predict_curves(model, xs, ys, hp33, stroke_grid):
    """给定单组 33 维硬点坐标 + 一组行程网格，推理出对应的 8 条运动学曲线。
    这是"推理 API"层面的主入口：外部使用方（infer.py、网页后端）只需要给硬点和
    想要扫描的行程范围，不需要关心标准化/反标准化等内部细节。

    参数:
        model/xs/ys: 训练好的模型 + 标准化器（来自 load_bundle()）
        hp33: 长度33的硬点坐标数组
        stroke_grid: 长度n的行程点数组(mm)，想要在哪些行程点上求曲线值

    返回:
        (n, 8) 该组硬点在给定行程网格上的 8 个运动学量预测值（原始物理量纲）
    """
    hp = np.asarray(hp33, float).reshape(1, -1)
    sg = np.asarray(stroke_grid, float).reshape(-1, 1)
    # 把同一组硬点复制 n 份，分别搭配 n 个不同的 stroke 值，拼成 (n, 34) 的批量输入
    # （PointMLP/MLP 的前向传播本质上是逐行独立处理，这样一次前向就能算出整条曲线）
    X = np.hstack([np.repeat(hp, len(sg), 0), sg])
    return pred_orig(model, xs, ys, X)


def design_position_validation(model, xs, ys, X, Y, te_mask, meta):
    """设计位(|wc_stroke|<tol)处，模型/实测 toe/camber 与预设对照。"""
    names = meta["target_names"]
    wc = X[:, -1]
    mask = te_mask & (np.abs(wc) < cfg.DESIGN_POS_TOL_MM)
    res = {"tol_mm": cfg.DESIGN_POS_TOL_MM, "n_points": int(mask.sum()),
           "preset_init_toe_deg": cfg.INIT_TOE_DEG,
           "preset_init_camber_deg": cfg.INIT_CAMBER_DEG}
    if mask.any():
        yp = pred_orig(model, xs, ys, X[mask])
        for name, preset in (("toe", cfg.INIT_TOE_DEG), ("camber", cfg.INIT_CAMBER_DEG)):
            j = names.index(name)
            res[f"{name}_pred_mean"] = float(yp[:, j].mean())
            res[f"{name}_actual_mean"] = float(Y[mask, j].mean())
    log("\n设计位校验(|wc_stroke|<{}mm, 测试集 {} 点):".format(
        res["tol_mm"], res["n_points"]))
    if "toe_pred_mean" in res:
        log("  前束 toe : 预设 {:.3f}° | 实测 {:.3f}° | 模型 {:.3f}°".format(
            cfg.INIT_TOE_DEG, res["toe_actual_mean"], res["toe_pred_mean"]))
        log("  外倾 camber: 预设 {:.3f}° | 实测 {:.3f}° | 模型 {:.3f}°".format(
            cfg.INIT_CAMBER_DEG, res["camber_actual_mean"], res["camber_pred_mean"]))
    return res


def plot_example(model, xs, ys, X, Y, groups, te_mask, meta):
    """挑测试集里的第一条几何曲线，画出预测vs真实的对照图(8个子图，每个输出量一个)，
    并把该曲线的完整数据导出为 CSV，作为"训练完成后的直观效果展示"。
    图上绿色星标标注 toe/camber 在设计位(stroke=0)处的预设参考值，用于目视核对
    模型在这个关键工况点是否符合设计输入。"""
    names = meta["target_names"]
    sid = int(np.unique(groups[te_mask])[0])
    idx = np.where(groups == sid)[0]
    idx = idx[np.argsort(X[idx, -1])]
    hp33, stroke, y_true = X[idx[0], :-1], X[idx, -1], Y[idx]
    y_pred = predict_curves(model, xs, ys, hp33, stroke)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, name in enumerate(names):
        ax = axes[i // 4][i % 4]
        ax.plot(stroke, y_true[:, i], "b-", lw=2, label="actual")
        ax.plot(stroke, y_pred[:, i], "r--", lw=2, label="predicted")
        # 预设参考点（设计位 wc_stroke=0）
        if name == "toe":
            ax.scatter([0], [cfg.INIT_TOE_DEG], c="g", marker="*", s=140,
                       zorder=5, label=f"preset init {cfg.INIT_TOE_DEG}")
        if name == "camber":
            ax.scatter([0], [cfg.INIT_CAMBER_DEG], c="g", marker="*", s=140,
                       zorder=5, label=f"preset init {cfg.INIT_CAMBER_DEG}")
        ax.set_title(f"{name}  (R2={r2(y_true[:, i], y_pred[:, i]):.4f})")
        ax.set_xlabel("wc_stroke"); ax.grid(True, alpha=0.3)
        if name in ("toe", "camber") or i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Test sample Run#{sid}: predicted vs actual (PyTorch MLP) | "
                 f"tire R={cfg.TIRE_STATIC_RADIUS_MM}mm", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT_DIR, f"curves_test_sample_{sid}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    log(f"已保存示例曲线图: {p}")

    import csv
    cp = os.path.join(OUT_DIR, f"curves_test_sample_{sid}.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wc_stroke"] + [f"{n}_true" for n in names]
                   + [f"{n}_pred" for n in names])
        for k in range(len(stroke)):
            w.writerow([stroke[k]] + list(y_true[k]) + list(y_pred[k]))
    log(f"已保存示例曲线数据: {cp}")


def load_bundle():
    """从 MODEL_PATH 加载完整的训练产物：模型权重(重建集成/单模型)+标准化器+meta。
    这是本文件及其它下游脚本(infer.py/export_test_samples.py/visualize_results.py等)
    读取已训练模型的统一入口，兼容旧版单模型 checkpoint(无"members"字段时回退到
    "state_dict"字段)。

    返回:
        (model, xs, ys, meta)：model 已 eval()，xs/ys 是标准化器，meta 是训练时数据集的元信息
    """
    _install_numpy_core_shim()                 # 先装兼容 shim，再反序列化
    ckpt = _torch_load_safe(MODEL_PATH, weights_only=False)
    a = ckpt["arch"]
    kind = a.get("model_kind", a.get("kind", "point"))
    mems = []
    state_list = ckpt.get("members") or [ckpt["state_dict"]]   # 兼容旧单模型
    for sd in state_list:
        mm = build_model(kind, a["n_in"], a["n_out"],
                         hidden=tuple(a.get("hidden", (256, 256, 128))),
                         emb=a.get("emb", 24), trunk=tuple(a.get("trunk", (256, 128))),
                         p_drop=a["p_drop"], residual=a.get("residual", False),
                         hard_idx=a.get("hard_idx"), head_hidden=a.get("head_hidden", 64))
        mm.load_state_dict(sd); mm.eval(); mems.append(mm)
    model = mems[0] if len(mems) == 1 else Ensemble(mems)
    model.eval()
    model._y_lo, model._y_hi = ckpt.get("y_lo"), ckpt.get("y_hi")
    xs, ys = Standardizer.__new__(Standardizer), Standardizer.__new__(Standardizer)
    xs.mean, xs.std = ckpt["x_mean"], ckpt["x_std"]
    ys.mean, ys.std = ckpt["y_mean"], ckpt["y_std"]
    return model, xs, ys, ckpt["meta"]


def infer_only():
    """--infer-only 模式：跳过训练，直接加载已有 model.pt，对测试集第一条曲线画一次
    示例推理图，用于快速核对已训练模型的效果而不用重新跑训练。"""
    if not os.path.isfile(MODEL_PATH):
        log("未找到 model.pt，请先训练。"); return
    model, xs, ys, meta = load_bundle()
    X, Y, groups, _, _ = load_data()
    _, _, te, _ = split_by_group(groups)
    plot_example(model, xs, ys, X, Y, groups, te, meta)


def main():
    """命令行入口：解析参数，组装 target_weights/hard_targets 后调用 train() 或 infer_only()。
    完整参数列表见各 ap.add_argument 的 help 说明；--infer-only 时跳过训练，
    直接加载已有 model.pt 做一次示例推理绘图。"""
    ap = argparse.ArgumentParser(description="PyTorch 训练/推理 悬架运动学 MLP")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--time-budget", type=float, default=220.0)
    ap.add_argument("--model", choices=["point", "flat"], default="point",
                    help="point=逐点三维编码器(默认,推荐); flat=打平33维MLP")
    ap.add_argument("--ensemble", type=int, default=5, help="集成成员数(降方差,默认5)")
    ap.add_argument("--data", default=None, help="数据集目录(默认 nn_data；可指向 nn_data_gen)")
    ap.add_argument("--batch", type=int, default=512, help="批大小(大数据集建议 2048)")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout 比例(抑制过拟合,稀疏数据集建议>0)")
    ap.add_argument("--weight-decay", type=float, default=2e-4, help="Adam 权重衰减(L2正则)")
    ap.add_argument("--emb", type=int, default=32, help="point 模型逐点编码维度")
    ap.add_argument("--trunk", default="512,256,128", help="point 模型主干隐藏层,逗号分隔")
    ap.add_argument("--hidden", default="512,256,128", help="flat 模型隐藏层,逗号分隔")
    ap.add_argument("--lr", type=float, default=2e-3, help="学习率")
    ap.add_argument("--excel-weight", type=float, default=1.0,
                    help="仅合并数据集(含source字段)生效：Excel真实数据相对求解器生成数据的采样权重倍数,"
                         "用于抵消Excel数据量少的劣势(默认1.0=不加权)")
    ap.add_argument("--residual", action="store_true",
                    help="主干换成残差块(ResBlock)堆叠,配合更深的--trunk/--hidden使用,"
                         "缓解加深网络时的梯度消失")
    ap.add_argument("--target-weights", default=None,
                    help="按量加权损失+早停判据,格式\"量名:权重,量名:权重\",如"
                         "\"caster:2.0,caster_arm:2.0,scrub_radius:2.0\"。"
                         "未提及的量权重默认1.0。不传(默认)=全部等权重,与原行为完全一致。"
                         "用于改善 caster/caster_arm/scrub_radius 这类物理上对硬点误差更敏感、"
                         "容易被其它量的高R²掩盖真实短板的量。")
    ap.add_argument("--caster-weight", type=float, default=None,
                    help="快捷方式:同时给 caster/caster_arm 设置权重(会被--target-weights覆盖)")
    ap.add_argument("--scrub-weight", type=float, default=None,
                    help="快捷方式:给 scrub_radius 设置权重(会被--target-weights覆盖)")
    ap.add_argument("--hard-targets", default=None,
                    help="给指定量额外配独立的2层小MLP输出头(不与其它量共享最后一层),格式"
                         "\"量名,量名\",如 \"caster_arm,scrub_radius\"。"
                         "单纯加大--target-weights到一定程度后收益递减甚至让其它量变差"
                         "(共享输出层容量没变)，独立头是更根本的解法，推荐与--target-weights"
                         "搭配使用。不传(默认)=不加独立头，与原行为完全一致。")
    ap.add_argument("--head-hidden", type=int, default=64,
                    help="--hard-targets 独立头的隐藏层宽度(默认64)")
    args = ap.parse_args()
    if args.data:
        globals()["DATA_DIR"] = os.path.abspath(args.data)
        log(f"使用数据集目录: {DATA_DIR}")

    target_weights = {}
    if args.caster_weight is not None:
        target_weights["caster"] = args.caster_weight
        target_weights["caster_arm"] = args.caster_weight
    if args.scrub_weight is not None:
        target_weights["scrub_radius"] = args.scrub_weight
    if args.target_weights:
        for kv in args.target_weights.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split(":")
            target_weights[k.strip()] = float(v.strip())
    target_weights = target_weights or None

    hard_targets = [s.strip() for s in args.hard_targets.split(",") if s.strip()] \
        if args.hard_targets else None

    if args.infer_only:
        infer_only()
    else:
        trunk = tuple(int(v) for v in args.trunk.split(","))
        hidden = tuple(int(v) for v in args.hidden.split(","))
        train(epochs=args.epochs, time_budget=args.time_budget,
              model_kind=args.model, n_ensemble=args.ensemble, batch=args.batch,
              p_drop=args.dropout, weight_decay=args.weight_decay,
              emb=args.emb, trunk=trunk, hidden=hidden, lr=args.lr,
              excel_weight=args.excel_weight, residual=args.residual,
              target_weights=target_weights, hard_targets=hard_targets,
              head_hidden=args.head_hidden)



if __name__ == "__main__":
    main()
