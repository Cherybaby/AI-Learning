#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量导出测试集样本的「预测 vs 真值」曲线数据 + 逐样本 MAE（Attention/Seq2Seq 模型版），
供 web_app_attention.py 的拖动选样本对照页面使用（避免网页每次请求都重新跑模型前向）。

与 export_test_samples.py 的区别：export_test_samples.py 针对 train_nn.py 训练的
逐点 MLP/PointMLP 模型（长格式数据，按曲线分组）；本脚本针对 train_nn_attention.py
训练的 Seq2SeqAttnKC 模型（规整序列数据，按几何分组，接口是 load_data_seq/pred_seq）。
两者输出的 JSON 结构完全一致，网页前端(compare.html)无需区分。

用法：
  cd /workspace/Suspension_AI_Design/训练推理
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples_attention.py                 # 默认导出全部测试集样本
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples_attention.py --max-samples 30
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples_attention.py --out nn_out_attention/test_samples.json

输出 JSON 结构：
  {
    "target_names": [...], "labels": {...}, "presets": {...},
    "samples": [
      {"sample_id": <几何编号>, "stroke": [...],
       "true": {name: [...]}, "pred": {name: [...]},
       "mae": {name: float}} , ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import train_nn_attention as tn

Q_LABEL = tn.Q_LABEL


def main():
    """导出测试集样本的「预测 vs 真值」曲线及逐样本 MAE（Attention/Seq2Seq 模型版）到 JSON。

    用途：
        针对 train_nn_attention.py 训练的 Seq2SeqAttnKC 模型，遍历测试集中每条几何，
        整条序列一次性预测其 8 个运动学量随 wc_stroke 变化的曲线，与真值对比并计算
        逐目标量 MAE。输出 JSON 结构与 export_test_samples.py 完全一致，供
        web_app_attention.py 的对照页面直接读取，前端无需区分模型类型。

    与 MLP 版的接口差异：
        本脚本使用 load_data_seq（规整序列数据）/ split_geo（按几何索引划分）/
        pred_seq（整序列前向），而非 MLP 版的 load_data / split_by_group / predict_curves。

    参数：
        无（通过 argparse 从命令行读取）：
          --out          输出 JSON 路径（默认 <OUT_DIR>/test_samples.json）。
          --max-samples  最多导出的测试集样本数，0 表示不限（导出全部测试集）。

    返回值：
        None。产出文件：
          - <out>  JSON，结构见模块 docstring（含 target_names/labels/presets/samples）。

    异常：
        SystemExit —— 未找到已训练模型文件（tn.MODEL_PATH）时抛出，提示先训练。
    """
    ap = argparse.ArgumentParser(description="导出测试集样本 预测vs真值 曲线 + MAE(Attention模型版)，供 web_app_attention 使用")
    ap.add_argument("--out", default=None, help="输出 JSON 路径(默认 nn_out_attention/test_samples.json)")
    ap.add_argument("--max-samples", type=int, default=0, help="最多导出的测试集样本数(默认0=不限,导出全部测试集)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(tn.OUT_DIR, "test_samples.json")

    # 前置检查：模型文件必须存在，否则无法做前向预测
    if not os.path.isfile(tn.MODEL_PATH):
        raise SystemExit(f"未找到模型 {tn.MODEL_PATH}，请先训练(train_nn_attention.py)。")

    # 加载模型与标准化器：xs/ys 为输入/输出归一化参数，ss 为行程(stroke)归一化参数，meta 元信息
    model, xs, ys, ss, meta = tn.load_bundle()
    # 加载规整序列数据：hp=各几何硬点坐标, stroke=行程序列, Yseq=真值序列, gids=几何编号(末返回值不用)
    hp, stroke, Yseq, gids, _ = tn.load_data_seq()
    # 按几何总数划分 train/val/test 的索引，仅取测试集索引 te（其余不用）
    _, _, te = tn.split_geo(len(gids))
    names = meta["target_names"]           # 8 个目标量的名称

    # 测试集样本索引（指向 hp/stroke/Yseq/gids 的行）
    test_ids = te
    # 若设置了上限且测试集更多，用固定随机种子(0)抽样以可复现，并排序保持稳定顺序
    if args.max_samples and len(test_ids) > args.max_samples:
        test_ids = np.random.default_rng(0).choice(test_ids, args.max_samples, replace=False)
        test_ids.sort()

    # 逐个测试集几何：整序列预测并与真值比较，累积到 samples 列表
    samples = []
    for i in test_ids:
        sid = int(gids[i])                    # 该样本对应的几何原始编号
        st = stroke[i]                        # 该几何的行程序列（横坐标）
        y_true = Yseq[i]                      # 该几何的 8 量真值曲线序列
        # 整序列前向：注意 hp/stroke 需切成 [i] 的单样本 batch，取结果的第 0 条
        y_pred = tn.pred_seq(model, xs, ys, ss, hp[[i]], stroke[[i]])[0]
        # 逐目标量计算平均绝对误差 MAE
        mae = {n: float(np.mean(np.abs(y_true[:, j] - y_pred[:, j]))) for j, n in enumerate(names)}
        samples.append({
            "sample_id": sid,                                              # 几何原始编号
            "stroke": st.tolist(),                                         # 行程横坐标
            "true": {n: y_true[:, j].tolist() for j, n in enumerate(names)},  # 各量真值曲线
            "pred": {n: y_pred[:, j].tolist() for j, n in enumerate(names)},  # 各量预测曲线
            "mae": mae,                                                    # 各量 MAE
        })
        # 实时打印该样本的整体平均 MAE，便于观察进度
        print(f"  样本 Run#{sid}: 平均MAE={np.mean(list(mae.values())):.4f}", flush=True)

    # 组装最终 JSON 载荷：结构与 MLP 版完全一致，保证前端复用
    payload = {
        "target_names": names,
        "labels": {n: Q_LABEL.get(n, n) for n in names},   # 目标量的中文/可读标签映射
        "presets": meta.get("presets", {}),
        "samples": samples,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 确保输出目录存在
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)          # 保留中文，不转义为 \uXXXX
    print(f"已导出 {len(samples)} 个测试集样本 -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
