#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量导出测试集样本的「预测 vs 真值」曲线数据 + 逐样本 MAE，供 web_app2.py 的
拖动选样本对照页面使用（避免网页每次请求都重新跑模型前向）。

用法：
  cd /workspace/Suspension_AI_Design
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples.py                 # 默认导出全部测试集样本
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples.py --max-samples 30
  PYTHONPATH=/workspace/.pylibs python3 export_test_samples.py --data nn_data --out nn_out/test_samples.json

输出 JSON 结构：
  {
    "target_names": [...], "labels": {...}, "presets": {...},
    "samples": [
      {"sample_id": <groups中的原始编号>, "stroke": [...],
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

import train_nn as tn

Q_LABEL = tn.Q_LABEL


def main():
    """导出测试集样本的「预测 vs 真值」曲线及逐样本 MAE（MLP 模型版）到 JSON。

    用途：
        针对 train_nn.py 训练的逐点 MLP 模型，遍历测试集中每条几何，
        用模型预测其 8 个运动学量随 wc_stroke 变化的曲线，并与真值对比、
        计算逐目标量的 MAE。结果一次性写入 JSON，供 web_app2.py 的对照
        页面直接读取（避免网页每次请求都重新跑模型前向）。

    参数：
        无（通过 argparse 从命令行读取）：
          --data         数据集目录，覆盖 train_nn.DATA_DIR（默认 nn_data）。
          --out          输出 JSON 路径（默认 <OUT_DIR>/test_samples.json）。
          --max-samples  最多导出的测试集样本数，0 表示不限（导出全部测试集）。

    返回值：
        None。产出文件：
          - <out>  JSON，结构见模块 docstring（含 target_names/labels/presets/samples）。

    异常：
        SystemExit —— 未找到已训练模型文件（tn.MODEL_PATH）时抛出，提示先训练。
    """
    ap = argparse.ArgumentParser(description="导出测试集样本 预测vs真值 曲线 + MAE，供 web_app2 使用")
    ap.add_argument("--data", default=None, help="数据集目录(默认 nn_data)")
    ap.add_argument("--out", default=None, help="输出 JSON 路径(默认 nn_out/test_samples.json)")
    ap.add_argument("--max-samples", type=int, default=0, help="最多导出的测试集样本数(默认0=不限,导出全部测试集)")
    args = ap.parse_args()

    # 若指定了 --data，则覆盖 train_nn 模块内的全局数据目录，使后续 load_data 从此处读取
    if args.data:
        tn.DATA_DIR = os.path.abspath(args.data)
    out_path = args.out or os.path.join(tn.OUT_DIR, "test_samples.json")

    # 前置检查：模型文件必须存在，否则无法做前向预测
    if not os.path.isfile(tn.MODEL_PATH):
        raise SystemExit(f"未找到模型 {tn.MODEL_PATH}，请先训练(train_nn.py)。")

    # 加载模型与标准化器：model 模型本体，xs/ys 分别是输入/输出的归一化参数，meta 元信息
    model, xs, ys, meta = tn.load_bundle()
    # 加载长格式数据：X 输入、Y 真值、groups 每行所属几何编号（后两个返回值本脚本不用）
    X, Y, groups, _, _ = tn.load_data()
    # 按几何分组划分 train/val/test，仅取测试集掩码 te_mask（其余划分结果不用）
    _, _, te_mask, _ = tn.split_by_group(groups)
    names = meta["target_names"]           # 8 个目标量的名称

    # 测试集中包含的所有几何编号（去重）
    test_ids = np.unique(groups[te_mask])
    # 若设置了上限且测试集样本更多，用固定随机种子(0)抽样，保证可复现，并排序保持稳定顺序
    if args.max_samples and len(test_ids) > args.max_samples:
        test_ids = np.random.default_rng(0).choice(test_ids, args.max_samples, replace=False)
        test_ids.sort()

    # 逐个测试集几何：还原其曲线并预测，累积到 samples 列表
    samples = []
    for sid in test_ids:
        idx = np.where(groups == sid)[0]      # 该几何对应的所有行下标
        idx = idx[np.argsort(X[idx, -1])]     # 按 wc_stroke（X 末列）升序，使曲线点有序
        # 拆出：hp33=33 个硬点坐标(该几何各行相同，取首行)、stroke=行程序列、y_true=8 量真值曲线
        hp33, stroke, y_true = X[idx[0], :-1], X[idx, -1], Y[idx]
        # 用模型预测该几何在整条行程上的 8 量曲线
        y_pred = tn.predict_curves(model, xs, ys, hp33, stroke)
        # 逐目标量计算平均绝对误差 MAE（对整条曲线求 |真值-预测| 的均值）
        mae = {n: float(np.mean(np.abs(y_true[:, i] - y_pred[:, i]))) for i, n in enumerate(names)}
        samples.append({
            "sample_id": int(sid),                                        # 几何原始编号
            "stroke": stroke.tolist(),                                     # 行程横坐标
            "true": {n: y_true[:, i].tolist() for i, n in enumerate(names)},  # 各量真值曲线
            "pred": {n: y_pred[:, i].tolist() for i, n in enumerate(names)},  # 各量预测曲线
            "mae": mae,                                                    # 各量 MAE
        })
        # 实时打印该样本的整体平均 MAE，便于观察进度
        print(f"  样本 Run#{sid}: 平均MAE={np.mean(list(mae.values())):.4f}", flush=True)

    # 组装最终 JSON 载荷：目标名、显示标签、预设方案、以及全部样本数据
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
