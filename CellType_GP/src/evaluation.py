#!/usr/bin/env python

"""
单方法评估脚本：评估预测的 CTGP 数值与真实值的拟合度与存在性判断。
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # sklearn 可用于额外指标（如 ROC-AUC），若不可用则降级
    from sklearn.metrics import roc_auc_score, average_precision_score

    _has_sklearn = True
except Exception:
    _has_sklearn = False


def load_tables(truth_path: Path, pred_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取真值与预测表，按照索引和列进行对齐。"""
    # `index_col=0` 读取时把第一列作为 spot 索引，保持与宽表格式一致。
    truth = pd.read_csv(truth_path, index_col=0)
    pred = pd.read_csv(pred_path, index_col=0)

    # 对齐行列，保证一一对应
    truth, pred = truth.align(pred, join="inner", axis=0)
    truth, pred = truth.align(pred, join="inner", axis=1)
    return truth, pred


def compute_regression_metrics(truth: pd.DataFrame, pred: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """计算拟合度指标，包括整体与按特征划分的结果。"""
    # stack() 将宽表转换为长表：索引变为 (spot, feature)，便于批量计算。
    truth_stack = truth.stack().rename("truth")
    pred_stack = pred.stack().rename("pred")

    # 合并两列并丢弃 NaN，只评估双方同时有值的 spot-feature 对。
    merged = pd.concat([truth_stack, pred_stack], axis=1, join="inner").dropna()

    metrics: Dict[str, float] = {}
    if len(merged) > 1:
        metrics["pearson"] = merged["truth"].corr(merged["pred"], method="pearson")  # 线性相关系数
        metrics["spearman"] = merged["truth"].corr(merged["pred"], method="spearman")  # 秩相关
    else:
        metrics["pearson"] = np.nan
        metrics["spearman"] = np.nan

    diff = merged["pred"] - merged["truth"]
    metrics["mae"] = float(np.abs(diff).mean()) if len(diff) else np.nan  # 平均绝对误差
    metrics["rmse"] = float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan  # 均方根误差
    metrics["n_pairs"] = len(merged)  # 样本数，用于判断是否有足够比较点

    # 按列（CTGP）统计，便于深入分析
    per_feature = []
    for feature, group in merged.groupby(level=1):
        # group 包含同一 CTGP 在所有 spot 的比较结果
        if len(group) > 1:
            pearson = group["truth"].corr(group["pred"], method="pearson")
            spearman = group["truth"].corr(group["pred"], method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan
        diff_f = group["pred"] - group["truth"]
        per_feature.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "mae": float(np.abs(diff_f).mean()),
                "rmse": float(np.sqrt(np.mean(diff_f**2))),
                "n_pairs": len(group),
            }
        )

    per_feature_df = pd.DataFrame(per_feature).sort_values("pearson", ascending=False)
    return metrics, per_feature_df


def compute_presence_metrics(truth: pd.DataFrame, pred: pd.DataFrame, threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    """对比各方法在预测“是否存在”上的表现。"""
    # 真值中 NaN 代表“不存在”，因此 notna() 能直接构建存在标签。
    truth_presence = truth.notna().stack().astype(int).rename("truth_presence")
    # 预测值先填充缺失为 0，再 stack() 得到与真值同结构的长表。
    pred_presence = pred.fillna(0).stack().rename("pred_score")

    # 对齐真值标签与预测分数，并基于阈值转换为二分类结果。
    aligned = pd.concat([truth_presence, pred_presence], axis=1, join="inner")
    aligned["pred_presence"] = (aligned["pred_score"] > threshold).astype(int)

    # 统计混淆矩阵条目，用于计算常见二分类指标。
    tp = int(((aligned["pred_presence"] == 1) & (aligned["truth_presence"] == 1)).sum())
    fp = int(((aligned["pred_presence"] == 1) & (aligned["truth_presence"] == 0)).sum())
    tn = int(((aligned["pred_presence"] == 0) & (aligned["truth_presence"] == 0)).sum())
    fn = int(((aligned["pred_presence"] == 0) & (aligned["truth_presence"] == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    prob_metrics = {"roc_auc": np.nan, "average_precision": np.nan}
    if _has_sklearn and aligned["truth_presence"].nunique() > 1:
        try:
            prob_metrics["roc_auc"] = roc_auc_score(aligned["truth_presence"], aligned["pred_score"])
            prob_metrics["average_precision"] = average_precision_score(aligned["truth_presence"], aligned["pred_score"])
        except Exception:
            pass

    summary = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        **prob_metrics,
    }

    # 按特征统计，帮助定位问题 CTGP
    per_feature_rows = []
    for feature, group in aligned.groupby(level=1):
        tp_f = int(((group["pred_presence"] == 1) & (group["truth_presence"] == 1)).sum())
        fp_f = int(((group["pred_presence"] == 1) & (group["truth_presence"] == 0)).sum())
        tn_f = int(((group["pred_presence"] == 0) & (group["truth_presence"] == 0)).sum())
        fn_f = int(((group["pred_presence"] == 0) & (group["truth_presence"] == 1)).sum())

        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 0.0
        rec_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) else 0.0
        f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) else 0.0
        acc_f = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f) if (tp_f + tn_f + fp_f + fn_f) else 0.0

        per_feature_rows.append(
            {
                "feature": feature,
                "tp": tp_f,
                "fp": fp_f,
                "tn": tn_f,
                "fn": fn_f,
                "precision": prec_f,
                "recall": rec_f,
                "f1": f1_f,
                "accuracy": acc_f,
            }
        )

    per_feature_df = pd.DataFrame(per_feature_rows).sort_values("f1", ascending=False)
    return summary, per_feature_df





def main() -> None:
    parser = argparse.ArgumentParser(description="评估单个 CTGP 预测结果。")
    parser.add_argument("--truth", type=Path, required=True, help="真实宽表 CSV 路径")
    parser.add_argument("--prediction", type=Path, required=True, help="预测宽表 CSV 路径")
    parser.add_argument("--method-name", type=str, default="method", help="方法名称，用于输出标识")
    parser.add_argument("--presence-threshold", type=float, default=0.0, help="判断存在性的阈值")
    parser.add_argument("--scatter-path", type=Path, default=None, help="散点图输出路径（缺省为仅展示）")
    parser.add_argument("--summary-path", type=Path, default=None, help="整体指标输出 CSV 路径")
    parser.add_argument("--per-feature-regression", type=Path, default=None, help="按特征输出拟合指标 CSV 路径")
    parser.add_argument("--per-feature-presence", type=Path, default=None, help="按特征输出存在性指标 CSV 路径")
    parser.add_argument("--no-show", action="store_true", help="只保存图像，不弹出窗口")

    args = parser.parse_args()

    # Step1: 统一对齐真值与预测表，防止错行错列。
    truth, pred = load_tables(args.truth, args.prediction)

    # 1) 拟合度评估
    # 将整体指标打印到控制台，便于快速观察。
    reg_summary, reg_per_feature = compute_regression_metrics(truth, pred)
    print("===== 拟合度指标 =====")
    for key, value in reg_summary.items():
        print(f"{key:>10}: {value}")

    # 如需按特征保存指标，写入 CSV，方便后续在 notebook 中过滤分析。
    if args.per_feature_regression:
        args.per_feature_regression.parent.mkdir(parents=True, exist_ok=True)
        reg_per_feature.to_csv(args.per_feature_regression, index=False)
        print(f"📄 已保存按特征拟合指标：{args.per_feature_regression}")

    # 散点图使用上一函数生成需要的数据
    merged_data = pd.concat(
        [
            truth.stack().rename("truth"),
            pred.stack().rename("pred"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    # show 参数可在 notebook 中保持交互，也可用于批处理保存图片。
    plot_regression_scatter(merged_data, args.method_name, args.scatter_path, show=not args.no_show)

    # 2) 存在性评估
    # presence_threshold 控制“预测值多大视作存在”，满足实验需求。
    presence_summary, presence_per_feature = compute_presence_metrics(truth, pred, args.presence_threshold)
    print("\n===== 存在性指标 =====")
    for key, value in presence_summary.items():
        print(f"{key:>10}: {value}")

    if args.per_feature_presence:
        args.per_feature_presence.parent.mkdir(parents=True, exist_ok=True)
        presence_per_feature.to_csv(args.per_feature_presence, index=False)
        print(f"📄 已保存按特征存在性指标：{args.per_feature_presence}")

    # 综合结果输出
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(
            [
                {
                    "method": args.method_name,
                    **{f"reg_{k}": v for k, v in reg_summary.items()},
                    **{f"presence_{k}": v for k, v in presence_summary.items()},
                    "threshold": args.presence_threshold,
                }
            ]
        )
        summary_df.to_csv(args.summary_path, index=False)
        # 汇总文件便于跨方法比较时直接拼接或合并多行记录。
        print(f"📄 已保存整体指标汇总：{args.summary_path}")


if __name__ == "__main__":
    main()
