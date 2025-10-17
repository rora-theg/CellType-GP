#!/usr/bin/env python

"""
CellType-GP 模型评估脚本
用于对比模型预测结果 (delta, vectorized) 与真实值 (truth)。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 尝试导入 sklearn
try:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
    _sklearn_available = True
except Exception:
    _sklearn_available = False


# -----------------------------------------------------------------------------
# 评估函数
# -----------------------------------------------------------------------------
def evaluate_deconvolution_enhanced(
    truth_wide, pred_wide, presence_mask=None,
    presence_threshold=0.0, min_pairs=5,
    compute_prob_metrics=True, verbose=True,
    plot_top=6, score_transform=None,
    plot_title=None, ax=None
):
    """改良版评估函数（支持结构性缺失、Pearson/Spearman/MAE/RMSE 等）

    参数:
        plot_title : str | None
            绘图标题；若为空则使用默认标题。
        ax : matplotlib.axes.Axes | None
            若提供则在给定坐标轴上绘制直方图，可用于外部子图布局。
    """

    # --- 对齐索引 ---
    truth_wide, pred_wide = truth_wide.align(pred_wide, join="inner", axis=0)
    truth_wide, pred_wide = truth_wide.align(pred_wide, join="inner", axis=1)

    # --- 推断 presence ---
    if presence_mask is None:
        presence_mask = ~truth_wide.isna()
    else:
        presence_mask = presence_mask.reindex_like(truth_wide).astype(bool)

    # --- 可选变换 ---
    if score_transform:
        truth_proc = score_transform(truth_wide.copy())
        pred_proc = score_transform(pred_wide.copy())
    else:
        truth_proc = truth_wide.copy()
        pred_proc = pred_wide.copy()

    results = []
    total_spots = len(truth_proc.index)

    for feat in truth_proc.columns:
        tcol, pcol = truth_proc[feat], pred_proc[feat]
        pres_true = presence_mask[feat]

        mask = pres_true & ~(tcol.isna() | pcol.isna())
        n_pairs = int(mask.sum())

        if n_pairs > 0:
            tvals, pvals = tcol[mask].astype(float), pcol[mask].astype(float)
            pearson = tvals.corr(pvals, method="pearson")
            spearman = spearmanr(tvals, pvals).correlation if len(tvals) > 2 else np.nan
            rmse = np.sqrt(np.mean((tvals - pvals) ** 2))
            mae = np.mean(np.abs(tvals - pvals))
        else:
            pearson = spearman = rmse = mae = np.nan

        coverage = pres_true.mean()

        # --- 二分类指标 ---
        pred_presence = ((pcol.fillna(0)) > presence_threshold).astype(int)
        true_presence = pres_true.astype(int)

        TP = int(((pred_presence == 1) & (true_presence == 1)).sum())
        FP = int(((pred_presence == 1) & (true_presence == 0)).sum())
        TN = int(((pred_presence == 0) & (true_presence == 0)).sum())
        FN = int(((pred_presence == 0) & (true_presence == 1)).sum())

        prec = rec = f1 = auroc = aupr = np.nan
        if _sklearn_available and true_presence.nunique() > 1:
            try:
                prec, rec, f1, _ = precision_recall_fscore_support(true_presence, pred_presence, average="binary", zero_division=0)
                if compute_prob_metrics:
                    auroc = roc_auc_score(true_presence, pcol.fillna(0))
                    aupr = average_precision_score(true_presence, pcol.fillna(0))
            except Exception:
                pass

        results.append({
            "feature": feat,
            "PearsonR": pearson,
            "SpearmanR": spearman,
            "RMSE": rmse,
            "MAE": mae,
            "N_pairs": n_pairs,
            "coverage": coverage,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "Precision": prec, "Recall": rec, "F1": f1,
            "AUROC": auroc, "AUPR": aupr
        })

    df = pd.DataFrame(results).sort_values("PearsonR", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"✅ 共 {len(df)} 个特征，spots={total_spots}")
        lowcov = (df['N_pairs'] < min_pairs).sum()
        print(f"低覆盖 (<{min_pairs}) 特征: {lowcov}")
        if ax is None:
            fig, ax_local = plt.subplots(figsize=(7, 4))
            show_plot = True
        else:
            ax_local = ax
            show_plot = False

        sns.histplot(df["PearsonR"].dropna(), bins=25, kde=True, color="steelblue", ax=ax_local)
        ax_local.set_title(plot_title or "Distribution of PearsonR")
        ax_local.set_xlabel("PearsonR")
        if show_plot:
            plt.tight_layout()
            plt.show()

    return df


# -----------------------------------------------------------------------------
# 主评估程序
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = "/home/vs_theg/ST_program/CellType_GP/DATA/"
    truth_path = os.path.join(base_dir, "truth_output/truth_result(wide).csv")

    # 载入真值与预测
    truth = pd.read_csv(truth_path, index_col=0)
    pred_delta = pd.read_csv(os.path.join(base_dir, "pred_result(delta_loo).csv"), index_col=0)
    pred_vectorized = pd.read_csv(os.path.join(base_dir, "pred_result(vectorized).csv"), index_col=0)

    # 评估 delta 模型
    print("\n📈 评估 delta 模型...")
    eval_delta = evaluate_deconvolution_enhanced(truth, pred_delta, verbose=True)
    eval_delta.to_csv(os.path.join(base_dir, "eval_result(delta).csv"), index=False)
    print(f"✅ 保存: {base_dir}eval_result(delta).csv")

    # 评估 vectorized 模型
    print("\n📈 评估 vectorized 模型...")
    eval_vectorized = evaluate_deconvolution_enhanced(truth, pred_vectorized, verbose=True)
    eval_vectorized.to_csv(os.path.join(base_dir, "eval_result(vectorized).csv"), index=False)
    print(f"✅ 保存: {base_dir}eval_result(vectorized).csv")

    # 对比两个模型整体性能
    print("\n🎯 平均性能对比：")
    summary = pd.DataFrame({
        "Metric": ["PearsonR", "SpearmanR", "RMSE", "MAE", "Precision", "Recall", "F1", "AUROC", "AUPR"],
        "Delta_mean": [eval_delta[m].mean(skipna=True) for m in ["PearsonR","SpearmanR","RMSE","MAE","Precision","Recall","F1","AUROC","AUPR"]],
        "Vectorized_mean": [eval_vectorized[m].mean(skipna=True) for m in ["PearsonR","SpearmanR","RMSE","MAE","Precision","Recall","F1","AUROC","AUPR"]],
    })
    print(summary)
    summary.to_csv(os.path.join(base_dir, "eval_summary_comparison.csv"), index=False)
    print(f"\n📊 已保存平均指标对比文件：{base_dir}eval_summary_comparison.csv")
