import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def score_gene_programs(
    adata,
    gene_sets: dict,
    platform: str,
    output_dir: str = "results_scores",
    library_id: str = "spatial",
    log_transform: bool = True,
    normalize: bool = False,
    norm_method: str = "none",  # {'minmax','zscore','none'}
    seed: int = 42,
    draw_umap: bool = True,
    draw_spatial: bool = True,
):
    """
    对空间转录组数据进行基因集打分（log1p + 平均表达）。

    参数：
        adata : AnnData
            Xenium 或 Visium 对象。
        gene_sets : dict[str, list[str]]
            每个键为基因集名称，值为基因列表。
        platform : str
            'xenium' 或 'visium'，用于命名输出。
        output_dir : str
            结果保存的主目录。
        library_id : str
            squidpy 绘制 spatial 图时的 library_id。
        log_transform : bool
            是否对 count 矩阵 log1p。
        normalize : bool
            是否将得分归一化到 [0,1]。
        seed : int
            随机数种子，确保可重复性。
    """

    np.random.seed(seed)
    sc.settings.seed = seed

    # ---- 颜色定义 ----
    cmap = LinearSegmentedColormap.from_list(
        'custom', ["blue", "#7BAFDE", "white", "orange", "red"]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ---- 准备矩阵 ----
    if adata.raw is not None and hasattr(adata.raw, "X"):
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
    # 转成可操作矩阵
    if not isinstance(X, np.ndarray):
        try:
            X = X.toarray()
        except Exception:
            X = np.asarray(X[:])  # for h5 datasets
    
    # log1p 转换
    if log_transform:
        X = np.log1p(X)
    
    # DataFrame 转换
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=var_names)

    # ---- 打分循环 ----
    for score_name, gene_list in gene_sets.items():
        if not gene_list:
            print(f"⏭ 跳过空的基因集: {score_name}")
            continue

        print(f"🔹 正在处理: {score_name}")

        # 检查基因存在性
        existing_genes = [g for g in gene_list if g in expr_df.columns]
        missing_genes = [g for g in gene_list if g not in expr_df.columns]

        if missing_genes:
            print(f"⚠️ 缺失基因 ({score_name}): {', '.join(missing_genes)}")

        if len(existing_genes) == 0:
            print(f"❌ 跳过 {score_name}: 无有效基因。")
            continue

        # ---- 计算平均得分 ----
        scores = expr_df[existing_genes].mean(axis=1)

        # ---- 归一化 ----
        if normalize:
            if norm_method == "minmax":
                scaler = MinMaxScaler()
                scores = scaler.fit_transform(scores.values.reshape(-1, 1)).flatten()
            elif norm_method == "zscore":
                v = scores.values
                scores = (v - v.mean()) / (v.std() + 1e-8)
            elif norm_method == "none":
                pass
            else:
                print(f"⚠️ 未知归一化方法 {norm_method}，已跳过归一化。")

        # 保存结果
        adata.obs[score_name + ("_norm" if normalize else "")] = scores

        # ---- 输出文件目录 ----
        score_dir = output_dir / f"{platform}_{score_name}"
        score_dir.mkdir(exist_ok=True, parents=True)

        # ---- 绘图 ----
        color_key = score_name + ("_norm" if normalize else "")
        # UMAP（若未计算 UMAP 或禁用绘图，则跳过）
        if draw_umap and ("X_umap" in adata.obsm_keys()):
            fig_umap = sc.pl.umap(
                adata,
                color=color_key,
                cmap=cmap,
                show=False,
                size=3,
                title=f"{platform.upper()} UMAP: {score_name}"
            )
            fig_umap.figure.set_size_inches(4, 3)
            fig_umap.figure.savefig(score_dir / "UMAP.png", dpi=300, bbox_inches="tight")
            plt.close(fig_umap.figure)
        elif draw_umap:
            print("ℹ️ 跳过 UMAP 绘图：未找到 adata.obsm['X_umap']。")

        # Spatial
        # Spatial（需要 Visium 空间元数据；异常时跳过）
        if draw_spatial:
            try:
                fig, ax = plt.subplots(figsize=(4, 4))
                sq.pl.spatial_scatter(
                    adata,
                    library_id=library_id,
                    color=color_key,
                    shape=None,
                    wspace=0.15,
                    title=f"{platform.upper()} Spatial: {score_name}",
                    ax=ax
                )
                fig.savefig(score_dir / "Spatial.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"ℹ️ 跳过空间绘图（{score_name}）：{e}")

        print(f"✅ 完成: {score_name} ({len(existing_genes)} 基因)\n")

    print(f"\n🎉 所有基因集打分完成（使用平均表达），结果已保存至 {output_dir}/\n")
