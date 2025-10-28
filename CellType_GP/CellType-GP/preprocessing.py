#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理管线（可脚本运行）：
  - Xenium/Visium 读入、基因集打分
  - Xenium 真值计算（按 spot × broad_annotation 分组均值）
  - Visium 与细胞比例矩阵对齐，导出标准 npz（供模型使用）

注意：本版本移除了绝对路径与 os.chdir，改为 argparse 参数化；
      依赖的 compute_truth_score 当前含交互式清洗，下一步将改为无交互参数。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

from score_gene_program import score_gene_programs
from compute_truth_score import compute_truth_score

# -------------------------------------------------------------------------
# 2. 定义所有 17 个 Cluster 的基因集
# -------------------------------------------------------------------------
DCIS_1_genes = [
    'DCIS1:HPX', 'CEACAM6', 'ESR1', 'HOOK2', 'CEACAM8', 'GATA3', 'TFAP2A', 'FLNB', 'KLF5', 'CD9',
    'TPD52', 'CLDN4', 'SMS', 'DNTTIP1', 'QARS', 'C6orf132', 'KLRF1', 'LYPD3', 'SDC4', 'RHOH'
]

DCIS_2_genes = [
    'AGR3', 'S100A14', 'CEACAM8', 'KRT8', 'CCND1', 'CDH1', 'TCIM', 'AQP3', 'TACSTD2', 'LYPD3',
    'SERHL2', 'ESR1', 'CEACAM6', 'BACE2', 'DSP', 'SERPINA3', 'RORC', 'ERBB2', 'CLDN4', 'DMKN'
]

Prolif_Invasive_genes = [
    'CENPF', 'MKI67', 'TOP2A', 'PCLAF', 'STC1', 'RTKN2', 'TUBA4A', 'MDM2', 'HMGA1', 'C2orf42',
    'POLR2J3', 'PTRHD1', 'SRPK1', 'EIF4EBP1', 'SQLE', 'SH3YL1', 'THAP2', 'NPM3', 'LAG3', 'FOXA1'
]

Invasive_Tumor_genes = [
    'ABCC11', 'SERHL2', 'TCIM', 'FASN', 'AR', 'PTRHD1', 'TRAF4', 'USP53', 'SCD', 'SQLE',
    'MYO5B', 'DNAAF1', 'FOXA1', 'EPCAM', 'CTTN', 'MLPH', 'ELF3', 'ANKRD30A', 'ENAH', 'KARS'
]

# 合并所有基因集用于评分
gene_sets_to_score = {
    'DCIS_1_score': DCIS_1_genes,
    'DCIS_2_score': DCIS_2_genes,
    'Prolif_Invasive_score': Prolif_Invasive_genes,
    'Invasive_Tumor_score': Invasive_Tumor_genes,
}
def main():
    parser = argparse.ArgumentParser(description="Preprocess ST data: scoring + truth + aligned NPZ")
    parser.add_argument("--xenium", type=Path, required=True, help="Xenium AnnData (.h5ad/.h5) path")
    parser.add_argument("--visium", type=Path, required=True, help="Visium AnnData (.h5ad/.h5) path")
    parser.add_argument("--fractions", type=Path, required=True, help="Spot×celltype fractions CSV (index=spot)")
    parser.add_argument("--x2v-map", type=Path, required=True, help="Xenium→Visium barcode mapping CSV")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory (e.g., CellType_GP/DATA)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 3. Xenium 数据打分
    # ---------------------------------------------------------------------
    adata_x = sc.read(str(args.xenium))
    score_gene_programs(adata_x, gene_sets_to_score, platform="xenium", output_dir=str(outdir / "xenium_scores"))
    adata_x.write(str(outdir / "xdata_processed.h5"))

    # ---------------------------------------------------------------------
    # 4. Xenium 真值计算（合并条形码映射；清洗后按 group 平均）
    # ---------------------------------------------------------------------
    x2v = pd.read_csv(args.x2v_map)
    # 自动检测 Xenium 细胞 ID 列 / auto-detect xenium cell id column
    possible_keys = ["cell_id", "Barcode", "cell_ID", "xenium_cell_id"]
    left_key = next((k for k in possible_keys if k in adata_x.obs.columns), None)
    if left_key is None:
        raise ValueError("❌ 在 adata_x.obs 中找不到匹配的 cell id 列（尝试列：" + ", ".join(possible_keys) + ")")

    # 合并到 obs 上（左连接），保留 Visium 条形码 / merge mapping into obs
    adata_x.obs = adata_x.obs.merge(
        x2v[["xenium_cell_id", "transcript_level_visium_barcode"]],
        how="left",
        left_on=left_key,
        right_on="xenium_cell_id"
    )
    if "xenium_cell_id" in adata_x.obs.columns:
        adata_x.obs.drop(columns=["xenium_cell_id"], inplace=True)

    before = adata_x.n_obs
    adata_x = adata_x[adata_x.obs["transcript_level_visium_barcode"].notna()].copy()
    after = adata_x.n_obs
    print(f"✅ 去除无 transcript_level_visium_barcode 的细胞：删除 {before - after}，剩余 {after}")

    # 计算真值（注意：当前 compute_truth_score 含交互式清洗，下一步将去交互化）
    compute_truth_score(
        adata_x,
        output_dir=str(outdir / "truth_output"),
        spot_col='transcript_level_visium_barcode',
        celltype_col='broad_annotation',
        drop_columns=[]
    )

    # ---------------------------------------------------------------------
    # 5. Visium 数据打分
    # ---------------------------------------------------------------------
    adata_v = sc.read(str(args.visium))
    score_gene_programs(adata_v, gene_sets_to_score, platform="visium", output_dir=str(outdir / "visium_scores"))
    adata_v.write(str(outdir / "vdata_processed.h5"))

    # ---------------------------------------------------------------------
    # 6. 对齐 + 生成标准化 NPZ 文件
    # ---------------------------------------------------------------------
    fractions = pd.read_csv(args.fractions, index_col=0)

    # 对齐顺序（严格使用交集，避免 KeyError）/ strict intersection
    target_spots = pd.Index(fractions.index.astype(str))
    vis_spots = adata_v.obs_names.astype(str)
    common = target_spots.intersection(vis_spots)
    missing_in_visium = target_spots.difference(vis_spots)
    missing_in_fraction = vis_spots.difference(target_spots)
    if len(missing_in_visium):
        print(f"⚠️ 在 Visium 中缺失 {len(missing_in_visium)} 个 fractions 索引（已丢弃）：前3个示例：{list(missing_in_visium[:3])}")
    if len(missing_in_fraction):
        print(f"⚠️ 在 fractions 中缺失 {len(missing_in_fraction)} 个 Visium 索引（已丢弃）：前3个示例：{list(missing_in_fraction[:3])}")

    # 重排为相同顺序 / reorder both
    fractions = fractions.loc[common]
    adata_v = adata_v[common].copy()
    print(f"✅ Visium 与 fractions 已对齐，共 {adata_v.n_obs} 个公共 spot。")

    # 生成矩阵 / build matrices
    # 程序列按定义顺序提取，保证与 gene_sets_to_score 对应；过滤不存在列
    desired_cols = [f"{k}_score_norm" for k in gene_sets_to_score.keys()]
    score_cols = [c for c in desired_cols if c in adata_v.obs.columns]
    if len(score_cols) == 0:
        raise RuntimeError("❌ 未在 Visium obs 中找到任何 *_score_norm 列，请先成功打分。")

    visium_score = adata_v.obs[score_cols].values.T            # (P, S)
    coords = adata_v.obsm["spatial"] if "spatial" in adata_v.obsm_keys() else np.zeros((adata_v.n_obs, 2))

    spot_names = fractions.index.values.astype(str)
    # 细胞类型名称从 fractions 列派生，确保顺序一致 / derive ct names from fractions columns
    celltype_names = np.array(fractions.columns.values, dtype=str)
    program_names = np.array(score_cols, dtype=str)

    # 保存坐标 CSV（便于可视化）
    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df.to_csv(outdir / "vdata_spatial_coords.csv", index=False)
    print("✅ 已保存到", outdir / "vdata_spatial_coords.csv", "形状：", coords_df.shape)

    # 保存 npz（标准输入包）
    npz_path = outdir / "spot_data_full.npz"
    np.savez_compressed(
        npz_path,
        spot_cluster_fraction_matrix=fractions.values.astype(np.float32),
        coords=coords.astype(np.float32),
        visium_score=visium_score.astype(np.float32),
        spot_names=spot_names,
        celltype_names=celltype_names,
        program_names=program_names,
    )
    print("🎉 成功保存：", npz_path)

    print("✅ 矩阵形状：")
    print("  spot_cluster_fraction_matrix:", fractions.shape)
    print("  visium_score:", visium_score.shape)
    print("  coords:", coords.shape)


if __name__ == "__main__":
    main()
