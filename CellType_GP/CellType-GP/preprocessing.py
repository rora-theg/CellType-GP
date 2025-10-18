#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc

# -------------------------------------------------------------------------
# 1. 添加模块路径并导入函数
# -------------------------------------------------------------------------
sys.path.append("/home/vs_theg/ST_program/CellType_GP/CellType-GP/")
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
# -------------------------------------------------------------------------
# 3. Xenium 数据打分
# -------------------------------------------------------------------------
os.chdir('/home/vs_theg/ST_program/CellType_GP/DATA/')
adata_x = sc.read("/home/vs_theg/ST_program/CellType_GP/DATA/xdata.h5")
score_gene_programs(adata_x, gene_sets_to_score, platform="xenium", output_dir="xenium_scores")
adata_x.write("/home/vs_theg/ST_program/CellType_GP/DATA/xdata_processed.h5")
# -------------------------------------------------------------------------
# 4. Xenium 真值计算
# -------------------------------------------------------------------------
xenium_to_visium_transcript_mapping = pd.read_csv(
    '/home/vs_theg/ST_program/CellType_GP/DATA/xenium_to_visium_transcript_mapping.csv'
)
possible_keys = ["cell_id", "Barcode", "cell_ID", "xenium_cell_id"]
left_key = next((k for k in possible_keys if k in adata_x.obs.columns), None)
if left_key is None:
    raise ValueError("❌ 在 adata_x.obs 中找不到匹配的 cell id 列。")

adata_x.obs = adata_x.obs.merge(
    xenium_to_visium_transcript_mapping[["xenium_cell_id", "transcript_level_visium_barcode"]],
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

compute_truth_score(adata_x)

# -------------------------------------------------------------------------
# 5. Visium 数据打分
# -------------------------------------------------------------------------
adata_v = sc.read("/home/vs_theg/ST_program/CellType_GP/DATA/vdata.h5")
score_gene_programs(adata_v, gene_sets_to_score, platform="visium", output_dir="visium_scores")
adata_v.write("/home/vs_theg/ST_program/CellType_GP/DATA/vdata_processed.h5")
# -------------------------------------------------------------------------
# 6. 对齐 + 生成标准化 NPZ 文件
# -------------------------------------------------------------------------
spot_cluster_fraction_matrix = pd.read_csv(
    '/home/vs_theg/ST_program/CellType_GP/DATA/spot_cluster_fraction_matrix.csv', index_col=0
)

# 对齐顺序
target_spots = spot_cluster_fraction_matrix.index
adata_v = adata_v[target_spots.intersection(adata_v.obs_names)].copy()
adata_v = adata_v[target_spots]  # 保证顺序一致
print(f"✅ Visium 数据已对齐，共 {adata_v.n_obs} 个 spot。")

# 生成矩阵
score_cols = [c for c in adata_v.obs.columns if c.endswith("_score_norm")]
visium_score = adata_v.obs[score_cols].values.T
coords = adata_v.obsm["spatial"]
spot_names = spot_cluster_fraction_matrix.index.values
celltype_names = np.array(['B_Cells', 'DCIS', 'Endothelial', 'Invasive_Tumor',
                           'Myeloid', 'Stromal', 'T_cells'])
program_names = np.array(score_cols)
print(program_names)

# 保存
import pandas as pd
coords_df = pd.DataFrame(coords, columns=["x", "y"])
coords_df.to_csv("/home/vs_theg/ST_program/CellType_GP/DATA/vdata_spatial_coords.csv", index=False)
print("✅ 已保存到 vdata_spatial_coords.csv，形状：", coords_df.shape)

np.savez_compressed(
    '/home/vs_theg/ST_program/CellType_GP/DATA/spot_data_full.npz',
    spot_cluster_fraction_matrix=spot_cluster_fraction_matrix.values,
    coords=coords,
    visium_score=visium_score,
    spot_names=spot_names,
    celltype_names=celltype_names,
    program_names=program_names
)
print("🎉 成功保存：/home/vs_theg/ST_program/CellType_GP/DATA/spot_data_full.npz")

print("✅ 矩阵形状：")
print("spot_cluster_fraction_matrix:", spot_cluster_fraction_matrix.shape)
print("visium_score:", visium_score.shape)
print("coords:", coords.shape)
