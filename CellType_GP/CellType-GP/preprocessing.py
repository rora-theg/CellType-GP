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
Stromal_genes = [
    'MUC6', 'PIM1', 'UCP1', 'CSF3', 'GJB2', 'TCEAL7', 'CLIC6', 'LEP', 'DPT', 'SFRP4',
    'LRRC15', 'LUM', 'POSTN', 'MMP2', 'CCDC80', 'FBLN1', 'PDGFRA', 'TAC1', 'ADH1B', 'CXCL12',
    'ADIPOQ', 'IGF1', 'LPL', 'MEDAG', 'CRISPLD2', 'PCOLCE', 'TIMP4', 'AKR1C1'
]

Prolif_Invasive_Tumor_genes = [
    'ANKRD28', 'TUBA4A', 'STC1', 'RTKN2', 'PCLAF', 'CENPF', 'MKI67', 'TOP2A', 'DNAAF1', 'RHOH',
    'TOMM7', 'NPM3', 'KARS', 'THAP2', 'C2orf42', 'HMGA1', 'PTRHD1', 'MDM2', 'POLR2J3', 'CDH1',
    'ELF3', 'MYO5B', 'USP53', 'KRT7', 'OCIAD2', 'CCND1', 'CCDC6', 'CTTN', 'FASN', 'SQLE',
    'EPCAM', 'AR', 'FOXA1', 'ERBB2', 'SCD', 'ANKRD30A', 'MLPH', 'SH3YL1', 'SRPK1', 'NARS',
    'ZNF562', 'TMEM147', 'LARS', 'TRAF4', 'EIF4EBP1', 'ENAH', 'DAPK3', 'REXO4', 'SLC4A1', 'LAG3'
]

Perivascular_Like_genes = [
    'AHSP', 'S100A4', 'ZEB2', 'PDGFRB', 'FBLIM1', 'LGALSL', 'SSTR2', 'AVPR1A', 'NDUFA4L2', 'ANKRD29',
    'FSTL3', 'ACTA2', 'MYH11', 'FOXC2'
]

Myoepi_KRT15_genes = [
    'RORC', 'SLC5A6', 'ADAM9', 'TACSTD2', 'JUP', 'CXCL16', 'TRIB1', 'SVIL', 'KRT16', 'KRT6B',
    'C5orf46', 'PTN', 'DSC2', 'TUBB2B', 'SLC25A37', 'CXCL5', 'KRT15', 'SFRP1', 'MYBPC1', 'S100A8',
    'ALDH1A3', 'OPRPN', 'ELF5', 'KRT23', 'PIGR', 'CDC42EP1', 'SCGB2A1', 'NCAM1'
]

Myoepi_ACTA2_genes = [
    'DMKN', 'DSP', 'ACTG2', 'DST', 'OXTR', 'CLCA2', 'KRT14', 'KRT5', 'MMP1', 'RUNX1',
    'EGFR', 'PGR', 'MYLK'
]

Mast_Cells_genes = [
    'CD69', 'FAM107B', 'HDC', 'KIT', 'LIF', 'TPSAB1', 'CPA3', 'CTSG', 'CYP1A1', 'PDE4A'
]

Macrophages_2_genes = [
    'MPO', 'SMAP2', 'FCER1G', 'AIF1', 'CD68', 'TYROBP', 'ITGAM', 'CCL8', 'PDK4', 'IL2RA',
    'C1QA', 'CD14', 'MRC1', 'C1QC', 'CD163'
]

Macrophages_1_genes = [
    'C15orf48', 'APOBEC3A', 'FCER1A', 'ITGAX', 'APOC1', 'MMP12', 'CD1C', 'CLEC9A', 'LYZ', 'CX3CR1',
    'MNDA', 'LY86', 'HAVCR2', 'FCGR3A', 'FGL2', 'IGSF6', 'CD86'
]

LAMP3_DCs_genes = [
    'MAP3K8', 'PELI1', 'SERPINB9', 'CRHBP', 'FAM49A', 'PDCD1LG2', 'CD274', 'DUSP5', 'CCR7', 'CD80',
    'CD83', 'BASP1', 'VOPP1', 'LPXN'
]

IRF7_DCs_genes = [
    'CLECL1', 'ERN1', 'DERL3', 'SPIB', 'PTGDS', 'IL3RA', 'GZMB', 'PLD4', 'LILRA4', 'TCL1A',
    'CXCR4', 'GPR183', 'SELL', 'CD4', 'GLIPR1'
]

Invasive_Tumor_genes = [
    'ABCC11', 'SERHL2', 'TCIM'
]

Endothelial_genes = [
    'LDHB', 'WARS', 'BACE2', 'ZEB1', 'EDNRB', 'ANGPT2', 'CAV1', 'HOXD8', 'CAVIN2', 'EDN1',
    'RAPGEF3', 'SNAI1', 'EGFL7', 'NOSTRIN', 'AQP1', 'SOX18', 'CLDN5', 'BTNL9', 'VWF', 'MMRN2',
    'ESM1', 'CLEC14A', 'HOXD9', 'RAMP2', 'KDR', 'SOX17', 'CD93', 'PECAM1', 'PPARG', 'TCF15',
    'TCF4', 'AKR1C3'
]

DCIS_2_genes = [
    'AGR3', 'S100A14', 'KRT8', 'AQP3'
]

DCIS_1_genes = [
    'SERPINA3', 'ESR1', 'CEACAM6', 'CEACAM8', 'C6orf132', 'LYPD3', 'QARS', 'SEC24A', 'TPD52', 'HPX',
    'HOOK2', 'SMS', 'KLF5', 'TFAP2A', 'CD9', 'GATA3', 'FLNB', 'CLDN4', 'TRAPPC3', 'DNTTIP1',
    'SDC4', 'CTH'
]

CD8_T_Cells_genes = [
    'ADGRE5', 'GZMK', 'PRF1', 'CCL5', 'GZMA', 'CD8A', 'NKG7', 'CD8B', 'KLRC1', 'KLRD1',
    'DUSP2', 'GNLY', 'PTPRC', 'APOBEC3B', 'CYTIP', 'IL2RG', 'PDCD1', 'TIGIT', 'CD3G', 'TRAC',
    'CD3E', 'CD247', 'CD3D', 'KLRF1'
]

CD4_T_Cells_genes = [
    'LTB', 'SLAMF1', 'FOXP3', 'IL7R', 'CTLA4', 'KLRB1', 'CCL20', 'TCF7'
]

B_Cells_genes = [
    'ITM2C', 'CCPG1', 'TIFA', 'SEC11C', 'TENT5C', 'RAB30', 'MZB1', 'CD27', 'PRDM1', 'SLAMF7',
    'CD79B', 'MS4A1', 'CD19', 'CD79A', 'BANK1', 'TNFRSF17'
]

gene_sets_to_score = {
    'Stromal_score': Stromal_genes,
    'Prolif_Invasive_Tumor_score': Prolif_Invasive_Tumor_genes,
    'Perivascular_Like_score': Perivascular_Like_genes,
    'Myoepi_KRT15_score': Myoepi_KRT15_genes,
    'Myoepi_ACTA2_score': Myoepi_ACTA2_genes,
    'Mast_Cells_score': Mast_Cells_genes,
    'Macrophages_2_score': Macrophages_2_genes,
    'Macrophages_1_score': Macrophages_1_genes,
    'LAMP3_DCs_score': LAMP3_DCs_genes,
    'IRF7_DCs_score': IRF7_DCs_genes,
    'Invasive_Tumor_score': Invasive_Tumor_genes,
    'Endothelial_score': Endothelial_genes,
    'DCIS_2_score': DCIS_2_genes,
    'DCIS_1_score': DCIS_1_genes,
    'CD8_T_Cells_score': CD8_T_Cells_genes,
    'CD4_T_Cells_score': CD4_T_Cells_genes,
    'B_Cells_score': B_Cells_genes,
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
