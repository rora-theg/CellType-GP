#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys
sys.path.append("/home/vs_theg/ST_program/cell_program_deconvolution/cell_program_deconvolution/")  # 添加上级目录到Python路径
from deconvolution.model import DeconvModel
from deconvolution.graph_utils import build_laplacian
from deconvolution.train import train_model
from deconvolution.visualize import plot_spatial, plot_program_contribution


# Step 1: Load data
data = np.load("/home/vs_theg/ST_program/CellType_GP/DATA/spot_data_full.npz", allow_pickle=True)
Y = torch.tensor(data["visium_score"], dtype=torch.float32)         # (P, S)
X = torch.tensor(data["spot_cluster_fraction_matrix"], dtype=torch.float32)         # (S, T)
coords = data["coords"]                                  # (S, 2)

spot_names = data['spot_names']
celltype_names = data['celltype_names']
program_names = data['program_names']

# 查看数据前几行
print("Y (shape: {}):".format(Y.shape))
print(Y[:5, :5])  # 查看前5行前5列
print("\nX (shape: {}):".format(X.shape))
print(X[:5, :5])  # 查看前5行前5列
print("\ncoords (shape: {}):".format(coords.shape))
print(coords[:5])  # 查看前5行

P, S = Y.shape
S_, T = X.shape
assert S == S_

L = build_laplacian(coords, k=6)
model = DeconvModel(T=T, P=P, S=S, X_tensor=X, L=L)


# Step 2: Train model
train_model(model, Y_obs=Y, num_epochs=1500, lambda1=1e-4, lambda2=1e-4)             

# Step 3: Visualize results
Y_tps = model.Y_tps.detach()
plot_spatial(Y_tps, coords, cell_type=0, program_index=0)
plot_program_contribution(Y_tps, program_index=0)

import numpy as np
import pandas as pd


# Step 4: Convert Y_tps to wide DataFrame and save
# 假设 Y_tps 形状为 (T, P, S)
Y_tps = model.Y_tps.detach().cpu().numpy()
T, P, S = Y_tps.shape

# 确保索引
celltype_names = np.array(['B_Cells', 'DCIS', 'Endothelial', 'Invasive_Tumor',
                           'Myeloid', 'Stromal', 'T_cells'])
program_names = np.array(['Stromal_score_norm', 'Prolif_Invasive_Tumor_score_norm',
                           'Perivascular_Like_score_norm', 'Myoepi_KRT15_score_norm',
                             'Myoepi_ACTA2_score_norm', 'Mast_Cells_score_norm', 
                             'Macrophages_2_score_norm', 'Macrophages_1_score_norm', 
                             'LAMP3_DCs_score_norm', 'IRF7_DCs_score_norm', 
                             'Invasive_Tumor_score_norm', 'Endothelial_score_norm', 
                             'DCIS_2_score_norm', 'DCIS_1_score_norm', 
                             'CD8_T_Cells_score_norm', 'CD4_T_Cells_score_norm', 
                             'B_Cells_score_norm'])
spot_names = data['spot_names']
# 1️⃣ 重新排列维度： (S, T, P)
Y_tps_reordered = np.transpose(Y_tps, (2, 0, 1))  # S x T x P

# 2️⃣ 展平每个 spot 的 (T×P) 向量
Y_tps_flat = Y_tps_reordered.reshape(S, T * P)

# 3️⃣ 生成列名
columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]

# 4️⃣ 构造 DataFrame
y_tps_matrix_df = pd.DataFrame(Y_tps_flat, index=spot_names, columns=columns)

# 5️⃣ 保存结果
y_tps_matrix_df.to_csv("Y_tps_result.csv")
print("✅ 已生成矩阵表：3953 × (7×17)")
print(y_tps_matrix_df.shape)
y_tps_matrix_df.to_csv('/home/vs_theg/ST_program/CellType_GP/DATA/train1500_result(wide).csv')