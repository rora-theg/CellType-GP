#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import random

# Step 0: 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"✅ 已设置随机种子: {seed}")


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
history = train_model(model, Y_obs=Y, num_epochs=3000)

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
program_names = np.array(['Prolif_Invasive_Tumor_score_norm',
                             'Invasive_Tumor_score_norm', 
                             'DCIS_2_score_norm', 'DCIS_1_score_norm'
                             ])
spot_names = data['spot_names']
# 1️⃣ 重新排列维度： (S, T, P)
Y_tps_reordered = np.transpose(Y_tps, (2, 0, 1))  # S x T x P

# 2️⃣ 展平每个 spot 的 (T×P) 向量
Y_tps_flat = Y_tps_reordered.reshape(S, T * P)

# 3️⃣构造 DataFrame
columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]
y_tps_matrix_df = pd.DataFrame(Y_tps_flat, index=spot_names, columns=columns)

# 4️⃣ 保存结果
output_path = '/home/vs_theg/ST_program/CellType_GP/DATA/train1500_result(wide).csv'
y_tps_matrix_df.to_csv(output_path, index=True)
print(f"✅ 已生成矩阵表：{y_tps_matrix_df.shape[0]} × {y_tps_matrix_df.shape[1]}")
print(f"👉 保存路径: {output_path}")


# Step 5: Plot loss curve
import matplotlib.pyplot as plt
import numpy as np

history = np.array(history)
total_loss = history[:, 0]
recon_loss = history[:, 1]

plt.figure(figsize=(7,4))
plt.plot(total_loss, label='Total Loss', linewidth=1.5)
plt.plot(recon_loss, label='Reconstruction Loss', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()