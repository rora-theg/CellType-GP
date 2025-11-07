import numpy as np
import pandas as pd
import os

base = "/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/"

# 读取主要文件
pseudo_data = pd.read_csv(os.path.join(base, "pseudo_data.csv"), index_col=0)
true_p = pd.read_csv(os.path.join(base, "true_p.csv"), index_col=0)
sample_random = pd.read_csv(os.path.join(base, "sample_random.csv"), index_col=0)
mapping = pd.read_csv(os.path.join(base, "cell_to_spot_mapping.csv"))

# 打印维度检查
print("✅ pseudo_data:", pseudo_data.shape)
print("✅ true_p:", true_p.shape)
print("✅ sample_random:", sample_random.shape)
print("✅ mapping:", mapping.shape)

# 转成 numpy 格式并保存为 npz
np.savez_compressed(
    os.path.join(base, "sim_sc2sp_dataset.npz"),
    pseudo_data=pseudo_data.values,
    pseudo_data_genes=pseudo_data.index.to_numpy(),
    pseudo_data_spots=pseudo_data.columns.to_numpy(),
    true_p=true_p.values,
    true_p_spots=true_p.index.to_numpy(),
    sample_random=sample_random.values,
    sample_random_spots=sample_random.index.to_numpy(),
    mapping_cell=mapping["cell_id"].to_numpy(),
    mapping_spot=mapping["spot_id"].to_numpy(),
    mapping_celltype=mapping["celltype"].to_numpy()
)

print("🎯 Saved dataset to sim_sc2sp_dataset.npz ✅")
