import numpy as np

# 设置随机种子确保可复现
np.random.seed(42)

# 基本参数
P = 5   # 程序数量
S = 100  # spot 数量
T = 4   # 细胞类型数量

# 模拟坐标：在二维空间中有一定结构，比如两个区域聚类
coords = np.zeros((S, 2))
coords[:S//2] = np.random.normal(loc=[30, 30], scale=5, size=(S//2, 2))  # 区域1
coords[S//2:] = np.random.normal(loc=[70, 70], scale=5, size=(S//2, 2))  # 区域2

# 模拟细胞类型比例矩阵 X (S, T)，每个 spot 的细胞组成有空间依赖
X = np.zeros((S, T))
for i in range(S):
    if coords[i, 0] < 50:
        X[i] = np.random.dirichlet([0.7, 0.2, 0.05, 0.05])
    else:
        X[i] = np.random.dirichlet([0.05, 0.1, 0.6, 0.25])

# 设定每种细胞类型对每个程序的典型激活水平 (T, P)
Y_profile = np.array([
    [2.0, 1.5, 0.2, 0.1, 0.1],  # cell type 0
    [0.1, 0.1, 1.8, 0.2, 0.2],  # cell type 1
    [0.1, 0.1, 0.2, 2.0, 1.5],  # cell type 2
    [0.2, 0.3, 0.3, 0.3, 0.3]   # cell type 3
])

# 构造 Y = X @ Y_profile + 噪声
Y = X @ Y_profile  # (S, T) × (T, P) = (S, P)
Y += np.random.normal(scale=0.1, size=Y.shape)
Y = Y.T  # 转置为 (P, S)

# 保存为新的示例文件
save_path = "/mnt/data/structured_inputs.npz"
np.savez(save_path, Y=Y, X=X, coords=coords)

save_path