"""
celltype_gp_models.py
---------------------
模块功能：
    基于不同模型方法计算细胞类型特异性基因程序贡献 (Y_tps)
    并转换为与 truth_result 对应的宽格式结果表。

输入要求：
    npz 文件中包含：
        - visium_score: (P, S)
        - spot_cluster_fraction_matrix: (S, T)
        - coords: (S, 2)
        - spot_names, celltype_names, program_names
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# ====================================================
# 🧠 核心函数：不同计算方法
# ====================================================

# 由于留一法在数值上相减两次相近预测，误差被严重抵消，建议弃用 delta_loo 或改为相对差定义，主用 residual_vectorized 做 ctGP 分解。
# def delta_loo(Y, X):
#     """贡献差值法 (Y_tps = y_full - y_loo)"""
#     P, S = Y.shape
#     S2, T = X.shape
#     assert S == S2
#     Y_tps = torch.zeros((T, P, S))

#     X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-6)
#     ridge = Ridge(alpha=0.1)

#     for p in range(P):
#         y = Y[p, :].numpy()
#         ridge.fit(X_scaled.numpy(), y)
#         y_full = ridge.predict(X_scaled.numpy())
#         for t in range(T):
#             X_loo = np.delete(X_scaled.numpy(), t, axis=1)
#             ridge.fit(X_loo, y)
#             y_loo = ridge.predict(X_loo)
#             Y_tps[t, p, :] = torch.tensor(y_full - y_loo)
#     return Y_tps


def residual_vectorized(Y, X):
    P, S = Y.shape
    S2, T = X.shape
    assert S == S2
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-6)

    ridge = Ridge(alpha=0.1)
    ridge.fit(X_scaled, Y.T)   
    coefs = torch.tensor(ridge.coef_.T, dtype=torch.float32)  # (T, P)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    Y_tps = torch.empty((T, P, S))
    for t in range(T):
        mask = torch.ones(T, dtype=torch.bool); mask[t] = False
        X_other = X_scaled[:, mask]
        beta_other = coefs[mask, :]
        Y_tps[t] = (Y.T - X_other @ beta_other).T
    return Y_tps


# 空间依赖性建模法（取消）

# ====================================================
# 🧾 工具函数
# ====================================================

def Ytps_to_wide_df(Y_tps, spot_names, celltype_names, program_names):
    """把 Y_tps 转为 truth_result 样式的宽表格"""
    T, P, S = Y_tps.shape
    Y_tps_np = Y_tps.permute(2, 0, 1).reshape(S, T * P).numpy()
    columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]
    return pd.DataFrame(Y_tps_np, index=spot_names, columns=columns)


# ====================================================
# 🚀 主函数入口
# ====================================================

def run_model(npz_path, method="vectorized", save_path="train_result.csv"):
    data = np.load(npz_path, allow_pickle=True)
    Y = torch.tensor(data["visium_score"], dtype=torch.float32)  # (P, S)
    X = torch.tensor(data["spot_cluster_fraction_matrix"], dtype=torch.float32)  # (S, T)
    coords = data["coords"] # 若需空间依赖性建模，则因导入coords
    spot_names = data["spot_names"]
    celltype_names = data["celltype_names"]
    program_names = data["program_names"]

    if method == "delta":
        Y_tps = delta_loo(Y, X)
    elif method == "vectorized":
        Y_tps = residual_vectorized(Y, X)
    else:
        raise ValueError("method 必须是 ['delta', 'vectorized'] 之一")

    df = Ytps_to_wide_df(Y_tps, spot_names, celltype_names, program_names)
    df.to_csv(save_path)
    print(f"✅ 已保存结果到 {save_path}，形状 {df.shape}")
    return df



# ====================================================
# 🧭 命令行入口（可选）
# ====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run celltype-GP model.")
    parser.add_argument("--input", type=str, required=True, help="Path to npz file")
    parser.add_argument("--method", type=str, default="vectorized", help="Method name")
    parser.add_argument("--save", type=str, default="train_result.csv", help="Output path")
    args = parser.parse_args()

    run_model(args.input, args.method, args.save)

