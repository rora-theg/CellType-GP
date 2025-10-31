"""
celltype_gp_models.py
---------------------
模块功能：
    基于不同模型方法计算细胞类型特异性基因程序贡献 (Y_tps)
    并转换为与 truth_result 对应的宽格式结果表。

Inputs:
    npz 文件需包含：
        - visium_score: (P, S) 程序打分矩阵（每列为一个 spot）
        - spot_cluster_fraction_matrix: (S, T) 细胞类型比例矩阵
        - coords: (S, 2) 空间坐标（本文件当前未使用）
        - spot_names, celltype_names, program_names

Method notes / 方法说明：
    - "vectorized" (默认)：加性贡献分解（Additive contribution）。
        先用 Ridge 回归拟合 Y ≈ X·β；随后将预测值按细胞类型分解为
        对每个 t 的贡献项：contrib_t(s,p) = X_s,t · β_t,p。
        该分解与线性模型可加性一致、稳定、向量化高效。

    - "lofo_refit"：逐细胞类型 LOFO 重拟合（Leave-one-feature-out）。
        对每个 t，基于 X_-t 拟合 β_-t 并计算残差 Y - X_-t·β_-t。
        该方法更贴近“剔除 t 后剩余解释力”，但计算开销为 O(T)。

"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge



# ====================================================
# 🧠 核心函数：不同计算方法
# ====================================================

def _standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    列标准化（zero-mean, unit-variance），返回 (X_std, mean, std)。
    Column-wise standardization for stable Ridge fitting.
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sigma, mu, sigma

def clr_transform(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对组合数据矩阵做 CLR 变换。"""
    X_safe = np.clip(X, eps, None)
    geom_mean = np.exp(np.mean(np.log(X_safe), axis=1, keepdims=True))
    return np.log(X_safe / geom_mean)

def contribution_vectorized(Y: np.ndarray, X: np.ndarray, alpha: float = 0.1, 
                            use_clr: bool = False,eps: float = 1e-6,
                            standardize: bool = False,
                            positive: bool = True) -> np.ndarray:
    """
    向量化加性贡献分解（默认方法）
    Vectorized additive contribution decomposition.

    参数 / Args:
        Y: (P, S) 程序得分矩阵，每列一个 spot。
        X: (S, T) 细胞类型比例矩阵。
        alpha: Ridge 正则系数。
        use_clr: 是否对 X 做 CLR 变换（建议 True），消除细胞比例矩阵的完全共线性问题。
        eps: CLR 变换中的最小值截断，防止 log(0）。
        standardize: 是否对 X 做列标准化（建议 True）。
        positive: 是否对 Ridge 回归的系数施加非负约束。

    返回 / Returns:
        Y_tps: (T, P, S)，每个细胞类型 t 对 (程序 p, spot s) 的加性贡献。

    说明 / Notes:
        先拟合 Y ≈ X·β（multi-target Ridge），随后按 t 将预测分解为贡献项：
        contrib[t, p, s] = X_std[s, t] * β[t, p]；所有 t 的 contrib 求和即为 Y_hat。
        该分解与线性模型可加性相符，比 LOFO 残差更稳定且易向量化。

        如果后续需要对结果进行解释，需将CLR空间的β系数反映射回比例空间。
    """
    P, S = Y.shape
    S2, T = X.shape
    assert S == S2, f"S mismatch: Y(P,S)={Y.shape} vs X(S,T)={X.shape}"

    X_in = X.copy()
    if use_clr:
        X_in = clr_transform(X_in,eps)
    if standardize:
        X_in, _, _ = _standardize_columns(X_in)


    # sklearn 期望形状：fit(X: (S,T), Y: (S,P))
    ridge = Ridge(alpha=alpha, positive=positive)
    ridge.fit(X_in, Y.T)
    # sklearn.coef_ 形状为 (n_targets=P, n_features=T)
    beta = ridge.coef_.T  # (T, P)


    # 贡献分解：对每个 t，contrib_t(s,p) = X_in[s,t] * beta[t,p]
    # 通过爱因斯坦求和向量化：
    # 输入 (S,T) 与 (T,P)，输出 (T,P,S)
    contrib = np.einsum('st,tp->tps', X_in, beta, optimize=True)
    return contrib.astype(np.float32, copy=False)


def lofo_refit_residual(Y: np.ndarray, X: np.ndarray, alpha: float = 0.1, 
                        use_clr: bool = False,eps: float = 1e-6,
                        standardize: bool = False) -> np.ndarray:
    """
    逐 t LOFO 重拟合并计算残差：Y_tps[t,:,:] = Y - X_-t·β_-t。
    支持 CLR 与标准化。
    True LOFO residual per feature (slower, O(T) fits)).
    """
    P, S = Y.shape
    S2, T = X.shape
    assert S == S2, f"S mismatch: Y(P,S)={Y.shape} vs X(S,T)={X.shape}"

    Y_tps = np.empty((T, P, S), dtype=np.float32)
    for t in range(T):
        mask = np.ones(T, dtype=bool)
        mask[t] = False
        X_other = X[:, mask]
        Xin = X_other


        if use_clr:
            Xin = clr_transform(Xin, eps)
        if standardize:
            Xin, _, _ = _standardize_columns(Xin)
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xin, Y.T)                     # (S, T-1) -> (S, P)
        Y_hat_other = ridge.predict(Xin).T      # -> (P, S)
        Y_tps[t] = (Y - Y_hat_other).astype(np.float32)
    return Y_tps


# 空间依赖性建模法（取消）

# ====================================================
# 🧾 工具函数
# ====================================================

def Ytps_to_wide_df(Y_tps, spot_names, celltype_names, program_names):
    """
    把 Y_tps (T,P,S) 转为 truth_result 样式的宽表格（S, T*P）。
    Tolerates numpy 或 torch 输入，输出为 pandas.DataFrame。
    """
    # 允许 torch/numpy 两种输入
    if isinstance(Y_tps, torch.Tensor):
        T, P, S = Y_tps.shape
        Y_np = Y_tps.permute(2, 0, 1).detach().cpu().numpy().reshape(S, T * P)
    else:
        Y_np = np.asarray(Y_tps)
        T, P, S = Y_np.shape
        Y_np = np.transpose(Y_np, (2, 0, 1)).reshape(S, T * P)

    columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]
    return pd.DataFrame(Y_np, index=spot_names, columns=columns)


# ====================================================
# 🚀 主函数入口
# ====================================================

def run_model(npz_path, method: str = "vectorized", save_path: str = "train_result.csv",
              alpha: float = 0.1, standardize: bool = False , use_clr: bool = False,
              positive: bool = True) -> pd.DataFrame:
    """
    运行 ctGP 分解，并导出宽表。

    参数 / Args:
        npz_path: 输入 npz 路径，需含 visium_score/X/名字等字段。
        method: 'vectorized'（默认，可解释加性分解）或 'lofo_refit'（逐 t 重拟合残差）。
        save_path: 输出 CSV 路径。
        alpha: Ridge 正则系数。
        standardize: 是否标准化 X 的列。
        use_clr：是否对X做CLR变换。
    """
    data = np.load(npz_path, allow_pickle=True)
    Y = data["visium_score"].astype(np.float64)                 # (P, S)
    X = data["spot_cluster_fraction_matrix"].astype(np.float64)  # (S, T)
    # coords 目前未使用；如扩展空间项，可在此传入
    # coords = data.get("coords")
    spot_names = data["spot_names"]
    celltype_names = data["celltype_names"]
    program_names = data["program_names"]

    if method == "vectorized":
        Y_tps = contribution_vectorized(Y, X, alpha=alpha, standardize=standardize, use_clr=use_clr, eps=1e-6,positive=positive)
    elif method == "lofo_refit":
        Y_tps = lofo_refit_residual(Y, X, alpha=alpha, standardize=standardize, use_clr=use_clr, eps=1e-6,positive=positive)
    else:
        raise ValueError("method 必须是 ['vectorized', 'lofo_refit'] 之一")

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
    parser.add_argument("--method", type=str, default="vectorized", choices=["vectorized", "lofo_refit"], help="Method name")
    parser.add_argument("--save", type=str, default="train_result.csv", help="Output path")
    parser.add_argument("--alpha", type=float, default=0.1, help="Ridge regularization strength")
    parser.add_argument("--no-standardize", action="store_true", help="Disable column standardization for X")
    args = parser.parse_args()
    run_model(args.input, args.method, args.save, alpha=args.alpha, standardize=not args.no_standardize)
