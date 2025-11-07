#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch 反卷积原型的脚本化入口：
  - 从标准 npz（由 preprocessing.py 生成）读取数据
  - 构建图拉普拉斯 L 与 DeconvModel
  - 训练、导出 ctGP 宽表、保存损失曲线

本版去除了绝对路径，使用 argparse 参数化，名称顺序统一来自 npz。
"""

import argparse
from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """固定随机种子 / set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="CTGP deconvolution (PyTorch prototype)")
    parser.add_argument("--npz", type=Path, required=True, help="Input npz (spot_data_full.npz)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--k", type=int, default=6, help="kNN for Laplacian")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda1", type=float, default=1e-4)
    parser.add_argument("--lambda2", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true", help="Do not show plots (only save)")
    args = parser.parse_args()

    set_seed(args.seed)

    # 解决导入路径（相对本仓库结构）/ add package root to sys.path
    import sys
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "cell_program_deconvolution" / "src"))
    from cell_program_deconvolution.deconvolution.model import DeconvModel
    from cell_program_deconvolution.deconvolution.graph_utils import build_laplacian
    from cell_program_deconvolution.deconvolution.train import train_model
    from cell_program_deconvolution.deconvolution.visualize import plot_spatial, plot_program_contribution

    # Step 1: Load data
    data = np.load(args.npz, allow_pickle=True)
    Y = torch.tensor(data["visium_score"], dtype=torch.float32)                # (P, S)
    X = torch.tensor(data["spot_cluster_fraction_matrix"], dtype=torch.float32)  # (S, T)
    coords = data["coords"]                                                     # (S, 2)

    spot_names = data["spot_names"]
    celltype_names = data["celltype_names"]
    program_names = data["program_names"]

    P, S = Y.shape
    S_, T = X.shape
    assert S == S_, f"S mismatch: Y.shape={Y.shape}, X.shape={X.shape}"
    print(f"Loaded: S={S}, T={T}, P={P}")

    # Step 2: Build model
    L = build_laplacian(coords, k=args.k)
    model = DeconvModel(T=T, P=P, S=S, X_tensor=X, L=L)

    # Step 3: Train
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    history = train_model(model, Y_obs=Y, num_epochs=args.epochs, lr=args.lr,
                          lambda1=args.lambda1, lambda2=args.lambda2)

    # Step 4: Export Y_tps wide
    Y_tps_np = model.Y_tps.detach().cpu().numpy()  # (T, P, S)
    T_, P_, S_ = Y_tps_np.shape
    assert (T_, P_, S_) == (T, P, S)
    Y_tps_flat = np.transpose(Y_tps_np, (2, 0, 1)).reshape(S, T * P)
    columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]
    import pandas as pd
    df_wide = pd.DataFrame(Y_tps_flat, index=spot_names, columns=columns)
    csv_path = outdir / "ctgp_deconv(wide).csv"
    df_wide.to_csv(csv_path)
    print(f"✅ 导出宽表：{df_wide.shape} -> {csv_path}")

    # Step 5: Plots (optional show)
    import matplotlib.pyplot as plt
    _hist = np.array(history)
    total_loss = _hist[:, 0]
    recon_loss = _hist[:, 1]
    plt.figure(figsize=(7, 4))
    plt.plot(total_loss, label='Total Loss', linewidth=1.5)
    plt.plot(recon_loss, label='Reconstruction Loss', linewidth=1.5)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Curve')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    lc_path = outdir / "loss_curve.png"
    plt.savefig(lc_path, dpi=300)
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # 示例：单视图（避免阻塞，可只保存图）
    try:
        Y_tps = model.Y_tps.detach()
        plot_spatial(Y_tps, coords, cell_type=0, program_index=0)
        plot_program_contribution(Y_tps, program_index=0)
    except Exception as e:
        print(f"ℹ️ 跳过示例可视化：{e}")


if __name__ == "__main__":
    main()
