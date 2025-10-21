import matplotlib.pyplot as plt
import torch

def plot_spatial(Y_tps, coords, cell_type, program_index, spot_size=10):
    """绘制指定细胞类型中某个程序在空间位置上的活性分布。"""
    # 先切片出目标细胞类型与程序后再移至 CPU 并转换为 NumPy，方便 matplotlib 绘图。
    values = Y_tps[cell_type, program_index, :].detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=spot_size)
    plt.title(f"Program {program_index} in Cell Type {cell_type}")
    plt.colorbar()
    plt.show()

def plot_program_contribution(Y_tps, program_index):
    """统计并展示各细胞类型对指定程序的总贡献。"""
    # 在空间维度求和，以比较不同细胞类型对该程序的整体贡献。
    total_contrib = Y_tps[:, program_index, :].sum(dim=1).detach().cpu().numpy()
    plt.bar(range(len(total_contrib)), total_contrib)
    plt.title(f"Program {program_index} Cell Type Contributions")
    plt.xlabel("Cell Type")
    plt.ylabel("Total Contribution")
    plt.show()
