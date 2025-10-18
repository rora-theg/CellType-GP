import matplotlib.pyplot as plt
import torch

def plot_spatial(Y_tps, coords, cell_type, program_index, spot_size=10):
    values = Y_tps[cell_type, program_index, :].detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=spot_size)
    plt.title(f"Program {program_index} in Cell Type {cell_type}")
    plt.colorbar()
    plt.show()

def plot_program_contribution(Y_tps, program_index):
    total_contrib = Y_tps[:, program_index, :].sum(dim=1).detach().cpu().numpy()
    plt.bar(range(len(total_contrib)), total_contrib)
    plt.title(f"Program {program_index} Cell Type Contributions")
    plt.xlabel("Cell Type")
    plt.ylabel("Total Contribution")
    plt.show()
