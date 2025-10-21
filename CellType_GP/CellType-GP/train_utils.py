#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_utils.py
----------------
用于 CellType-GP 模型的通用训练工具模块。
包含：
    1. train_model() —— 主训练函数
    2. 自动 early stopping
    3. 学习率动态调整 (ReduceLROnPlateau)
    4. 训练曲线可视化与日志保存

版本: v1.0
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange


def train_model_2(
    model,
    Y_obs,
    num_epochs=3000,
    lr=1e-3,
    lambda1=1e-4,
    lambda2=1e-2,
    early_stop=True,
    patience=300,
    tol=1e-5,
    verbose=True,
    save_dir="./training_logs",
):
    """
    =============================================
    主训练函数（带早停与学习率调度）
    =============================================

    参数：
    ----------
    model : torch.nn.Module
        已定义好的深度学习模型，必须实现 model.loss(Y_obs, lambda1, lambda2) 接口，
        返回 (loss, recon_loss)。

    Y_obs : torch.Tensor
        观测数据张量，形状通常为 (spots × gene_programs)。

    num_epochs : int, 默认 3000
        训练轮次（最大迭代次数）。

    lr : float, 默认 1e-3
        初始学习率。

    lambda1 : float, 默认 1e-4
        稀疏约束（L1）正则项权重。

    lambda2 : float, 默认 1e-2
        空间平滑（Laplacian）正则项权重。

    early_stop : bool, 默认 True
        是否启用早停机制（loss 长期无改善自动停止）。

    patience : int, 默认 300
        早停容忍次数，即连续多少个 epoch 没有改善后停止。

    tol : float, 默认 1e-5
        改善阈值，小于该值的波动视为“无提升”。

    verbose : bool, 默认 True
        是否打印训练进度条与损失信息。

    save_dir : str, 默认 "./training_logs"
        保存日志与曲线图的目录。

    返回：
    ----------
    history : dict
        包含训练历史的字典：
            - 'train_loss': 总损失（含正则）
            - 'recon_loss': 重构误差
            - 'lr': 学习率变化
            - 'best_loss': 最优 loss 值
            - 'best_epoch': 对应 epoch
    """

    # ========== 初始化 ==========
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    # 每次反向传播后，用 Adam 算法、学习率 = lr，来调整 model 的参数

    # 学习率调度器：当验证集的 loss 长期不改善时，自动将学习率减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 4
    )

    # 记录训练过程
    history = {"train_loss": [], "recon_loss": [], "lr": []}

    best_loss = np.inf  # 最优loss初始化为无穷大
    best_epoch = 0      # 记录最优epoch
    no_improve = 0      # 无提升计数器

    # tqdm 进度条
    pbar = trange(num_epochs, disable=not verbose, ncols=100)

    # ========== 主训练循环 ==========
    for epoch in pbar:
        # 1. 前向传播 + 计算loss
        loss, recon = model.loss(Y_obs, lambda1, lambda2)

        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. 记录当前学习率与loss
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(loss.item())
        history["recon_loss"].append(recon.item())
        history["lr"].append(current_lr)

        # 4. 更新学习率调度器
        scheduler.step(loss.item())

        # 5. 进度条显示
        if verbose:
            pbar.set_description(
                f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Recon {recon.item():.4f} | LR {current_lr:.2e}"
            )

        # 6. 早停逻辑
        if early_stop:
            if loss.item() < best_loss - tol:
                best_loss = loss.item()
                best_epoch = epoch
                no_improve = 0
                # 保存当前模型参数
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            else:
                no_improve += 1

            if no_improve > patience:
                if verbose:
                    print(f"\n🔹 早停触发：在第 {epoch} 轮停止训练，最优loss={best_loss:.6f}")
                break

    # ========== 训练结束后处理 ==========
    history = {k: np.array(v) for k, v in history.items()}
    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch

    # 保存loss曲线
    fig_path = os.path.join(save_dir, "loss_curve.png")
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Total Loss", color="tab:blue")
    plt.plot(history["recon_loss"], label="Reconstruction Loss", color="tab:orange")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 保存日志文件
    log_path = os.path.join(save_dir, "train_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("====== 训练参数与结果日志 ======\n")
        f.write(f"学习率初始值: {lr}\n")
        f.write(f"λ1 (稀疏正则): {lambda1}\n")
        f.write(f"λ2 (空间平滑正则): {lambda2}\n")
        f.write(f"最大轮次: {num_epochs}\n")
        f.write(f"早停阈值: {tol}, 容忍次数: {patience}\n")
        f.write(f"最优Epoch: {best_epoch}\n")
        f.write(f"最优Loss: {best_loss:.6f}\n")
        f.write(f"最终学习率: {history['lr'][-1]:.6e}\n")
        f.write("\n====== 历史Loss前5条记录 ======\n")
        for i in range(min(5, len(history['train_loss']))):
            f.write(f"Epoch {i}: loss={history['train_loss'][i]:.6f}\n")
        f.write("\n训练完成！\n")

    if verbose:
        print(f"\n✅ 训练完成：最优loss={best_loss:.6f}，最优epoch={best_epoch}")
        print(f"📉 Loss曲线已保存至: {fig_path}")
        print(f"📝 训练日志已保存至: {log_path}")

    return history


# ==========
